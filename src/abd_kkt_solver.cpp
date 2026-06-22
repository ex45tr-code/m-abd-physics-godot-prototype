#include "abd_kkt_solver.h"
#include "ipc_physics_server.h"
#include "abd_joint.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <queue>
#include <set>

// ════════════════════════════════════════════════════════════════════════════
//  abd_kkt_solver.cpp  —  M-ABD Section 5: Dual-space KKT
//
//  전체 흐름:
//  1. build_constraint_blocks()  : ConstraintBlock 배열 구성
//  2. compute_constraint_gradient() per block  : ∇C 계산 (Eq.30-35)
//  3. precompute_dual()  : H^{-1}∇C^T, D, b 계산
//  4. topology 판별 → solve_chain / solve_tree / solve_loop / solve_graph
//  5. apply_dual_correction() : δq -= H^{-1}∇C^T δλ
//
// ════════════════════════════════════════════════════════════════════════════

namespace godot {

static inline float absf_(float a) { return a < 0.f ? -a : a; }
static inline float sqrtf_s(float v) { return v <= 0.f ? 0.f : sqrtf(v); }

// ════════════════════════════════════════════════════════════════════════════
//  DynMat
// ════════════════════════════════════════════════════════════════════════════

bool DynMat::cholesky_factor() {
    for (int i = 0; i < rows; ++i) {
        float sum = v[i*cols+i];
        for (int k = 0; k < i; ++k) sum -= v[i*cols+k]*v[i*cols+k];
        if (sum <= 0.f) sum = 1e-12f;
        float L_ii = sqrtf(sum);
        v[i*cols+i] = L_ii;
        float inv = 1.f/L_ii;
        for (int j = i+1; j < rows; ++j) {
            float s2 = v[j*cols+i];
            for (int k = 0; k < i; ++k) s2 -= v[j*cols+k]*v[i*cols+k];
            v[j*cols+i] = s2*inv;
        }
    }
    return true;
}

void DynMat::cholesky_solve(std::vector<float> &b) const {
    for (int i = 0; i < rows; ++i) {
        float s = b[i];
        for (int k = 0; k < i; ++k) s -= v[i*cols+k]*b[k];
        b[i] = s/v[i*cols+i];
    }
    for (int i = rows-1; i >= 0; --i) {
        float s = b[i];
        for (int k = i+1; k < rows; ++k) s -= v[k*cols+i]*b[k];
        b[i] = s/v[i*cols+i];
    }
}

bool DynMat::lu_factor(std::vector<int> &piv) {
    piv.resize(rows);
    std::iota(piv.begin(), piv.end(), 0);
    for (int i = 0; i < rows; ++i) {
        int max_r = i;
        float max_v = absf_(v[i*cols+i]);
        for (int r = i+1; r < rows; ++r) {
            if (absf_(v[r*cols+i]) > max_v) { max_v = absf_(v[r*cols+i]); max_r = r; }
        }
        if (max_r != i) {
            std::swap(piv[i], piv[max_r]);
            for (int c = 0; c < cols; ++c) std::swap(v[i*cols+c], v[max_r*cols+c]);
        }
        if (absf_(v[i*cols+i]) < 1e-14f) v[i*cols+i] = 1e-14f; // regularize
        float inv = 1.f/v[i*cols+i];
        for (int r = i+1; r < rows; ++r) {
            v[r*cols+i] *= inv;
            for (int c = i+1; c < cols; ++c)
                v[r*cols+c] -= v[r*cols+i]*v[i*cols+c];
        }
    }
    return true;
}

void DynMat::lu_solve(const std::vector<int> &piv, std::vector<float> &b) const {
    std::vector<float> pb(rows);
    for (int i = 0; i < rows; ++i) pb[i] = b[piv[i]];
    for (int i = 0; i < rows; ++i) {
        float s = pb[i];
        for (int k = 0; k < i; ++k) s -= v[i*cols+k]*pb[k];
        pb[i] = s;
    }
    for (int i = rows-1; i >= 0; --i) {
        float s = pb[i];
        for (int k = i+1; k < rows; ++k) s -= v[i*cols+k]*pb[k];
        pb[i] = s/v[i*cols+i];
    }
    b = pb;
}

void DynMat::add(const DynMat &B) {
    for (int i = 0; i < (int)v.size(); ++i) v[i] += B.v[i];
}

DynMat DynMat::mul(const DynMat &A, const DynMat &B) {
    DynMat C(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i)
        for (int k = 0; k < A.cols; ++k) {
            float aik = A.v[i*A.cols+k];
            if (aik == 0.f) continue;
            for (int j = 0; j < B.cols; ++j)
                C.v[i*B.cols+j] += aik * B.v[k*B.cols+j];
        }
    return C;
}

DynMat DynMat::transpose() const {
    DynMat T(cols, rows);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            T.v[j*rows+i] = v[i*cols+j];
    return T;
}

// ════════════════════════════════════════════════════════════════════════════
//  Section 5.1: Constraint gradient helpers
// ════════════════════════════════════════════════════════════════════════════

// add one constraint row l to grad (C_k × 12):
//   ∂C[l]/∂q^{j*3+i} = R[row_comp, i] * cp_rest[j]   for j=0,1,2  (A-block)
//   ∂C[l]/∂q^{9+i}   = R[row_comp, i]                              (t-block)
static void add_grad_row(DynMat &grad, int l,
                          const RotMat3 &R, int row_comp,
                          const Vector3 &cp_rest, float sign)
{
    // A-block: j=0,1,2
    for (int j = 0; j < 3; ++j) {
        float yrj = (j==0)?cp_rest.x : (j==1)?cp_rest.y : cp_rest.z;
        for (int i = 0; i < 3; ++i)
            grad(l, j*3+i) += sign * R.m[row_comp][i] * yrj;
    }
    // t-block
    for (int i = 0; i < 3; ++i)
        grad(l, 9+i) += sign * R.m[row_comp][i];
}

// Eq.35 skew-symmetrize correction: modifies A-block rows
// (only the 9×9 A-sub-matrix of the grad row)
static void skew_symmetrize_row(DynMat &grad, int l, float sign_factor) {
    // The A-block of grad[l] is a 1×9 vector interpreted as flattened 3×3.
    // Skew part: G = (G - G^T)/2
    // col layout: q[0..2]=a1, q[3..5]=a2, q[6..8]=a3
    // 3×3 view: M[i][j] = grad(l, j*3+i)
    float M[3][3];
    for (int j=0;j<3;++j) for (int i=0;i<3;++i) M[i][j]=grad(l,j*3+i);
    float S[3][3];
    for (int i=0;i<3;++i) for (int j=0;j<3;++j)
        S[i][j] = sign_factor * 0.5f*(M[i][j]-M[j][i]);
    for (int j=0;j<3;++j) for (int i=0;i<3;++i)
        grad(l,j*3+i) = S[i][j];
}

// ════════════════════════════════════════════════════════════════════════════
//  compute_constraint_gradient  (Eq.30-35)
// ════════════════════════════════════════════════════════════════════════════

void ABDKKTSolver::compute_constraint_gradient(
    const ABDBody &ba, const ABDBody &bb,
    const ABDJoint &joint,
    ConstraintBlock &out)
{
    int c_dim = 0;
    switch (joint.type) {
        case ABDJointType::Ball:       c_dim = 3; break;
        case ABDJointType::Hinge:      c_dim = 5; break;
        case ABDJointType::Universal:  c_dim = 4; break;
        case ABDJointType::Prismatic:  c_dim = 5; break;
    }
    out.c_dim      = c_dim;
    out.body_alpha = joint.body_alpha;
    out.body_beta  = joint.body_beta;

    out.grad_alpha = DynMat(c_dim, 12);
    out.grad_beta  = DynMat(c_dim, 12);
    out.C_val.assign(c_dim, 0.f);

    float ya[12], yb[12];
    abd_compute_y(ba, ya);
    abd_compute_y(bb, yb);

    RotMat3 R = joint.build_R_joint(ba);

    joint.compute_C(ba.q.data(), bb.q.data(), ya, yb, R, out.C_val.data());

    switch (joint.type) {

    // ── Ball (Eq.19): C = y^α_{pa} - y^β_{pb}, R=I ──────────────────────
    case ABDJointType::Ball: {
        RotMat3 I = RotMat3::identity();
        for (int comp = 0; comp < 3; ++comp) {
            add_grad_row(out.grad_alpha, comp, I, comp, ba.cp_rest[joint.cp_alpha],  1.f);
            add_grad_row(out.grad_beta,  comp, I, comp, bb.cp_rest[joint.cp_beta],  -1.f);
        }
        break;
    }

    // ── Hinge (Eq.21, Eq.31-35) ─────────────────────────────────────────
    // 5 constraints: CP1 x,z  + CP2 x,z  + CP1 y (translation along axis)
    case ABDJointType::Hinge: {
        struct Row { int cp_a; int cp_b; int comp; } rows[5] = {
            {joint.cp_alpha_1, joint.cp_beta_1, 0},
            {joint.cp_alpha_1, joint.cp_beta_1, 2},
            {joint.cp_alpha_2, joint.cp_beta_2, 0},
            {joint.cp_alpha_2, joint.cp_beta_2, 2},
            {joint.cp_alpha_1, joint.cp_beta_1, 1},
        };
        for (int l = 0; l < 5; ++l) {
            add_grad_row(out.grad_alpha, l, R, rows[l].comp, ba.cp_rest[rows[l].cp_a],  1.f);
            add_grad_row(out.grad_beta,  l, R, rows[l].comp, bb.cp_rest[rows[l].cp_b], -1.f);
        }
        for (int l = 0; l < 5; ++l) {
            skew_symmetrize_row(out.grad_alpha, l,  1.f);
            skew_symmetrize_row(out.grad_beta,  l, -1.f);
        }
        break;
    }

    // ── Universal (Eq.23) ────────────────────────────────────────────────
    // C[0..2]: ball on CP1  +  C[3]: ỹ^β_2[y]=0
    case ABDJointType::Universal: {
        for (int comp = 0; comp < 3; ++comp) {
            add_grad_row(out.grad_alpha, comp, R, comp, ba.cp_rest[joint.cp_alpha],  1.f);
            add_grad_row(out.grad_beta,  comp, R, comp, bb.cp_rest[joint.cp_beta],  -1.f);
        }
        add_grad_row(out.grad_beta, 3, R, 1, bb.cp_rest[joint.cp_beta_2], -1.f);
        break;
    }

    // ── Prismatic (Eq.25) ────────────────────────────────────────────────
    case ABDJointType::Prismatic: {
        struct Row { int cp_a; int cp_b; int comp; } rows[5] = {
            {joint.cp_alpha_1, joint.cp_beta_1, 0},
            {joint.cp_alpha_1, joint.cp_beta_1, 2},
            {joint.cp_alpha_2, joint.cp_beta_2, 0},
            {joint.cp_alpha_2, joint.cp_beta_2, 2},
            {joint.cp_alpha,   joint.cp_beta,   0},
        };
        for (int l = 0; l < 5; ++l) {
            add_grad_row(out.grad_alpha, l, R, rows[l].comp, ba.cp_rest[rows[l].cp_a],  1.f);
            add_grad_row(out.grad_beta,  l, R, rows[l].comp, bb.cp_rest[rows[l].cp_b], -1.f);
        }
        for (int l = 0; l < 5; ++l) {
            skew_symmetrize_row(out.grad_alpha, l,  1.f);
            skew_symmetrize_row(out.grad_beta,  l, -1.f);
        }
        break;
    }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  unconstrained_dq  —  Algorithm 1: δq = H̄_A^{-1} f_A
//
//  논문 Algorithm 1 (p.5):
//  Lines 4-8:   f̃_A  = diag4(A^T) f_A     (rotate force into body frame)
//  Lines 10-11: δ̃q   = H̄_A^{-1} f̃_A     (solve in body frame)
//  Lines 12-14: δq   = diag4(A) δ̃q        (rotate result back to world)
//
//  BUGFIX: 이전 구현은 diag4(A^T) 적용 후 magnitude를 보존하는 잘못된
//  scale 계산을 했음. 올바른 구현은 단순히 A^T 곱 → solve → A 곱.
// ════════════════════════════════════════════════════════════════════════════

Vec12 ABDKKTSolver::unconstrained_dq(const ABDBody &b, const Vec12 &f_A) {
    float A[3][3]; b.get_A(A);

    // Step 1: f̃_A = diag4(A^T) f_A
    // 각 3-block k: f̃_k = A^T f_k
    Vec12 f_tilde;
    for (int k = 0; k < 4; ++k) {
        int off = k*3;
        float fx = f_A[off], fy = f_A[off+1], fz = f_A[off+2];
        // A^T * [fx,fy,fz]:  A^T[i][j] = A[j][i]
        f_tilde[off+0] = A[0][0]*fx + A[1][0]*fy + A[2][0]*fz;
        f_tilde[off+1] = A[0][1]*fx + A[1][1]*fy + A[2][1]*fz;
        f_tilde[off+2] = A[0][2]*fx + A[1][2]*fy + A[2][2]*fz;
    }

    // Step 2: δ̃q = H̄_A^{-1} f̃_A  (H̄_A is block-diagonal, Cholesky-factored)
    float rhs[12];
    for (int i = 0; i < 12; ++i) rhs[i] = f_tilde[i];
    b.H_bar_A.cholesky_solve(rhs);

    // Step 3: δq = diag4(A) δ̃q
    // 각 3-block k: δq_k = A * δ̃q_k
    Vec12 dq;
    for (int k = 0; k < 4; ++k) {
        int off = k*3;
        float dx = rhs[off], dy = rhs[off+1], dz = rhs[off+2];
        dq[off+0] = A[0][0]*dx + A[0][1]*dy + A[0][2]*dz;
        dq[off+1] = A[1][0]*dx + A[1][1]*dy + A[1][2]*dz;
        dq[off+2] = A[2][0]*dx + A[2][1]*dy + A[2][2]*dz;
    }
    return dq;
}

// ════════════════════════════════════════════════════════════════════════════
//  hinv_mat  —  H̄_A^{-1} * M  (12×C matrix)
//
//  같은 co-rotation 적용: H̄_A^{-1} in body frame, then rotate back
// ════════════════════════════════════════════════════════════════════════════

static DynMat hinv_mat(const ABDBody &b, const DynMat &M) {
    // M is 12×C (each column is a 12-vec)
    float A[3][3]; b.get_A(A);
    DynMat out(12, M.cols);

    for (int c = 0; c < M.cols; ++c) {
        // Extract column
        Vec12 col; for (int i=0;i<12;++i) col[i] = M.v[i*M.cols+c];

        // Rotate into body frame: f̃ = diag4(A^T) col
        Vec12 col_tilde;
        for (int k = 0; k < 4; ++k) {
            int off = k*3;
            float fx=col[off], fy=col[off+1], fz=col[off+2];
            col_tilde[off+0] = A[0][0]*fx + A[1][0]*fy + A[2][0]*fz;
            col_tilde[off+1] = A[0][1]*fx + A[1][1]*fy + A[2][1]*fz;
            col_tilde[off+2] = A[0][2]*fx + A[1][2]*fy + A[2][2]*fz;
        }

        // Solve H̄_A^{-1} col_tilde
        float rhs[12]; for(int i=0;i<12;++i) rhs[i]=col_tilde[i];
        b.H_bar_A.cholesky_solve(rhs);

        // Rotate back: diag4(A) rhs
        for (int k = 0; k < 4; ++k) {
            int off = k*3;
            float dx=rhs[off], dy=rhs[off+1], dz=rhs[off+2];
            out.v[(off+0)*M.cols+c] = A[0][0]*dx + A[0][1]*dy + A[0][2]*dz;
            out.v[(off+1)*M.cols+c] = A[1][0]*dx + A[1][1]*dy + A[1][2]*dz;
            out.v[(off+2)*M.cols+c] = A[2][0]*dx + A[2][1]*dy + A[2][2]*dz;
        }
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════════════
//  build_constraint_blocks
// ════════════════════════════════════════════════════════════════════════════

void ABDKKTSolver::build_constraint_blocks(
    const std::vector<ABDBody*> &bodies,
    const std::vector<ABDJoint> &joints,
    std::vector<ConstraintBlock> &out)
{
    out.clear();
    out.resize(joints.size());
    for (int k = 0; k < (int)joints.size(); ++k) {
        const ABDJoint &j = joints[k];
        out[k].joint_idx = k;
        int ia = j.body_alpha, ib = j.body_beta;
        if (ia < 0 || ib < 0 ||
            ia >= (int)bodies.size() || ib >= (int)bodies.size()) continue;
        compute_constraint_gradient(*bodies[ia], *bodies[ib], j, out[k]);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  precompute_dual  —  H^{-1}∇C^T, D = ∇C H^{-1} ∇C^T, b = ∇C H^{-1} f_A
//
//  Eq.27-28:
//    D δλ = b
//    D = ∇C_α H_α^{-1} ∇C_α^T + ∇C_β H_β^{-1} ∇C_β^T
//    b = ∇C_α H_α^{-1} f_α + ∇C_β H_β^{-1} f_β
// ════════════════════════════════════════════════════════════════════════════

void ABDKKTSolver::precompute_dual(
    const std::vector<ABDBody*> &bodies,
    std::vector<ConstraintBlock> &blocks,
    const std::vector<Vec12> &f_A_list,
    float /*h*/)
{
    for (auto &blk : blocks) {
        if (blk.c_dim == 0) continue;
        int ia = blk.body_alpha, ib = blk.body_beta;
        if (ia < 0 || ib < 0) continue;
        const ABDBody &ba = *bodies[ia];
        const ABDBody &bb = *bodies[ib];

        // Hinv_gradT: H^{-1} (∇C)^T  — shape 12 × C_k
        // We transpose grad first (C_k×12 → 12×C_k), then apply hinv_mat
        blk.Hinv_gradT_alpha = hinv_mat(ba, blk.grad_alpha.transpose());
        blk.Hinv_gradT_beta  = hinv_mat(bb, blk.grad_beta.transpose());

        // D = ∇C_α Hinv_gradT_α + ∇C_β Hinv_gradT_β  (C_k × C_k)
        blk.D = DynMat::mul(blk.grad_alpha, blk.Hinv_gradT_alpha);
        DynMat D_beta = DynMat::mul(blk.grad_beta, blk.Hinv_gradT_beta);
        blk.D.add(D_beta);

        // Regularize D diagonal to prevent singular systems
        //for (int i = 0; i < blk.c_dim; ++i)
        //    blk.D(i,i) += 1e-10f;

        // b = ∇C_α (H_α^{-1} f_α) + ∇C_β (H_β^{-1} f_β)
        // = ∇C_α * dq_a_unconstrained + ∇C_β * dq_b_unconstrained
        blk.b_dual.assign(blk.c_dim, 0.f);
        Vec12 dq_a = unconstrained_dq(ba, f_A_list[ia]);
        Vec12 dq_b = unconstrained_dq(bb, f_A_list[ib]);
        for (int l = 0; l < blk.c_dim; ++l) {
            float va = 0.f, vb = 0.f;
            for (int i = 0; i < 12; ++i) {
                va += blk.grad_alpha(l,i) * dq_a[i];
                vb += blk.grad_beta (l,i) * dq_b[i];
            }
            blk.b_dual[l] = va + vb;
        }

        blk.dlambda.assign(blk.c_dim, 0.f);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  apply_dual_correction  —  Eq.27: δq -= H^{-1} ∇C^T δλ
// ════════════════════════════════════════════════════════════════════════════

void ABDKKTSolver::apply_dual_correction(
    const std::vector<ABDBody*> &bodies,
    const std::vector<ConstraintBlock> &blocks,
    std::vector<Vec12> &dq_list)
{
    for (const auto &blk : blocks) {
        if (blk.c_dim == 0) continue;
        int ia = blk.body_alpha, ib = blk.body_beta;
        if (ia < 0 || ib < 0) continue;

        // δq^α -= Hinv_gradT_alpha * δλ
        // Hinv_gradT_alpha is 12×C_k, stored row-major
        for (int i = 0; i < 12; ++i) {
            float corr_a = 0.f, corr_b = 0.f;
            for (int l = 0; l < blk.c_dim; ++l) {
                corr_a += blk.Hinv_gradT_alpha(i, l) * blk.dlambda[l];
                corr_b += blk.Hinv_gradT_beta (i, l) * blk.dlambda[l];
            }
            if (!bodies[ia]->is_static) dq_list[ia][i] -= corr_a;
            if (!bodies[ib]->is_static) dq_list[ib][i] -= corr_b;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Topology analysis
// ════════════════════════════════════════════════════════════════════════════

TopologyType ABDKKTSolver::analyze_topology(
    int n_bodies, const std::vector<ABDJoint> &joints)
{
    if (joints.empty()) return TopologyType::Independent;

    std::vector<std::vector<int>> adj(n_bodies);
    for (auto &j : joints) {
        if (j.body_alpha < 0 || j.body_beta < 0) continue;
        adj[j.body_alpha].push_back(j.body_beta);
        adj[j.body_beta].push_back(j.body_alpha);
    }

    std::vector<bool> visited(n_bodies, false);
    bool has_cycle = false;
    int n_components = 0;

    for (int start = 0; start < n_bodies; ++start) {
        if (visited[start]) continue;
        ++n_components;
        std::queue<int> q;
        std::unordered_map<int,int> parent;
        q.push(start); visited[start] = true; parent[start] = -1;
        int body_count = 0, edge_count = 0;
        while (!q.empty()) {
            int u = q.front(); q.pop(); ++body_count;
            for (int v : adj[u]) {
                edge_count++;
                if (!visited[v]) {
                    visited[v] = true;
                    parent[v] = u;
                    q.push(v);
                } else if (parent.count(u) && parent[u] != v) {
                    has_cycle = true;
                }
            }
        }
        edge_count /= 2;
        if (edge_count > body_count - 1) has_cycle = true;
    }

    bool is_chain = !has_cycle;
    if (is_chain) {
        for (int i = 0; i < n_bodies; ++i)
            if ((int)adj[i].size() > 2) { is_chain = false; break; }
    }

    if (!has_cycle && is_chain && n_components == 1) return TopologyType::Chain;
    if (!has_cycle) return TopologyType::Tree;
    if (n_components == 1 && has_cycle) return TopologyType::Loop;
    return TopologyType::Graph;
}

// ════════════════════════════════════════════════════════════════════════════
//  find_chain  —  topological order for chain solver
// ════════════════════════════════════════════════════════════════════════════

bool ABDKKTSolver::find_chain(
    int n_bodies,
    const std::vector<ABDJoint> &joints,
    std::vector<int> &body_order,
    std::vector<int> &joint_order)
{
    body_order.clear();
    joint_order.clear();
    if (n_bodies == 0) return true;

    // Build adjacency: body → list of (neighbor_body, joint_idx)
    std::vector<std::vector<std::pair<int,int>>> adj(n_bodies);
    for (int k = 0; k < (int)joints.size(); ++k) {
        const auto &j = joints[k];
        if (j.body_alpha < 0 || j.body_beta < 0) continue;
        adj[j.body_alpha].push_back({j.body_beta,  k});
        adj[j.body_beta ].push_back({j.body_alpha, k});
    }

    // Find endpoint (degree 0 or 1) to start from
    int start = 0;
    for (int i = 0; i < n_bodies; ++i)
        if (adj[i].size() <= 1) { start = i; break; }

    // Walk the chain
    std::vector<bool> visited(n_bodies, false);
    body_order.push_back(start);
    visited[start] = true;

    while (true) {
        int cur = body_order.back();
        bool found = false;
        for (auto &[nb, jk] : adj[cur]) {
            if (!visited[nb]) {
                visited[nb] = true;
                joint_order.push_back(jk);
                body_order.push_back(nb);
                found = true;
                break;
            }
        }
        if (!found) break;
    }

    return (int)body_order.size() == n_bodies;
}

// ════════════════════════════════════════════════════════════════════════════
//  build_tree  —  BFS tree for ABA pass
// ════════════════════════════════════════════════════════════════════════════

std::vector<ABDKKTSolver::TreeNode> ABDKKTSolver::build_tree(
    int n_bodies,
    const std::vector<ABDJoint> &joints,
    std::vector<int> &order_leaf_to_root)
{
    std::vector<TreeNode> nodes(n_bodies);
    for (int i = 0; i < n_bodies; ++i) nodes[i].body_idx = i;

    std::vector<std::vector<std::pair<int,int>>> adj(n_bodies);
    for (int k = 0; k < (int)joints.size(); ++k) {
        const auto &j = joints[k];
        if (j.body_alpha < 0 || j.body_beta < 0) continue;
        adj[j.body_alpha].push_back({j.body_beta, k});
        adj[j.body_beta ].push_back({j.body_alpha, k});
    }

    std::vector<bool> visited(n_bodies, false);
    std::queue<int> q;
    int root = 0;
    q.push(root); visited[root] = true;
    nodes[root].is_root    = true;
    nodes[root].parent_idx = -1;

    std::vector<int> bfs_order;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        bfs_order.push_back(u);
        for (auto &[v, jk] : adj[u]) {
            if (!visited[v]) {
                visited[v] = true;
                nodes[v].parent_idx      = u;
                nodes[v].joint_to_parent = jk;
                nodes[u].children.push_back(v);
                nodes[u].child_joints.push_back(jk);
                q.push(v);
            }
        }
    }

    order_leaf_to_root = bfs_order;
    std::reverse(order_leaf_to_root.begin(), order_leaf_to_root.end());
    return nodes;
}

// ════════════════════════════════════════════════════════════════════════════
//  find_loop_breakers
// ════════════════════════════════════════════════════════════════════════════

std::vector<int> ABDKKTSolver::find_loop_breakers(
    int n_bodies, const std::vector<ABDJoint> &joints)
{
    std::vector<bool> visited(n_bodies, false);
    std::vector<int> breakers;
    std::set<std::pair<int,int>> tree_edges;
    std::queue<int> q;
    q.push(0); visited[0] = true;

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int k = 0; k < (int)joints.size(); ++k) {
            int a = joints[k].body_alpha, b = joints[k].body_beta;
            if (a < 0 || b < 0) continue;
            int other = -1;
            if (a == u) other = b;
            else if (b == u) other = a;
            else continue;
            auto key = std::make_pair(std::min(u,other), std::max(u,other));
            if (!visited[other]) {
                visited[other] = true;
                tree_edges.insert(key);
                q.push(other);
            }
        }
    }
    for (int k = 0; k < (int)joints.size(); ++k) {
        int a = joints[k].body_alpha, b = joints[k].body_beta;
        if (a < 0 || b < 0) continue;
        auto key = std::make_pair(std::min(a,b), std::max(a,b));
        if (tree_edges.find(key) == tree_edges.end())
            breakers.push_back(k);
    }
    return breakers;
}

// ════════════════════════════════════════════════════════════════════════════
//  Section 5.2: Chain solver — block-tridiagonal Thomas (Eq.39-41)
//
//  Thomas forward sweep:
//    D^k_mod = D^k - B^{k-1,T} (D^{k-1}_mod)^{-1} B^{k-1}
//    b^k_mod = b^k - B^{k-1,T} (D^{k-1}_mod)^{-1} b^{k-1}_mod
//
//  Back substitution:
//    δλ^K = (D^K_mod)^{-1} b^K_mod
//    δλ^k = (D^k_mod)^{-1} (b^k_mod - B^k δλ^{k+1})
//
//  BUGFIX: 이전 구현은 piv를 루프 바깥에서 재사용 → 잘못된 LU 분해
//          각 k마다 독립적인 piv로 factor해야 함.
// ════════════════════════════════════════════════════════════════════════════

void ABDKKTSolver::solve_chain(
    const std::vector<ABDBody*> &bodies,
    const std::vector<ABDJoint> &joints,
    std::vector<ConstraintBlock> &blocks,
    std::vector<Vec12> &f_A_list,
    std::vector<Vec12> &dq_out,
    float h)
{
    std::vector<int> body_order, joint_order;
    if (!find_chain((int)bodies.size(), joints, body_order, joint_order)) {
        solve_graph(bodies, joints, blocks, f_A_list, dq_out, h);
        return;
    }

    int K = (int)joint_order.size();
    if (K == 0) {
        for (int i = 0; i < (int)bodies.size(); ++i)
            if (!bodies[i]->is_static)
                dq_out[i] = unconstrained_dq(*bodies[i], f_A_list[i]);
        return;
    }

    std::vector<ConstraintBlock*> cb(K);
    for (int k = 0; k < K; ++k) cb[k] = &blocks[joint_order[k]];

    // Compute off-diagonal B^k:
    // B^k ∈ ℝ^{C_k × C_{k+1}}
    // = ∇C^k_{shared} * H_{shared}^{-1} * (∇C^{k+1}_{shared})^T
    // where 'shared' is the body between joint k and joint k+1
    for (int k = 0; k < K-1; ++k) {
        int shared_body = body_order[k+1];
        const ABDBody &bs = *bodies[shared_body];

        // Determine which grad to use for the shared body in each joint
        // In joint k,   shared body = body_beta  (end of joint k)
        // In joint k+1, shared body = body_alpha (start of joint k+1)
        const DynMat &grad_k_shared    = (cb[k]->body_beta == shared_body)
                                          ? cb[k]->grad_beta
                                          : cb[k]->grad_alpha;
        const DynMat &grad_k1_shared   = (cb[k+1]->body_alpha == shared_body)
                                          ? cb[k+1]->grad_alpha
                                          : cb[k+1]->grad_beta;

        // B^k = grad_k_shared * H_shared^{-1} * grad_k1_shared^T
        DynMat Hinv_gk1T = hinv_mat(bs, grad_k1_shared.transpose()); // 12 × C_{k+1}
        cb[k]->B_fwd  = DynMat::mul(grad_k_shared, Hinv_gk1T);       // C_k × C_{k+1}
        cb[k+1]->B_bwd = cb[k]->B_fwd.transpose();                    // C_{k+1} × C_k
    }

    // Thomas forward sweep
    std::vector<DynMat>             D_mod(K);
    std::vector<std::vector<float>> b_mod(K);
    std::vector<std::vector<int>>   pivs(K); // BUGFIX: each k gets its own piv

    D_mod[0] = cb[0]->D;
    b_mod[0] = cb[0]->b_dual;
    D_mod[0].lu_factor(pivs[0]);

    for (int k = 1; k < K; ++k) {
        // Compute (D^{k-1}_mod)^{-1} B^{k-1}_fwd  column by column
        DynMat &Bfwd = cb[k-1]->B_fwd;
        DynMat Dinv_B(Bfwd.rows, Bfwd.cols);
        for (int c = 0; c < Bfwd.cols; ++c) {
            std::vector<float> col(Bfwd.rows);
            for (int r = 0; r < Bfwd.rows; ++r) col[r] = Bfwd(r, c);
            DynMat Dtmp = D_mod[k-1];
            std::vector<int> ptmp;
            Dtmp.lu_factor(ptmp);
            Dtmp.lu_solve(ptmp, col);
            for (int r = 0; r < Bfwd.rows; ++r) Dinv_B(r, c) = col[r];
        }

        // D^k_mod = D^k - B^{k,T} * Dinv_B  (B^{k,T} = B_bwd of k)
        D_mod[k] = cb[k]->D;
        DynMat BT_DinvB = DynMat::mul(cb[k-1]->B_bwd, Dinv_B);
        for (int r = 0; r < D_mod[k].rows; ++r)
            for (int c = 0; c < D_mod[k].cols; ++c)
                D_mod[k](r,c) -= BT_DinvB(r,c);

        // b^k_mod = b^k - B^{k,T} * (D^{k-1}_mod)^{-1} * b^{k-1}_mod
        b_mod[k] = cb[k]->b_dual;
        {
            std::vector<float> Dinv_b = b_mod[k-1];
            DynMat Dtmp = D_mod[k-1];
            std::vector<int> ptmp;
            Dtmp.lu_factor(ptmp);
            Dtmp.lu_solve(ptmp, Dinv_b);
            for (int r = 0; r < (int)b_mod[k].size(); ++r) {
                float s = 0.f;
                for (int c = 0; c < (int)Dinv_b.size(); ++c)
                    s += cb[k-1]->B_bwd(r, c) * Dinv_b[c];
                b_mod[k][r] -= s;
            }
        }

        D_mod[k].lu_factor(pivs[k]);
    }

    // Back substitution
    std::vector<std::vector<float>> dlambda(K);
    {
        dlambda[K-1] = b_mod[K-1];
        DynMat Dc = D_mod[K-1];
        std::vector<int> piv;
        Dc.lu_factor(piv);
        Dc.lu_solve(piv, dlambda[K-1]);
    }
    for (int k = K-2; k >= 0; --k) {
        // δλ^k = (D^k_mod)^{-1} (b^k_mod - B^k_fwd * δλ^{k+1})
        std::vector<float> rhs = b_mod[k];
        DynMat &Bfwd = cb[k]->B_fwd;
        for (int r = 0; r < (int)rhs.size(); ++r) {
            float s = 0.f;
            for (int c = 0; c < (int)dlambda[k+1].size(); ++c)
                s += Bfwd(r, c) * dlambda[k+1][c];
            rhs[r] -= s;
        }
        DynMat Dc = D_mod[k];
        std::vector<int> piv;
        Dc.lu_factor(piv);
        Dc.lu_solve(piv, rhs);
        dlambda[k] = rhs;
    }

    for (int k = 0; k < K; ++k)
        cb[k]->dlambda = dlambda[k];

    // δq = H^{-1}(f_A - ∇C^T δλ)
    for (int i = 0; i < (int)bodies.size(); ++i) {
        if (bodies[i]->is_static) { dq_out[i].fill(0.f); continue; }
        dq_out[i] = unconstrained_dq(*bodies[i], f_A_list[i]);
    }
    apply_dual_correction(bodies, blocks, dq_out);
}

// ════════════════════════════════════════════════════════════════════════════
//  Section 5.3: ABD-ABA tree solver (Algorithm 2)
//
//  Upward pass  (leaf → root, Eq.53-54):
//    U^j        = Ĥ^j_A * (∇C^j_child)^T          (12 × C_k)
//    D̂^j       = ∇C^j_child * Ĥ^j_A^{-1} * (∇C^j_child)^T  (C_k × C_k)
//    ΔH^j_A    = Ĥ^j_A - U^j (D̂^j)^{-1} U^{j,T}
//    Δf^j_A    = f̂^j_A - U^j (D̂^j)^{-1} ∇C^j_child f̂^j_A
//    Ĥ^p_A   += X^{j,T} ΔH^j_A X^j
//    f̂^p_A   += X^{j,T} Δf^j_A
//
//  Root solve: δq^root = Ĥ^root_A^{-1} f̂^root_A
//
//  Downward pass (root → leaf, Eq.56-57):
//    r^j    = -∇C^j_parent * δq^parent
//    Solve local KKT (12+C_k system) for (δq^j, δλ^j)
// ════════════════════════════════════════════════════════════════════════════

void ABDKKTSolver::solve_tree(
    const std::vector<ABDBody*> &bodies,
    const std::vector<ABDJoint> &joints,
    std::vector<ConstraintBlock> &blocks,
    std::vector<Vec12> &f_A_list,
    std::vector<Vec12> &dq_out,
    float h)
{
    int N = (int)bodies.size();
    std::vector<int> order;
    std::vector<TreeNode> tree = build_tree(N, joints, order);

    // Initialize each node's condensed Hessian and force
    for (int j = 0; j < N; ++j) {
        tree[j].Ĥ_A = DynMat(12, 12);
        const ABDBody &b = *bodies[j];
        for (int r = 0; r < 12; ++r)
            for (int c = 0; c < 12; ++c)
                tree[j].Ĥ_A(r,c) = b.H_bar_A.v[r*12+c];
        tree[j].f_hat = f_A_list[j];
    }

    // ── Upward pass ────────────────────────────────────────────
    for (int j : order) {
        TreeNode &nd = tree[j];
        if (nd.is_root) continue;
        int jk = nd.joint_to_parent;
        int p  = nd.parent_idx;
        ConstraintBlock &blk = blocks[jk];
        if (blk.c_dim == 0) continue;

        bool child_is_alpha = (blk.body_alpha == j);
        const DynMat &grad_child = child_is_alpha ? blk.grad_alpha : blk.grad_beta;

        // U^j = Ĥ^j_A * (∇C^j_child)^T  (12 × C_k)
        // = Ĥ^j_A * grad_child^T
        DynMat gradT = grad_child.transpose(); // 12 × C_k
        DynMat U(12, blk.c_dim);
        for (int r = 0; r < 12; ++r)
            for (int c = 0; c < blk.c_dim; ++c) {
                float s = 0.f;
                for (int i = 0; i < 12; ++i)
                    s += nd.Ĥ_A(r,i) * gradT(i,c);
                U(r,c) = s;
            }

        // D̂^j = ∇C^j_child * Ĥ^j_A^{-1} * (∇C^j_child)^T
        // = grad_child * (Ĥ^j_A^{-1} * grad_child^T)
        // Solve Ĥ^j_A * X = grad_child^T  → X = Ĥ^j_A^{-1} gradT
        DynMat Hinv_gradT(12, blk.c_dim);
        for (int c = 0; c < blk.c_dim; ++c) {
            std::vector<float> col(12);
            for (int r = 0; r < 12; ++r) col[r] = gradT(r,c);
            // Solve Ĥ^j_A col using LU (not Cholesky since Ĥ may be non-SPD after accumulation)
            DynMat Htmp = nd.Ĥ_A;
            std::vector<int> piv;
            Htmp.lu_factor(piv);
            Htmp.lu_solve(piv, col);
            for (int r = 0; r < 12; ++r) Hinv_gradT(r,c) = col[r];
        }
        DynMat D_hat = DynMat::mul(grad_child, Hinv_gradT); // C_k × C_k

        // Regularize
        for (int i = 0; i < blk.c_dim; ++i) D_hat(i,i) += 1e-10f;

        // (D̂^j)^{-1}: solve systems
        // ΔH^j_A = Ĥ^j_A - U * D̂^{-1} * U^T
        // Compute Dinv_UT: solve D̂^j * X = U^T  (size C_k × 12)
        DynMat UT = U.transpose(); // C_k × 12
        DynMat Dinv_UT(blk.c_dim, 12);
        {
            DynMat Dtmp = D_hat;
            std::vector<int> piv;
            Dtmp.lu_factor(piv);
            for (int c = 0; c < 12; ++c) {
                std::vector<float> col(blk.c_dim);
                for (int r = 0; r < blk.c_dim; ++r) col[r] = UT(r,c);
                //DynMat Dt2 = D_hat;
				DynMat Dt2 = Dtmp;
                std::vector<int> p2;
                //Dt2.lu_factor(p2);
                Dt2.lu_solve(p2, col);
                for (int r = 0; r < blk.c_dim; ++r) Dinv_UT(r,c) = col[r];
            }
        }
        DynMat U_Dinv_UT = DynMat::mul(U, Dinv_UT); // 12×12

        // ΔH^j_A = Ĥ^j - U D̂^{-1} U^T
        DynMat dH(12,12);
        for (int r=0;r<12;++r) for(int c=0;c<12;++c)
            dH(r,c) = nd.Ĥ_A(r,c) - U_Dinv_UT(r,c);

        // Δf^j_A = f̂^j - U * D̂^{-1} * (∇C^j_child * f̂^j)
        std::vector<float> Cf(blk.c_dim, 0.f);
        for (int l=0; l<blk.c_dim; ++l)
            for (int i=0; i<12; ++i)
                Cf[l] += grad_child(l,i) * nd.f_hat[i];
        {
            DynMat Dtmp = D_hat; std::vector<int> piv; Dtmp.lu_factor(piv);
            Dtmp.lu_solve(piv, Cf);
        }
        Vec12 df_j;
        for (int i=0;i<12;++i) {
            float s=0.f;
            for (int l=0;l<blk.c_dim;++l) s += U(i,l)*Cf[l];
            df_j[i] = nd.f_hat[i] - s;
        }

        // Propagate to parent via X^j = diag4(R_rel):
        // R_rel = A^{parent,T} * A^{child}  (Eq.53 note)
        float Ap[3][3]; bodies[p]->get_A(Ap);
        float Aj[3][3]; bodies[j]->get_A(Aj);
        float Rrel[3][3] = {};
        for(int r=0;r<3;++r) for(int c2=0;c2<3;++c2)
            for(int k2=0;k2<3;++k2) Rrel[r][c2] += Ap[k2][r]*Aj[k2][c2];

        // Ĥ^p_A += X^{j,T} ΔH^j X^j
        // X = diag4(R_rel): apply R_rel to each 3-block
        // X^{j,T} = diag4(R_rel^T)
        // For symmetric: Ĥ^p += R^T ΔH R  (block-wise)
        // Efficient: column-by-column apply R to dH, then row-by-row apply R^T
        DynMat dH_Xj(12,12);
        for (int c2=0;c2<12;++c2) {
            int kb = c2/3, comp = c2%3;
            for (int r=0;r<12;++r) {
                int rb = r/3, rcomp = r%3;
                // Apply R_rel to column c2 of dH: result in same block
                // dH_Xj[r][c2] = dH[r][kb*3+0]*R[0][comp] + dH[r][kb*3+1]*R[1][comp] + dH[r][kb*3+2]*R[2][comp]
                dH_Xj(r,c2) = dH(r,kb*3+0)*Rrel[0][comp]
                             + dH(r,kb*3+1)*Rrel[1][comp]
                             + dH(r,kb*3+2)*Rrel[2][comp];
            }
        }
        // Now apply R^T on left (rows): Ĥ^p += (R^T dH_Xj)
        for (int r=0;r<12;++r) {
            int rb = r/3, rcomp = r%3;
            for (int c2=0;c2<12;++c2) {
                // row r of R^T * dH_Xj:
                // = R[0][rcomp]*dH_Xj[rb*3+0][c2] + R[1][rcomp]*dH_Xj[rb*3+1][c2] + R[2][rcomp]*dH_Xj[rb*3+2][c2]
                float s = Rrel[0][rcomp]*dH_Xj(rb*3+0,c2)
                        + Rrel[1][rcomp]*dH_Xj(rb*3+1,c2)
                        + Rrel[2][rcomp]*dH_Xj(rb*3+2,c2);
                tree[p].Ĥ_A(r,c2) += s;
            }
        }

        // f̂^p += X^{j,T} Δf^j: apply R^T block-wise
        for (int k2=0;k2<4;++k2) {
            int off = k2*3;
            float dx=df_j[off], dy=df_j[off+1], dz=df_j[off+2];
            // R^T * [dx,dy,dz]
            tree[p].f_hat[off+0] += Rrel[0][0]*dx + Rrel[1][0]*dy + Rrel[2][0]*dz;
            tree[p].f_hat[off+1] += Rrel[0][1]*dx + Rrel[1][1]*dy + Rrel[2][1]*dz;
            tree[p].f_hat[off+2] += Rrel[0][2]*dx + Rrel[1][2]*dy + Rrel[2][2]*dz;
        }
    }

    // ── Root solve ─────────────────────────────────────────────
    int root_idx = -1;
    for (int i=0;i<N;++i) if (tree[i].is_root) { root_idx=i; break; }
    if (root_idx < 0) root_idx = order.back();

    {
        TreeNode &rnd = tree[root_idx];
        DynMat Hroot = rnd.Ĥ_A;
        std::vector<int> piv; Hroot.lu_factor(piv);
        std::vector<float> rhs(12);
        for(int i=0;i<12;++i) rhs[i]=rnd.f_hat[i];
        Hroot.lu_solve(piv, rhs);
        for(int i=0;i<12;++i) dq_out[root_idx][i]=rhs[i];
    }
    if (bodies[root_idx]->is_static) dq_out[root_idx].fill(0.f);

    // ── Downward pass ──────────────────────────────────────────
    std::vector<int> root_to_leaf = order;
    std::reverse(root_to_leaf.begin(), root_to_leaf.end());

    for (int j : root_to_leaf) {
        TreeNode &nd = tree[j];
        if (nd.is_root) continue;
        int jk = nd.joint_to_parent;
        int p  = nd.parent_idx;
        ConstraintBlock &blk = blocks[jk];

        bool child_is_alpha = (blk.body_alpha == j);
        const DynMat &grad_child  = child_is_alpha ? blk.grad_alpha : blk.grad_beta;
        const DynMat &grad_parent = child_is_alpha ? blk.grad_beta  : blk.grad_alpha;

        // r^j = -∇C^j_parent * δq^parent  (Eq.56)
        std::vector<float> rj(blk.c_dim, 0.f);
        for (int l=0;l<blk.c_dim;++l)
            for (int i=0;i<12;++i)
                rj[l] -= grad_parent(l,i)*dq_out[p][i];

        // Solve local KKT:  (12+C_k) system (Eq.57)
        // [ Ĥ^j    (∇C^j_child)^T ] [ δq^j ] = [ f̂^j ]
        // [ ∇C^j_child      0     ] [ δλ^j ]   [ r^j  ]
        int sz = 12 + blk.c_dim;
        DynMat KKT(sz, sz);
        for(int r=0;r<12;++r) for(int c=0;c<12;++c) KKT(r,c)=nd.Ĥ_A(r,c);
        for(int r=0;r<12;++r) for(int c=0;c<blk.c_dim;++c)
            KKT(r, 12+c) = grad_child(c,r); // (∇C)^T
        for(int r=0;r<blk.c_dim;++r) for(int c=0;c<12;++c)
            KKT(12+r, c) = grad_child(r,c);

        std::vector<float> rhs_kkt(sz,0.f);
        for(int i=0;i<12;++i) rhs_kkt[i]=nd.f_hat[i];
        for(int l=0;l<blk.c_dim;++l) rhs_kkt[12+l]=rj[l];

        std::vector<int> piv; KKT.lu_factor(piv);
        KKT.lu_solve(piv, rhs_kkt);

        if (!bodies[j]->is_static)
            for(int i=0;i<12;++i) dq_out[j][i]=rhs_kkt[i];
        else
            dq_out[j].fill(0.f);
        blk.dlambda.assign(blk.c_dim,0.f);
        for(int l=0;l<blk.c_dim;++l) blk.dlambda[l]=rhs_kkt[12+l];
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Section 5.4: Loop — Schur complement (Eq.58)
// ════════════════════════════════════════════════════════════════════════════

void ABDKKTSolver::solve_loop(
    const std::vector<ABDBody*> &bodies,
    const std::vector<ABDJoint> &joints,
    std::vector<ConstraintBlock> &blocks,
    std::vector<Vec12> &f_A_list,
    std::vector<Vec12> &dq_out,
    float h)
{
    // Find loop-breaker joint indices
    std::vector<int> breaker_jidx = find_loop_breakers((int)bodies.size(), joints);
    if (breaker_jidx.empty()) {
        solve_tree(bodies, joints, blocks, f_A_list, dq_out, h);
        return;
    }

    std::vector<bool> in_tree(joints.size(), true);
    for (int k : breaker_jidx) in_tree[k] = false;

    // Collect tree joints
    std::vector<ABDJoint> tree_joints;
    std::vector<int> tree_jmap; // tree_joints[i] = joints[tree_jmap[i]]
    for (int k = 0; k < (int)joints.size(); ++k)
        if (in_tree[k]) { tree_joints.push_back(joints[k]); tree_jmap.push_back(k); }

    // Solve tree subsystem → get δq_tree
    std::vector<ConstraintBlock> tree_blocks;
    build_constraint_blocks(bodies, tree_joints, tree_blocks);
    precompute_dual(bodies, tree_blocks, f_A_list, h);
    solve_tree(bodies, tree_joints, tree_blocks, f_A_list, dq_out, h);

    // For each loop-breaker: Schur correction
    // Schur: S δλ = b_D - C A^{-1} b_A
    // S ≈ D^breaker (approximation: ignore cross-coupling with tree)
    // C * dq_tree = ∇C * δq from tree solve
    for (int k : breaker_jidx) {
        auto &blk = blocks[k];
        if (blk.c_dim == 0) continue;
        int ia = blk.body_alpha, ib = blk.body_beta;
        if (ia < 0 || ib < 0) continue;

        std::vector<float> C_dq(blk.c_dim, 0.f);
        for (int l = 0; l < blk.c_dim; ++l) {
            for (int i = 0; i < 12; ++i) {
                C_dq[l] += blk.grad_alpha(l,i)*dq_out[ia][i];
                C_dq[l] += blk.grad_beta (l,i)*dq_out[ib][i];
            }
        }

        std::vector<float> rhs(blk.c_dim);
        for (int l = 0; l < blk.c_dim; ++l)
            rhs[l] = blk.b_dual[l] - C_dq[l];

        DynMat S = blk.D;
        std::vector<int> piv; S.lu_factor(piv);
        S.lu_solve(piv, rhs);
        blk.dlambda = rhs;
    }

    // Apply correction for breaker joints
    std::vector<ConstraintBlock> breaker_blocks;
    for (int k : breaker_jidx) breaker_blocks.push_back(blocks[k]);
    apply_dual_correction(bodies, breaker_blocks, dq_out);
}

// ════════════════════════════════════════════════════════════════════════════
//  Section 5.5: Graph — multi-directional Gauss-Seidel
// ════════════════════════════════════════════════════════════════════════════

void ABDKKTSolver::solve_graph(
    const std::vector<ABDBody*> &bodies,
    const std::vector<ABDJoint> &joints,
    std::vector<ConstraintBlock> &blocks,
    std::vector<Vec12> &f_A_list,
    std::vector<Vec12> &dq_out,
    float h)
{
    int N = (int)bodies.size();
    for (int i = 0; i < N; ++i) {
        if (bodies[i]->is_static) { dq_out[i].fill(0.f); continue; }
        dq_out[i] = unconstrained_dq(*bodies[i], f_A_list[i]);
    }
    if (joints.empty()) return;

    static constexpr int GS_ITERS = 10;
    for (int iter = 0; iter < GS_ITERS; ++iter) {
        for (int k = 0; k < (int)blocks.size(); ++k) {
            auto &blk = blocks[k];
            if (blk.c_dim == 0) continue;
            int ia = blk.body_alpha, ib = blk.body_beta;
            if (ia < 0 || ib < 0) continue;

            // Compute residual of constraint in current dq:
            // r = b_dual - ∇C_α dq_α - ∇C_β dq_β
            std::vector<float> rhs(blk.c_dim, 0.f);
            for (int l = 0; l < blk.c_dim; ++l) {
                float Cdq = 0.f;
                for (int i = 0; i < 12; ++i) {
                    Cdq += blk.grad_alpha(l,i)*dq_out[ia][i];
                    Cdq += blk.grad_beta (l,i)*dq_out[ib][i];
                }
                rhs[l] = blk.b_dual[l] - Cdq;
            }

            DynMat D = blk.D;
            std::vector<int> piv; D.lu_factor(piv);
            D.lu_solve(piv, rhs);

            // δ(δλ) = rhs - δλ_current
            std::vector<float> d_dlambda(blk.c_dim);
            for (int l = 0; l < blk.c_dim; ++l)
                d_dlambda[l] = rhs[l] - blk.dlambda[l];
            blk.dlambda = rhs;

            // Apply incremental correction to dq
            for (int i = 0; i < 12; ++i) {
                float ca = 0.f, cb = 0.f;
                for (int l = 0; l < blk.c_dim; ++l) {
                    ca += blk.Hinv_gradT_alpha(i, l) * d_dlambda[l];
                    cb += blk.Hinv_gradT_beta (i, l) * d_dlambda[l];
                }
                if (!bodies[ia]->is_static) dq_out[ia][i] -= ca;
                if (!bodies[ib]->is_static) dq_out[ib][i] -= cb;
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  ABDKKTSolver::solve  —  main entry point
// ════════════════════════════════════════════════════════════════════════════

void ABDKKTSolver::solve(
    const std::vector<ABDBody*> &bodies,
    const std::vector<ABDJoint> &joints,
    std::vector<Vec12>          &f_A_list,
    std::vector<Vec12>          &dq_out,
    float h)
{
    int N = (int)bodies.size();
    dq_out.resize(N);

    if (joints.empty()) {
        for (int i = 0; i < N; ++i) {
            if (bodies[i]->is_static) { dq_out[i].fill(0.f); continue; }
            dq_out[i] = unconstrained_dq(*bodies[i], f_A_list[i]);
        }
        return;
    }

    std::vector<ConstraintBlock> blocks;
    build_constraint_blocks(bodies, joints, blocks);
    precompute_dual(bodies, blocks, f_A_list, h);

    TopologyType topo = analyze_topology(N, joints);

    switch (topo) {
    case TopologyType::Independent:
        for (int i = 0; i < N; ++i) {
            if (bodies[i]->is_static) { dq_out[i].fill(0.f); continue; }
            dq_out[i] = unconstrained_dq(*bodies[i], f_A_list[i]);
        }
        break;
    case TopologyType::Chain:
        solve_chain(bodies, joints, blocks, f_A_list, dq_out, h);
        break;
    case TopologyType::Tree:
        solve_tree(bodies, joints, blocks, f_A_list, dq_out, h);
        break;
    case TopologyType::Loop:
        solve_loop(bodies, joints, blocks, f_A_list, dq_out, h);
        break;
    case TopologyType::Graph:
        solve_graph(bodies, joints, blocks, f_A_list, dq_out, h);
        break;
    }
}

} // namespace godot
