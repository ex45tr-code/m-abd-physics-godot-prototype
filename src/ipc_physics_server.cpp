#include "ipc_physics_server.h"
#include "ipc_body_state.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace godot {

// ════════════════════════════════════════════════════════════════════════════
//  수학 헬퍼
// ════════════════════════════════════════════════════════════════════════════

static inline float absf(float a)              { return a < 0.f ? -a : a; }
static inline float maxf(float a, float b)     { return a > b ? a : b; }
static inline float sqrtf_s(float v)           { return v <= 0.f ? 0.f : sqrtf(v); }

// ════════════════════════════════════════════════════════════════════════════
//  Mat12 — 12×12 Cholesky factorization & solve
// ════════════════════════════════════════════════════════════════════════════

bool Mat12::cholesky_factor() {
    // In-place LL^T decomposition, lower triangle stored
    for (int i = 0; i < 12; ++i) {
        float sum = v[i*12+i];
        for (int k = 0; k < i; ++k) sum -= v[i*12+k] * v[i*12+k];
        //if (sum <= 0.f) sum = 1e-12f;
        float L_ii = sqrtf(sum);
        v[i*12+i] = L_ii;
        float inv_L_ii = 1.f / L_ii;
        for (int j = i+1; j < 12; ++j) {
            float s2 = v[j*12+i];
            for (int k = 0; k < i; ++k) s2 -= v[j*12+k] * v[i*12+k];
            v[j*12+i] = s2 * inv_L_ii;
        }
    }
    return true;
}

void Mat12::cholesky_solve(float rhs[12]) const {
    // Forward: L y = rhs
    for (int i = 0; i < 12; ++i) {
        float s = rhs[i];
        for (int k = 0; k < i; ++k) s -= v[i*12+k] * rhs[k];
        rhs[i] = s / v[i*12+i];
    }
    // Backward: L^T x = y
    for (int i = 11; i >= 0; --i) {
        float s = rhs[i];
        for (int k = i+1; k < 12; ++k) s -= v[k*12+i] * rhs[k];
        rhs[i] = s / v[i*12+i];
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  ABDBody methods
// ════════════════════════════════════════════════════════════════════════════

void ABDBody::get_A(float A[3][3]) const {
    // q = [a1, a2, a3, t], A = [a1|a2|a3] column-major
    for (int col = 0; col < 3; ++col)
        for (int row = 0; row < 3; ++row)
            A[row][col] = q[col*3 + row];
}

void ABDBody::set_A(const float A[3][3]) {
    for (int col = 0; col < 3; ++col)
        for (int row = 0; row < 3; ++row)
            q[col*3 + row] = A[row][col];
}

Vector3 ABDBody::world_point(const Vector3 &x_rest) const {
    float A[3][3]; get_A(A);
    Vector3 t = get_t();
    return Vector3(
        A[0][0]*x_rest.x + A[0][1]*x_rest.y + A[0][2]*x_rest.z + t.x,
        A[1][0]*x_rest.x + A[1][1]*x_rest.y + A[1][2]*x_rest.z + t.y,
        A[2][0]*x_rest.x + A[2][1]*x_rest.y + A[2][2]*x_rest.z + t.z
    );
}

// DBG helper: print all world verts of a body once per body/step
static void DBG_print_body_verts(const ABDBody &b, int body_idx, int step) {
    UtilityFunctions::print("[DBG_VERTS] step=", step, " body=", body_idx,
        " static=", b.is_static,
        " t=(", b.get_t().x, ",", b.get_t().y, ",", b.get_t().z, ")",
        " n_verts=", (int)b.verts_local.size(),
        " n_faces=", (int)b.faces_sim.size());
    for (int i = 0; i < (int)b.verts_local.size(); ++i) {
        Vector3 wp = b.world_point(b.verts_local[i]);
        UtilityFunctions::print("  v[", i, "] local=(", b.verts_local[i].x, ",",
            b.verts_local[i].y, ",", b.verts_local[i].z, ")",
            " world=(", wp.x, ",", wp.y, ",", wp.z, ")");
    }
    for (int i = 0; i < (int)b.faces_sim.size(); ++i) {
        const auto &f = b.faces_sim[i];
        Vector3 wa = b.world_point(b.verts_local[f.vi[0]]);
        UtilityFunctions::print("  face[", i, "] vi=(", f.vi[0], ",", f.vi[1], ",", f.vi[2], ")",
            " n=(", f.n.x, ",", f.n.y, ",", f.n.z, ")",
            " v0_world=(", wa.x, ",", wa.y, ",", wa.z, ")");
    }
}

// ── Polar decomposition R = A(AᵀA)^{-1/2} (Eq.9) ────────────────────────
// Higham iteration: R_{k+1} = (R_k + R_k^{-T}) / 2
Basis ABDBody::polar_rotation() const {
    float R[3][3]; get_A(R);

    for (int iter = 0; iter < 10; ++iter) {
        float det =
            R[0][0]*(R[1][1]*R[2][2] - R[1][2]*R[2][1])
          - R[0][1]*(R[1][0]*R[2][2] - R[1][2]*R[2][0])
          + R[0][2]*(R[1][0]*R[2][1] - R[1][1]*R[2][0]);
        if (absf(det) < 1e-12f) break;
        float id = 1.f / det;

        // cofactor matrix (= R^{-T} * det)
        float C[3][3];
        C[0][0] =  (R[1][1]*R[2][2] - R[1][2]*R[2][1]) * id;
        C[0][1] = -(R[1][0]*R[2][2] - R[1][2]*R[2][0]) * id;
        C[0][2] =  (R[1][0]*R[2][1] - R[1][1]*R[2][0]) * id;
        C[1][0] = -(R[0][1]*R[2][2] - R[0][2]*R[2][1]) * id;
        C[1][1] =  (R[0][0]*R[2][2] - R[0][2]*R[2][0]) * id;
        C[1][2] = -(R[0][0]*R[2][1] - R[0][1]*R[2][0]) * id;
        C[2][0] =  (R[0][1]*R[1][2] - R[0][2]*R[1][1]) * id;
        C[2][1] = -(R[0][0]*R[1][2] - R[0][2]*R[1][0]) * id;
        C[2][2] =  (R[0][0]*R[1][1] - R[0][1]*R[1][0]) * id;

        float diff = 0.f;
        float Rn[3][3];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                // R^{-T}[r][c] = C[c][r]  (transpose of inverse)
                Rn[r][c] = 0.5f * (R[r][c] + C[c][r]);
                diff += (Rn[r][c]-R[r][c])*(Rn[r][c]-R[r][c]);
                R[r][c] = Rn[r][c];
            }
        if (diff < 1e-10f) break;
    }

    Basis B;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            B[r][c] = R[r][c];
    return B;
}

// ── fast_rotate: Ra ≈ (||a||/||Aa||) Aa  (Eq.17) ────────────────────────
Vector3 ABDBody::fast_rotate(const float A[3][3], const Vector3 &a) {
    Vector3 Aa(
        A[0][0]*a.x + A[0][1]*a.y + A[0][2]*a.z,
        A[1][0]*a.x + A[1][1]*a.y + A[1][2]*a.z,
        A[2][0]*a.x + A[2][1]*a.y + A[2][2]*a.z
    );
    float len_a  = a.length();
    float len_Aa = Aa.length();
    if (len_Aa < 1e-12f) return Aa;
    return Aa * (len_a / len_Aa);
}

// ── apply_diag4(R, in_q, out_q): out_q = diag₄(R)·in_q (Eq.10) ─────────
void ABDBody::apply_diag4(const Basis &R, const Vec12 &in_q, Vec12 &out_q) {
    for (int blk = 0; blk < 4; ++blk) {
        const int b = blk * 3;
        Vector3 vi(in_q[b], in_q[b+1], in_q[b+2]);
        Vector3 vo = R.xform(vi);
        out_q[b]   = vo.x;
        out_q[b+1] = vo.y;
        out_q[b+2] = vo.z;
    }
}

// ── G(A)^T W_ext → affine force contribution (Eq.42, Eq.44) ─────────────
// G(A) ∈ ℝ^{6×12}:
//   ω = (1/2) Σ_k q_k × q̇_k
//   v = ṫ
// G(A)^T [τ; f]:
//   블록 k (k=0,1,2): (1/2)(q_k × τ) 방향 기여  →  ½ [q_k]_× ^T τ = ½ τ × q_k
//                                                    (skew-sym transpose)
//   블록 3 (t):       f  (translation → linear force)
//
// 정확한 유도 (Eq.42):
//   G = [ ½[q1]×  ½[q2]×  ½[q3]×  0 ]
//       [   0       0       0      I3]
//   G^T [τ;f] = [½[q1]×^T τ; ½[q2]×^T τ; ½[q3]×^T τ; f]
//             = [½ τ×q1 - sign... ]
// [q_k]_× τ = q_k × τ (skew-sym: [a]× b = a×b)
// [q_k]_×^T τ = -q_k × τ = τ × q_k
void ABDBody::apply_G_transpose(const float A[3][3],
                                const Vector3 &tau, const Vector3 &f,
                                Vec12 &out_f_A) {
    // A columns: a_k = A[:,k]
    for (int k = 0; k < 3; ++k) {
        Vector3 a_k(A[0][k], A[1][k], A[2][k]);
        // G^T 블록 k: ½ * (τ × a_k)
        //  [q_k]_× = skew(q_k),  [q_k]_×^T τ = -q_k × τ = τ × q_k
        Vector3 contrib = tau.cross(a_k) * 0.5f;
        out_f_A[k*3]   += contrib.x;
        out_f_A[k*3+1] += contrib.y;
        out_f_A[k*3+2] += contrib.z;
    }
    // G^T 블록 3 (translation): f
    out_f_A[9]  += f.x;
    out_f_A[10] += f.y;
    out_f_A[11] += f.z;
}

// ── Transform3D sync ──────────────────────────────────────────────────────

Transform3D ABDBody::get_transform() const {
    float A[3][3]; get_A(A);
    Basis basis;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            basis[r][c] = A[r][c];
    return Transform3D(basis, get_t());
}

void ABDBody::set_transform(const Transform3D &xf) {
    float A[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            A[r][c] = xf.basis[r][c];
    set_A(A);
    set_t(xf.origin);
}

// ── Velocity ──────────────────────────────────────────────────────────────

Vector3 ABDBody::get_linear_velocity() const {
    return Vector3(qdot[9], qdot[10], qdot[11]);
}
void ABDBody::set_linear_velocity(const Vector3 &v) {
    qdot[9] = v.x; qdot[10] = v.y; qdot[11] = v.z;
}

// ω = (1/2) Σ_k a_k × ȧ_k  (Eq.43)
Vector3 ABDBody::get_angular_velocity() const {
    Vector3 omega;
    for (int k = 0; k < 3; ++k) {
        Vector3 ak  (q[k*3],    q[k*3+1],    q[k*3+2]);
        Vector3 akdot(qdot[k*3], qdot[k*3+1], qdot[k*3+2]);
        omega += ak.cross(akdot);
    }
    return omega * 0.5f;
}

// ȧ_k = ω × a_k  for rigid body
void ABDBody::set_angular_velocity(const Vector3 &omega) {
    for (int k = 0; k < 3; ++k) {
        Vector3 ak(q[k*3], q[k*3+1], q[k*3+2]);
        Vector3 adot = omega.cross(ak);
        qdot[k*3]   = adot.x;
        qdot[k*3+1] = adot.y;
        qdot[k*3+2] = adot.z;
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  ABDBody::build_mass_matrix  — M_A = Jᵀ M J  (Eq.7)
//
//  균일 밀도 박스, COM at origin, half-extents (hx,hy,hz):
//
//  M_A는 12×12 블록 대각 행렬 (각 3×3 블록):
//    블록 (k,k): s_k · I₃   (k=0,1,2 → A 열벡터 블록)
//    블록 (3,3): m · I₃      (translation 블록)
//    오프대각: 0 (대칭 박스, COM at origin)
//
//  s_k = ∫ x̄_k² dm = m · h_k² / 3
//  (균일 분포 1D: (1/2h)∫_{-h}^{h} x² dx = h²/3)
// ════════════════════════════════════════════════════════════════════════════

void ABDBody::build_mass_matrix() {
    M_A.set_zero();
    float m  = mass;
    float hx = half_extents.x;
    float hy = half_extents.y;
    float hz = half_extents.z;

    float s[3] = {
        m * hx * hx / 3.f,
        m * hy * hy / 3.f,
        m * hz * hz / 3.f
    };

    for (int k = 0; k < 3; ++k) {
        int b = k * 3;
        M_A(b,   b)   = s[k];
        M_A(b+1, b+1) = s[k];
        M_A(b+2, b+2) = s[k];
    }
    // translation block
    M_A(9,  9)  = m;
    M_A(10, 10) = m;
    M_A(11, 11) = m;
}

// ════════════════════════════════════════════════════════════════════════════
//  ABDBody::build_stiffness_matrix  — K̄_A = Jᵀ K̄ J  (Eq.11, Eq.15)
//
//  선형 탄성 (small deformation prior, Eq.15):
//    ∂Ψ/∂A = μ(A + Aᵀ - 2I) + λ tr(A-I) I
//
//  K̄_A = V · C   (C: 9×9 material tangent, isotropic linear elasticity)
//    C_ijkl = μ(δ_ik δ_jl + δ_il δ_jk) + λ δ_ij δ_kl
//
//  vec(A) index (column-major): (i,j) → i + 3j
//    q[0..2]=col0, q[3..5]=col1, q[6..8]=col2
//  translation DOFs (q[9..11]): stiffness = 0
// ════════════════════════════════════════════════════════════════════════════

void ABDBody::build_stiffness_matrix() {
    K_bar_A.set_zero();

    float mu     = ABD_MU_LAME;
    float lambda = ABD_LAMBDA_LAME;
    float V      = 8.f * half_extents.x * half_extents.y * half_extents.z;

    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) {
            int idx_ij = i + 3*j;
            for (int l = 0; l < 3; ++l)
                for (int k = 0; k < 3; ++k) {
                    int idx_kl = k + 3*l;
                    float C_ijkl =
                        mu * ((i==k?1.f:0.f)*(j==l?1.f:0.f)
                            + (i==l?1.f:0.f)*(j==k?1.f:0.f))
                        + lambda * (i==j?1.f:0.f) * (k==l?1.f:0.f);
                    K_bar_A(idx_ij, idx_kl) = V * C_ijkl;
                }
        }
    // translation DOFs: 0 (already from set_zero)
}

// ════════════════════════════════════════════════════════════════════════════
//  ABDBody::build_H_bar — H̄_A = (1/h²)M_A + K̄_A, Cholesky (Eq.17)
// ════════════════════════════════════════════════════════════════════════════

void ABDBody::build_H_bar() { update_H_bar(ABD_H); }

void ABDBody::update_H_bar(float h) {
    float inv_h2 = 1.f / (h * h);
    for (int r = 0; r < 12; ++r)
        for (int c = 0; c < 12; ++c)
            H_bar_A(r, c) = inv_h2 * M_A(r, c) + K_bar_A(r, c);
    // 수치 안정성 정규화
    //for (int i = 0; i < 12; ++i)
    //    H_bar_A(i, i) += 1e-6f;
    H_bar_A.cholesky_factor();
}

// ════════════════════════════════════════════════════════════════════════════
//  ABDBody::build_control_tetrahedron  (Section 4.1, Eq.18)
//
//  y = T q  (12×12, constant)
//  Control points (rest-shape): 4 non-degenerate vertices inscribed in box
//    ȳ₁ = ( hx,  hy,  hz)
//    ȳ₂ = (-hx,  hy, -hz)
//    ȳ₃ = (-hx, -hy,  hz)
//    ȳ₄ = ( hx, -hy, -hz)
//
//  y_k = A ȳ_k + t  →  y_k[i] = Σ_j q[j*3+i]*ȳ_k[j] + q[9+i]
//  T[k*3+i, j*3+i] = ȳ_k[j],  T[k*3+i, 9+i] = 1
// ════════════════════════════════════════════════════════════════════════════

void ABDBody::build_control_tetrahedron() {
    float hx = half_extents.x;
    float hy = half_extents.y;
    float hz = half_extents.z;

    cp_rest[0] = Vector3( hx,  hy,  hz);
    cp_rest[1] = Vector3(-hx,  hy, -hz);
    cp_rest[2] = Vector3(-hx, -hy,  hz);
    cp_rest[3] = Vector3( hx, -hy, -hz);

    T_mat.set_zero();
    for (int k = 0; k < 4; ++k) {
        float yb[3] = {cp_rest[k].x, cp_rest[k].y, cp_rest[k].z};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j)
                T_mat(k*3+i, j*3+i) = yb[j];
            T_mat(k*3+i, 9+i) = 1.f;
        }
    }

    // T_inv via Gauss-Jordan on 12×12
    float work[144], inv[144];
    for (int i = 0; i < 144; ++i) {
        work[i] = T_mat.v[i];
        inv[i]  = (i / 12 == i % 12) ? 1.f : 0.f;
    }
    for (int col = 0; col < 12; ++col) {
        int pivot = col;
        float max_val = absf(work[col*12+col]);
        for (int row = col+1; row < 12; ++row) {
            if (absf(work[row*12+col]) > max_val) {
                max_val = absf(work[row*12+col]); pivot = row;
            }
        }
        if (pivot != col) {
            for (int c2 = 0; c2 < 12; ++c2) {
                float t1 = work[col*12+c2]; work[col*12+c2] = work[pivot*12+c2]; work[pivot*12+c2] = t1;
                float t2 = inv [col*12+c2]; inv [col*12+c2] = inv [pivot*12+c2]; inv [pivot*12+c2] = t2;
            }
        }
        float diag = work[col*12+col];
        if (absf(diag) < 1e-12f) continue;
        float id = 1.f / diag;
        for (int c2 = 0; c2 < 12; ++c2) {
            work[col*12+c2] *= id;
            inv [col*12+c2] *= id;
        }
        for (int row = 0; row < 12; ++row) {
            if (row == col) continue;
            float fac = work[row*12+col];
            for (int c2 = 0; c2 < 12; ++c2) {
                work[row*12+c2] -= fac * work[col*12+c2];
                inv [row*12+c2] -= fac * inv [col*12+c2];
            }
        }
    }
    for (int i = 0; i < 144; ++i) T_inv.v[i] = inv[i];
}

// ════════════════════════════════════════════════════════════════════════════
//  ABDBody::build_J_bar  — J̄ (volume-weighted Jacobian, Eq.14)
//
//  For uniform ABD box body:
//  J̄ is the 12×12 constant matrix that converts
//  ∂Ψ/∂A (9-vector) → f_A (12-vector).
//
//  For a box body (single element), F_e = A everywhere, so:
//    J̄ᵀ f = V · [∂Ψ/∂A ; 0₃]
//  We store J̄ = V · I₁₂ (scaled identity).
//  The actual elastic gradient is computed analytically in _compute_f_A.
// ════════════════════════════════════════════════════════════════════════════

void ABDBody::build_J_bar() {
    J_bar.set_zero();
    float V = 8.f * half_extents.x * half_extents.y * half_extents.z;
    for (int i = 0; i < 12; ++i)
        J_bar(i, i) = V;
}

// ════════════════════════════════════════════════════════════════════════════
//  ShapeData helpers
// ════════════════════════════════════════════════════════════════════════════

void ShapeData::rebuild_bsphere() {
    if (verts.empty()) { bsphere_radius = 0.f; return; }
    float r2 = 0.f;
    for (auto &v : verts) r2 = maxf(r2, (float)v.length_squared());
    bsphere_radius = sqrtf(r2);
}

void ShapeData::rebuild_edges_from_faces() {
    edges.clear();
    std::set<std::pair<int,int>> edge_set;
    for (auto &f : faces)
        for (int k = 0; k < 3; ++k) {
            int a = f.vi[k], b = f.vi[(k+1)%3];
            if (a > b) std::swap(a, b);
            edge_set.insert({a, b});
        }
    for (auto &e : edge_set) edges.push_back({e.first, e.second});
}

// ════════════════════════════════════════════════════════════════════════════
//  IPCPhysicsServer
// ════════════════════════════════════════════════════════════════════════════

IPCPhysicsServer::IPCPhysicsServer() {}

void IPCPhysicsServer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_body_count"), &IPCPhysicsServer::get_body_count);
    // M-ABD Joint API (callable from GDScript)
    ClassDB::bind_method(D_METHOD("abd_create_ball_joint",
        "body_a", "body_b", "pivot_world"),
        &IPCPhysicsServer::abd_create_ball_joint);
    ClassDB::bind_method(D_METHOD("abd_create_hinge_joint",
        "body_a", "body_b", "pivot_world", "axis_world"),
        &IPCPhysicsServer::abd_create_hinge_joint);
    ClassDB::bind_method(D_METHOD("abd_create_universal_joint",
        "body_a", "body_b", "pivot_world", "axis1_world", "axis2_world"),
        &IPCPhysicsServer::abd_create_universal_joint);
    ClassDB::bind_method(D_METHOD("abd_create_prismatic_joint",
        "body_a", "body_b", "pivot_world", "slide_axis_world"),
        &IPCPhysicsServer::abd_create_prismatic_joint);
}

int IPCPhysicsServer::get_body_count() const { return (int)_body_list.size(); }

RID IPCPhysicsServer::_make_rid() {
    uint64_t id = _next_rid_id++;
	RID r;
	static_assert(sizeof(RID) == sizeof(uint64_t), "RID size mismatch");
	memcpy(&r, &id, sizeof(uint64_t));
	return r;
}

// ── Space ─────────────────────────────────────────────────────────────────

RID IPCPhysicsServer::_space_create() {
    RID rid = _make_rid();
    _spaces.insert(rid.get_id());
    return rid;
}
void IPCPhysicsServer::_space_set_active(const RID &, bool) {}
bool IPCPhysicsServer::_space_is_active(const RID &p_space) const {
    return _spaces.count(p_space.get_id()) > 0;
}

// ── Shape builders ────────────────────────────────────────────────────────

ShapeData IPCPhysicsServer::_build_box_shape(const Vector3 &he) {
    ShapeData sd; sd.kind = ShapeKind::Box;
    float hx=he.x, hy=he.y, hz=he.z;
    sd.verts = {
        Vector3(-hx,-hy,-hz), Vector3( hx,-hy,-hz),
        Vector3( hx, hy,-hz), Vector3(-hx, hy,-hz),
        Vector3(-hx,-hy, hz), Vector3( hx,-hy, hz),
        Vector3( hx, hy, hz), Vector3(-hx, hy, hz)
    };
    std::vector<std::array<int,3>> tris = {
        {0,2,1},{0,3,2}, // -z
        {4,5,6},{4,6,7}, // +z
        {0,1,5},{0,5,4}, // -y
        {3,6,2},{3,7,6}, // +y
        {0,4,7},{0,7,3}, // -x
        {1,2,6},{1,6,5}  // +x
    };
    static const Vector3 normals[12] = {
        {0,0,-1},{0,0,-1},{0,0,1},{0,0,1},
        {0,-1,0},{0,-1,0},{0,1,0},{0,1,0},
        {-1,0,0},{-1,0,0},{1,0,0},{1,0,0}
    };
    for (int i = 0; i < 12; ++i) {
        ShapeFace f; f.vi = tris[i]; f.n = normals[i];
        sd.faces.push_back(f);
    }
    sd.rebuild_edges_from_faces();
    sd.rebuild_bsphere();
    return sd;
}

ShapeData IPCPhysicsServer::_build_convex_shape(const PackedVector3Array &pts) {
    ShapeData sd; sd.kind = ShapeKind::Convex;
    for (int i = 0; i < pts.size(); ++i) sd.verts.push_back(pts[i]);
    sd.rebuild_bsphere();
    return sd;
}

ShapeData IPCPhysicsServer::_build_concave_shape(const PackedVector3Array &faces_arr) {
    ShapeData sd; sd.kind = ShapeKind::Concave;
    int n = faces_arr.size();
    for (int i = 0; i < n; ++i) sd.verts.push_back(faces_arr[i]);
    for (int i = 0; i+2 < n; i += 3) {
        ShapeFace f; f.vi = {i, i+1, i+2};
        Vector3 a = faces_arr[i+1]-faces_arr[i], b = faces_arr[i+2]-faces_arr[i];
        Vector3 nn = a.cross(b); float ln = nn.length();
        f.n = (ln > 1e-8f) ? nn/ln : Vector3(0,1,0);
        sd.faces.push_back(f);
    }
    sd.rebuild_edges_from_faces();
    sd.rebuild_bsphere();
    return sd;
}

RID IPCPhysicsServer::_box_shape_create() {
    RID rid = _make_rid();
    _shapes[rid].kind = ShapeKind::Box;
    return rid;
}
RID IPCPhysicsServer::_convex_polygon_shape_create() {
    RID rid = _make_rid();
    _shapes[rid].kind = ShapeKind::Convex;
    return rid;
}
RID IPCPhysicsServer::_concave_polygon_shape_create() {
    RID rid = _make_rid();
    _shapes[rid].kind = ShapeKind::Concave;
    return rid;
}

void IPCPhysicsServer::_shape_set_data(const RID &p_shape, const Variant &p_data) {
    auto it = _shapes.find(p_shape);
    if (it == _shapes.end()) return;
    ShapeData &sd = it->second;

    if (sd.kind == ShapeKind::Box)
        sd = _build_box_shape((Vector3)p_data);
    else if (sd.kind == ShapeKind::Convex)
        sd = _build_convex_shape((PackedVector3Array)p_data);
    else if (sd.kind == ShapeKind::Concave)
        sd = _build_concave_shape((PackedVector3Array)p_data);

    for (auto &kv : _body_map) {
        ABDBody &b = *kv.second;
        bool used = (b.pending_shape_rid == p_shape);
        if (!used) for (auto &sr : b.shapes) if (sr.shape_rid == p_shape) { used = true; break; }
        if (used) _rebuild_body_sim_mesh(b);
    }
}

PhysicsServer3D::ShapeType IPCPhysicsServer::_shape_get_type(const RID &p_shape) const {
    auto it = _shapes.find(p_shape);
    if (it == _shapes.end()) return PhysicsServer3D::SHAPE_CUSTOM;
    switch (it->second.kind) {
        case ShapeKind::Box:     return PhysicsServer3D::SHAPE_BOX;
        case ShapeKind::Convex:  return PhysicsServer3D::SHAPE_CONVEX_POLYGON;
        case ShapeKind::Concave: return PhysicsServer3D::SHAPE_CONCAVE_POLYGON;
        default:                 return PhysicsServer3D::SHAPE_CUSTOM;
    }
}

Variant IPCPhysicsServer::_shape_get_data(const RID &p_shape) const {
    auto it = _shapes.find(p_shape);
    if (it == _shapes.end()) return Variant();
    const ShapeData &sd = it->second;
    if (sd.kind == ShapeKind::Box && sd.verts.size() >= 8)
        return sd.verts[6]; // (hx,hy,hz) corner
    return Variant();
}

// ── Body ──────────────────────────────────────────────────────────────────

RID IPCPhysicsServer::_body_create() {
    RID rid = _make_rid();
    _body_storage.emplace_back();
    ABDBody *b = &_body_storage.back();
    b->body_rid = rid;

    // Initialize: q = identity (A=I, t=0)
    b->q.fill(0.f); b->q_prev.fill(0.f); b->qdot.fill(0.f);
    b->f_ext_accum.fill(0.f);
    b->q[0] = 1.f; b->q[4] = 1.f; b->q[8] = 1.f; // A = I

    // Create direct body state
    IPCBodyState *state = memnew(IPCBodyState);
	state->body = b;
	b->direct_state = state;
	_body_states[rid] = state;

    _body_map[rid] = b;
    _body_list.push_back(b);
    return rid;
}

void IPCPhysicsServer::_body_set_space(const RID &, const RID &) {}
RID  IPCPhysicsServer::_body_get_space(const RID &) const { return RID(); }

void IPCPhysicsServer::_body_set_mode(const RID &p_body, PhysicsServer3D::BodyMode p_mode) {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end()) return;
    it->second->mode = p_mode;
    it->second->is_static = (p_mode == PhysicsServer3D::BODY_MODE_STATIC);
}

PhysicsServer3D::BodyMode IPCPhysicsServer::_body_get_mode(const RID &p_body) const {
    auto it = _body_map.find(p_body);
    return it == _body_map.end() ? PhysicsServer3D::BODY_MODE_STATIC : it->second->mode;
}

void IPCPhysicsServer::_body_add_shape(const RID &p_body, const RID &p_shape,
                                        const Transform3D &p_transform, bool p_disabled) {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end()) return;
    ABDBody &b = *it->second;
    BodyShapeRef ref; ref.shape_rid = p_shape; ref.local_xf = p_transform; ref.disabled = p_disabled;
    b.shapes.push_back(ref);
    if (_shapes.count(p_shape)) _rebuild_body_sim_mesh(b);
    else b.pending_shape_rid = p_shape;
}

void IPCPhysicsServer::_body_set_shape(const RID &p_body, int32_t idx, const RID &p_shape) {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end()) return;
    ABDBody &b = *it->second;
    if (idx < 0 || idx >= (int)b.shapes.size()) return;
    b.shapes[idx].shape_rid = p_shape;
    _rebuild_body_sim_mesh(b);
}

void IPCPhysicsServer::_body_set_shape_transform(const RID &p_body, int32_t idx, const Transform3D &xf) {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end()) return;
    ABDBody &b = *it->second;
    if (idx < 0 || idx >= (int)b.shapes.size()) return;
    b.shapes[idx].local_xf = xf;
    _rebuild_body_sim_mesh(b);
}

void IPCPhysicsServer::_body_set_shape_disabled(const RID &p_body, int32_t idx, bool dis) {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end()) return;
    ABDBody &b = *it->second;
    if (idx < 0 || idx >= (int)b.shapes.size()) return;
    b.shapes[idx].disabled = dis;
}

int32_t IPCPhysicsServer::_body_get_shape_count(const RID &p_body) const {
    auto it = _body_map.find(p_body);
    return it == _body_map.end() ? 0 : (int)it->second->shapes.size();
}
RID IPCPhysicsServer::_body_get_shape(const RID &p_body, int32_t idx) const {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end()) return RID();
    const ABDBody &b = *it->second;
    if (idx < 0 || idx >= (int)b.shapes.size()) return RID();
    return b.shapes[idx].shape_rid;
}
Transform3D IPCPhysicsServer::_body_get_shape_transform(const RID &p_body, int32_t idx) const {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end()) return Transform3D();
    const ABDBody &b = *it->second;
    if (idx < 0 || idx >= (int)b.shapes.size()) return Transform3D();
    return b.shapes[idx].local_xf;
}
void IPCPhysicsServer::_body_remove_shape(const RID &p_body, int32_t idx) {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end()) return;
    ABDBody &b = *it->second;
    if (idx < 0 || idx >= (int)b.shapes.size()) return;
    b.shapes.erase(b.shapes.begin() + idx);
    _rebuild_body_sim_mesh(b);
}
void IPCPhysicsServer::_body_clear_shapes(const RID &p_body) {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end()) return;
    it->second->shapes.clear();
    it->second->verts_local.clear();
    it->second->faces_sim.clear();
}

void IPCPhysicsServer::_body_set_state(const RID &p_body, PhysicsServer3D::BodyState p_state,
                                        const Variant &p_value) {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end()) return;
    ABDBody &b = *it->second;
    switch (p_state) {
        case PhysicsServer3D::BODY_STATE_TRANSFORM:
            b.set_transform((Transform3D)p_value);
            b.q_prev = b.q;
            break;
        case PhysicsServer3D::BODY_STATE_LINEAR_VELOCITY:
            b.set_linear_velocity((Vector3)p_value); break;
        case PhysicsServer3D::BODY_STATE_ANGULAR_VELOCITY:
            b.set_angular_velocity((Vector3)p_value); break;
        default: break;
    }
}

Variant IPCPhysicsServer::_body_get_state(const RID &p_body, PhysicsServer3D::BodyState p_state) const {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end()) return Variant();
    const ABDBody &b = *it->second;
    switch (p_state) {
        case PhysicsServer3D::BODY_STATE_TRANSFORM:         return b.get_transform();
        case PhysicsServer3D::BODY_STATE_LINEAR_VELOCITY:   return b.get_linear_velocity();
        case PhysicsServer3D::BODY_STATE_ANGULAR_VELOCITY:  return b.get_angular_velocity();
        case PhysicsServer3D::BODY_STATE_SLEEPING:          return false;
        default: return Variant();
    }
}

// ── Impulse / Force ───────────────────────────────────────────────────────

void IPCPhysicsServer::_body_apply_central_impulse(const RID &p_body, const Vector3 &imp) {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end() || it->second->is_static) return;
    ABDBody &b = *it->second;
    float inv_m = (b.mass > 0.f) ? 1.f/b.mass : 0.f;
    b.qdot[9]  += imp.x * inv_m;
    b.qdot[10] += imp.y * inv_m;
    b.qdot[11] += imp.z * inv_m;
}

void IPCPhysicsServer::_body_apply_impulse(const RID &p_body,
                                            const Vector3 &imp, const Vector3 &pos) {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end() || it->second->is_static) return;
    ABDBody &b = *it->second;
    // Linear part
    float inv_m = (b.mass > 0.f) ? 1.f/b.mass : 0.f;
    b.qdot[9]  += imp.x * inv_m;
    b.qdot[10] += imp.y * inv_m;
    b.qdot[11] += imp.z * inv_m;
    // Angular part: r = pos - COM, torque = r × imp
    Vector3 r = pos - b.get_t();
    Vector3 tau = r.cross(imp);
    // Apply via ȧ_k += ω_delta × a_k, ω_delta = I^{-1} τ
    float hx = b.half_extents.x, hy = b.half_extents.y, hz = b.half_extents.z;
    float m = b.mass;
    if (m > 0.f) {
        Vector3 Iinv(3.f/(m*(hy*hy+hz*hz)),
                     3.f/(m*(hx*hx+hz*hz)),
                     3.f/(m*(hx*hx+hy*hy)));
        Vector3 domega(tau.x*Iinv.x, tau.y*Iinv.y, tau.z*Iinv.z);
        for (int k = 0; k < 3; ++k) {
            Vector3 ak(b.q[k*3], b.q[k*3+1], b.q[k*3+2]);
            Vector3 da = domega.cross(ak);
            b.qdot[k*3]   += da.x;
            b.qdot[k*3+1] += da.y;
            b.qdot[k*3+2] += da.z;
        }
    }
}

void IPCPhysicsServer::_body_apply_torque_impulse(const RID &p_body, const Vector3 &imp) {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end() || it->second->is_static) return;
    ABDBody &b = *it->second;
    float hx = b.half_extents.x, hy = b.half_extents.y, hz = b.half_extents.z;
    float m = b.mass;
    if (m <= 0.f) return;
    Vector3 Iinv(3.f/(m*(hy*hy+hz*hz)),
                 3.f/(m*(hx*hx+hz*hz)),
                 3.f/(m*(hx*hx+hy*hy)));
    Vector3 domega(imp.x*Iinv.x, imp.y*Iinv.y, imp.z*Iinv.z);
    for (int k = 0; k < 3; ++k) {
        Vector3 ak(b.q[k*3], b.q[k*3+1], b.q[k*3+2]);
        Vector3 da = domega.cross(ak);
        b.qdot[k*3]   += da.x;
        b.qdot[k*3+1] += da.y;
        b.qdot[k*3+2] += da.z;
    }
}

// force는 f_ext_accum에 누적 (매 스텝 solver에서 소비 후 초기화)
void IPCPhysicsServer::_body_apply_central_force(const RID &p_body, const Vector3 &f) {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end() || it->second->is_static) return;
    ABDBody &b = *it->second;
    // G^T [0; f] 적용: translation block만
    b.f_ext_accum[9]  += f.x;
    b.f_ext_accum[10] += f.y;
    b.f_ext_accum[11] += f.z;
}

void IPCPhysicsServer::_body_apply_force(const RID &p_body,
                                          const Vector3 &f, const Vector3 &pos) {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end() || it->second->is_static) return;
    ABDBody &b = *it->second;
    // G^T [r×f; f] where r = pos - COM
    Vector3 r = pos - b.get_t();
    Vector3 tau = r.cross(f);
    float A[3][3]; b.get_A(A);
    ABDBody::apply_G_transpose(A, tau, f, b.f_ext_accum);
}

void IPCPhysicsServer::_body_apply_torque(const RID &p_body, const Vector3 &t) {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end() || it->second->is_static) return;
    ABDBody &b = *it->second;
    float A[3][3]; b.get_A(A);
    ABDBody::apply_G_transpose(A, t, Vector3(), b.f_ext_accum);
}

void IPCPhysicsServer::_body_set_state_sync_callback(const RID &p_body,
                                                      const Callable &p_callable) {
    _sync_callbacks[p_body] = p_callable;
}

PhysicsDirectBodyState3D* IPCPhysicsServer::_body_get_direct_state(const RID &p_body) {
    auto it = _body_map.find(p_body);
    return it == _body_map.end() ? nullptr : it->second->direct_state;
}

void IPCPhysicsServer::_body_attach_object_instance_id(const RID &p_body, uint64_t p_id) {
    auto it = _body_map.find(p_body);
    if (it != _body_map.end()) it->second->instance_id = p_id;
}
uint64_t IPCPhysicsServer::_body_get_object_instance_id(const RID &p_body) const {
    auto it = _body_map.find(p_body);
    return it != _body_map.end() ? it->second->instance_id : 0;
}

void IPCPhysicsServer::_body_set_param(const RID &p_body,
                                        PhysicsServer3D::BodyParameter p_param,
                                        const Variant &p_value) {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end()) return;
    ABDBody &b = *it->second;
    if (p_param == PhysicsServer3D::BODY_PARAM_MASS)
        _recalc_mass_props(b, (float)(double)p_value);
}

Variant IPCPhysicsServer::_body_get_param(const RID &p_body,
                                           PhysicsServer3D::BodyParameter p_param) const {
    auto it = _body_map.find(p_body);
    if (it == _body_map.end()) return Variant();
    const ABDBody &b = *it->second;
    if (p_param == PhysicsServer3D::BODY_PARAM_MASS)    return (double)b.mass;
    if (p_param == PhysicsServer3D::BODY_PARAM_GRAVITY_SCALE) return 1.0;
    return Variant();
}

void IPCPhysicsServer::_body_reset_mass_properties(const RID &p_body) {
    auto it = _body_map.find(p_body);
    if (it != _body_map.end()) _recalc_mass_props(*it->second, it->second->mass);
}

void IPCPhysicsServer::_free_rid(const RID &p_rid) {
    _shapes.erase(p_rid);
	{
        auto sit = _body_states.find(p_rid);
        if (sit != _body_states.end()) { memdelete(sit->second); _body_states.erase(sit); }
    }
	
    auto it = _body_map.find(p_rid);
    if (it != _body_map.end()) {
        ABDBody *b = it->second;
        _body_map.erase(it);
        _body_list.erase(std::find(_body_list.begin(), _body_list.end(), b));
        for (auto sit = _body_storage.begin(); sit != _body_storage.end(); ++sit)
		{
            if (&(*sit) == b) 
			{ 
				_body_storage.erase(sit); 
				break; 
			}
		}
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

void IPCPhysicsServer::_recalc_mass_props(ABDBody &b, float mass) {
    b.mass     = mass;
    b.inv_mass = (mass > 0.f) ? 1.f/mass : 0.f;
    b.build_mass_matrix();
    b.build_stiffness_matrix();
    b.update_H_bar(_last_h);
}

void IPCPhysicsServer::_rebuild_body_sim_mesh(ABDBody &b) {
    b.verts_local.clear();
    b.faces_sim.clear();

    for (auto &sr : b.shapes) {
        if (sr.disabled) continue;
        auto sit = _shapes.find(sr.shape_rid);
        if (sit == _shapes.end()) continue;
        const ShapeData &sd = sit->second;

        int base = (int)b.verts_local.size();
        for (auto &v : sd.verts)
            b.verts_local.push_back(sr.local_xf.xform(v));
        for (auto &f : sd.faces) {
            ShapeFace sf;
            sf.vi = {f.vi[0]+base, f.vi[1]+base, f.vi[2]+base};
            sf.n  = sr.local_xf.basis.xform(f.n).normalized();
            b.faces_sim.push_back(sf);
        }
        // DBG: verify face normals point outward
        UtilityFunctions::print("[DBG_MESH] body rebuild: n_verts=", (int)b.verts_local.size(),
            " n_faces=", (int)b.faces_sim.size(),
            " half_ext=(", b.half_extents.x, ",", b.half_extents.y, ",", b.half_extents.z, ")");
        for (int _fi = 0; _fi < (int)b.faces_sim.size() && _fi < 12; ++_fi) {
            const auto &_sf = b.faces_sim[_fi];
            // Centroid of this face
            Vector3 _c = (b.verts_local[_sf.vi[0]] + b.verts_local[_sf.vi[1]] + b.verts_local[_sf.vi[2]]) * (1.f/3.f);
            // Should point away from body center (0,0,0 in local space)
            float _dot = _c.dot(_sf.n);
            UtilityFunctions::print("  face[", _fi, "] n=(", _sf.n.x, ",", _sf.n.y, ",", _sf.n.z, ")",
                " centroid=(", _c.x, ",", _c.y, ",", _c.z, ")",
                " n·centroid=", _dot, " (>0 means outward, <0 means INWARD/WRONG)");
        }
        // half_extents from box shape
        if (sd.kind == ShapeKind::Box && sd.verts.size() >= 8) {
            b.half_extents = sd.verts[6]; // (+hx,+hy,+hz) vertex
            b.half_extents.x *= absf(sr.local_xf.basis[0][0]);
            b.half_extents.y *= absf(sr.local_xf.basis[1][1]);
            b.half_extents.z *= absf(sr.local_xf.basis[2][2]);
        }
    }

    b.build_mass_matrix();
    b.build_stiffness_matrix();
    b.build_control_tetrahedron();
    b.build_J_bar();
    b.update_H_bar(_last_h);
    b.sim_started = false;
}

// ── BVH ───────────────────────────────────────────────────────────────────

void IPCPhysicsServer::_bvh_rebuild() {
    _bvh_pool.clear(); _bvh_root = -1;
    int N = (int)_body_list.size();
    if (N == 0) return;
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    _bvh_root = _bvh_build(indices, 0, N);
}

int IPCPhysicsServer::_bvh_build(std::vector<int> &indices, int lo, int hi) {
    if (lo >= hi) return -1;
    int node_idx = (int)_bvh_pool.size();
    _bvh_pool.emplace_back();

    if (hi - lo == 1) {
        int bi = indices[lo];
        const ABDBody &b = *_body_list[bi];
        BVHNode &node = _bvh_pool[node_idx];
        node.body_idx = bi;
        node.center   = b.get_t();
        float br = 0.f;
        for (auto &v : b.verts_local)
            br = maxf(br, (b.world_point(v) - node.center).length());
        node.radius = br + 0.05f;
        return node_idx;
    }

    Vector3 center_sum;
    for (int i = lo; i < hi; ++i) center_sum += _body_list[indices[i]]->get_t();
    Vector3 mid = center_sum / (float)(hi - lo);

    Vector3 extent;
    for (int i = lo; i < hi; ++i) {
        Vector3 d = _body_list[indices[i]]->get_t() - mid;
        extent.x = maxf(extent.x, absf(d.x));
        extent.y = maxf(extent.y, absf(d.y));
        extent.z = maxf(extent.z, absf(d.z));
    }
    int axis = (extent.y > extent.x && extent.y > extent.z) ? 1 : (extent.z > extent.x ? 2 : 0);

    std::sort(indices.begin()+lo, indices.begin()+hi, [&](int a, int b_) {
        Vector3 ta = _body_list[a]->get_t(), tb = _body_list[b_]->get_t();
        return axis==0 ? ta.x<tb.x : axis==1 ? ta.y<tb.y : ta.z<tb.z;
    });

    int mid_idx = lo + (hi-lo)/2;
    int left  = _bvh_build(indices, lo,      mid_idx);
    int right = _bvh_build(indices, mid_idx, hi);

    BVHNode &node = _bvh_pool[node_idx];
    node.body_idx = -1; node.left = left; node.right = right;
    if (left >= 0 && right >= 0) {
        const BVHNode &l = _bvh_pool[left], &r = _bvh_pool[right];
        node.center = (l.center + r.center) * 0.5f;
        node.radius = maxf((node.center-l.center).length()+l.radius,
                           (node.center-r.center).length()+r.radius);
    } else if (left >= 0) {
        node.center = _bvh_pool[left].center;
        node.radius = _bvh_pool[left].radius;
    }
    return node_idx;
}

void IPCPhysicsServer::_bvh_collect_internal_pairs(int node, std::vector<std::pair<int,int>> &out) const {
    if (node < 0) return;
    const BVHNode &n = _bvh_pool[node];
    if (n.left >= 0 && n.right >= 0) {
        _bvh_query_pairs(n.left, n.right, out);
        _bvh_collect_internal_pairs(n.left, out);
        _bvh_collect_internal_pairs(n.right, out);
    }
}

void IPCPhysicsServer::_bvh_query_pairs(int a, int b, std::vector<std::pair<int,int>> &out) const {
    if (a < 0 || b < 0) return;
    const BVHNode &na = _bvh_pool[a], &nb = _bvh_pool[b];
    if ((na.center-nb.center).length() > na.radius+nb.radius+ABD_CONTACT_D_HAT) return;
    if (na.body_idx >= 0 && nb.body_idx >= 0) {
        if (na.body_idx != nb.body_idx) out.push_back({na.body_idx, nb.body_idx});
        return;
    }
    if (na.body_idx >= 0) {
        _bvh_query_pairs(a, nb.left,  out);
        _bvh_query_pairs(a, nb.right, out);
    } else if (nb.body_idx >= 0) {
        _bvh_query_pairs(na.left,  b, out);
        _bvh_query_pairs(na.right, b, out);
    } else {
        _bvh_query_pairs(na.left,  nb.left,  out);
        _bvh_query_pairs(na.left,  nb.right, out);
        _bvh_query_pairs(na.right, nb.left,  out);
        _bvh_query_pairs(na.right, nb.right, out);
    }
}

// ── Contact detection ─────────────────────────────────────────────────────

float IPCPhysicsServer::_pt_distance(const Vector3 &p,
                                      const Vector3 &a, const Vector3 &b_v,
                                      const Vector3 &c) const {
    Vector3 ab=b_v-a, ac=c-a, ap=p-a;
    float d1=ab.dot(ap), d2=ac.dot(ap);
    if (d1<=0.f && d2<=0.f) return (p-a).length();
    Vector3 bp=p-b_v; float d3=ab.dot(bp), d4=ac.dot(bp);
    if (d3>=0.f && d4<=d3) return (p-b_v).length();
    Vector3 cp_v=p-c; float d5=ab.dot(cp_v), d6=ac.dot(cp_v);
    if (d6>=0.f && d5<=d6) return (p-c).length();
    float vc=d1*d4-d3*d2;
    if (vc<=0.f && d1>=0.f && d3<=0.f) {
        float v=d1/(d1-d3); return (p-(a+v*ab)).length();
    }
    float vb=d5*d2-d1*d6;
    if (vb<=0.f && d2>=0.f && d6<=0.f) {
        float v=d2/(d2-d6); return (p-(a+v*ac)).length();
    }
    float va=d3*d6-d5*d4;
    if (va<=0.f && (d4-d3)>=0.f && (d5-d6)>=0.f) {
        float v=(d4-d3)/((d4-d3)+(d5-d6)); return (p-(b_v+v*(c-b_v))).length();
    }
    float denom=1.f/(va+vb+vc), v2=vb*denom, w2=vc*denom;
    return (p-(a+ab*v2+ac*w2)).length();
}

// ── Contact detection helpers ─────────────────────────────────────────────
//
// 볼록 메쉬에 대한 꼭짓점 침투 판정 (올바른 알고리즘):
//
// 꼭짓점 p가 body b의 볼록 메쉬에 얼마나 침투했는지 판정.
// 각 face에 대해 world-space signed distance를 계산.
//   sd_i = (p - face_vertex_i) · world_normal_i
//   sd > 0: 그 면의 바깥쪽  (p가 그 면 밖에 있음)
//   sd < 0: 그 면의 안쪽
//
// 볼록 메쉬 내부 판정: 모든 면에서 sd < 0  (=모든 면의 안쪽)
// 접촉 판정: max_i(sd_i) < d_hat  (=가장 얕은 침투면이 d_hat 이내)
//
// contact face는 max sd를 주는 면 (가장 얕은 침투 = 분리 방향에 가장 가까운 면).
// contact dist = max_sd  (양수=분리됨, 음수=침투, dist < d_hat 이면 force 발생).
//
// 주의: faces_sim.n은 local-space normal이므로,
//       world-space로 변환해야 함: n_world = A * n_local  (A가 현재 body rotation).

// 꼭짓점 p (world) vs 볼록 메쉬 body의 모든 면.
// d_hat 이내인 face를 모두 out_contacts에 추가.
// 반환값: 하나라도 hit이면 true.
//
// 이전 단일-face 버전(max_sd만 반환)은 코너/모서리에 낀 경우
// 두 번째 이후 면의 반발력이 누락되는 문제가 있었음.
// → d_hat 이내 모든 face를 독립 contact pair로 등록해 해결.
struct FaceContact {
    float   dist;
    int     face_idx;
    Vector3 n_world;
};

static void convex_vertex_contacts_all(
        const Vector3 &p_world,
        const ABDBody &body,
        float d_hat,
        std::vector<FaceContact> &out_contacts)
{
    float A[3][3]; body.get_A(A);
    int nf = (int)body.faces_sim.size();

    // 볼록 메쉬 내부 판정: 모든 면에서 sd < 0 이어야 내부.
    // max_sd >= 0 이면 이미 외부 → 접촉 없음.
    // max_sd < d_hat 이면 적어도 하나의 면이 d_hat 이내.
    //
    // 각 면 개별적으로:
    //   sd_i = (p - face_v0_i) · n_w_i
    //   sd_i < d_hat → 이 면 방향으로 contact force 필요
    //
    // 단, 볼록 메쉬 외부인지 먼저 확인 (max_sd >= 0 이면 외부).

    float max_sd = -1e30f;
    for (int fi = 0; fi < nf; ++fi) {
        const ShapeFace &f = body.faces_sim[fi];
        const Vector3 &nl = f.n;
        Vector3 n_w(
            A[0][0]*nl.x + A[0][1]*nl.y + A[0][2]*nl.z,
            A[1][0]*nl.x + A[1][1]*nl.y + A[1][2]*nl.z,
            A[2][0]*nl.x + A[2][1]*nl.y + A[2][2]*nl.z
        );
        float n_len = n_w.length();
        if (n_len < 1e-8f) continue;
        n_w = n_w / n_len;
        Vector3 face_v0 = body.world_point(body.verts_local[f.vi[0]]);
        float sd = (p_world - face_v0).dot(n_w);
        if (sd > max_sd) max_sd = sd;
    }

    // max_sd >= d_hat: 메쉬와 충분히 분리 → contact 없음
    if (max_sd >= d_hat) return;
    // max_sd >= 0: 메쉬 외부이지만 d_hat 이내 → 가장 가까운 면만
    // max_sd < 0:  메쉬 내부 (침투) → 모든 d_hat 이내 면 등록

    for (int fi = 0; fi < nf; ++fi) {
        const ShapeFace &f = body.faces_sim[fi];
        const Vector3 &nl = f.n;
        Vector3 n_w(
            A[0][0]*nl.x + A[0][1]*nl.y + A[0][2]*nl.z,
            A[1][0]*nl.x + A[1][1]*nl.y + A[1][2]*nl.z,
            A[2][0]*nl.x + A[2][1]*nl.y + A[2][2]*nl.z
        );
        float n_len = n_w.length();
        if (n_len < 1e-8f) continue;
        n_w = n_w / n_len;
        Vector3 face_v0 = body.world_point(body.verts_local[f.vi[0]]);
        float sd = (p_world - face_v0).dot(n_w);

        if (max_sd >= 0.f) {
            // 외부: max_sd 면(가장 가까운 면)만 등록
            if (sd == max_sd && sd < d_hat) {
                out_contacts.push_back({sd, fi, n_w});
            }
        } else {
            // 내부(침투): d_hat 이내 모든 면 등록 → 코너 대응
            if (sd < d_hat) {
                out_contacts.push_back({sd, fi, n_w});
            }
        }
    }
}

void IPCPhysicsServer::_find_contact_pairs(std::vector<ABDContactPair> &out) const {
    static int s_cp_step = 0; ++s_cp_step;
    const bool CPDBG = (s_cp_step <= 5) || (s_cp_step % 30 == 0);

    out.clear();
    if (_bvh_root < 0) {
        if (CPDBG) UtilityFunctions::print("[DBG_CP] step=", s_cp_step, " bvh_root<0, skip");
        return;
    }
    std::vector<std::pair<int,int>> broad;
    _bvh_collect_internal_pairs(_bvh_root, broad);

    if (CPDBG) {
        UtilityFunctions::print("[DBG_CP] step=", s_cp_step,
            " n_bodies=", (int)_body_list.size(),
            " broad_pairs=", (int)broad.size(),
            " ABD_CONTACT_D_HAT=", ABD_CONTACT_D_HAT);
        for (int _bi = 0; _bi < (int)_body_list.size(); ++_bi) {
            const ABDBody &_b = *_body_list[_bi];
            Vector3 _t = _b.get_t();
            UtilityFunctions::print("  body[", _bi, "] static=", _b.is_static,
                " t=(", _t.x, ",", _t.y, ",", _t.z, ")",
                " verts=", (int)_b.verts_local.size(),
                " faces=", (int)_b.faces_sim.size());
        }
    }

    for (auto &[ia, ib] : broad) {
        const ABDBody &ba = *_body_list[ia];
        const ABDBody &bb = *_body_list[ib];
        if (ba.is_static && bb.is_static) continue;

        if (CPDBG) UtilityFunctions::print("  [DBG_CP] broad pair ia=", ia, " ib=", ib,
            " static=(", ba.is_static, ",", bb.is_static, ")");

        // ── ba 꼭짓점 vs bb 볼록 메쉬 ──────────────────────────────────────
        for (int vi = 0; vi < (int)ba.verts_local.size(); ++vi) {
            Vector3 wp = ba.world_point(ba.verts_local[vi]);
            std::vector<FaceContact> fc;
            convex_vertex_contacts_all(wp, bb, ABD_CONTACT_D_HAT, fc);

            if (CPDBG && vi == 0) {
                UtilityFunctions::print("    [DBG_CP] ia=", ia, " vi=0 vs bb:",
                    " hit=", !fc.empty(), " n_contacts=", (int)fc.size(),
                    " wp=(", wp.x, ",", wp.y, ",", wp.z, ")");
            }

            for (const FaceContact &fci : fc) {
                ABDContactPair cp;
                cp.body_a = ia; cp.body_b = ib;
                cp.pa     = vi;
                cp.tri    = bb.faces_sim[fci.face_idx].vi;
                cp.dist   = fci.dist;
                cp.n_hat  = fci.n_world;
                out.push_back(cp);
            }
        }

        // ── bb 꼭짓점 vs ba 볼록 메쉬 ──────────────────────────────────────
        for (int vi = 0; vi < (int)bb.verts_local.size(); ++vi) {
            Vector3 wp = bb.world_point(bb.verts_local[vi]);
            std::vector<FaceContact> fc;
            convex_vertex_contacts_all(wp, ba, ABD_CONTACT_D_HAT, fc);

            if (CPDBG && vi == 0) {
                UtilityFunctions::print("    [DBG_CP] ib=", ib, " vi=0 vs ba:",
                    " hit=", !fc.empty(), " n_contacts=", (int)fc.size(),
                    " wp=(", wp.x, ",", wp.y, ",", wp.z, ")");
            }

            for (const FaceContact &fci : fc) {
                ABDContactPair cp;
                cp.body_a = ib; cp.body_b = ia;
                cp.pa     = vi;
                cp.tri    = ba.faces_sim[fci.face_idx].vi;
                cp.dist   = fci.dist;
                cp.n_hat  = fci.n_world;
                out.push_back(cp);
            }
        }
    }

    if (CPDBG) UtilityFunctions::print("[DBG_CP] total contacts registered=", (int)out.size());
}

// ── Step ──────────────────────────────────────────────────────────────────

void IPCPhysicsServer::_set_active(bool p_active) { _is_active = p_active; }

void IPCPhysicsServer::_step(float p_step) {
    if (!_is_active) return;
    _last_h = p_step;
    _abd_step(p_step);
    _dispatch_sync_callbacks();
}

void IPCPhysicsServer::_dispatch_sync_callbacks() {
    for (auto &kv : _sync_callbacks) {
        auto bit = _body_map.find(kv.first);
        if (bit == _body_map.end()) continue;
        ABDBody *b = bit->second;
        if (b->direct_state) kv.second.call(b->direct_state);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Joint implementation  (Section 4)
// ════════════════════════════════════════════════════════════════════════════

// ── _find_closest_cp: world point → nearest CP index (0-3) ───────────────
int IPCPhysicsServer::_find_closest_cp(const ABDBody &b, const Vector3 &world_pt) {
    int best = 0;
    float best_d2 = 1e30f;
    for (int k = 0; k < 4; ++k) {
        float d2 = (abd_get_cp(b, k) - world_pt).length_squared();
        if (d2 < best_d2) { best_d2 = d2; best = k; }
    }
    return best;
}

// ── _joint_alloc: allocate joint and populate body indices ────────────────
ABDJoint *IPCPhysicsServer::_joint_alloc(const RID &joint_rid,
                                          const RID &body_a, const RID &body_b,
                                          ABDJointType type) {
    _joint_storage.emplace_back();
    ABDJoint *j = &_joint_storage.back();
    _joint_map[joint_rid] = j;

    // Map body RIDs to list indices
    auto ia = _body_map.find(body_a);
    auto ib = _body_map.find(body_b);
    if (ia != _body_map.end()) {
        for (int i = 0; i < (int)_body_list.size(); ++i)
            if (_body_list[i] == ia->second) { j->body_alpha = i; break; }
    }
    if (ib != _body_map.end()) {
        for (int i = 0; i < (int)_body_list.size(); ++i)
            if (_body_list[i] == ib->second) { j->body_beta = i; break; }
    }
    j->type = type;
    return j;
}

// ── Godot joint API ───────────────────────────────────────────────────────

RID IPCPhysicsServer::_joint_create() {
    RID rid = _make_rid();
    // 빈 슬롯 예약 — 실제 joint는 _joint_make_* 에서 채움
    // 헤더에 RID 등록만 해두고 storage는 _joint_make_*에서 생성
    return rid;
}

void IPCPhysicsServer::_joint_clear(const RID &p_joint) {
    auto it = _joint_map.find(p_joint);
    if (it == _joint_map.end()) return;
    ABDJoint *j = it->second;
    _joint_map.erase(it);
    _joint_type_map.erase(p_joint);
    for (auto sit = _joint_storage.begin(); sit != _joint_storage.end(); ++sit)
        if (&(*sit) == j) { _joint_storage.erase(sit); break; }
}

// Ball joint ← pin (Eq.19, 3-DOF constraint)
void IPCPhysicsServer::_joint_make_pin(const RID &p_joint,
                                        const RID &p_body_a, const Vector3 &p_local_a,
                                        const RID &p_body_b, const Vector3 &p_local_b) {
    _joint_clear(p_joint);
    auto ia = _body_map.find(p_body_a);
    auto ib = _body_map.find(p_body_b);
    if (ia == _body_map.end() || ib == _body_map.end()) return;

    ABDJoint *j = _joint_alloc(p_joint, p_body_a, p_body_b, ABDJointType::Ball);
    _joint_type_map[p_joint] = PhysicsServer3D::JOINT_TYPE_PIN;

    // Find CPs closest to local pivot points (transformed to world)
    Vector3 pivot_a = ia->second->get_transform().xform(p_local_a);
    Vector3 pivot_b = ib->second->get_transform().xform(p_local_b);
    j->cp_alpha = _find_closest_cp(*ia->second, pivot_a);
    j->cp_beta  = _find_closest_cp(*ib->second, pivot_b);
}

// Hinge joint (Eq.21, 5-DOF constraint)
void IPCPhysicsServer::_joint_make_hinge(const RID &p_joint,
                                          const RID &p_body_a, const Transform3D &p_hinge_a,
                                          const RID &p_body_b, const Transform3D &p_hinge_b) {
    _joint_clear(p_joint);
    auto ia = _body_map.find(p_body_a);
    auto ib = _body_map.find(p_body_b);
    if (ia == _body_map.end() || ib == _body_map.end()) return;

    ABDJoint *j = _joint_alloc(p_joint, p_body_a, p_body_b, ABDJointType::Hinge);
    _joint_type_map[p_joint] = PhysicsServer3D::JOINT_TYPE_HINGE;

    // Hinge axis: local x of the hinge transform → world
    Vector3 axis_world = ia->second->get_transform().basis.xform(p_hinge_a.basis.get_column(0)).normalized();
    j->hinge_axis_world = axis_world;

    // Pivot point (origin of hinge transform in world)
    Vector3 pivot = ia->second->get_transform().xform(p_hinge_a.origin);

    // Pick two CPs on the rotation axis (one on each side of pivot)
    // Strategy: find the two CPs nearest to the axis line
    // For simplicity: find 2 CPs closest to pivot for each body
    j->cp_alpha_1 = _find_closest_cp(*ia->second, pivot);
    j->cp_alpha_2 = (j->cp_alpha_1 + 1) % 4;
    j->cp_beta_1  = _find_closest_cp(*ib->second, pivot);
    j->cp_beta_2  = (j->cp_beta_1  + 1) % 4;
}

void IPCPhysicsServer::_joint_make_hinge_simple(const RID &p_joint,
                                                 const RID &p_body_a,
                                                 const Vector3 &p_pivot_a,
                                                 const Vector3 &p_axis_a,
                                                 const RID &p_body_b,
                                                 const Vector3 &p_pivot_b,
                                                 const Vector3 &p_axis_b) {
    _joint_clear(p_joint);
    auto ia = _body_map.find(p_body_a);
    auto ib = _body_map.find(p_body_b);
    if (ia == _body_map.end() || ib == _body_map.end()) return;

    ABDJoint *j = _joint_alloc(p_joint, p_body_a, p_body_b, ABDJointType::Hinge);
    _joint_type_map[p_joint] = PhysicsServer3D::JOINT_TYPE_HINGE;

    // Axis in world space
    Vector3 axis_world = ia->second->get_transform().basis.xform(p_axis_a).normalized();
    j->hinge_axis_world = axis_world;

    Vector3 pivot_a_w = ia->second->get_transform().xform(p_pivot_a);
    Vector3 pivot_b_w = ib->second->get_transform().xform(p_pivot_b);

    j->cp_alpha_1 = _find_closest_cp(*ia->second, pivot_a_w);
    j->cp_alpha_2 = (j->cp_alpha_1 + 1) % 4;
    j->cp_beta_1  = _find_closest_cp(*ib->second, pivot_b_w);
    j->cp_beta_2  = (j->cp_beta_1  + 1) % 4;
}

// Slider (prismatic) joint (Eq.25, 5-DOF constraint)
void IPCPhysicsServer::_joint_make_slider(const RID &p_joint,
                                           const RID &p_body_a, const Transform3D &p_local_a,
                                           const RID &p_body_b, const Transform3D &p_local_b) {
    _joint_clear(p_joint);
    auto ia = _body_map.find(p_body_a);
    auto ib = _body_map.find(p_body_b);
    if (ia == _body_map.end() || ib == _body_map.end()) return;

    ABDJoint *j = _joint_alloc(p_joint, p_body_a, p_body_b, ABDJointType::Prismatic);
    _joint_type_map[p_joint] = PhysicsServer3D::JOINT_TYPE_SLIDER;

    // Slide axis: local x of the ref transform
    Vector3 axis = ia->second->get_transform().basis.xform(p_local_a.basis.get_column(0)).normalized();
    j->prismatic_axis = axis;

    Vector3 pivot = ia->second->get_transform().xform(p_local_a.origin);
    j->cp_alpha   = _find_closest_cp(*ia->second, pivot);
    j->cp_alpha_1 = j->cp_alpha;
    j->cp_alpha_2 = (j->cp_alpha + 1) % 4;
    j->cp_beta    = _find_closest_cp(*ib->second, pivot);
    j->cp_beta_1  = j->cp_beta;
    j->cp_beta_2  = (j->cp_beta + 1) % 4;
}

// Cone-twist → Universal joint (Eq.23, 4-DOF constraint)
void IPCPhysicsServer::_joint_make_cone_twist(const RID &p_joint,
                                               const RID &p_body_a, const Transform3D &p_local_a,
                                               const RID &p_body_b, const Transform3D &p_local_b) {
    _joint_clear(p_joint);
    auto ia = _body_map.find(p_body_a);
    auto ib = _body_map.find(p_body_b);
    if (ia == _body_map.end() || ib == _body_map.end()) return;

    ABDJoint *j = _joint_alloc(p_joint, p_body_a, p_body_b, ABDJointType::Universal);
    _joint_type_map[p_joint] = PhysicsServer3D::JOINT_TYPE_CONE_TWIST;

    // Two orthogonal axes from the local transforms
    Basis ba_world = ia->second->get_transform().basis * p_local_a.basis;
    j->univ_axis_1 = ba_world.get_column(0).normalized();  // x axis
    j->univ_axis_2 = ba_world.get_column(2).normalized();  // z axis

    Vector3 pivot = ia->second->get_transform().xform(p_local_a.origin);
    j->cp_alpha  = _find_closest_cp(*ia->second, pivot);
    j->cp_beta   = _find_closest_cp(*ib->second, pivot);
    j->cp_beta_2 = (j->cp_beta + 1) % 4;
}

PhysicsServer3D::JointType IPCPhysicsServer::_joint_get_type(const RID &p_joint) const {
    auto it = _joint_type_map.find(p_joint);
    return it != _joint_type_map.end() ? it->second : PhysicsServer3D::JOINT_TYPE_PIN;
}

// ── M-ABD 전용 joint 생성 API ─────────────────────────────────────────────

RID IPCPhysicsServer::abd_create_ball_joint(const RID &body_a, const RID &body_b,
                                             const Vector3 &pivot_world) {
    RID rid = _make_rid();
    auto ia = _body_map.find(body_a);
    auto ib = _body_map.find(body_b);
    if (ia == _body_map.end() || ib == _body_map.end()) return rid;

    ABDJoint *j = _joint_alloc(rid, body_a, body_b, ABDJointType::Ball);
    _joint_type_map[rid] = PhysicsServer3D::JOINT_TYPE_PIN;
    j->cp_alpha = _find_closest_cp(*ia->second, pivot_world);
    j->cp_beta  = _find_closest_cp(*ib->second, pivot_world);
    return rid;
}

RID IPCPhysicsServer::abd_create_hinge_joint(const RID &body_a, const RID &body_b,
                                              const Vector3 &pivot_world,
                                              const Vector3 &axis_world) {
    RID rid = _make_rid();
    auto ia = _body_map.find(body_a);
    auto ib = _body_map.find(body_b);
    if (ia == _body_map.end() || ib == _body_map.end()) return rid;

    ABDJoint *j = _joint_alloc(rid, body_a, body_b, ABDJointType::Hinge);
    _joint_type_map[rid] = PhysicsServer3D::JOINT_TYPE_HINGE;
    j->hinge_axis_world = axis_world.normalized();
    j->cp_alpha_1 = _find_closest_cp(*ia->second, pivot_world);
    j->cp_alpha_2 = (j->cp_alpha_1 + 1) % 4;
    j->cp_beta_1  = _find_closest_cp(*ib->second, pivot_world);
    j->cp_beta_2  = (j->cp_beta_1  + 1) % 4;
    return rid;
}

RID IPCPhysicsServer::abd_create_universal_joint(const RID &body_a, const RID &body_b,
                                                  const Vector3 &pivot_world,
                                                  const Vector3 &axis1_world,
                                                  const Vector3 &axis2_world) {
    RID rid = _make_rid();
    auto ia = _body_map.find(body_a);
    auto ib = _body_map.find(body_b);
    if (ia == _body_map.end() || ib == _body_map.end()) return rid;

    ABDJoint *j = _joint_alloc(rid, body_a, body_b, ABDJointType::Universal);
    _joint_type_map[rid] = PhysicsServer3D::JOINT_TYPE_CONE_TWIST;
    j->univ_axis_1 = axis1_world.normalized();
    j->univ_axis_2 = axis2_world.normalized();
    j->cp_alpha  = _find_closest_cp(*ia->second, pivot_world);
    j->cp_beta   = _find_closest_cp(*ib->second, pivot_world);
    j->cp_beta_2 = (j->cp_beta + 1) % 4;
    return rid;
}

RID IPCPhysicsServer::abd_create_prismatic_joint(const RID &body_a, const RID &body_b,
                                                  const Vector3 &pivot_world,
                                                  const Vector3 &slide_axis_world) {
    RID rid = _make_rid();
    auto ia = _body_map.find(body_a);
    auto ib = _body_map.find(body_b);
    if (ia == _body_map.end() || ib == _body_map.end()) return rid;

    ABDJoint *j = _joint_alloc(rid, body_a, body_b, ABDJointType::Prismatic);
    _joint_type_map[rid] = PhysicsServer3D::JOINT_TYPE_SLIDER;
    j->prismatic_axis = slide_axis_world.normalized();
    j->cp_alpha   = _find_closest_cp(*ia->second, pivot_world);
    j->cp_alpha_1 = j->cp_alpha;
    j->cp_alpha_2 = (j->cp_alpha + 1) % 4;
    j->cp_beta    = _find_closest_cp(*ib->second, pivot_world);
    j->cp_beta_1  = j->cp_beta;
    j->cp_beta_2  = (j->cp_beta + 1) % 4;
    return rid;
}

// ── _apply_joint_penalties: Section 4 penalty (legacy, now replaced by KKT) ──
// Joint constraints are now enforced via ABDKKTSolver in _abd_step.
// This stub is kept for compatibility but not called in the main loop.

} // namespace godot
