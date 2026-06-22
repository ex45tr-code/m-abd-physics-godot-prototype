#include "abd_joint.h"
#include "ipc_physics_server.h"

#include <cmath>
#include <cstring>

// ════════════════════════════════════════════════════════════════════════════
//  abd_joint.cpp  —  M-ABD Section 4 Joint Constraints (Penalty formulation)
//
//  논문 Section 4.2-4.5의 모든 joint를 penalty 에너지로 구현.
//  KKT (Section 5)는 다음 단계.
//
//  공통 구조:
//  ──────────
//  1. y = Tq  로 CP 좌표 계산
//  2. joint rotation R (R_H, R_U, R_P) 빌드
//  3. constraint C 계산 (selection matrix S 적용)
//  4. gradient ∇C 계산 (Eq.30-35)
//  5. f_A += -κ (∇C)^T C
//
//  ∂C/∂q 계산 방법 (Eq.30-31):
//  ────────────────────────────
//  C = S · diag4(R) · T · q  →  ∂C/∂q = S · diag4(R) · T
//  (R constant로 treat 시)
//
//  정확한 gradient (Eq.31-32)에서 R-derivative term도 포함하지만
//  penalty 근사에서는 R=const로 treat해도 충분히 수렴함.
//  (논문 Section 4.6: "treating R_Joint as constant")
//
//  따라서:
//    (∂C/∂q^α)^T = T^α^T · diag4(R^T) · S^α^T
//    (∂C/∂q^β)^T = T^β^T · diag4(R^T) · S^β^T  (opposite sign for β)
//
//  Implementation note:
//  ─────────────────────
//  S는 밀집 행렬이 아니라 CP-index와 component 인덱스로 자연스럽게 표현됨.
//  각 constraint 행은 "y^α_k [component] - y^β_k [component] = 0" 형태이므로
//  gradient back-projection을 add_J_y_T_r()으로 직접 계산한다.
//
// ════════════════════════════════════════════════════════════════════════════

namespace godot {

static inline float absf_(float a) { return a < 0.f ? -a : a; }

// ════════════════════════════════════════════════════════════════════════════
//  abd_compute_y  —  y = Tq:  y_k = A ȳ_k + t  (k=0..3)
// ════════════════════════════════════════════════════════════════════════════
void abd_compute_y(const ABDBody &b, float y_out[12]) {
    float A[3][3]; b.get_A(A);
    Vector3 t = b.get_t();
    for (int k = 0; k < 4; ++k) {
        const Vector3 &yr = b.cp_rest[k];
        y_out[k*3+0] = A[0][0]*yr.x + A[0][1]*yr.y + A[0][2]*yr.z + t.x;
        y_out[k*3+1] = A[1][0]*yr.x + A[1][1]*yr.y + A[1][2]*yr.z + t.y;
        y_out[k*3+2] = A[2][0]*yr.x + A[2][1]*yr.y + A[2][2]*yr.z + t.z;
    }
}

Vector3 abd_get_cp(const ABDBody &b, int k) {
    return b.world_point(b.cp_rest[k]);
}

// ════════════════════════════════════════════════════════════════════════════
//  ABDJoint::get_y  —  CP k world position from body
// ════════════════════════════════════════════════════════════════════════════
Vector3 ABDJoint::get_y(const ABDBody &b, int k) {
    return abd_get_cp(b, k);
}

// ════════════════════════════════════════════════════════════════════════════
//  ABDJoint::add_J_y_T_r  —  (∂y_k/∂q)^T r  → f  (Section 4.1, Eq.18)
//
//  y_k = Tq では:
//    y_k[i] = Σ_j (A[i][j] * ȳ_k[j]) + t[i]
//           = Σ_j q[j*3+i] * ȳ_k[j] + q[9+i]
//
//  따라서:
//    ∂y_k[i]/∂q[j*3+r] = δ_{ir} * ȳ_k[j]
//    ∂y_k[i]/∂q[9+r]   = δ_{ir}
//
//  (J_y_k^T r)[j*3+i] += ȳ_k[j] * r[i]   (j=0,1,2; i=0,1,2)
//  (J_y_k^T r)[9+i]   += r[i]              (i=0,1,2)
// ════════════════════════════════════════════════════════════════════════════
void ABDJoint::add_J_y_T_r(const ABDBody &b, int k,
                             const Vector3 &r, float out_f[12]) {
    const Vector3 &yr = b.cp_rest[k];
    float yr_arr[3] = {yr.x, yr.y, yr.z};
    float r_arr[3]  = {r.x, r.y, r.z};
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            out_f[j*3+i] += yr_arr[j] * r_arr[i];
    out_f[9]  += r.x;
    out_f[10] += r.y;
    out_f[11] += r.z;
}

// ════════════════════════════════════════════════════════════════════════════
//  ABDJoint::build_R_joint  —  joint rotation from current body state
// ════════════════════════════════════════════════════════════════════════════
RotMat3 ABDJoint::build_R_joint(const ABDBody &ba) const {
    switch (type) {
    case ABDJointType::Ball:
        // Ball: no rotation needed
        return RotMat3::identity();

    case ABDJointType::Hinge: {
        // R_H = rotation that maps hinge_axis_world → local y (논문 Eq.20)
        // hinge_axis_world is already stored as unit vector
        return RotMat3::from_axis_to_y(hinge_axis_world);
    }

    case ABDJointType::Universal: {
        // R_U = [a1×a2, a1, a2]^T  (논문 Eq.22)
        // maps a1 → local y,  a2 → local z
        Vector3 a1 = univ_axis_1;
        Vector3 a2 = univ_axis_2;
        // Use hinge mapping on a1
        return RotMat3::from_axis_to_y(a1);
    }

    case ABDJointType::Prismatic: {
        // R_P same structure as R_H (논문 Eq.24)
        return RotMat3::from_axis_to_y(prismatic_axis);
    }
    }
    return RotMat3::identity();
}

// ════════════════════════════════════════════════════════════════════════════
//  ABDJoint::compute_C  —  constraint vector
//
//  Ball joint (논문 Eq.19, Section 4.2):
//  ──────────────────────────────────────
//  C = S_B^α T^α q^α - S_B^β T^β q^β = y^α_{cp_α} - y^β_{cp_β}
//  (3 constraints: shared pivot point)
//
//  Hinge joint (논문 Eq.21, Section 4.3):
//  ────────────────────────────────────────
//  5-DOF: S_H ∈ ℝ^{5×12}
//  Local frame ỹ = diag4(R_H) · y
//  CP1 (on axis, α and β): ỹ^α_{cp1,x} = ỹ^β_{cp1,x},  ỹ_z match
//  CP2 (on axis, α and β): ỹ^α_{cp2,x} = ỹ^β_{cp2,x},  ỹ_z match
//  5 constraints total (one redundant excluded):
//    C[0] = ỹ^α_1[x] - ỹ^β_1[x]
//    C[1] = ỹ^α_1[z] - ỹ^β_1[z]
//    C[2] = ỹ^α_2[x] - ỹ^β_2[x]
//    C[3] = ỹ^α_2[z] - ỹ^β_2[z]
//    C[4] = ỹ^α_1[y] - ỹ^β_1[y]   (translation along axis)
//  (논문 Fig.4 참고: CP1, CP2 on rotation axis, x/z constrained)
//
//  Universal joint (논문 Eq.23, Section 4.4):
//  ────────────────────────────────────────────
//  4-DOF: S_U ∈ ℝ^{4×12}
//  C = S_U^α diag4(R_U) T^α q^α - S_U^β diag4(R_U) T^β q^β
//  Local ỹ = R_U y:
//    C[0..2] = ỹ^α_1 - ỹ^β_1        (ball constraint on CP1, 3 constraints)
//    C[3]    = ỹ^β_2[z]              (y2^β ⊥ a1: z-component = 0 in local frame)
//
//  Prismatic joint (논문 Eq.25, Section 4.5):
//  ────────────────────────────────────────────
//  5-DOF: S_P ∈ ℝ^{5×12}
//  Local ỹ = R_P y:
//    C[0] = ỹ^α_2[x] - ỹ^β_2[x]    (edge alignment x)
//    C[1] = ỹ^α_2[z] - ỹ^β_2[z]    (edge alignment z)
//    C[2] = ỹ^α_1[x] - ỹ^β_1[x]    (edge alignment x)
//    C[3] = ỹ^α_1[z] - ỹ^β_1[z]    (edge alignment z)
//    C[4] = ỹ^α_4[x] - ỹ^β_4[x]    (current plane perp constraint)
//  (논문 Fig.5 참고, x̃,z̃ components constrained, ỹ free)
//
// ════════════════════════════════════════════════════════════════════════════

int ABDJoint::compute_C(
    const float /*q_alpha*/[12], const float /*q_beta*/[12],
    const float y_alpha[12],    const float y_beta[12],
    const RotMat3 &R,
    float out_C[JOINT_MAX_C]) const
{
    for (int i = 0; i < JOINT_MAX_C; ++i) out_C[i] = 0.f;

    switch (type) {
    // ── Ball joint (Eq.19) ────────────────────────────────────
    case ABDJointType::Ball: {
        // C = y^α_{cp_α} - y^β_{cp_β}
        int oa = cp_alpha * 3;
        int ob = cp_beta  * 3;
        out_C[0] = y_alpha[oa+0] - y_beta[ob+0];
        out_C[1] = y_alpha[oa+1] - y_beta[ob+1];
        out_C[2] = y_alpha[oa+2] - y_beta[ob+2];
        return 3;
    }

    // ── Hinge joint (Eq.21) ───────────────────────────────────
    case ABDJointType::Hinge: {
        // Local CP positions: ỹ = R_H * y_k
        auto rotY = [&](const float y[12], int k) -> Vector3 {
            Vector3 yk(y[k*3+0], y[k*3+1], y[k*3+2]);
            return R.apply(yk);
        };
        Vector3 ya1 = rotY(y_alpha, cp_alpha_1);
        Vector3 ya2 = rotY(y_alpha, cp_alpha_2);
        Vector3 yb1 = rotY(y_beta,  cp_beta_1);
        Vector3 yb2 = rotY(y_beta,  cp_beta_2);

        // 5 constraints (x,z for each CP + y translation of CP1):
        out_C[0] = ya1.x - yb1.x;  // CP1 x
        out_C[1] = ya1.z - yb1.z;  // CP1 z
        out_C[2] = ya2.x - yb2.x;  // CP2 x
        out_C[3] = ya2.z - yb2.z;  // CP2 z
        out_C[4] = ya1.y - yb1.y;  // CP1 y (translation along axis)
        return 5;
    }

    // ── Universal joint (Eq.23) ───────────────────────────────
    case ABDJointType::Universal: {
        // R_U built from axis a1
        // Ball constraint on CP1 (3 DOF)
        int oa = cp_alpha * 3;
        int ob = cp_beta  * 3;
        Vector3 ya1(y_alpha[oa+0], y_alpha[oa+1], y_alpha[oa+2]);
        Vector3 yb1(y_beta [ob+0], y_beta [ob+1], y_beta [ob+2]);
        Vector3 ya1_loc = R.apply(ya1);
        Vector3 yb1_loc = R.apply(yb1);
        out_C[0] = ya1_loc.x - yb1_loc.x;
        out_C[1] = ya1_loc.y - yb1_loc.y;
        out_C[2] = ya1_loc.z - yb1_loc.z;

        // Orthogonality: ỹ^β_2 · a1 = 0  (z-component in local frame = 0)
        // CP2 of β in local frame: z-component should be 0
        int ob2 = cp_beta_2 * 3;
        Vector3 yb2(y_beta[ob2+0], y_beta[ob2+1], y_beta[ob2+2]);
        Vector3 yb2_loc = R.apply(yb2);
        // a2 ⊥ a1: a2 dot a1 = 0
        // In local frame, a1 maps to y-axis, so a2 must have no y-component
        out_C[3] = yb2_loc.y;  // y2^β ⊥ a1 → y-component = 0 in local frame
        return 4;
    }

    // ── Prismatic joint (Eq.25) ───────────────────────────────
    case ABDJointType::Prismatic: {
        // Local frame ỹ = R_P * y_k
        auto rotY = [&](const float y[12], int k) -> Vector3 {
            Vector3 yk(y[k*3+0], y[k*3+1], y[k*3+2]);
            return R.apply(yk);
        };
        Vector3 ya1 = rotY(y_alpha, cp_alpha_1);
        Vector3 ya2 = rotY(y_alpha, cp_alpha_2);
        Vector3 yb1 = rotY(y_beta,  cp_beta_1);
        Vector3 yb2 = rotY(y_beta,  cp_beta_2);

        // 5 constraints (edges aligned, translation along axis only):
        out_C[0] = ya1.x - yb1.x;  // edge 1 x-alignment
        out_C[1] = ya1.z - yb1.z;  // edge 1 z-alignment
        out_C[2] = ya2.x - yb2.x;  // edge 2 x-alignment
        out_C[3] = ya2.z - yb2.z;  // edge 2 z-alignment
        // Translation perpendicular to axis: x-plane constraint
        // CP4 (t-block = translation): x,z fixed between bodies
        {
            int oa4 = cp_alpha * 3;  // use cp_alpha as translation ref CP
            int ob4 = cp_beta  * 3;
            Vector3 ya4 = rotY(y_alpha, cp_alpha);
            Vector3 yb4 = rotY(y_beta,  cp_beta);
            out_C[4] = ya4.x - yb4.x;
        }
        return 5;
    }
    }
    return 0;
}

// ════════════════════════════════════════════════════════════════════════════
//  ABDJoint::apply_penalty
//
//  f_A^α += -κ · (∂C/∂q^α)^T · C
//  f_A^β += -κ · (∂C/∂q^β)^T · C
//
//  ∂C/∂q^α の計算 (treating R_joint as constant, Section 4.6):
//    C[i] = (R · y^α_k)[component] - (R · y^β_k)[component]
//    ∂C[i]/∂q^α = R · ∂y^α_k/∂q^α   (각 constraint 행마다)
//
//  (∂C/∂q^α)^T C = Σ_i C[i] · (∂C[i]/∂q^α)^T
//                = Σ_i C[i] · (∂y^α_k/∂q^α)^T · R^T · e_s
//  여기서 e_s = component selector (x,y,z)
//
//  add_J_y_T_r(b, k, R^T * (C[i]*e_s), f_alpha) 로 구현.
// ════════════════════════════════════════════════════════════════════════════

void ABDJoint::apply_penalty(
    const ABDBody &ba, const ABDBody &bb,
    float f_alpha[12], float f_beta[12]) const
{
    // Compute current CP coords
    float ya[12], yb[12];
    abd_compute_y(ba, ya);
    abd_compute_y(bb, yb);

    // Build joint rotation
    RotMat3 R = build_R_joint(ba);

    // Compute constraint
    float C[JOINT_MAX_C] = {};
    int n_c = compute_C(ba.q.data(), bb.q.data(), ya, yb, R, C);

    if (n_c == 0) return;

    // Check if constraint is satisfied (skip if tiny)
    float c_norm2 = 0.f;
    for (int i = 0; i < n_c; ++i) c_norm2 += C[i]*C[i];
    if (c_norm2 < 1e-20f) return;

    // ── Gradient back-projection ──────────────────────────────
    // Per-joint, per-constraint: figure out which CP/component
    // and apply (∂C/∂q)^T C via add_J_y_T_r

    switch (type) {

    // ── Ball (3 constraints on CP positions) ─────────────────
    case ABDJointType::Ball: {
        // C = y^α_{pa} - y^β_{pb}
        // ∂C/∂q^α = J_y_{pa}^T,  ∂C/∂q^β = -J_y_{pb}^T
        // f^α += -κ (J_y_{pa})^T C  →  r = -κ C
        Vector3 r(C[0], C[1], C[2]);
        r = r * (-kappa);
        add_J_y_T_r(ba, cp_alpha, r,  f_alpha);
        add_J_y_T_r(bb, cp_beta,  -r, f_beta);   // opposite sign
        break;
    }

    // ── Hinge (5 constraints) ─────────────────────────────────
    case ABDJointType::Hinge: {
        // C[0,1]: CP1 x,z of α vs β  (R applied)
        // C[2,3]: CP2 x,z
        // C[4]:   CP1 y
        //
        // ∂C[i]/∂q^α = (R^T e_s) projected via J_y
        // where e_s is the component selector (x=0,z=2,y=1)

        static const int cp_indices_a[5] = {0, 0, 1, 1, 0};  // which CP (relative)
        static const int cp_indices_b[5] = {0, 0, 1, 1, 0};
        static const int comp[5]         = {0, 2, 0, 2, 1};  // x,z,x,z,y

        for (int i = 0; i < 5; ++i) {
            float ci = C[i] * (-kappa);

            // Component selector in local frame
            Vector3 e_s(comp[i]==0?1.f:0.f, comp[i]==1?1.f:0.f, comp[i]==2?1.f:0.f);

            // Back to world frame: R^T e_s
            Vector3 r_world = R.apply_T(e_s);

            // α: which CP?
            int ka = (i < 2) ? cp_alpha_1 : (i < 4 ? cp_alpha_2 : cp_alpha_1);
            int kb = (i < 2) ? cp_beta_1  : (i < 4 ? cp_beta_2  : cp_beta_1 );

            add_J_y_T_r(ba, ka,  r_world * ci,   f_alpha);
            add_J_y_T_r(bb, kb, -r_world * ci,   f_beta);
        }
        break;
    }

    // ── Universal (4 constraints) ─────────────────────────────
    case ABDJointType::Universal: {
        // C[0..2]: ball on CP1 (R applied)
        // C[3]: ỹ^β_2 y-component

        // Ball part (C[0..2])
        for (int i = 0; i < 3; ++i) {
            float ci = C[i] * (-kappa);
            Vector3 e_s(i==0?1.f:0.f, i==1?1.f:0.f, i==2?1.f:0.f);
            Vector3 r_world = R.apply_T(e_s);
            add_J_y_T_r(ba, cp_alpha,  r_world * ci,  f_alpha);
            add_J_y_T_r(bb, cp_beta,  -r_world * ci,  f_beta);
        }
        // Orthogonality (C[3]): ỹ^β_2[y] = 0
        {
            float ci = C[3] * (-kappa);
            Vector3 e_y(0.f,1.f,0.f);
            Vector3 r_world = R.apply_T(e_y);
            // Only β CP2 contributes
            add_J_y_T_r(bb, cp_beta_2, r_world * ci, f_beta);
        }
        break;
    }

    // ── Prismatic (5 constraints) ─────────────────────────────
    case ABDJointType::Prismatic: {
        // C[0,1]: edge1 x,z  (cp_alpha_1 vs cp_beta_1)
        // C[2,3]: edge2 x,z  (cp_alpha_2 vs cp_beta_2)
        // C[4]: translation perp x

        static const int comp_p[5] = {0, 2, 0, 2, 0};  // x,z,x,z,x

        for (int i = 0; i < 5; ++i) {
            float ci = C[i] * (-kappa);
            Vector3 e_s(comp_p[i]==0?1.f:0.f, 0.f, comp_p[i]==2?1.f:0.f);
            Vector3 r_world = R.apply_T(e_s);

            int ka, kb;
            if      (i < 2) { ka = cp_alpha_1; kb = cp_beta_1; }
            else if (i < 4) { ka = cp_alpha_2; kb = cp_beta_2; }
            else             { ka = cp_alpha;   kb = cp_beta;   }

            add_J_y_T_r(ba, ka,  r_world * ci,  f_alpha);
            add_J_y_T_r(bb, kb, -r_world * ci,  f_beta);
        }
        break;
    }
    }
}

} // namespace godot
