#pragma once
// ... (M-ABD Section 4 Joint Definitions)
#include "abd_types.h"
#include <array>
#include <cstdint>
#include <cmath>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/basis.hpp>
#include <godot_cpp/variant/transform3d.hpp>

namespace godot {

struct ABDBody;

// ════════════════════════════════════════════════════════════════════════════
//  JointType
// ════════════════════════════════════════════════════════════════════════════
enum class ABDJointType {
    Ball,       // Section 4.2 — 3 DOF removed
    Hinge,      // Section 4.3 — 5 DOF removed
    Universal,  // Section 4.4 — 4 DOF removed
    Prismatic,  // Section 4.5 — 5 DOF removed
};

// ════════════════════════════════════════════════════════════════════════════
//  Selection matrix S_B  (S ∈ ℝ^{r×12})
//  Picks r rows from the 12-dim y vector.
//  We store it as a list of (row, 12-col) pairs with float coefficients.
//  For the joints here all entries are ±1 or 0, so we use a compact form.
// ════════════════════════════════════════════════════════════════════════════

// Max constraint rows per joint = 5
static constexpr int JOINT_MAX_C = 6;

// Selection entry: picks linear combination of y components
struct SelRow {
    float coeff[12] = {};
};

// ════════════════════════════════════════════════════════════════════════════
//  Rotation helpers
// ════════════════════════════════════════════════════════════════════════════

// Rodrigues rotation matrix from axis v (unit) and pre-computed terms
// R_H / R_U / R_P  (논문 Eq.20, Eq.22, Eq.24)
// R = I + [v]× + [v]×² · (1/(1+a_y))
// where v = a × e_y,  a_y = a · e_y  (or similar depending on joint)
struct RotMat3 {
    float m[3][3] = {};

    // Build from axis-angle Rodrigues (논문 Eq.20)
    // a: unit rotation axis of the hinge
    // e_ref: reference axis (e_y = [0,1,0] for hinge)
    // v = a × e_ref, a_y = a · e_ref
    static RotMat3 from_axis_to_y(const Vector3 &a);
    static RotMat3 identity();

    // Apply: out = R * v
    Vector3 apply(const Vector3 &v) const;
    // Transpose apply: out = R^T * v
    Vector3 apply_T(const Vector3 &v) const;
};

inline RotMat3 RotMat3::identity() {
    RotMat3 r;
    r.m[0][0]=r.m[1][1]=r.m[2][2]=1.f;
    return r;
}

// Eq.20: R_H = I_3 + [v]× + [v]×² / (1+a_y)
// a: unit hinge axis, aligns to local y axis
// v = a × e_y  (cross with [0,1,0])
inline RotMat3 RotMat3::from_axis_to_y(const Vector3 &a) {
    RotMat3 R;
    // v = a × e_y = (a.z, 0, -a.x)
    Vector3 e_y(0.f,1.f,0.f);
    Vector3 v = a.cross(e_y);
    float   ay = a.dot(e_y);
    float   denom = 1.f + ay;

    R.m[0][0]=1.f; R.m[1][1]=1.f; R.m[2][2]=1.f;
    if (fabsf(denom) < 1e-8f) return R; // already aligned

    // [v]×:
    //  0   -v.z   v.y
    //  v.z  0    -v.x
    // -v.y  v.x   0
    float vx=v.x, vy=v.y, vz=v.z;
    // R += [v]×
    R.m[0][1] -= vz; R.m[0][2] += vy;
    R.m[1][0] += vz; R.m[1][2] -= vx;
    R.m[2][0] -= vy; R.m[2][1] += vx;
    // R += [v]×² / (1+ay)
    // [v]×² = v v^T - (v·v) I
    float vdotv = vx*vx+vy*vy+vz*vz;
    float inv_d = 1.f/denom;
    R.m[0][0] += (vx*vx - vdotv)*inv_d;
    R.m[0][1] += vx*vy*inv_d;
    R.m[0][2] += vx*vz*inv_d;
    R.m[1][0] += vy*vx*inv_d;
    R.m[1][1] += (vy*vy - vdotv)*inv_d;
    R.m[1][2] += vy*vz*inv_d;
    R.m[2][0] += vz*vx*inv_d;
    R.m[2][1] += vz*vy*inv_d;
    R.m[2][2] += (vz*vz - vdotv)*inv_d;
    return R;
}

inline Vector3 RotMat3::apply(const Vector3 &v_in) const {
    return Vector3(
        m[0][0]*v_in.x+m[0][1]*v_in.y+m[0][2]*v_in.z,
        m[1][0]*v_in.x+m[1][1]*v_in.y+m[1][2]*v_in.z,
        m[2][0]*v_in.x+m[2][1]*v_in.y+m[2][2]*v_in.z
    );
}

inline Vector3 RotMat3::apply_T(const Vector3 &v_in) const {
    return Vector3(
        m[0][0]*v_in.x+m[1][0]*v_in.y+m[2][0]*v_in.z,
        m[0][1]*v_in.x+m[1][1]*v_in.y+m[2][1]*v_in.z,
        m[0][2]*v_in.x+m[1][2]*v_in.y+m[2][2]*v_in.z
    );
}

// ════════════════════════════════════════════════════════════════════════════
//  ABDJoint
//
//  두 body α, β를 연결하는 joint.
//  body_alpha, body_beta: 연결된 ABDBody 포인터
//
//  CP index 규약 (논문 Fig.4 기준):
//    Ball:       y1^α = y1^β  (3 constraints)
//    Hinge(5DOF): CP1,CP2 on rotation axis → 5 constraints
//    Universal:   CP1 shared + y2^β ⊥ a1 → 4 constraints
//    Prismatic:   edge alignment + translation along axis → 5 constraints
//
//  각 joint는 자신의 constraint C와 gradient ∇C를 계산해
//  penalty force f_A^α, f_A^β에 기여한다.
// ════════════════════════════════════════════════════════════════════════════

struct ABDJoint {
    ABDJointType type;

    // connected bodies (indices into IPCPhysicsServer::_body_list)
    int body_alpha = -1;
    int body_beta  = -1;

    // penalty stiffness (can be set per-joint)
    float kappa = 1e6f;

    // ── Joint-specific configuration ─────────────────────────

    // Ball joint (Sec 4.2):
    //   pivot_alpha: CP index on body α (0-3) that is the ball pivot
    //   pivot_beta:  CP index on body β (0-3)
    int cp_alpha = 0;  // CP index on α
    int cp_beta  = 0;  // CP index on β

    // Hinge joint (Sec 4.3):
    //   hinge_axis_world: prescribed rotation axis (world space, unit)
    //   cp_alpha_1/2: two CPs on α on the rotation axis
    //   cp_beta_1/2:  two CPs on β on the rotation axis
    Vector3 hinge_axis_world = Vector3(0,1,0);
    int cp_alpha_1 = 0, cp_alpha_2 = 1;
    int cp_beta_1  = 0, cp_beta_2  = 1;

    // Universal joint (Sec 4.4):
    //   a1, a2: two orthogonal rotation axes (world space, unit)
    //   cp_alpha_1: shared CP (ball constraint part)
    //   cp_beta_2:  CP on β for orthogonality constraint
    Vector3 univ_axis_1 = Vector3(1,0,0);
    Vector3 univ_axis_2 = Vector3(0,0,1);

    // Prismatic joint (Sec 4.5):
    //   prismatic_axis: translation axis (world space, unit)
    //   cp indices for edge alignment constraints
    Vector3 prismatic_axis = Vector3(0,1,0);

    // ── Godot meta ────────────────────────────────────────────
    uint64_t joint_id = 0;  // unique ID

    // ════════════════════════════════════════════════════════
    //  Core interface (implemented in abd_joint.cpp)
    // ════════════════════════════════════════════════════════

    // Compute constraint vector C (dim = DOF_removed)
    // q^α, q^β: current 12-vecs
    // y^α, y^β: current CP coords (Tq, 12-vecs)
    // R_joint: joint rotation (pre-computed in apply_penalty)
    // out_C: output constraint violation
    // returns number of active constraints
    int compute_C(
        const float q_alpha[12], const float q_beta[12],
        const float y_alpha[12], const float y_beta[12],
        const RotMat3 &R_joint,
        float out_C[JOINT_MAX_C]) const;

    // Compute penalty force contribution to f_A^α and f_A^β
    // (gradient back-projection: f_A += -κ (∂C/∂q)^T C)
    void apply_penalty(
        const ABDBody &ba, const ABDBody &bb,
        float f_alpha[12], float f_beta[12]) const;

    // ── Helpers ───────────────────────────────────────────────
    // Build joint rotation R from current body state
    RotMat3 build_R_joint(const ABDBody &ba) const;

    // Extract a_k from body q (k=0,1,2 → column of A)
    static Vector3 get_a(const float q[12], int k) {
        return Vector3(q[k*3], q[k*3+1], q[k*3+2]);
    }

    // Get CP position y_k = T q (k=0..3)
    // y_k = A ȳ_k + t   (ȳ_k = rest-shape CP of the body)
    static Vector3 get_y(const ABDBody &b, int k);

    // ∂y_k/∂q: Jacobian of CP k w.r.t. q
    // J_y_k ∈ ℝ^{3×12}
    // y_k[i] = Σ_j q[j*3+i] * ȳ_k[j] + q[9+i]
    // ∂y_k[i]/∂q[j*3+r] = δ_{ir} * ȳ_k[j]
    // ∂y_k[i]/∂q[9+r]   = δ_{ir}
    // → J_y_k applied to residual r (∈ ℝ³) gives 12-vec:
    //   (J_y_k^T r)[j*3+i] = ȳ_k[j] * r[i]
    //   (J_y_k^T r)[9+i]   = r[i]
    static void add_J_y_T_r(const ABDBody &b, int k,
                             const Vector3 &r, float out_f[12]);
};

// ════════════════════════════════════════════════════════════════════════════
//  Free functions used by the solver
// ════════════════════════════════════════════════════════════════════════════

// Get CP world position from body
Vector3 abd_get_cp(const ABDBody &b, int k);

// Compute T matrix: y = Tq  (12×12)
// Already done in ABDBody::build_control_tetrahedron, but reusable here
void abd_compute_y(const ABDBody &b, float y_out[12]);

} // namespace godot
