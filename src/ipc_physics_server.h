#pragma once

// ════════════════════════════════════════════════════════════════════════════
//  M-ABD Physics Server  (godot-cpp 4.5)
//
//  논문: "M-ABD: Scalable, Efficient, and Robust Multi-Affine-Body Dynamics"
//        He et al., arXiv:2603.08079v2 (2026)
//
//  핵심 개념
//  ─────────
//  · 각 rigid body를 12-DOF affine coordinate q = [vec(A), t] ∈ ℝ¹² 로 표현
//    (논문 Section 3.1, Eq.4)
//  · Co-rotational 공식으로 Hessian H̄_A = (1/h²)M_A + K̄_A 를 상수화 →
//    사전 인수분해 후 전 스텝에서 재사용 (Section 3.3-3.4, Eq.16-17)
//  · 충돌: Simple penalty
//  · joint: Section 4 — Ball/Hinge/Universal/Prismatic (penalty enforcement)
//
//  좌표 규약
//  ─────────
//  q = [a1, a2, a3, t]  (각각 ℝ³, A의 세 열벡터 + translation)
//  논문 Eq.4:  x_i = J_i q,  J_i = [x̄_i^T ⊗ I_3,  I_3]
//
//  Algorithm 1 파티셔닝 (논문 p.5):
//  q = [q1, q2, q3, q4]^T,  q1=a1, q2=a2, q3=a3, q4=t  (각 ℝ³)
//  A = [q1, q2, q3]  (3×3 matrix, column = a_k)
//
// ════════════════════════════════════════════════════════════════════════════

#include <godot_cpp/classes/physics_server3d_extension.hpp>
#include <godot_cpp/classes/physics_server3d_extension_motion_result.hpp>
#include <godot_cpp/classes/physics_server3d_manager.hpp>
#include <godot_cpp/classes/physics_server3d_rendering_server_handler.hpp>
#include <godot_cpp/classes/physics_direct_body_state3d.hpp>
#include <godot_cpp/classes/physics_direct_space_state3d.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <array>
#include <cstdint>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "abd_types.h"   // Vec12, vec12_zero

// forward-declare before including joint header
namespace godot { class IPCBodyState; struct ABDBody; }
#include "abd_joint.h"
// ABDKKTSolver은 solver.cpp에서만 사용 — forward declare로 충분
namespace godot { class ABDKKTSolver; }

namespace godot {

// ════════════════════════════════════════════════════════════════════════════
//  시뮬레이션 파라미터
// ════════════════════════════════════════════════════════════════════════════

static constexpr float ABD_GRAVITY_Y       = -9.81f;
static constexpr float ABD_H               = 1.f / 60.f;  // default time step
// Young's modulus E=1e8, Poisson ν=0.3 → Lamé μ, λ
// 논문 Eq.15: ∂Ψ/∂A = μ(A+Aᵀ−2I) + λ·tr(A−I)·I
static constexpr float ABD_MU_LAME         = 3.8461e7f;   // μ = E/(2(1+ν))
static constexpr float ABD_LAMBDA_LAME     = 5.7692e7f;   // λ = Eν/((1+ν)(1-2ν))
// 충돌 penalty stiffness
// kappa 선택 기준: contact force로 인한 dq ~ kappa*d_hat*h²/m.
// 이 값이 d_hat보다 작아야 한 스텝에 contact 역전이 없음.
// kappa < m/h² (=3600 at 60fps, m=1). 안전하게 절반 이하 사용.
static constexpr float ABD_CONTACT_KAPPA   = 1000.f;
static constexpr float ABD_CONTACT_D_HAT   = 0.5f;        // 충돌 감지 거리: 낙하 tunneling 방지용
// 감쇠 (수치 안정성용)
static constexpr float ABD_LIN_DAMP        = 0.999f;
static constexpr float ABD_ANG_DAMP        = 0.98f;

// ════════════════════════════════════════════════════════════════════════════
//  ShapeData
// ════════════════════════════════════════════════════════════════════════════

struct ShapeFace {
    std::array<int, 3> vi;
    Vector3            n;      // outward normal (unit)
};

enum class ShapeKind { Box, Convex, Concave, Unknown };

struct ShapeData {
    ShapeKind                       kind   = ShapeKind::Unknown;
    std::vector<Vector3>            verts;
    std::vector<std::array<int,2>>  edges;
    std::vector<ShapeFace>          faces;
    float                           bsphere_radius = 0.f;

    void rebuild_bsphere();
    void rebuild_edges_from_faces();
};

// ════════════════════════════════════════════════════════════════════════════
//  BodyShapeRef
// ════════════════════════════════════════════════════════════════════════════

struct BodyShapeRef {
    RID         shape_rid;
    Transform3D local_xf;
    bool        disabled = false;
};

// ════════════════════════════════════════════════════════════════════════════
//  ABDBody  ─  12-DOF Affine Body  (논문 Section 3.1)
//
//  q = [a1, a2, a3, t]  ∈ ℝ¹²
//    a1,a2,a3: 3×3 행렬 A 의 세 열벡터
//    t       : translation (world COM)
//
//  Algorithm 1 파티셔닝:
//    q1 = q[0..2]  = a1
//    q2 = q[3..5]  = a2
//    q3 = q[6..8]  = a3
//    q4 = q[9..11] = t
//
//  Pre-factorized system (Eq.16):
//    H̄_A · diag4(Rᵀ)·δq = diag4(Rᵀ)·f_A
//    H̄_A = (1/h²)M_A + K̄_A   (constant, factorized once)
//
// ════════════════════════════════════════════════════════════════════════════

// 12×12 symmetric positive definite matrix (full storage)
struct Mat12 {
    float v[144] = {};

    float &operator()(int r, int c)       { return v[r*12+c]; }
    float  operator()(int r, int c) const { return v[r*12+c]; }

    void set_zero() { for (auto &x : v) x = 0.f; }

    // Cholesky factorization (in-place, L·Lᵀ = this)
    bool cholesky_factor();

    // forward/back solve: L Lᵀ x = rhs  (in-place on rhs)
    void cholesky_solve(float rhs[12]) const;
};

// Vec12 is defined in abd_types.h (included above)

struct ABDBody {
    // ── Affine coordinate q = [a1, a2, a3, t] ────────────────
    Vec12  q;        // current state
    Vec12  q_prev;   // previous step state
    Vec12  qdot;     // generalized velocity  dq/dt

    // ── Pre-computed (constant) matrices ─────────────────────
    Mat12  M_A;           // generalized mass (constant)       Eq.7
    Mat12  K_bar_A;       // rest-shape generalized stiffness  Eq.11
    Mat12  H_bar_A;       // pre-factored H̄_A (Cholesky)      Eq.17
    Mat12  J_bar;         // volume-weighted Jacobian J̄        Eq.14

    // ── Control tetrahedron (논문 Section 4.1) ────────────────
    std::array<Vector3, 4> cp_rest;   // ȳ₁..ȳ₄ (rest-shape positions)
    Mat12  T_mat;         // y = T q   Eq.18
    Mat12  T_inv;         // q = T⁻¹ y

    // ── Mass / inertia ────────────────────────────────────────
    float   mass     = 1.f;
    float   inv_mass = 1.f;
    Vector3 half_extents = Vector3(0.5f, 0.5f, 0.5f);
    bool    is_static    = false;
    bool    sim_started  = false;

    // ── Accumulated external forces (per step, reset each step) ──
    // 논문 Eq.44: f_A,ext = G(A)ᵀ W_ext
    // 외부에서 apply_force/apply_torque 로 누적됨
    Vec12  f_ext_accum;   // accumulated external affine force

    // ── Shape mesh (simulation, local body frame) ─────────────
    std::vector<Vector3>     verts_local;
    std::vector<ShapeFace>   faces_sim;
    std::vector<BodyShapeRef> shapes;
    RID                       pending_shape_rid;

    // ── Godot meta ────────────────────────────────────────────
    IPCBodyState* direct_state = nullptr;
    RID      body_rid;
    uint64_t instance_id = 0;
    PhysicsServer3D::BodyMode mode = PhysicsServer3D::BODY_MODE_RIGID;

    // ── Convenience accessors ─────────────────────────────────
    // A matrix (3×3) from q — column-major: A[:,k] = q[3k..3k+2]
    void get_A(float A[3][3]) const;
    void set_A(const float A[3][3]);

    Vector3 get_t() const { return Vector3(q[9], q[10], q[11]); }
    void    set_t(const Vector3 &t_in) { q[9]=t_in.x; q[10]=t_in.y; q[11]=t_in.z; }

    // world position of rest-shape point x̄: x = A x̄ + t (Eq.1)
    Vector3 world_point(const Vector3 &x_rest) const;

    // polar decomposition R = A (AᵀA)^{-1/2} (논문 Eq.9, Higham iteration)
    Basis polar_rotation() const;

    // fast rotation approximation: Ra ≈ (||a||/||Aa||) Aa (논문 Eq.17)
    static Vector3 fast_rotate(const float A[3][3], const Vector3 &a);

    // diag₄(R): q-space block-diagonal rotation (Eq.10)
    static void apply_diag4(const Basis &R, const Vec12 &in_q, Vec12 &out_q);

    // ── Sync with Godot Transform3D ───────────────────────────
    Transform3D get_transform() const;
    void        set_transform(const Transform3D &xf);

    // ── Velocity helpers ──────────────────────────────────────
    Vector3 get_linear_velocity() const;
    void    set_linear_velocity(const Vector3 &v);

    // ω = (1/2) Σ_k  q_k × q̇_k  (논문 Eq.43)
    Vector3 get_angular_velocity() const;
    void    set_angular_velocity(const Vector3 &omega);

    // ── G(A) matrix: ABD velocity → spatial twist (Eq.42) ────
    // G(A) ∈ ℝ^{6×12}:  V = G(A) q̇,  V = [ω, v]^T
    // ω = (1/2)(q1×q̇1 + q2×q̇2 + q3×q̇3),  v = ṫ
    // G(A)^T W_ext → affine generalized force (Eq.44)
    // out_f_A: adds G(A)^T [tau, f]^T to it
    static void apply_G_transpose(const float A[3][3],
                                  const Vector3 &tau, const Vector3 &f,
                                  Vec12 &out_f_A);

    // ── Matrix builders ───────────────────────────────────────
    void build_mass_matrix();         // M_A = Jᵀ M J
    void build_stiffness_matrix();    // K̄_A  (linear elasticity, Eq.15)
    void build_H_bar();               // H̄_A = (1/h²)M_A + K̄_A, Cholesky factor
    void build_control_tetrahedron(); // cp_rest + T_mat + T_inv
    void build_J_bar();               // J̄

    void update_H_bar(float h);
};

// ════════════════════════════════════════════════════════════════════════════
//  Contact pair
// ════════════════════════════════════════════════════════════════════════════

struct ABDContactPair {
    int body_a = -1, body_b = -1;
    int          pa = 0;            // vertex index in body_a
    std::array<int,3> tri = {};     // triangle in body_b
    float        dist    = 0.f;
    Vector3      n_hat;             // contact normal (a→b)
};

// ════════════════════════════════════════════════════════════════════════════
//  BVH (broad-phase)
// ════════════════════════════════════════════════════════════════════════════

struct BVHNode {
    int   body_idx = -1;
    Vector3 center;
    float radius   = 0.f;
    int   left     = -1;
    int   right    = -1;
};

// ════════════════════════════════════════════════════════════════════════════
//  RID helpers
// ════════════════════════════════════════════════════════════════════════════

struct RIDHash {
    size_t operator()(const RID &r) const { return std::hash<uint64_t>{}(r.get_id()); }
};
struct RIDEqual {
    bool operator()(const RID &a, const RID &b) const { return a == b; }
};
template<typename V>
using RIDMap = std::unordered_map<RID, V, RIDHash, RIDEqual>;

// ════════════════════════════════════════════════════════════════════════════
//  IPCPhysicsServer
// ════════════════════════════════════════════════════════════════════════════

class IPCPhysicsServer : public PhysicsServer3DExtension {
    GDCLASS(IPCPhysicsServer, PhysicsServer3DExtension)

public:
    IPCPhysicsServer();
	~IPCPhysicsServer(){};

    static void _bind_methods();

    int  get_body_count() const;

    // ════════════════════════════════════════════════════════
    //  PhysicsServer3DExtension overrides
    // ════════════════════════════════════════════════════════

    void _set_active(bool p_active) override;
    void _init() override {}

    void _step(float p_step) override;
    void _sync()             override {}
    void _flush_queries()    override {}
    void _end_sync()         override {}
    void _finish()           override {}
    bool _is_flushing_queries() const override { return false; }
    int32_t _get_process_info(PhysicsServer3D::ProcessInfo) override { return 0; }

    // ── space ────────────────────────────────────────────────
    RID  _space_create() override;
    void _space_set_active(const RID &, bool) override;
    bool _space_is_active(const RID &) const override;
    void _space_set_param(const RID &, PhysicsServer3D::SpaceParameter, float) override {}
    float _space_get_param(const RID &, PhysicsServer3D::SpaceParameter) const override { return 0.f; }
    PhysicsDirectSpaceState3D* _space_get_direct_state(const RID &) override { return nullptr; }
    void _space_set_debug_contacts(const RID &, int32_t) override {}
    PackedVector3Array _space_get_contacts(const RID &) const override { return {}; }
    int32_t _space_get_contact_count(const RID &) const override { return 0; }

    // ── shape ────────────────────────────────────────────────
    RID  _world_boundary_shape_create() override              { return RID(); }
    RID  _separation_ray_shape_create() override              { return RID(); }
    RID  _sphere_shape_create() override                      { return RID(); }
    RID  _box_shape_create() override;
    RID  _capsule_shape_create() override                     { return RID(); }
    RID  _cylinder_shape_create() override                    { return RID(); }
    RID  _convex_polygon_shape_create() override;
    RID  _concave_polygon_shape_create() override;
    RID  _heightmap_shape_create() override                   { return RID(); }
    RID  _custom_shape_create() override                      { return RID(); }

    void    _shape_set_data(const RID &p_shape, const Variant &p_data) override;
    void    _shape_set_custom_solver_bias(const RID &, float) override {}
    PhysicsServer3D::ShapeType _shape_get_type(const RID &p_shape) const override;
    Variant _shape_get_data(const RID &p_shape) const override;
    float   _shape_get_custom_solver_bias(const RID &) const override { return 0.f; }
    void    _shape_set_margin(const RID &, float) override {}
    float   _shape_get_margin(const RID &) const override { return 0.f; }

    // ── body ─────────────────────────────────────────────────
    RID  _body_create() override;
    void _body_set_space(const RID &p_body, const RID &p_space) override;
    RID  _body_get_space(const RID &p_body) const override;
    void _body_set_mode(const RID &p_body, PhysicsServer3D::BodyMode p_mode) override;
    PhysicsServer3D::BodyMode _body_get_mode(const RID &p_body) const override;
    void _body_add_shape(const RID &p_body, const RID &p_shape,
                         const Transform3D &p_transform, bool p_disabled) override;
    void _body_set_shape(const RID &p_body, int32_t p_shape_idx, const RID &p_shape) override;
    void _body_set_shape_transform(const RID &p_body, int32_t p_shape_idx,
                                   const Transform3D &p_transform) override;
    void _body_set_shape_disabled(const RID &p_body, int32_t p_shape_idx, bool p_disabled) override;
    int32_t     _body_get_shape_count(const RID &p_body) const override;
    RID         _body_get_shape(const RID &p_body, int32_t p_shape_idx) const override;
    Transform3D _body_get_shape_transform(const RID &p_body, int32_t p_shape_idx) const override;
    void _body_remove_shape(const RID &p_body, int32_t p_shape_idx) override;
    void _body_clear_shapes(const RID &p_body) override;
    void _body_set_state(const RID &p_body, PhysicsServer3D::BodyState p_state,
                         const Variant &p_value) override;
    Variant _body_get_state(const RID &p_body,
                            PhysicsServer3D::BodyState p_state) const override;

    void _body_apply_central_impulse(const RID &p_body, const Vector3 &p_impulse) override;
    void _body_apply_impulse(const RID &p_body, const Vector3 &p_impulse,
                             const Vector3 &p_position) override;
    void _body_apply_torque_impulse(const RID &p_body, const Vector3 &p_impulse) override;
    void _body_apply_central_force(const RID &p_body, const Vector3 &p_force) override;
    void _body_apply_force(const RID &p_body, const Vector3 &p_force,
                           const Vector3 &p_position) override;
    void _body_apply_torque(const RID &p_body, const Vector3 &p_torque) override;

    void    _body_add_constant_central_force(const RID &, const Vector3 &) override {}
    void    _body_add_constant_force(const RID &, const Vector3 &, const Vector3 &) override {}
    void    _body_add_constant_torque(const RID &, const Vector3 &) override  {}
    void    _body_set_constant_force(const RID &, const Vector3 &) override   {}
    Vector3 _body_get_constant_force(const RID &) const override              { return {}; }
    void    _body_set_constant_torque(const RID &, const Vector3 &) override  {}
    Vector3 _body_get_constant_torque(const RID &) const override             { return {}; }
    void _body_set_axis_velocity(const RID &, const Vector3 &) override       {}
    void _body_set_axis_lock(const RID &, PhysicsServer3D::BodyAxis, bool) override {}
    bool _body_is_axis_locked(const RID &, PhysicsServer3D::BodyAxis) const override { return false; }
    void _body_add_collision_exception(const RID &, const RID &) override     {}
    void _body_remove_collision_exception(const RID &, const RID &) override  {}
    TypedArray<RID> _body_get_collision_exceptions(const RID &) const override { return {}; }
    void    _body_set_max_contacts_reported(const RID &, int32_t) override    {}
    int32_t _body_get_max_contacts_reported(const RID &) const override       { return 0; }
    void  _body_set_contacts_reported_depth_threshold(const RID &, float) override {}
    float _body_get_contacts_reported_depth_threshold(const RID &) const override  { return 0.f; }
    void _body_set_omit_force_integration(const RID &, bool) override         {}
    bool _body_is_omitting_force_integration(const RID &) const override      { return false; }
    void _body_set_state_sync_callback(const RID &p_body,
                                       const Callable &p_callable) override;
    void _body_set_force_integration_callback(const RID &, const Callable &,
                                              const Variant &) override {}
    void _body_set_ray_pickable(const RID &, bool) override                   {}
    bool _body_test_motion(const RID &, const Transform3D &, const Vector3 &,
                           float, int32_t, bool, bool,
                           PhysicsServer3DExtensionMotionResult *) const override { return false; }
    PhysicsDirectBodyState3D* _body_get_direct_state(const RID &p_body) override;

    void     _body_attach_object_instance_id(const RID &p_body, uint64_t p_id) override;
    uint64_t _body_get_object_instance_id(const RID &p_body) const override;
    void _body_set_enable_continuous_collision_detection(const RID &, bool) override {}
    bool _body_is_continuous_collision_detection_enabled(const RID &) const override { return false; }
    void     _body_set_collision_layer(const RID &, uint32_t) override        {}
    uint32_t _body_get_collision_layer(const RID &) const override            { return 0xFFFFFFFF; }
    void     _body_set_collision_mask(const RID &, uint32_t) override         {}
    uint32_t _body_get_collision_mask(const RID &) const override             { return 0xFFFFFFFF; }
    void  _body_set_collision_priority(const RID &, float) override           {}
    float _body_get_collision_priority(const RID &) const override            { return 1.f; }
    void     _body_set_user_flags(const RID &, uint32_t) override             {}
    uint32_t _body_get_user_flags(const RID &) const override                 { return 0; }
    void    _body_set_param(const RID &p_body, PhysicsServer3D::BodyParameter p_param,
                            const Variant &p_value) override;
    Variant _body_get_param(const RID &p_body,
                            PhysicsServer3D::BodyParameter p_param) const override;
    void _body_reset_mass_properties(const RID &p_body) override;

    // ── area (stub) ───────────────────────────────────────────
    RID  _area_create() override                                { return _make_rid(); }
    void _area_set_space(const RID &, const RID &) override     {}
    RID  _area_get_space(const RID &) const override            { return RID(); }
    void _area_add_shape(const RID &, const RID &, const Transform3D &, bool) override {}
    void _area_set_shape(const RID &, int32_t, const RID &) override {}
    void _area_set_shape_transform(const RID &, int32_t, const Transform3D &) override {}
    void _area_set_shape_disabled(const RID &, int32_t, bool) override {}
    int32_t _area_get_shape_count(const RID &) const override   { return 0; }
    RID     _area_get_shape(const RID &, int32_t) const override{ return RID(); }
    Transform3D _area_get_shape_transform(const RID &, int32_t) const override { return {}; }
    void _area_remove_shape(const RID &, int32_t) override      {}
    void _area_clear_shapes(const RID &) override               {}
    void _area_attach_object_instance_id(const RID &, uint64_t) override {}
    uint64_t _area_get_object_instance_id(const RID &) const override { return 0; }
    void _area_set_param(const RID &, PhysicsServer3D::AreaParameter, const Variant &) override {}
    Variant _area_get_param(const RID &, PhysicsServer3D::AreaParameter) const override { return Variant(); }
    void _area_set_transform(const RID &, const Transform3D &) override {}
    Transform3D _area_get_transform(const RID &) const override { return {}; }
    void _area_set_collision_layer(const RID &, uint32_t) override {}
    uint32_t _area_get_collision_layer(const RID &) const override { return 0; }
    void _area_set_collision_mask(const RID &, uint32_t) override {}
    uint32_t _area_get_collision_mask(const RID &) const override { return 0; }
    void _area_set_monitorable(const RID &, bool) override      {}
    void _area_set_ray_pickable(const RID &, bool) override     {}
    void _area_set_monitor_callback(const RID &, const Callable &) override {}
    void _area_set_area_monitor_callback(const RID &, const Callable &) override {}

    // ── soft body (stub) ──────────────────────────────────────
    RID  _soft_body_create() override                           { return RID(); }
    void _soft_body_update_rendering_server(const RID &, PhysicsServer3DRenderingServerHandler *) override {}
    void _soft_body_set_space(const RID &, const RID &) override {}
    RID  _soft_body_get_space(const RID &) const override       { return RID(); }
    void _soft_body_set_ray_pickable(const RID &, bool) override {}
    void     _soft_body_set_collision_layer(const RID &, uint32_t) override {}
    uint32_t _soft_body_get_collision_layer(const RID &) const override { return 0; }
    void     _soft_body_set_collision_mask(const RID &, uint32_t) override {}
    uint32_t _soft_body_get_collision_mask(const RID &) const override { return 0; }
    void _soft_body_add_collision_exception(const RID &, const RID &) override {}
    void _soft_body_remove_collision_exception(const RID &, const RID &) override {}
    TypedArray<RID> _soft_body_get_collision_exceptions(const RID &) const override { return {}; }
    void    _soft_body_set_state(const RID &, PhysicsServer3D::BodyState, const Variant &) override {}
    Variant _soft_body_get_state(const RID &, PhysicsServer3D::BodyState) const override { return Variant(); }
    void _soft_body_set_transform(const RID &, const Transform3D &) override {}
    void    _soft_body_set_simulation_precision(const RID &, int32_t) override {}
    int32_t _soft_body_get_simulation_precision(const RID &) const override { return 0; }
    void  _soft_body_set_total_mass(const RID &, float) override {}
    float _soft_body_get_total_mass(const RID &) const override { return 0.f; }
    void  _soft_body_set_linear_stiffness(const RID &, float) override {}
    float _soft_body_get_linear_stiffness(const RID &) const override { return 0.f; }
    void  _soft_body_set_shrinking_factor(const RID &, float) override {}
    float _soft_body_get_shrinking_factor(const RID &) const override { return 0.f; }
    void  _soft_body_set_pressure_coefficient(const RID &, float) override {}
    float _soft_body_get_pressure_coefficient(const RID &) const override { return 0.f; }
    void  _soft_body_set_damping_coefficient(const RID &, float) override {}
    float _soft_body_get_damping_coefficient(const RID &) const override { return 0.f; }
    void  _soft_body_set_drag_coefficient(const RID &, float) override {}
    float _soft_body_get_drag_coefficient(const RID &) const override { return 0.f; }
    void _soft_body_set_mesh(const RID &, const RID &) override {}
    AABB _soft_body_get_bounds(const RID &) const override      { return AABB(); }
    void    _soft_body_move_point(const RID &, int32_t, const Vector3 &) override {}
    Vector3 _soft_body_get_point_global_position(const RID &, int32_t) const override { return {}; }
    void _soft_body_remove_all_pinned_points(const RID &) override {}
    void _soft_body_pin_point(const RID &, int32_t, bool) override {}
    bool _soft_body_is_point_pinned(const RID &, int32_t) const override { return false; }
    void _soft_body_apply_point_impulse(const RID &, int32_t, const Vector3 &) override {}
    void _soft_body_apply_point_force(const RID &, int32_t, const Vector3 &) override {}
    void _soft_body_apply_central_impulse(const RID &, const Vector3 &) override {}
    void _soft_body_apply_central_force(const RID &, const Vector3 &) override {}

    // ── joint (Section 4 — Ball/Hinge/Universal/Prismatic) ──────
    // Godot PhysicsServer3D joint API → M-ABD ABDJoint 매핑:
    //   _joint_make_pin         → Ball joint (3-DOF)
    //   _joint_make_hinge       → Hinge joint (5-DOF)
    //   _joint_make_cone_twist  → Universal joint (4-DOF)
    //   _joint_make_slider      → Prismatic joint (5-DOF)
    //   _joint_make_generic_6dof → 사용자가 직접 ABDJoint 구성 가능

    RID  _joint_create() override;
    void _joint_clear(const RID &p_joint) override;

    // Ball joint ← pin joint API
    void _joint_make_pin(const RID &p_joint,
                         const RID &p_body_a, const Vector3 &p_local_a,
                         const RID &p_body_b, const Vector3 &p_local_b) override;
    void  _pin_joint_set_param(const RID &, PhysicsServer3D::PinJointParam, float) override {}
    float _pin_joint_get_param(const RID &, PhysicsServer3D::PinJointParam) const override { return 0.f; }
    void    _pin_joint_set_local_a(const RID &, const Vector3 &) override {}
    Vector3 _pin_joint_get_local_a(const RID &) const override  { return {}; }
    void    _pin_joint_set_local_b(const RID &, const Vector3 &) override {}
    Vector3 _pin_joint_get_local_b(const RID &) const override  { return {}; }

    // Hinge joint (5-DOF)
    void _joint_make_hinge(const RID &p_joint,
                           const RID &p_body_a, const Transform3D &p_hinge_a,
                           const RID &p_body_b, const Transform3D &p_hinge_b) override;
    void _joint_make_hinge_simple(const RID &p_joint,
                                  const RID &p_body_a,
                                  const Vector3 &p_pivot_a, const Vector3 &p_axis_a,
                                  const RID &p_body_b,
                                  const Vector3 &p_pivot_b, const Vector3 &p_axis_b) override;
    void  _hinge_joint_set_param(const RID &, PhysicsServer3D::HingeJointParam, float) override {}
    float _hinge_joint_get_param(const RID &, PhysicsServer3D::HingeJointParam) const override { return 0.f; }
    void _hinge_joint_set_flag(const RID &, PhysicsServer3D::HingeJointFlag, bool) override {}
    bool _hinge_joint_get_flag(const RID &, PhysicsServer3D::HingeJointFlag) const override { return false; }

    // Prismatic / Slider joint (5-DOF)
    void _joint_make_slider(const RID &p_joint,
                            const RID &p_body_a, const Transform3D &p_local_ref_a,
                            const RID &p_body_b, const Transform3D &p_local_ref_b) override;
    void  _slider_joint_set_param(const RID &, PhysicsServer3D::SliderJointParam, float) override {}
    float _slider_joint_get_param(const RID &, PhysicsServer3D::SliderJointParam) const override { return 0.f; }

    // Universal joint ← cone-twist API (4-DOF)
    void _joint_make_cone_twist(const RID &p_joint,
                                const RID &p_body_a, const Transform3D &p_local_ref_a,
                                const RID &p_body_b, const Transform3D &p_local_ref_b) override;
    void  _cone_twist_joint_set_param(const RID &, PhysicsServer3D::ConeTwistJointParam, float) override {}
    float _cone_twist_joint_get_param(const RID &, PhysicsServer3D::ConeTwistJointParam) const override { return 0.f; }

    // Generic 6DOF → stub (all constraints off = free)
    void _joint_make_generic_6dof(const RID &, const RID &, const Transform3D &,
                                  const RID &, const Transform3D &) override {}
    void  _generic_6dof_joint_set_param(const RID &, Vector3::Axis,
                                        PhysicsServer3D::G6DOFJointAxisParam, float) override {}
    float _generic_6dof_joint_get_param(const RID &, Vector3::Axis,
                                        PhysicsServer3D::G6DOFJointAxisParam) const override { return 0.f; }
    void _generic_6dof_joint_set_flag(const RID &, Vector3::Axis,
                                      PhysicsServer3D::G6DOFJointAxisFlag, bool) override {}
    bool _generic_6dof_joint_get_flag(const RID &, Vector3::Axis,
                                      PhysicsServer3D::G6DOFJointAxisFlag) const override { return false; }

    PhysicsServer3D::JointType _joint_get_type(const RID &p_joint) const override;
    void    _joint_set_solver_priority(const RID &, int32_t) override {}
    int32_t _joint_get_solver_priority(const RID &) const override { return 0; }
    void _joint_disable_collisions_between_bodies(const RID &, bool) override {}
    bool _joint_is_disabled_collisions_between_bodies(const RID &) const override { return false; }

    // ── M-ABD 전용 joint 생성 API (GDScript에서 호출 가능) ────
    // body_a_rid, body_b_rid: RigidBody3D의 get_rid()
    // pivot_world: world-space pivot point
    // axis_world: world-space axis (hinge/prismatic only)
    // returns joint RID
    RID abd_create_ball_joint(const RID &body_a, const RID &body_b,
                              const Vector3 &pivot_world);
    RID abd_create_hinge_joint(const RID &body_a, const RID &body_b,
                               const Vector3 &pivot_world,
                               const Vector3 &axis_world);
    RID abd_create_universal_joint(const RID &body_a, const RID &body_b,
                                   const Vector3 &pivot_world,
                                   const Vector3 &axis1_world,
                                   const Vector3 &axis2_world);
    RID abd_create_prismatic_joint(const RID &body_a, const RID &body_b,
                                   const Vector3 &pivot_world,
                                   const Vector3 &slide_axis_world);

    void _free_rid(const RID &p_rid) override;

private:
    bool _is_active = true;
    float _last_h   = ABD_H;

    std::unordered_set<uint64_t> _spaces;
    RIDMap<ShapeData>     _shapes;
    std::list<ABDBody>    _body_storage;
    RIDMap<ABDBody*>      _body_map;
    std::vector<ABDBody*> _body_list;
    RIDMap<Callable>      _sync_callbacks;
    // Owns IPCBodyState objects (memnew'd, freed in _free_rid / destructor).
    // PhysicsDirectBodyState3DExtension inherits Object, NOT RefCounted,
    // so Ref<> cannot be used — we manage lifetime manually with memnew/memdelete.
    RIDMap<IPCBodyState*> _body_states;

    // ── Joint storage ─────────────────────────────────────────
    std::list<ABDJoint>   _joint_storage;
    RIDMap<ABDJoint*>     _joint_map;
    // Joint type registry (for _joint_get_type)
    RIDMap<PhysicsServer3D::JointType> _joint_type_map;

    // BVH
    std::vector<BVHNode> _bvh_pool;
    int                  _bvh_root = -1;

    uint64_t _next_rid_id = 1;
    RID      _make_rid();

    // ── Shape builders ────────────────────────────────────────
    static ShapeData _build_box_shape(const Vector3 &half_extents);
    static ShapeData _build_convex_shape(const PackedVector3Array &points);
    static ShapeData _build_concave_shape(const PackedVector3Array &faces);

    void _rebuild_body_sim_mesh(ABDBody &b);
    void _recalc_mass_props(ABDBody &b, float mass);

    // ── BVH ───────────────────────────────────────────────────
    void  _bvh_rebuild();
    int   _bvh_build(std::vector<int> &indices, int lo, int hi);
    void  _bvh_query_pairs(int a, int b, std::vector<std::pair<int,int>> &out) const;
    void  _bvh_collect_internal_pairs(int node, std::vector<std::pair<int,int>> &out) const;

    // ── Contact detection ─────────────────────────────────────
    void  _find_contact_pairs(std::vector<ABDContactPair> &out) const;
    float _pt_distance(const Vector3 &p,
                       const Vector3 &a, const Vector3 &b, const Vector3 &c) const;

    // ── ABD solver (ipc_physics_server_solver.cpp) ────────────
    void _abd_step(float h);

    // Compute total affine force f_A for one body
    // = f_ext + gravity (via G^T) + inertia momentum (1/h)*M_A*q̇
    //   - elastic restoring K̄_A*(q - q_rest)
    // 이것이 Algorithm 1의 f_A = J̄ᵀ f  (Eq.14, Eq.44 통합)
    static void _compute_f_A(ABDBody &b, float h, Vec12 &f_A_out);

    // One Newton step: Algorithm 1 (논문 Section 3.4, Alg.1)
    // 입력: f_A (complete RHS including inertia/elastic/external)
    // 출력: δq
    static void _abd_newton_step(ABDBody &b, const Vec12 &f_A, Vec12 &dq_out);

    // Contact penalty force contribution → f_A
    void _add_contact_forces(ABDBody &b, int body_idx,
                             const std::vector<ABDContactPair> &pairs,
                             Vec12 &f_A) const;

    // Apply δq, update velocity (implicit Euler)
    static void _integrate(ABDBody &b, const Vec12 &dq, float h);

    void _dispatch_sync_callbacks();

    // ── Joint helpers ─────────────────────────────────────────
    // Find closest CP index on body to a world-space point
    static int _find_closest_cp(const ABDBody &b, const Vector3 &world_pt);

    // Build ABDJoint from two body RIDs and configure CPs
    ABDJoint *_joint_alloc(const RID &joint_rid,
                           const RID &body_a, const RID &body_b,
                           ABDJointType type);
};

} // namespace godot

// Include after class definition to break circular dependency:
//   ipc_body_state.h → ipc_physics_server.h (for ABDBody)
//   ipc_physics_server.h needs IPCBodyState only for Ref<> storage type
#include "ipc_body_state.h"
