#pragma once

#include <godot_cpp/classes/physics_direct_body_state3d_extension.hpp>
#include <godot_cpp/classes/physics_direct_space_state3d.hpp>
#include <godot_cpp/variant/basis.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/core/class_db.hpp>

namespace godot {

// Forward declaration — 실제 정의는 ipc_physics_server.h
struct ABDBody;

class IPCBodyState : public PhysicsDirectBodyState3DExtension {
    GDCLASS(IPCBodyState, PhysicsDirectBodyState3DExtension)

public:
    ABDBody* body = nullptr;

    static void _bind_methods() {}

    // ── transform / velocity ─────────────────────────────────
    Transform3D _get_transform() const override;
    void        _set_transform(const Transform3D &p_transform) override;
    Vector3     _get_linear_velocity() const override;
    void        _set_linear_velocity(const Vector3 &p_velocity) override;
    Vector3     _get_angular_velocity() const override;
    void        _set_angular_velocity(const Vector3 &p_velocity) override;
    float       _get_inverse_mass() const override;
    Vector3     _get_inverse_inertia() const override;

    // ── stub overrides ────────────────────────────────────────
    Vector3 _get_total_gravity() const override                                              { return Vector3(0, -9.81f, 0); }
    float   _get_total_linear_damp() const override                                          { return 0.f; }
    float   _get_total_angular_damp() const override                                         { return 0.f; }
    Vector3 _get_center_of_mass() const override                                             { return Vector3(); }
    Vector3 _get_center_of_mass_local() const override                                       { return Vector3(); }
    Basis   _get_principal_inertia_axes() const override                                     { return Basis(); }
    Basis   _get_inverse_inertia_tensor() const override                                     { return Basis(); }

    Vector3 _get_velocity_at_local_position(const Vector3 &) const override                  { return Vector3(); }

    void _apply_central_impulse(const Vector3 &) override                                    {}
    void _apply_impulse(const Vector3 &, const Vector3 &) override                           {}
    void _apply_torque_impulse(const Vector3 &) override                                     {}
    void _apply_central_force(const Vector3 &) override                                      {}
    void _apply_force(const Vector3 &, const Vector3 &) override                             {}
    void _apply_torque(const Vector3 &) override                                             {}

    void    _add_constant_central_force(const Vector3 &) override                            {}
    void    _add_constant_force(const Vector3 &, const Vector3 &) override                   {}
    void    _add_constant_torque(const Vector3 &) override                                   {}
    void    _set_constant_force(const Vector3 &) override                                    {}
    Vector3 _get_constant_force() const override                                             { return Vector3(); }
    void    _set_constant_torque(const Vector3 &) override                                   {}
    Vector3 _get_constant_torque() const override                                            { return Vector3(); }

    void     _set_sleep_state(bool) override                                                 {}
    bool     _is_sleeping() const override                                                   { return false; }

    void     _set_collision_layer(uint32_t) override                                         {}
    uint32_t _get_collision_layer() const override                                           { return 0xFFFFFFFF; }
    void     _set_collision_mask(uint32_t) override                                          {}
    uint32_t _get_collision_mask() const override                                            { return 0xFFFFFFFF; }

    int32_t  _get_contact_count() const override                                             { return 0; }
    Vector3  _get_contact_local_position(int32_t) const override                             { return Vector3(); }
    Vector3  _get_contact_local_normal(int32_t) const override                               { return Vector3(); }
    Vector3  _get_contact_impulse(int32_t) const override                                    { return Vector3(); }
    int32_t  _get_contact_local_shape(int32_t) const override                                { return 0; }
    Vector3  _get_contact_local_velocity_at_position(int32_t) const override                 { return Vector3(); }
    RID      _get_contact_collider(int32_t) const override                                   { return RID(); }
    Vector3  _get_contact_collider_position(int32_t) const override                          { return Vector3(); }
    uint64_t _get_contact_collider_id(int32_t) const override                                { return 0; }
    Object*  _get_contact_collider_object(int32_t) const override                            { return nullptr; }
    int32_t  _get_contact_collider_shape(int32_t) const override                             { return 0; }
    Vector3  _get_contact_collider_velocity_at_position(int32_t) const override              { return Vector3(); }

    float    _get_step() const override                                                      { return 1.f / 60.f; }
    void     _integrate_forces() override                                                    {}
    PhysicsDirectSpaceState3D* _get_space_state() override                                   { return nullptr; }
};

} // namespace godot
