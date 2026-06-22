#include "ipc_body_state.h"
#include "ipc_physics_server.h"

namespace godot {

Transform3D IPCBodyState::_get_transform() const {
    if (!body) return Transform3D();
    return body->get_transform();
}

void IPCBodyState::_set_transform(const Transform3D &p_transform) {
    if (!body) return;
    body->set_transform(p_transform);
    body->q_prev = body->q;
}

Vector3 IPCBodyState::_get_linear_velocity() const {
    if (!body) return Vector3();
    return body->get_linear_velocity();
}

void IPCBodyState::_set_linear_velocity(const Vector3 &p_velocity) {
    if (!body) return;
    body->set_linear_velocity(p_velocity);
}

Vector3 IPCBodyState::_get_angular_velocity() const {
    if (!body) return Vector3();
    return body->get_angular_velocity();
}

void IPCBodyState::_set_angular_velocity(const Vector3 &p_velocity) {
    if (!body) return;
    body->set_angular_velocity(p_velocity);
}

float IPCBodyState::_get_inverse_mass() const {
    if (!body) return 0.f;
    return body->inv_mass;
}

Vector3 IPCBodyState::_get_inverse_inertia() const {
    if (!body) return Vector3();
    // Box inertia inverse (diagonal, body frame)
    float hx = body->half_extents.x;
    float hy = body->half_extents.y;
    float hz = body->half_extents.z;
    float m  = body->mass;
    if (m <= 0.f) return Vector3();
    return Vector3(
        3.f / (m * (hy*hy + hz*hz)),
        3.f / (m * (hx*hx + hz*hz)),
        3.f / (m * (hx*hx + hy*hy))
    );
}

} // namespace godot
