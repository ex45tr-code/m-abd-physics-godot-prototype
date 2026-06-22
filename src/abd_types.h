#pragma once
// ════════════════════════════════════════════════════════════════════════════
//  abd_types.h  —  공유 기본 타입 (순환 include 방지)
// ════════════════════════════════════════════════════════════════════════════
#include <array>
#include <cmath>

namespace godot {

// 12-DOF affine coordinate vector
using Vec12 = std::array<float, 12>;
inline Vec12 vec12_zero() { Vec12 v; v.fill(0.f); return v; }

} // namespace godot
