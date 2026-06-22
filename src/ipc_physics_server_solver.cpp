#include "ipc_physics_server.h"
#include "ipc_body_state.h"
#include "abd_joint.h"
#include "abd_kkt_solver.h"

#include <godot_cpp/variant/utility_functions.hpp>
#include <cmath>

// ════════════════════════════════════════════════════════════════════════════
//  ipc_physics_server_solver.cpp
//
//  M-ABD Co-rotated Single-body Newton Step
//  논문 Section 3.4, Algorithm 1 (arXiv:2603.08079v2, p.5)
//
//  [수정 사항 - _add_contact_forces 의 dist_clamped 버그]
//
//  원본 코드:
//    float dist_clamped = (cp.dist < 0.f) ? 0.f : cp.dist;
//    float pen = ABD_CONTACT_D_HAT - dist_clamped;
//
//  이 코드에서 dist < 0 (침투) 이면 dist_clamped = 0 으로 고정되어
//  pen = ABD_CONTACT_D_HAT 가 항상 일정한 값이 됨.
//  즉 침투 깊이가 얼마든 상관없이 같은 force → 위치 피드백 없음.
//
//  올바른 코드:
//    float pen = ABD_CONTACT_D_HAT - cp.dist;
//    if (pen <= 0.f) continue;
//
//  cp.dist 의 의미:
//    > 0: 두 표면이 분리됨 (gap)
//    = 0: 접촉
//    < 0: 침투 (interpenetration depth = |dist|)
//
//  pen = d_hat - dist:
//    dist >> d_hat : pen << 0 → skip (멀리 떨어짐)
//    dist = d_hat  : pen = 0  → skip (임계 거리)
//    dist = 0      : pen = d_hat  (접촉 시작)
//    dist < 0      : pen > d_hat  (침투량에 비례하여 더 강한 반발력)
//
//  이것으로 50m짜리 가짜 contact (sd=-50 으로 로그에 나오는 것)도
//  pen = 0.02 - (-50) 가 아니라, 실제로 _find_contact_pairs 에서
//  cp.dist = +50 (양수, 분리됨) 이면 pen = 0.02 - 50 = -49.98 < 0 → skip.
//  (로그의 sd=-50 은 body_b 면에서의 signed distance 출력이고
//   실제 contact pair 의 cp.dist 는 다를 수 있으므로 확인 필요)
//
// ════════════════════════════════════════════════════════════════════════════

namespace godot {

static inline float sqrtf_s(float v) { return v <= 0.f ? 0.f : sqrtf(v); }

// ════════════════════════════════════════════════════════════════════════════
//  _compute_f_A
// ════════════════════════════════════════════════════════════════════════════

void IPCPhysicsServer::_compute_f_A(ABDBody &b, float h, Vec12 &f_A_out) {
    f_A_out.fill(0.f);
    if (b.is_static) return;

    float inv_h = 1.f / h;

    // [A] Gravity
    float A_mat[3][3]; b.get_A(A_mat);
    Vector3 gravity_f(0.f, b.mass * ABD_GRAVITY_Y, 0.f);
    ABDBody::apply_G_transpose(A_mat, Vector3(), gravity_f, f_A_out);

    // [A2] Accumulated external forces
    for (int i = 0; i < 12; ++i)
        f_A_out[i] += b.f_ext_accum[i];
    b.f_ext_accum.fill(0.f);

    // [B] Inertia: (1/h) M_A qdot  (논문 Eq.6: q̂ = q + h*qdot)
    for (int r = 0; r < 12; ++r) {
        float contrib = 0.f;
        for (int c = 0; c < 12; ++c)
            contrib += b.M_A(r, c) * b.qdot[c];
        f_A_out[r] += inv_h * contrib;
    }

    // [C] Elastic restoring: -∂Ψ/∂q  (논문 Eq.15)
    {
        float mu     = ABD_MU_LAME;
        float lambda = ABD_LAMBDA_LAME;
        float V      = 8.f * b.half_extents.x * b.half_extents.y * b.half_extents.z;
        float A[3][3]; b.get_A(A);
        float trA_minus_I = (A[0][0]-1.f) + (A[1][1]-1.f) + (A[2][2]-1.f);
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) {
                float dPsi = mu * (A[i][j] + A[j][i] - 2.f*(i==j ? 1.f : 0.f))
                           + lambda * trA_minus_I * (i==j ? 1.f : 0.f);
                f_A_out[j*3 + i] -= V * dPsi;
            }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  _abd_newton_step  — Algorithm 1
// ════════════════════════════════════════════════════════════════════════════

void IPCPhysicsServer::_abd_newton_step(ABDBody &b, const Vec12 &f_A, Vec12 &dq_out) {
    dq_out.fill(0.f);
    if (b.is_static) return;

    static int s_ns = 0; ++s_ns;
    const bool NSDBG = (s_ns <= 40) || (s_ns % 120 == 0);

    if (NSDBG) {
        UtilityFunctions::print("[DBG_NS] newton_step #", s_ns,
            " f_A_t=(", f_A[9], ",", f_A[10], ",", f_A[11], ")",
            " f_A_a0=(", f_A[0], ",", f_A[1], ",", f_A[2], ")");
    }

    float A[3][3]; b.get_A(A);
    if (NSDBG) UtilityFunctions::print("[DBG_NS]   A=[[", A[0][0], ",", A[0][1], ",", A[0][2], "],",
        "[", A[1][0], ",", A[1][1], ",", A[1][2], "],",
        "[", A[2][0], ",", A[2][1], ",", A[2][2], "]]");

    // Lines 4-8: diag4(A^T) f_A with length preservation
    Vec12 f_rot;
    for (int k = 0; k < 4; ++k) {
        const int off = k * 3;
        Vector3 fk(f_A[off], f_A[off+1], f_A[off+2]);
        float l2 = fk.dot(fk);
        Vector3 fk_rot(
            A[0][0]*fk.x + A[1][0]*fk.y + A[2][0]*fk.z,
            A[0][1]*fk.x + A[1][1]*fk.y + A[2][1]*fk.z,
            A[0][2]*fk.x + A[1][2]*fk.y + A[2][2]*fk.z
        );
        float len_rot = fk_rot.length();
        float scale   = (len_rot > 1e-14f) ? sqrtf_s(l2) / len_rot : 1.f;
        f_rot[off]   = fk_rot.x * scale;
        f_rot[off+1] = fk_rot.y * scale;
        f_rot[off+2] = fk_rot.z * scale;
    }

    // Line 9: H̄_A δp = f_rot
    float rhs[12];
    for (int i = 0; i < 12; ++i) rhs[i] = f_rot[i];
    b.H_bar_A.cholesky_solve(rhs);

    // Lines 10-14: diag4(A) δp with length preservation
    for (int k = 0; k < 4; ++k) {
        const int off = k * 3;
        Vector3 dp(rhs[off], rhs[off+1], rhs[off+2]);
        float l2 = dp.dot(dp);
        Vector3 Adp(
            A[0][0]*dp.x + A[0][1]*dp.y + A[0][2]*dp.z,
            A[1][0]*dp.x + A[1][1]*dp.y + A[1][2]*dp.z,
            A[2][0]*dp.x + A[2][1]*dp.y + A[2][2]*dp.z
        );
        float len_Adp = Adp.length();
        float scale   = (len_Adp > 1e-14f) ? sqrtf_s(l2) / len_Adp : 1.f;
        dq_out[off]   = Adp.x * scale;
        dq_out[off+1] = Adp.y * scale;
        dq_out[off+2] = Adp.z * scale;
    }

    if (NSDBG) UtilityFunctions::print("[DBG_NS]   dq_t=(", dq_out[9], ",", dq_out[10], ",", dq_out[11], ")",
        " dq_a0=(", dq_out[0], ",", dq_out[1], ",", dq_out[2], ")");
}

// ════════════════════════════════════════════════════════════════════════════
//  _add_contact_forces
//
//  [FIX] dist_clamped 제거.
//  pen = d_hat - cp.dist  (부호 있는 거리 그대로 사용)
//  pen <= 0 이면 skip (충분히 멀거나 d_hat 경계 밖)
//  pen > 0  이면 force_mag = kappa * pen 적용
// ════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════
//  _add_contact_forces
//
//  contact force를 f_A에 더함.
//  dist: contact pair의 max-signed-distance (양수=분리, 음수=침투)
//  pen  = d_hat - dist
//  pen > 0 일 때만 force 발생.
//
//  force_mag = kappa * pen^2 / dist  (IPC barrier gradient, log barrier 근사)
//  → 단, dist < epsilon 이면 일반 linear penalty로 fallback.
//
//  주의: 이 force는 "다음 스텝의 위치를 바꿀 추진력"이 아니라
//        "현재 침투를 막는 반발력"이므로, 속도가 이미 튕기는 방향이면 무시.
// ════════════════════════════════════════════════════════════════════════════

void IPCPhysicsServer::_add_contact_forces(ABDBody &b, int body_idx,
                                            const std::vector<ABDContactPair> &pairs,
                                            Vec12 &f_A) const {
    for (const auto &cp : pairs) {
        // d = max signed distance (양수=분리, 음수=침투)
        // d >= d_hat → force = 0
        //
        // IPC log barrier는 Newton iteration + line search 없이 단일 스텝에서
        // 쓰면 d가 작을 때 force가 폭발(~kappa*(d_hat-d)²/d)하여 불안정.
        // 대신 quadratic penalty 사용:
        //   force_mag = kappa * (d_hat - d)
        //   d → d_hat : force → 0
        //   d = 0     : force = kappa * d_hat
        //   d < 0     : force > kappa * d_hat  (침투 시 더 강해짐)
        // 이 공식은 bounded increment이므로 단일 Newton step에서 안정적.

        float d     = cp.dist;
        float d_hat = ABD_CONTACT_D_HAT;

        if (d >= d_hat) continue;

        // quadratic penalty: force ∝ (d_hat - d)
        float force_mag = ABD_CONTACT_KAPPA * (d_hat - d);

        // n_hat: body_b face의 world outward normal
        // body_a 꼭짓점을 n_hat 방향으로 반발
        if (cp.body_a == body_idx) {
            Vector3 fw = cp.n_hat * force_mag;
            f_A[9]  += fw.x;
            f_A[10] += fw.y;
            f_A[11] += fw.z;
        } else if (cp.body_b == body_idx) {
            Vector3 fw = cp.n_hat * (-force_mag);
            f_A[9]  += fw.x;
            f_A[10] += fw.y;
            f_A[11] += fw.z;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  _integrate  — implicit Euler
// ════════════════════════════════════════════════════════════════════════════

void IPCPhysicsServer::_integrate(ABDBody &b, const Vec12 &dq, float h) {
    float inv_h = 1.f / h;
    b.q_prev = b.q;
    for (int i = 0; i < 12; ++i) b.q[i] += dq[i];
    for (int i = 0; i < 9;  ++i) b.qdot[i] = dq[i] * inv_h * ABD_ANG_DAMP;
    for (int i = 9; i < 12; ++i) b.qdot[i] = dq[i] * inv_h * ABD_LIN_DAMP;
}

// ════════════════════════════════════════════════════════════════════════════
//  _abd_step
// ════════════════════════════════════════════════════════════════════════════

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

void IPCPhysicsServer::_abd_step(float h) {
    static int s_step = 0; ++s_step;
    const bool SDBG = (s_step <= 10) || (s_step % 30 == 0);

    if (SDBG) UtilityFunctions::print("[DBG_STEP] ===== step=", s_step, " h=", h,
        " n_bodies=", (int)_body_list.size(), " =====");

    // ── 1. Lazy init ─────────────────────────────────────────
    for (ABDBody *bp : _body_list) {
        ABDBody &b = *bp;
        if (b.sim_started) continue;
        if (b.verts_local.empty() && !b.shapes.empty())
            _rebuild_body_sim_mesh(b);
        if (b.verts_local.empty()) {
            b.half_extents = Vector3(0.5f, 0.5f, 0.5f);
            ShapeData sd = _build_box_shape(b.half_extents);
            b.verts_local = sd.verts;
            b.faces_sim   = sd.faces;
        }
        b.build_mass_matrix();
        b.build_stiffness_matrix();
        b.build_control_tetrahedron();
        b.build_J_bar();
        b.update_H_bar(h);
        b.q_prev = b.q;
        b.f_ext_accum.fill(0.f);
        b.sim_started = true;
    }

    if (s_step == 1) {
        for (int _bi = 0; _bi < (int)_body_list.size(); ++_bi)
            DBG_print_body_verts(*_body_list[_bi], _bi, s_step);
    }

    if (SDBG) {
        for (int _bi = 0; _bi < (int)_body_list.size(); ++_bi) {
            const ABDBody &_b = *_body_list[_bi];
            Vector3 _t = _b.get_t();
            UtilityFunctions::print("[DBG_STEP] body[", _bi, "]",
                " static=", _b.is_static,
                " sim_started=", _b.sim_started,
                " pos=(", _t.x, ",", _t.y, ",", _t.z, ")",
                " vel=(", _b.qdot[9], ",", _b.qdot[10], ",", _b.qdot[11], ")",
                " mass=", _b.mass,
                " half_ext=(", _b.half_extents.x, ",", _b.half_extents.y, ",", _b.half_extents.z, ")",
                " n_verts=", (int)_b.verts_local.size(),
                " n_faces=", (int)_b.faces_sim.size());
        }
    }

    // ── 2. H̄_A 재인수분해 ────────────────────────────────────
    static float s_cached_h = -1.f;
    if (h != s_cached_h) {
        for (ABDBody *bp : _body_list)
            if (!bp->is_static) bp->update_H_bar(h);
        s_cached_h = h;
    }

    // ── 3. Contact detection ──────────────────────────────────
    _bvh_rebuild();
    std::vector<ABDContactPair> contacts;
    _find_contact_pairs(contacts);

    if (SDBG) UtilityFunctions::print("[DBG_STEP] contacts=", (int)contacts.size());

    // ── 4. f_A ───────────────────────────────────────────────
    int N = (int)_body_list.size();
    std::vector<Vec12> f_A_list(N);

    for (int bi = 0; bi < N; ++bi) {
        ABDBody &b = *_body_list[bi];
        if (b.is_static) { f_A_list[bi].fill(0.f); continue; }

        Vec12 f_before_contact;
        _compute_f_A(b, h, f_A_list[bi]);
        f_before_contact = f_A_list[bi];
        _add_contact_forces(b, bi, contacts, f_A_list[bi]);

        if (SDBG) {
            const Vec12 &fa = f_A_list[bi];
            UtilityFunctions::print("[DBG_FA] body[", bi, "]",
                " f_A_t_before_contact=(", f_before_contact[9], ",", f_before_contact[10], ",", f_before_contact[11], ")",
                " f_A_t_after_contact=(", fa[9], ",", fa[10], ",", fa[11], ")",
                " contact_delta_t=(", fa[9]-f_before_contact[9], ",", fa[10]-f_before_contact[10], ",", fa[11]-f_before_contact[11], ")");
            UtilityFunctions::print("[DBG_FA] body[", bi, "]",
                " f_A_a0=(", fa[0], ",", fa[1], ",", fa[2], ")",
                " f_A_a1=(", fa[3], ",", fa[4], ",", fa[5], ")",
                " f_A_a2=(", fa[6], ",", fa[7], ",", fa[8], ")");
        }
    }

    // ── 5. δq ─────────────────────────────────────────────────
    std::vector<Vec12> dq_list(N);
    if (_joint_storage.empty()) {
        for (int bi = 0; bi < N; ++bi) {
            ABDBody &b = *_body_list[bi];
            if (b.is_static) { dq_list[bi].fill(0.f); continue; }
            _abd_newton_step(b, f_A_list[bi], dq_list[bi]);
        }
    } else {
        std::vector<ABDJoint> joints_vec(_joint_storage.begin(), _joint_storage.end());
        ABDKKTSolver kkt;
        kkt.solve(_body_list, joints_vec, f_A_list, dq_list, h);
    }

    // ── 5.5. (제거됨) ────────────────────────────────────────

    // ── 6. Integrate ─────────────────────────────────────────
    for (int bi = 0; bi < N; ++bi) {
        ABDBody &b = *_body_list[bi];
        if (b.is_static) continue;
        if (SDBG) {
            Vector3 pos_before = b.get_t();
            UtilityFunctions::print("[DBG_INTEG] body[", bi, "] BEFORE",
                " pos=(", pos_before.x, ",", pos_before.y, ",", pos_before.z, ")",
                " dq_t=(", dq_list[bi][9], ",", dq_list[bi][10], ",", dq_list[bi][11], ")");
        }
        _integrate(b, dq_list[bi], h);
        if (SDBG) {
            Vector3 pos_after = b.get_t();
            UtilityFunctions::print("[DBG_INTEG] body[", bi, "] AFTER",
                " pos=(", pos_after.x, ",", pos_after.y, ",", pos_after.z, ")",
                " vel=(", b.qdot[9], ",", b.qdot[10], ",", b.qdot[11], ")");
        }
    }
}

} // namespace godot
