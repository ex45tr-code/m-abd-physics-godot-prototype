#pragma once
// ════════════════════════════════════════════════════════════════════════════
//  abd_kkt_solver.h  —  M-ABD Section 5: Dual-space KKT Solver
// ════════════════════════════════════════════════════════════════════════════

#include "abd_types.h"  // Vec12
#include "abd_joint.h"  // ABDJoint, RotMat3, abd_compute_y 등
#include <array>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace godot {

struct ABDBody;   // ipc_physics_server.h 에 정의
struct ABDJoint;  // abd_joint.h 에 정의

// ════════════════════════════════════════════════════════════════════════════
//  DynMat  —  동적 크기 dense 행렬 (row-major)
//  KKT dual matrix, constraint gradient 등에 사용
//  크기: joint당 최대 5×12, dual matrix 최대 5×5
// ════════════════════════════════════════════════════════════════════════════

struct DynMat {
    int rows = 0, cols = 0;
    std::vector<float> v;

    DynMat() = default;
    DynMat(int r, int c) : rows(r), cols(c), v(r*c, 0.f) {}

    float &operator()(int r, int c)       { return v[r*cols+c]; }
    float  operator()(int r, int c) const { return v[r*cols+c]; }

    void zero() { std::fill(v.begin(), v.end(), 0.f); }

    // Cholesky factor (in-place, lower triangle)
    bool cholesky_factor();
    // Forward-backward solve: L L^T x = b (in-place on b)
    void cholesky_solve(std::vector<float> &b) const;
    // LU with partial pivoting (for non-SPD)
    bool lu_factor(std::vector<int> &piv);
    void lu_solve(const std::vector<int> &piv, std::vector<float> &b) const;
    // A += B
    void add(const DynMat &B);
    // C = A * B
    static DynMat mul(const DynMat &A, const DynMat &B);
    // C = A^T
    DynMat transpose() const;
};

// Vec12 is defined in ipc_physics_server.h (included above)

// ════════════════════════════════════════════════════════════════════════════
//  ConstraintBlock
//    ∇C^k_α = ∂C^k/∂q^α  ∈ ℝ^{C_k × 12}
//    ∇C^k_β = ∂C^k/∂q^β  ∈ ℝ^{C_k × 12}
// ════════════════════════════════════════════════════════════════════════════

struct ConstraintBlock {
    int joint_idx  = -1;   // index in joint list
    int body_alpha = -1;   // body index α
    int body_beta  = -1;   // body index β
    int c_dim      = 0;    // C_k (constraint rows)

    // ∇C^k_α ∈ ℝ^{C_k×12},  ∇C^k_β ∈ ℝ^{C_k×12}
    DynMat grad_alpha;   // C_k × 12
    DynMat grad_beta;    // C_k × 12

    // Current constraint value C^k ∈ ℝ^{C_k}
    std::vector<float> C_val;

    // Dual variable δλ^k ∈ ℝ^{C_k}
    std::vector<float> dlambda;

    // Pre-computed: H^{-1} ∇C^T  (12 × C_k per body)
    DynMat Hinv_gradT_alpha;
    DynMat Hinv_gradT_beta;

    // Dual matrix block D^k = ∇C H^{-1} ∇C^T ∈ ℝ^{C_k × C_k}
    DynMat D;
    // Off-diagonal: B^k = ∇C^k H^{-1} ∇C^{k+1,T} ∈ ℝ^{C_k × C_{k+1}}  (chain)
    DynMat B_fwd;  // to next joint
    DynMat B_bwd;  // to prev joint

    // RHS in dual system: b^k = ∇C H^{-1} f_A ∈ ℝ^{C_k}
    std::vector<float> b_dual;
};

// ════════════════════════════════════════════════════════════════════════════
//  TopologyType — joint graph topology
// ════════════════════════════════════════════════════════════════════════════

enum class TopologyType {
    Independent,   // no joints
    Chain,         // single linear chain (Sec 5.2)
    Tree,          // acyclic tree (Sec 5.3, ABD-ABA)
    Loop,          // single loop (Sec 5.4, Schur)
    Graph,         // general (Sec 5.5, Gauss-Seidel)
};

// ════════════════════════════════════════════════════════════════════════════
//  ABDKKTSolver — Section 5 dual-space KKT solver
// ════════════════════════════════════════════════════════════════════════════

class ABDKKTSolver {
public:
    ABDKKTSolver() = default;

    // ── Main entry point ────────────────────────────────────────
    // bodies, joints: 현재 시뮬레이션 상태
    // f_A_list: [in/out] 각 body의 affine force (Algorithm 1 전에 KKT가 수정)
    // h: timestep
    // Returns δq for each body (Algorithm 1의 결과를 KKT로 교정)
    void solve(
        const std::vector<ABDBody*> &bodies,
        const std::vector<ABDJoint> &joints,
        std::vector<Vec12>          &f_A_list,   // modified in-place
        std::vector<Vec12>          &dq_out,      // output
        float h
    );

    // ── Section 5.1: Constraint gradient ∇C (Eq.30-35) ─────────
    // Computes grad_alpha, grad_beta, C_val for one joint
    static void compute_constraint_gradient(
        const ABDBody &ba, const ABDBody &bb,
        const ABDJoint &joint,
        ConstraintBlock &out);

    // ── Topology analysis ────────────────────────────────────────
    static TopologyType analyze_topology(
        int n_bodies,
        const std::vector<ABDJoint> &joints);

private:
    // ── Section 5.2: Chain (block-tridiagonal Thomas) (Eq.39-41) ─
    void solve_chain(
        const std::vector<ABDBody*> &bodies,
        const std::vector<ABDJoint> &joints,
        std::vector<ConstraintBlock> &blocks,
        std::vector<Vec12> &f_A_list,
        std::vector<Vec12> &dq_out,
        float h);

    // ── Section 5.3: Tree — ABD-ABA (Algorithm 2) ────────────────
    void solve_tree(
        const std::vector<ABDBody*> &bodies,
        const std::vector<ABDJoint> &joints,
        std::vector<ConstraintBlock> &blocks,
        std::vector<Vec12> &f_A_list,
        std::vector<Vec12> &dq_out,
        float h);

    // ── Section 5.4: Loop — Schur complement (Eq.58) ─────────────
    void solve_loop(
        const std::vector<ABDBody*> &bodies,
        const std::vector<ABDJoint> &joints,
        std::vector<ConstraintBlock> &blocks,
        std::vector<Vec12> &f_A_list,
        std::vector<Vec12> &dq_out,
        float h);

    // ── Section 5.5: Graph — multi-directional Gauss-Seidel ──────
    void solve_graph(
        const std::vector<ABDBody*> &bodies,
        const std::vector<ABDJoint> &joints,
        std::vector<ConstraintBlock> &blocks,
        std::vector<Vec12> &f_A_list,
        std::vector<Vec12> &dq_out,
        float h);

    // ── Helpers ───────────────────────────────────────────────────
    // Build all ConstraintBlocks from joint list
    void build_constraint_blocks(
        const std::vector<ABDBody*> &bodies,
        const std::vector<ABDJoint> &joints,
        std::vector<ConstraintBlock> &out);

    // Pre-compute H^{-1} ∇C^T, D, b for each block
    void precompute_dual(
        const std::vector<ABDBody*> &bodies,
        std::vector<ConstraintBlock> &blocks,
        const std::vector<Vec12> &f_A_list,
        float h);

    // Apply δλ back to primal: δq^j -= H^{-1} ∇C^T δλ  (Eq.27)
    static void apply_dual_correction(
        const std::vector<ABDBody*> &bodies,
        const std::vector<ConstraintBlock> &blocks,
        std::vector<Vec12> &dq_list);

    // Build topology: adjacency and tree structure
    struct TreeNode {
        int body_idx   = -1;
        int parent_idx = -1;          // body index
        int joint_to_parent = -1;     // joint index
        std::vector<int> children;    // body indices
        std::vector<int> child_joints;// joint indices

        // ABD-ABA accumulators (Eq.53-54)
        DynMat Ĥ_A;    // 12×12 condensed Hessian
        Vec12  f_hat;   // condensed force
        bool   is_root = false;
    };

    static std::vector<TreeNode> build_tree(
        int n_bodies,
        const std::vector<ABDJoint> &joints,
        std::vector<int> &order_leaf_to_root);

    // Find chain order (bodies in chain order, joints in chain order)
    static bool find_chain(
        int n_bodies,
        const std::vector<ABDJoint> &joints,
        std::vector<int> &body_order,
        std::vector<int> &joint_order);

    // Find loop breaker bodies for Schur complement
    static std::vector<int> find_loop_breakers(
        int n_bodies,
        const std::vector<ABDJoint> &joints);

    // Single-body unconstrained Newton step (Algorithm 1, via stored H_bar)
    static Vec12 unconstrained_dq(const ABDBody &b, const Vec12 &f_A);
};

} // namespace godot
