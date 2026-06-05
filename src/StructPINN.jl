"""
    StructPINN

Differentiable hard-constraint layers for scientific machine learning in Julia.

A constraint layer maps a raw network output `zhat` to a corrected output
`zstar` that locally satisfies `c(z) = 0`, and is differentiable end to end by
differentiating the optimality (KKT) conditions rather than the solver
iterations. See PLAN.md for the full specification (invariants I1..I13 and the
gated milestones).

This module currently implements:
- `ProjectionResult`, the per-call status object (PLAN.md Section 6, I10).
- the affine projection layer, M1 (`AffineConstraint`).
- the nonlinear KKT projection spike, M0.5 (`NonlinearConstraint`).
- the weighted affine, sparse KKT, sparse box-affine, dense QP, and positivity layers, M6.
"""
module StructPINN

using LinearAlgebra
using SparseArrays
using ChainRulesCore

export ProjectionResult, issuccess
export AffineConstraint, DiagonalWeightedAffineConstraint
export SparseAffineConstraint, SparseDiagonalWeightedAffineConstraint
export SparseAffineProjectionCache
export SparseBoxAffineConstraint
export CachedSparseBoxAffineConstraint, WarmStartedSparseBoxAffineConstraint
export SparseBoxAffineWorkspace, reset_workspace!
export SparseLinearQPActiveSetConstraint, SparseLinearQPPrimalDualConstraint
export NonlinearConstraint, BoxConstraint
export WeightedSimplexConstraint, BoundedWeightedSimplexConstraint
export DenseLinearQPConstraint
export project, vjp, correct
export circle_constraint, tangential_constraint

include("projection_result.jl")
include("affine.jl")
include("sparse_kkt.jl")
include("sparse_box_affine.jl")
include("sparse_qp.jl")
include("nonlinear.jl")
include("box.jl")
include("simplex.jl")
include("bounded_simplex.jl")
include("dense_qp.jl")
include("ad.jl")

end # module
