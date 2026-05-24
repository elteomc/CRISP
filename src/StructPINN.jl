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
"""
module StructPINN

using LinearAlgebra
using ChainRulesCore

export ProjectionResult, issuccess
export AffineConstraint, NonlinearConstraint
export project, vjp, correct
export circle_constraint, tangential_constraint

include("projection_result.jl")
include("affine.jl")
include("nonlinear.jl")
include("ad.jl")

end # module
