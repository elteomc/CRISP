"""
    ProjectionResult{T,V}

Full result of a single constraint projection. The public layer returns only
`zstar`, but tests and benchmarks log the whole struct so that no failure is
silent (PLAN.md I10).

`status` is one of:

- `:success`             regular solved point, the gradient is valid.
- `:max_iters`           Newton did not converge in the iteration budget.
- `:line_search_fail`    backtracking could not reduce the residual.
- `:singular_constraint` rank-deficient constraint Jacobian *at the solved
                         point* (PLAN.md I4). No gradient is claimed.
- `:ill_conditioned`     near-singular KKT matrix at the solution (I5).
- `:nonunique_input`     the input lies where the projection map is set-valued
                         or nondifferentiable (for example the centre of a
                         sphere constraint), so no derivative is claimed (I13).

The last three are distinct conditions and must not be merged. `:singular_constraint`
certifies the *solved point*, `:nonunique_input` is an *input-domain* failure,
and `:max_iters` / `:line_search_fail` are solver nonconvergence.
"""
struct ProjectionResult{T<:Real, V<:AbstractVector{T}}
    zstar::V                  # corrected output
    lambda::V                 # multipliers, length m
    constraint_residual::T    # norm(c(zstar))
    stationarity_residual::T  # norm(zstar - zhat + J' lambda)
    correction_norm::T        # norm(zstar - zhat)
    iterations::Int
    jac_min_singular::T       # sigma_min(J(zstar)), the LICQ check (I4)
    kkt_cond_estimate::T      # estimate of cond(M) at the solution (I5)
    status::Symbol
end

"""
    issuccess(r::ProjectionResult) -> Bool

True only when the projection returned a regular solved point with a valid
gradient.
"""
issuccess(r::ProjectionResult) = r.status === :success
