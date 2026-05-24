"""
    NonlinearConstraint(c, J, Hlam)

A general nonlinear constraint `c : R^n -> R^m` with

- `c(z)`        the constraint values, a length-`m` vector,
- `J(z)`        the Jacobian `dc/dz`, an `m x n` matrix,
- `Hlam(z, l)`  the Lagrangian Hessian contribution `sum_i l_i Hess(c_i)(z)`,
                an `n x n` matrix.

The derivatives are supplied analytically here so the M0.5 spike has zero
external dependencies. M3 will swap in AD-provided derivatives.
"""
struct NonlinearConstraint{C,J,H}
    c::C
    J::J
    Hlam::H
end

"""
    circle_constraint(r) -> NonlinearConstraint

The toy constraint `c(z) = z'z - r^2` (a sphere of radius `r`). Regular at every
solution for `r > 0`, since `J(zstar) = 2 zstar'` has full row rank there. Used
for the regular and the nonunique-input M0.5 cases (PLAN.md Section 9).
"""
circle_constraint(r) = NonlinearConstraint(
    z -> [dot(z, z) - r^2],
    z -> reshape(2 .* z, 1, length(z)),
    (z, l) -> 2 * l[1] * Matrix(I, length(z), length(z)),
)

"""
    tangential_constraint(n=2) -> NonlinearConstraint

The toy constraint `c(z) = z_1^2` (feasible set is the hyperplane `z_1 = 0`).
Its Jacobian `J(zstar) = (2 z_1, 0, ...) = 0` at the feasible point, a genuine
LICQ failure *at the solution*. Used for the `:singular_constraint` M0.5 case.
"""
function tangential_constraint(n::Int = 2)
    NonlinearConstraint(
        z -> [z[1]^2],
        z -> reshape([i == 1 ? 2 * z[1] : zero(eltype(z)) for i in 1:length(z)], 1, length(z)),
        (z, l) -> (H = zeros(eltype(z), length(z), length(z)); H[1, 1] = 2 * l[1]; H),
    )
end

# Build the constraint Jacobian Jz and the KKT matrix M at (z, l).
# M = [ I + sum_i l_i Hess(c_i)(z)   J(z)' ;
#       J(z)                          0    ]   (PLAN.md 4.4), symmetric, indefinite.
function _kkt(c::NonlinearConstraint, z, l, n, m)
    Jz = c.J(z)
    M = [Matrix(I, n, n) + c.Hlam(z, l)   transpose(Jz);
         Jz                                zeros(eltype(z), m, m)]
    return Jz, M
end

"""
    project(c::NonlinearConstraint, zhat; kwargs...) -> ProjectionResult

Local Euclidean projection by Newton on the stacked KKT residual `F(z, l) = 0`,
warm-started at `(zhat, 0)`, with a backtracking line search on `norm(F)`. The
factorization is a generic LU via `\\` (indefinite-safe, never Cholesky, I7).

Status logic distinguishes the three failure notions the M0.5 spike exists to
separate (PLAN.md I4, I5, I13):
- converged with rank-deficient `J(zstar)`  -> `:singular_constraint`
- converged with ill-conditioned `M`        -> `:ill_conditioned`
- a singular `M` reached while still infeasible (no identifiable branch from a
  symmetric input) -> `:nonunique_input`
- feasible but stationarity unreachable because `J` is rank-deficient there ->
  `:singular_constraint`
"""
function project(c::NonlinearConstraint, zhat::AbstractVector;
                 maxiter = 50, tol = 1e-10, tol_c = 1e-8,
                 tol_rank = 1e-6, cond_max = 1e8, amin = 1e-12)
    zh = float.(zhat)
    T = eltype(zh)
    n = length(zh)
    m = length(c.c(zh))
    z = copy(zh)
    l = zeros(T, m)

    Fval(z, l) = vcat(z - zh + transpose(c.J(z)) * l, c.c(z))

    converged = false
    solve_failed = false
    ls_failed = false
    iters = 0
    for it in 1:maxiter
        iters = it
        Fv = Fval(z, l)
        Fn = norm(Fv)
        if Fn < tol
            converged = true
            break
        end
        _, M = _kkt(c, z, l, n, m)
        if minimum(svdvals(M)) < 1 / cond_max          # M effectively singular
            solve_failed = true
            break
        end
        step = M \ (-Fv)
        dz = step[1:n]
        dl = step[n+1:end]
        alpha = one(T)
        while alpha > amin && norm(Fval(z + alpha * dz, l + alpha * dl)) > Fn
            alpha /= 2
        end
        if alpha <= amin
            ls_failed = true
            break
        end
        z += alpha * dz
        l += alpha * dl
    end

    Jz = c.J(z)
    jacmin = minimum(svdvals(Jz))                 # sigma_min(J) at the final iterate
    jac_input = minimum(svdvals(c.J(zh)))         # sigma_min(J) at the input zhat
    _, Mf = _kkt(c, z, l, n, m)
    condM = cond(Mf)
    deficient = jacmin < tol_rank

    # The input-vs-solution distinction is what separates a degenerate *input*
    # (nonunique or nondifferentiable, I13) from a degenerate *solution* (LICQ
    # failure at the solved point, I4). A singular M reached from a rank-deficient
    # input Jacobian means zhat itself is the bad point; from a full-rank input it
    # means the iteration is approaching a degenerate solution. See PLAN.md M0.5.
    status =
        converged    ? (deficient ? :singular_constraint :
                        (condM > cond_max ? :ill_conditioned : :success)) :
        solve_failed ? (jac_input < tol_rank ? :nonunique_input : :singular_constraint) :
        ls_failed    ? :line_search_fail :
                       (deficient ? :singular_constraint : :max_iters)

    cres = norm(c.c(z))
    sres = norm(z - zh + transpose(Jz) * l)
    ProjectionResult(z, l, T(cres), T(sres), T(norm(z - zh)), iters,
                     T(jacmin), T(condM), status)
end

"""
    vjp(c::NonlinearConstraint, res, gbar) -> Vector

Adjoint vector-Jacobian product. Rebuilds the exact KKT Jacobian `M` at the
solved point `(res.zstar, res.lambda)` (PLAN.md I8), solves
`M [w_z; w_l] = [gbar; 0]`, and returns `w_z`. Errors unless the projection
succeeded, since no gradient is claimed otherwise (I10).
"""
function vjp(c::NonlinearConstraint, res::ProjectionResult, gbar::AbstractVector)
    res.status === :success || error("no gradient is claimed for status :$(res.status)")
    n = length(res.zstar)
    m = length(res.lambda)
    _, M = _kkt(c, res.zstar, res.lambda, n, m)
    w = M \ vcat(gbar, zeros(eltype(gbar), m))
    return w[1:n]
end
