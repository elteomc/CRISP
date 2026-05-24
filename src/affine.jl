"""
    AffineConstraint(A, b)

The affine constraint set `{z : A z = b}`, with `A` an `m x n` matrix of full
row rank in the regular case. This is milestone M1 in PLAN.md.
"""
struct AffineConstraint{M<:AbstractMatrix, V<:AbstractVector}
    A::M
    b::V
end

"""
    project(c::AffineConstraint, zhat; tol_rank=1e-6) -> ProjectionResult

Euclidean projection of `zhat` onto `{z : A z = b}`.

The implementation uses the thin QR of `A'` and never forms an explicit inverse
(PLAN.md I6). With `A' = Q R` (Q has orthonormal columns spanning the row space
of `A`), the projector onto `null(A)` is `P = I - Q Q'`, and

    zstar = (I - Q Q') zhat + Q (R' \\ b),     lambda = R \\ (Q' zhat - R' \\ b).

A rank-deficient or ill-conditioned `A` is detected through `sigma_min(A)` and
returns `:singular_constraint` rather than a silent wrong answer (I4).
"""
function project(c::AffineConstraint, zhat::AbstractVector; tol_rank = 1e-6)
    A = c.A
    b = c.b
    m = length(b)
    zh = float.(zhat)
    T = eltype(zh)

    smin = minimum(svdvals(A))
    if smin < tol_rank
        return ProjectionResult(zh, zeros(T, m), norm(A * zh - b),
                                zero(T), zero(T), 0, T(smin), T(Inf),
                                :singular_constraint)
    end

    F = qr(Matrix(transpose(A)))         # A' is n x m
    Q = Matrix(F.Q)[:, 1:m]              # thin Q, n x m
    R = F.R                              # m x m, upper triangular

    y = Q' * zh - (transpose(R) \ b)
    zstar = zh - Q * y
    lambda = R \ y

    cres = norm(A * zstar - b)
    sres = norm(zstar - zh + transpose(A) * lambda)
    ProjectionResult(zstar, lambda, T(cres), T(sres), T(norm(zstar - zh)),
                     1, T(smin), T(cond(A * transpose(A))), :success)
end

"""
    vjp(c::AffineConstraint, gbar; tol_rank=1e-6) -> Vector

Vector-Jacobian product. The Jacobian `dzstar/dzhat` is the symmetric orthogonal
projector `P = I - Q Q'` onto `null(A)`, so the VJP is `P gbar` (PLAN.md 4.2).
"""
function vjp(c::AffineConstraint, gbar::AbstractVector; tol_rank = 1e-6)
    A = c.A
    m = size(A, 1)
    if minimum(svdvals(A)) < tol_rank
        error("no gradient is claimed for a rank-deficient affine constraint")
    end
    F = qr(Matrix(transpose(A)))
    Q = Matrix(F.Q)[:, 1:m]
    return gbar - Q * (Q' * gbar)
end
