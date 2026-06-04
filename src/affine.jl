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
    DiagonalWeightedAffineConstraint(A, b, weights)

The affine constraint set `{z : A z = b}` with weighted Euclidean objective
`0.5 * sum(weights .* abs2.(z - zhat))`. Weights must be strictly positive.
This is the first weighted-norm M6 layer.
"""
struct DiagonalWeightedAffineConstraint{M<:AbstractMatrix, V<:AbstractVector, W<:AbstractVector}
    A::M
    b::V
    weights::W
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
    size(A, 1) == m || throw(DimensionMismatch("A row count must match length(b)"))
    size(A, 2) == length(zh) || throw(DimensionMismatch("A column count must match length(zhat)"))

    if m == 0
        return ProjectionResult(zh, zeros(T, 0), zero(T), zero(T), zero(T),
                                1, T(Inf), one(T), :success)
    end

    svals = svdvals(A)
    smin = length(svals) < m ? zero(T) : minimum(svals)
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
    size(A, 2) == length(gbar) || throw(DimensionMismatch("A column count must match length(gbar)"))
    if m == 0
        return copy(gbar)
    end
    svals = svdvals(A)
    if length(svals) < m || minimum(svals) < tol_rank
        error("no gradient is claimed for a rank-deficient affine constraint")
    end
    F = qr(Matrix(transpose(A)))
    Q = Matrix(F.Q)[:, 1:m]
    return gbar - Q * (Q' * gbar)
end

"""
    vjp(c::AffineConstraint, res::ProjectionResult, gbar) -> Vector

Three-argument form matching the nonlinear layer, used by the AD rule. The
affine VJP is input-independent, so `res` only enforces that no gradient is
claimed unless the projection succeeded (I10).
"""
function vjp(c::AffineConstraint, res::ProjectionResult, gbar::AbstractVector)
    res.status === :success || error("no gradient is claimed for status :$(res.status)")
    return vjp(c, gbar)
end

function _weighted_affine_inputs(c::DiagonalWeightedAffineConstraint, zhat::AbstractVector)
    A = c.A
    b = c.b
    weights = float.(c.weights)
    zh = float.(zhat)
    m = length(b)
    size(A, 1) == m || throw(DimensionMismatch("A row count must match length(b)"))
    size(A, 2) == length(zh) || throw(DimensionMismatch("A column count must match length(zhat)"))
    length(weights) == length(zh) || throw(DimensionMismatch("weights length must match zhat length"))
    all(isfinite, weights) || throw(ArgumentError("weights must be finite"))
    all(w -> w > zero(eltype(weights)), weights) || throw(ArgumentError("weights must be strictly positive"))
    return A, b, weights, zh, m
end

"""
    project(c::DiagonalWeightedAffineConstraint, zhat; tol_rank=1e-6)

Weighted Euclidean projection onto `{z : A z = b}`. The solve uses
`K = A * Diagonal(1 ./ weights) * A'` through column scaling, and returns
`:singular_constraint` when the affine constraint cannot have full row rank.
"""
function project(c::DiagonalWeightedAffineConstraint, zhat::AbstractVector; tol_rank = 1e-6)
    A, b, weights, zh, m = _weighted_affine_inputs(c, zhat)
    T = eltype(zh)
    if m == 0
        return ProjectionResult(zh, zeros(T, 0), zero(T), zero(T), zero(T),
                                1, T(Inf), one(T), :success)
    end

    svals = svdvals(A)
    smin = length(svals) < m ? zero(T) : minimum(svals)
    if smin < tol_rank
        return ProjectionResult(zh, zeros(T, m), norm(A * zh - b),
                                zero(T), zero(T), 0, T(smin), T(Inf),
                                :singular_constraint)
    end

    winv = one(T) ./ weights
    K = (A .* transpose(winv)) * transpose(A)
    kmin = minimum(svdvals(K))
    if kmin < tol_rank
        return ProjectionResult(zh, zeros(T, m), norm(A * zh - b),
                                zero(T), zero(T), 0, T(smin), T(Inf),
                                :singular_constraint)
    end

    lambda = K \ (A * zh - b)
    zstar = zh .- winv .* (transpose(A) * lambda)
    cres = norm(A * zstar - b)
    sres = norm(weights .* (zstar - zh) .+ transpose(A) * lambda)
    ProjectionResult(zstar, lambda, T(cres), T(sres), T(norm(zstar - zh)),
                     1, T(smin), T(cond(K)), :success)
end

"""
    vjp(c::DiagonalWeightedAffineConstraint, res, gbar) -> Vector

VJP for the weighted affine projection. For
`K = A * Diagonal(1 ./ weights) * A'`, the VJP is
`gbar - A' * (K \\ (A * Diagonal(1 ./ weights) * gbar))`.
"""
function vjp(c::DiagonalWeightedAffineConstraint, res::ProjectionResult, gbar::AbstractVector)
    res.status === :success || error("no gradient is claimed for status :$(res.status)")
    A, _, weights, gb, m = _weighted_affine_inputs(c, gbar)
    if m == 0
        return copy(gb)
    end
    svals = svdvals(A)
    if length(svals) < m || minimum(svals) < 1e-6
        error("no gradient is claimed for a rank-deficient weighted affine constraint")
    end
    winv = one(eltype(gb)) ./ weights
    K = (A .* transpose(winv)) * transpose(A)
    return gb .- transpose(A) * (K \ (A * (winv .* gb)))
end
