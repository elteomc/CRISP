"""
    SparseAffineConstraint(A, b)

The affine constraint set `{z : A z = b}` solved through the sparse saddle-point
system with `I` and `A'` in the top row and `A` and `0` in the bottom row. This
is the sparse equality M6 backend.
"""
struct SparseAffineConstraint{M<:SparseMatrixCSC, V<:AbstractVector}
    A::M
    b::V
end

SparseAffineConstraint(A::AbstractMatrix, b::AbstractVector) =
    SparseAffineConstraint(sparse(A), b)

"""
    SparseAffineProjectionCache(A, b)

Cached sparse KKT factorization for repeated Euclidean projections onto
`{z : A z = b}`. The cache is useful when many right-hand sides share the same
constraint matrix, for example inside Dykstra iterations.
"""
struct SparseAffineProjectionCache{M<:SparseMatrixCSC, V<:AbstractVector, K, F, T<:Real}
    A::M
    b::V
    K::K
    factor::F
    smin::T
    cond_est::T
    status::Symbol
end

function _sparse_affine_cache_inputs(Araw, braw)
    T = promote_type(float(eltype(Araw)), float(eltype(braw)), Float64)
    A = sparse(T.(Araw))
    b = T.(braw)
    size(A, 1) == length(b) || throw(DimensionMismatch("A row count must match length(b)"))
    all(isfinite, nonzeros(A)) || throw(ArgumentError("A must be finite"))
    all(isfinite, b) || throw(ArgumentError("b must be finite"))
    return A, b
end

function _factor_sparse_kkt(K)
    F = lu(K, check = false)
    iszero(F.status) || return nothing
    return F
end

function SparseAffineProjectionCache(Araw::AbstractMatrix, braw::AbstractVector;
                                     tol_rank = 1e-8, cond_max = 1e12,
                                     diagnostic_limit = 256)
    A, b = _sparse_affine_cache_inputs(Araw, braw)
    T = eltype(A)
    m, n = size(A)
    smin, rank_ok = _sparse_full_rank_diagnostic(A, m, n, tol_rank, diagnostic_limit)
    if !rank_ok
        return SparseAffineProjectionCache(A, b, nothing, nothing, T(smin),
                                           T(Inf), :singular_constraint)
    end
    if m == 0
        return SparseAffineProjectionCache(A, b, nothing, nothing, T(Inf),
                                           one(T), :success)
    end
    K = _sparse_kkt_matrix(A, nothing)
    F = _factor_sparse_kkt(K)
    if F === nothing
        return SparseAffineProjectionCache(A, b, K, nothing, T(smin),
                                           T(Inf), :singular_constraint)
    end
    cond_est = _sparse_kkt_cond_estimate(K, diagnostic_limit)
    return SparseAffineProjectionCache(A, b, K, F, T(smin), T(cond_est), :success)
end

SparseAffineProjectionCache(c::SparseAffineConstraint; kwargs...) =
    SparseAffineProjectionCache(c.A, c.b; kwargs...)

"""
    SparseDiagonalWeightedAffineConstraint(A, b, weights)

The affine constraint set `{z : A z = b}` under the diagonal weighted objective
`0.5 * sum(weights .* abs2.(z - zhat))`, solved through the sparse weighted KKT
system.
"""
struct SparseDiagonalWeightedAffineConstraint{M<:SparseMatrixCSC, V<:AbstractVector, W<:AbstractVector}
    A::M
    b::V
    weights::W
end

SparseDiagonalWeightedAffineConstraint(A::AbstractMatrix, b::AbstractVector,
                                       weights::AbstractVector) =
    SparseDiagonalWeightedAffineConstraint(sparse(A), b, weights)

function _sparse_affine_inputs(Araw, braw, zhat::AbstractVector)
    zh = float.(zhat)
    T = eltype(zh)
    A = sparse(T.(Araw))
    b = T.(braw)
    n = length(zh)
    m = length(b)
    size(A, 1) == m || throw(DimensionMismatch("A row count must match length(b)"))
    size(A, 2) == n || throw(DimensionMismatch("A column count must match length(zhat)"))
    all(isfinite, nonzeros(A)) || throw(ArgumentError("A must be finite"))
    all(isfinite, b) || throw(ArgumentError("b must be finite"))
    return A, b, zh, m, n
end

function _sparse_weighted_affine_inputs(c::SparseDiagonalWeightedAffineConstraint,
                                        zhat::AbstractVector)
    A, b, zh, m, n = _sparse_affine_inputs(c.A, c.b, zhat)
    weights = eltype(zh).(c.weights)
    length(weights) == n || throw(DimensionMismatch("weights length must match zhat length"))
    all(isfinite, weights) || throw(ArgumentError("weights must be finite"))
    all(w -> w > zero(eltype(weights)), weights) ||
        throw(ArgumentError("weights must be strictly positive"))
    return A, b, weights, zh, m, n
end

function _sparse_full_rank_diagnostic(A, m, n, tol_rank, diagnostic_limit)
    T = eltype(A)
    if m == 0
        return T(Inf), true
    elseif m > n
        return zero(T), false
    elseif max(m, n) > diagnostic_limit
        return T(NaN), true
    end

    svals = svdvals(Matrix(A))
    smin = length(svals) < m ? zero(T) : minimum(svals)
    return T(smin), smin >= tol_rank
end

function _sparse_kkt_matrix(A, weights)
    T = eltype(A)
    m, n = size(A)
    W = weights === nothing ? spdiagm(0 => ones(T, n)) : spdiagm(0 => weights)
    At = sparse(transpose(A))
    return [W At
            A spzeros(T, m, m)]
end

function _sparse_kkt_solve(K, rhs)
    F = _factor_sparse_kkt(K)
    F === nothing && return nothing
    return _sparse_factor_solve(F, rhs)
end

function _sparse_factor_solve(F, rhs)
    sol = F \ rhs
    all(isfinite, sol) || return nothing
    return sol
end

function _sparse_kkt_cond_estimate(K, diagnostic_limit)
    T = eltype(K)
    size(K, 1) > diagnostic_limit && return T(NaN)
    return T(cond(Matrix(K)))
end

function _sparse_affine_fail_result(A, b, zh, status, smin, cond_est, iterations)
    T = eltype(zh)
    residual = norm(A * zh - b)
    return ProjectionResult(zh, zeros(T, length(b)), T(residual), zero(T),
                            zero(T), iterations, T(smin), T(cond_est), status)
end

function _sparse_affine_result(A, b, weights, zh, sol, smin, cond_est,
                               cond_max, tol_res)
    T = eltype(zh)
    n = length(zh)
    zstar = sol[1:n]
    lambda = sol[(n + 1):end]
    stationarity = weights === nothing ?
        zstar .- zh .+ sparse(transpose(A)) * lambda :
        weights .* (zstar .- zh) .+ sparse(transpose(A)) * lambda
    cres = norm(A * zstar - b)
    sres = norm(stationarity)
    status = :success
    if isfinite(cond_est) && cond_est > cond_max
        status = :ill_conditioned
    elseif cres > tol_res || sres > tol_res
        status = :ill_conditioned
    end
    return ProjectionResult(zstar, lambda, T(cres), T(sres), T(norm(zstar - zh)),
                            1, T(smin), T(cond_est), status)
end

"""
    project(c::SparseAffineConstraint, zhat)

Sparse KKT projection onto `{z : A z = b}`. Small systems get dense SVD and
condition diagnostics. Larger systems rely on sparse LU success for regularity
and store `NaN` diagnostics instead of densifying.
"""
function project(c::SparseAffineConstraint, zhat::AbstractVector;
                 tol_rank = 1e-8, cond_max = 1e12, tol_res = 1e-8,
                 diagnostic_limit = 256)
    A, b, zh, m, n = _sparse_affine_inputs(c.A, c.b, zhat)
    T = eltype(zh)
    if m == 0
        return ProjectionResult(zh, zeros(T, 0), zero(T), zero(T), zero(T),
                                1, T(Inf), one(T), :success)
    end

    smin, rank_ok = _sparse_full_rank_diagnostic(A, m, n, tol_rank, diagnostic_limit)
    if !rank_ok
        return _sparse_affine_fail_result(A, b, zh, :singular_constraint,
                                          smin, T(Inf), 0)
    end

    cache = SparseAffineProjectionCache(A, b, tol_rank = tol_rank,
                                        cond_max = cond_max,
                                        diagnostic_limit = diagnostic_limit)
    cache.status === :singular_constraint &&
        return _sparse_affine_fail_result(A, b, zh, cache.status,
                                          cache.smin, cache.cond_est, 0)
    sol = _sparse_factor_solve(cache.factor, vcat(zh, b))
    if sol === nothing
        return _sparse_affine_fail_result(A, b, zh, :singular_constraint,
                                          smin, T(Inf), 0)
    end

    return _sparse_affine_result(A, b, nothing, zh, sol, smin, cache.cond_est,
                                 cond_max, tol_res)
end

function project(cache::SparseAffineProjectionCache, zhat::AbstractVector;
                 cond_max = 1e12, tol_res = 1e-8)
    zh = float.(zhat)
    T = eltype(zh)
    A = sparse(T.(cache.A))
    b = T.(cache.b)
    m, n = size(A)
    length(zh) == n || throw(DimensionMismatch("zhat length must match cache width"))
    if m == 0
        return ProjectionResult(zh, zeros(T, 0), zero(T), zero(T), zero(T),
                                1, T(Inf), one(T), :success)
    end
    cache.status === :singular_constraint &&
        return _sparse_affine_fail_result(A, b, zh, cache.status,
                                          T(cache.smin), T(cache.cond_est), 0)
    sol = _sparse_factor_solve(cache.factor, vcat(zh, b))
    sol === nothing &&
        return _sparse_affine_fail_result(A, b, zh, :singular_constraint,
                                          T(cache.smin), T(Inf), 0)
    return _sparse_affine_result(A, b, nothing, zh, sol, T(cache.smin),
                                 T(cache.cond_est), cond_max, tol_res)
end

"""
    vjp(c::SparseAffineConstraint, res, gbar) -> Vector

Sparse KKT VJP for the equality projection. It solves the same sparse system
with `gbar` in the state block and returns the state block of the solution.
"""
function vjp(c::SparseAffineConstraint, res::ProjectionResult, gbar::AbstractVector)
    res.status === :success || error("no gradient is claimed for status :$(res.status)")
    A, _, gb, m, n = _sparse_affine_inputs(c.A, c.b, gbar)
    length(res.zstar) == n || throw(DimensionMismatch("gbar length must match zstar length"))
    if m == 0
        return copy(gb)
    end
    K = _sparse_kkt_matrix(A, nothing)
    sol = _sparse_kkt_solve(K, vcat(gb, zeros(eltype(gb), m)))
    sol === nothing && error("no gradient is claimed for a singular sparse KKT system")
    return sol[1:n]
end

"""
    project(c::SparseDiagonalWeightedAffineConstraint, zhat)

Sparse KKT projection onto `{z : A z = b}` under a diagonal weighted norm.
"""
function project(c::SparseDiagonalWeightedAffineConstraint, zhat::AbstractVector;
                 tol_rank = 1e-8, cond_max = 1e12, tol_res = 1e-8,
                 diagnostic_limit = 256)
    A, b, weights, zh, m, n = _sparse_weighted_affine_inputs(c, zhat)
    T = eltype(zh)
    if m == 0
        return ProjectionResult(zh, zeros(T, 0), zero(T), zero(T), zero(T),
                                1, T(Inf), one(T), :success)
    end

    smin, rank_ok = _sparse_full_rank_diagnostic(A, m, n, tol_rank, diagnostic_limit)
    if !rank_ok
        return _sparse_affine_fail_result(A, b, zh, :singular_constraint,
                                          smin, T(Inf), 0)
    end

    K = _sparse_kkt_matrix(A, weights)
    sol = _sparse_kkt_solve(K, vcat(weights .* zh, b))
    if sol === nothing
        return _sparse_affine_fail_result(A, b, zh, :singular_constraint,
                                          smin, T(Inf), 0)
    end

    cond_est = _sparse_kkt_cond_estimate(K, diagnostic_limit)
    return _sparse_affine_result(A, b, weights, zh, sol, smin, cond_est,
                                 cond_max, tol_res)
end

"""
    vjp(c::SparseDiagonalWeightedAffineConstraint, res, gbar) -> Vector

Sparse weighted affine VJP. It solves the same sparse weighted system with
`gbar` in the state block and returns the weighted state block.
"""
function vjp(c::SparseDiagonalWeightedAffineConstraint, res::ProjectionResult,
             gbar::AbstractVector)
    res.status === :success || error("no gradient is claimed for status :$(res.status)")
    A, _, weights, gb, m, n = _sparse_weighted_affine_inputs(c, gbar)
    length(res.zstar) == n || throw(DimensionMismatch("gbar length must match zstar length"))
    if m == 0
        return copy(gb)
    end
    K = _sparse_kkt_matrix(A, weights)
    sol = _sparse_kkt_solve(K, vcat(gb, zeros(eltype(gb), m)))
    sol === nothing && error("no gradient is claimed for a singular sparse KKT system")
    return weights .* sol[1:n]
end
