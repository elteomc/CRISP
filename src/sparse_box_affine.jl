"""
    SparseBoxAffineConstraint(A, b, lower, upper)

Projection onto `{z : A z = b, lower <= z <= upper}` with sparse equality rows
and componentwise bounds. The forward solve uses Dykstra iterations between the
sparse affine projection and the box projection. The backward pass uses the
active bound set and a sparse equality KKT solve on the free coordinates.
"""
struct SparseBoxAffineConstraint{M<:SparseMatrixCSC, V<:AbstractVector, L<:AbstractVector, U<:AbstractVector}
    A::M
    b::V
    lower::L
    upper::U
end

SparseBoxAffineConstraint(A::AbstractMatrix, b::AbstractVector,
                          lower::AbstractVector, upper::AbstractVector) =
    SparseBoxAffineConstraint(sparse(A), b, lower, upper)

"""
    CachedSparseBoxAffineConstraint(c)

Sparse box-affine constraint with a cached sparse affine KKT factorization for
the Dykstra affine step. The constraint set is identical to `c`, but repeated
projection calls avoid rebuilding and refactoring the same equality KKT matrix.
"""
struct CachedSparseBoxAffineConstraint{C<:SparseBoxAffineConstraint, AC<:SparseAffineProjectionCache}
    constraint::C
    affine_cache::AC
end

CachedSparseBoxAffineConstraint(c::SparseBoxAffineConstraint; kwargs...) =
    CachedSparseBoxAffineConstraint(c, SparseAffineProjectionCache(c.A, c.b; kwargs...))

"""
    SparseBoxAffineWorkspace(n)

Mutable workspace for repeated sparse box-affine projections. It stores the
last regular active set so the next call can try a validated active-face solve
before falling back to Dykstra.
"""
mutable struct SparseBoxAffineWorkspace
    lower_active::BitVector
    upper_active::BitVector
    active_cache::Any
    initialized::Bool
end

SparseBoxAffineWorkspace(n::Integer) =
    SparseBoxAffineWorkspace(falses(n), falses(n), nothing, false)

SparseBoxAffineWorkspace(c::SparseBoxAffineConstraint) =
    SparseBoxAffineWorkspace(length(c.lower))

"""
    WarmStartedSparseBoxAffineConstraint(c)

Cached sparse box-affine constraint with a mutable active-set workspace. Each
projection first tries the previous regular active face and falls back to
Dykstra if that face is no longer valid.
"""
mutable struct WarmStartedSparseBoxAffineConstraint{CC<:CachedSparseBoxAffineConstraint, WS<:SparseBoxAffineWorkspace}
    cached::CC
    workspace::WS
end

WarmStartedSparseBoxAffineConstraint(c::SparseBoxAffineConstraint; kwargs...) =
    WarmStartedSparseBoxAffineConstraint(CachedSparseBoxAffineConstraint(c; kwargs...),
                                         SparseBoxAffineWorkspace(c))

function reset_workspace!(ws::SparseBoxAffineWorkspace)
    fill!(ws.lower_active, false)
    fill!(ws.upper_active, false)
    ws.active_cache = nothing
    ws.initialized = false
    return ws
end

function _sparse_box_affine_inputs(c::SparseBoxAffineConstraint, zhat::AbstractVector)
    A, b, zh, m, n = _sparse_affine_inputs(c.A, c.b, zhat)
    T = eltype(zh)
    lo = T.(c.lower)
    hi = T.(c.upper)
    length(lo) == n || throw(DimensionMismatch("lower length must match zhat length"))
    length(hi) == n || throw(DimensionMismatch("upper length must match zhat length"))
    all(isfinite, lo) || throw(ArgumentError("lower bounds must be finite"))
    all(isfinite, hi) || throw(ArgumentError("upper bounds must be finite"))
    return A, b, lo, hi, zh, m, n
end

function _sparse_box_affine_interval_infeasible(A, b, lo, hi, tol_feas)
    T = eltype(b)
    amin = zeros(T, size(A, 1))
    amax = zeros(T, size(A, 1))
    for col in 1:size(A, 2)
        for j in nzrange(A, col)
            row = A.rowval[j]
            aij = A.nzval[j]
            if aij >= zero(T)
                amin[row] += aij * lo[col]
                amax[row] += aij * hi[col]
            else
                amin[row] += aij * hi[col]
                amax[row] += aij * lo[col]
            end
        end
    end
    return any(b .< amin .- tol_feas) || any(b .> amax .+ tol_feas)
end

function _sparse_box_affine_bounds_residual(z, lo, hi)
    T = eltype(z)
    return norm(vcat(max.(lo .- z, zero(T)), max.(z .- hi, zero(T))))
end

function _sparse_bound_rows(indices, signs, n, T)
    k = length(indices)
    rows = collect(1:k)
    return sparse(rows, indices, fill(T(signs), k), k, n)
end

function _sparse_box_affine_active_matrix(A, lower_active, upper_active)
    T = eltype(A)
    n = size(A, 2)
    lower_idx = findall(lower_active)
    upper_idx = findall(upper_active)
    L = _sparse_bound_rows(lower_idx, -1, n, T)
    U = _sparse_bound_rows(upper_idx, 1, n, T)
    return [A
            L
            U], lower_idx, upper_idx
end

function _sparse_box_affine_active_rhs(b, lo, hi, lower_active, upper_active)
    lower_idx = findall(lower_active)
    upper_idx = findall(upper_active)
    return vcat(b, .-lo[lower_idx], hi[upper_idx])
end

function _sparse_box_affine_update_workspace!(ws::SparseBoxAffineWorkspace,
                                              zstar, lo, hi, tol_active)
    length(ws.lower_active) == length(zstar) ||
        throw(DimensionMismatch("workspace length must match projection length"))
    lower = zstar .<= lo .+ tol_active
    upper = zstar .>= hi .- tol_active
    if !ws.initialized || ws.lower_active != lower || ws.upper_active != upper
        ws.active_cache = nothing
    end
    ws.lower_active .= lower
    ws.upper_active .= upper
    ws.initialized = true
    return ws
end

function _sparse_box_affine_try_warm(A, b, lo, hi, zh, ws, tol_active,
                                     tol_dual, tol_feas, tol_rank,
                                     cond_max, diagnostic_limit)
    ws === nothing && return nothing
    ws.initialized || return nothing
    length(ws.lower_active) == length(zh) ||
        throw(DimensionMismatch("workspace length must match zhat length"))
    C, _, _ = _sparse_box_affine_active_matrix(A, ws.lower_active,
                                               ws.upper_active)
    d = _sparse_box_affine_active_rhs(b, lo, hi, ws.lower_active,
                                      ws.upper_active)
    if ws.active_cache === nothing
        ws.active_cache = SparseAffineProjectionCache(C, d, tol_rank = tol_rank,
                                                      cond_max = cond_max,
                                                      diagnostic_limit = diagnostic_limit)
    end
    cache = ws.active_cache
    res = project(cache, zh, cond_max = cond_max, tol_res = tol_feas)
    res.status === :success || return nothing
    warm = _sparse_box_affine_result(A, b, lo, hi, zh, res.zstar, 0, true,
                                     tol_active, tol_dual, tol_feas)
    warm.status === :success || return nothing
    return warm
end

function _sparse_box_affine_multipliers(A, zh, zstar, lower_active, upper_active)
    T = eltype(zh)
    C, lower_idx, upper_idx = _sparse_box_affine_active_matrix(A, lower_active, upper_active)
    k = size(C, 1)
    if k == 0
        return zeros(T, 0), zeros(T, length(zh)), zeros(T, length(zh)), norm(zstar - zh), true
    end
    gram = C * sparse(transpose(C))
    rhs = C * (zh - zstar)
    eta = _sparse_kkt_solve(gram, rhs)
    eta === nothing && return zeros(T, size(A, 1)), zeros(T, length(zh)),
                             zeros(T, length(zh)), T(Inf), false

    m = size(A, 1)
    lambda_eq = eta[1:m]
    lower_mu = zeros(T, length(zh))
    upper_mu = zeros(T, length(zh))
    lower_mu[lower_idx] = eta[(m + 1):(m + length(lower_idx))]
    upper_start = m + length(lower_idx) + 1
    upper_mu[upper_idx] = eta[upper_start:end]
    stationarity = zstar .- zh .+ sparse(transpose(A)) * lambda_eq .- lower_mu .+ upper_mu
    return lambda_eq, lower_mu, upper_mu, norm(stationarity), true
end

function _sparse_box_affine_result(A, b, lo, hi, zh, zstar, iters, converged,
                                   tol_active, tol_dual, tol_feas)
    T = eltype(zh)
    lower_active = zstar .<= lo .+ tol_active
    upper_active = zstar .>= hi .- tol_active
    lambda_eq, lower_mu, upper_mu, sres, regular =
        _sparse_box_affine_multipliers(A, zh, zstar, lower_active, upper_active)
    cres = norm(vcat(A * zstar - b,
                     max.(lo .- zstar, zero(T)),
                     max.(zstar .- hi, zero(T))))
    weak_lower = any(lower_mu[lower_active] .<= tol_dual)
    weak_upper = any(upper_mu[upper_active] .<= tol_dual)
    status =
        !converged ? :max_iters :
        !regular ? :singular_constraint :
        (cres > tol_feas || sres > tol_feas) ? :ill_conditioned :
        (weak_lower || weak_upper) ? :nonunique_input :
        :success
    free_count = count(.!lower_active .& .!upper_active)
    return ProjectionResult(zstar, vcat(lambda_eq, lower_mu, upper_mu), T(cres),
                            T(sres), T(norm(zstar - zh)), iters, T(free_count),
                            T(NaN), status)
end

"""
    project(c::SparseBoxAffineConstraint, zhat)

Iterative projection onto a sparse affine and box intersection. Dykstra
iterations avoid enumerating inequality active sets. On `:success`, the active
set is regular and the KKT stationarity residual is checked.
"""
function project(c::SparseBoxAffineConstraint, zhat::AbstractVector;
                 maxiter = 5000, tol_feas = 1e-9, tol_step = 1e-10,
                 tol_active = 1e-8, tol_dual = 1e-8,
                 tol_rank = 1e-8, cond_max = 1e12,
                 diagnostic_limit = 256, affine_cache = nothing,
                 workspace = nothing, warm_start = workspace !== nothing)
    A, b, lo, hi, zh, m, n = _sparse_box_affine_inputs(c, zhat)
    T = eltype(zh)
    if any(lo .> hi)
        return ProjectionResult(zh, zeros(T, m + 2n), norm(max.(lo .- hi, zero(T))),
                                zero(T), zero(T), 0, zero(T), T(Inf),
                                :infeasible_constraint)
    end
    if _sparse_box_affine_interval_infeasible(A, b, lo, hi, tol_feas)
        return ProjectionResult(zh, zeros(T, m + 2n), norm(A * zh - b),
                                zero(T), zero(T), 0, zero(T), T(Inf),
                                :infeasible_constraint)
    end
    if m == 0
        box_res = project(BoxConstraint(lo, hi), zh, tol_active = tol_active)
        return ProjectionResult(box_res.zstar, vcat(zeros(T, 0), box_res.lambda),
                                box_res.constraint_residual,
                                box_res.stationarity_residual,
                                box_res.correction_norm, box_res.iterations,
                                box_res.jac_min_singular,
                                box_res.kkt_cond_estimate, box_res.status)
    end

    if warm_start
        warm = _sparse_box_affine_try_warm(A, b, lo, hi, zh, workspace,
                                           tol_active, tol_dual, tol_feas,
                                           tol_rank, cond_max,
                                           diagnostic_limit)
        if warm !== nothing
            _sparse_box_affine_update_workspace!(workspace, warm.zstar, lo,
                                                 hi, tol_active)
            return warm
        end
    end

    cache = affine_cache === nothing ?
        SparseAffineProjectionCache(A, b, tol_rank = tol_rank,
                                    cond_max = cond_max,
                                    diagnostic_limit = diagnostic_limit) :
        affine_cache
    x = copy(zh)
    p = zeros(T, n)
    q = zeros(T, n)
    converged = false
    iters = 0
    for it in 1:maxiter
        iters = it
        y_input = x .+ p
        affine_res = project(cache, y_input, cond_max = cond_max,
                             tol_res = tol_feas)
        affine_res.status === :success || return ProjectionResult(
            zh, zeros(T, m + 2n), norm(A * zh - b), zero(T), zero(T),
            it, zero(T), T(Inf), affine_res.status)
        y = affine_res.zstar
        p = y_input .- y
        box_input = y .+ q
        xnew = min.(max.(box_input, lo), hi)
        q = box_input .- xnew
        eqres = norm(A * xnew - b)
        bres = _sparse_box_affine_bounds_residual(xnew, lo, hi)
        step = norm(xnew - x) / max(one(T), norm(xnew))
        x = xnew
        if eqres <= tol_feas && bres <= tol_feas && step <= tol_step
            converged = true
            break
        end
    end

    res = _sparse_box_affine_result(A, b, lo, hi, zh, x, iters, converged,
                                    tol_active, tol_dual, tol_feas)
    if workspace !== nothing && res.status === :success
        _sparse_box_affine_update_workspace!(workspace, res.zstar, lo, hi,
                                             tol_active)
    end
    return res
end

function project(c::CachedSparseBoxAffineConstraint, zhat::AbstractVector; kwargs...)
    return project(c.constraint, zhat; affine_cache = c.affine_cache, kwargs...)
end

function project(c::WarmStartedSparseBoxAffineConstraint, zhat::AbstractVector; kwargs...)
    return project(c.cached.constraint, zhat;
                   affine_cache = c.cached.affine_cache,
                   workspace = c.workspace, kwargs...)
end

"""
    vjp(c::SparseBoxAffineConstraint, res, gbar) -> Vector

Active-set VJP for the sparse affine plus box projection. Bound coordinates are
fixed. Free coordinates receive the sparse affine tangent projection for
`A[:, free]`.
"""
function vjp(c::SparseBoxAffineConstraint, res::ProjectionResult, gbar::AbstractVector;
             tol_active = 1e-8, tol_rank = 1e-8, diagnostic_limit = 256)
    res.status === :success || error("no gradient is claimed for status :$(res.status)")
    A, _, lo, hi, gb, m, n = _sparse_box_affine_inputs(c, gbar)
    length(res.zstar) == n || throw(DimensionMismatch("gbar length must match zstar length"))
    free = (res.zstar .> lo .+ tol_active) .& (res.zstar .< hi .- tol_active)
    out = zeros(eltype(gb), n)
    if !any(free)
        return out
    end
    gfree = gb[free]
    if m == 0
        out[free] = gfree
        return out
    end
    Afree = A[:, free]
    tangent = project(SparseAffineConstraint(Afree, zeros(eltype(gb), m)), gfree,
                      tol_rank = tol_rank, diagnostic_limit = diagnostic_limit)
    tangent.status === :success ||
        error("no gradient is claimed for a rank-deficient sparse active set")
    out[free] = tangent.zstar
    return out
end

function vjp(c::CachedSparseBoxAffineConstraint, res::ProjectionResult,
             gbar::AbstractVector; kwargs...)
    return vjp(c.constraint, res, gbar; kwargs...)
end

function vjp(c::WarmStartedSparseBoxAffineConstraint, res::ProjectionResult,
             gbar::AbstractVector; kwargs...)
    return vjp(c.cached.constraint, res, gbar; kwargs...)
end
