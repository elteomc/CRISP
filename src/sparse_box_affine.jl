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
                 tol_rank = 1e-8, diagnostic_limit = 256)
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

    affine = SparseAffineConstraint(A, b)
    x = copy(zh)
    p = zeros(T, n)
    q = zeros(T, n)
    converged = false
    iters = 0
    for it in 1:maxiter
        iters = it
        y_input = x .+ p
        affine_res = project(affine, y_input, tol_rank = tol_rank,
                             tol_res = tol_feas, diagnostic_limit = diagnostic_limit)
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

    return _sparse_box_affine_result(A, b, lo, hi, zh, x, iters, converged,
                                     tol_active, tol_dual, tol_feas)
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
