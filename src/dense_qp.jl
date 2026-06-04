"""
    DenseLinearQPConstraint(Aeq, beq, G, h)

Projection onto the small dense polyhedron
`{z : Aeq * z = beq, G * z <= h}` with Euclidean objective
`0.5 * norm(z - zhat)^2`.

This M6 layer enumerates active inequality sets, solves the affine KKT system
for each candidate, and accepts a regular KKT point with primal feasibility and
nonnegative active multipliers. It is a correctness baseline for small dense
linear inequality systems.
"""
struct DenseLinearQPConstraint{A<:AbstractMatrix, B<:AbstractVector, G<:AbstractMatrix, H<:AbstractVector}
    Aeq::A
    beq::B
    G::G
    h::H
end

function DenseLinearQPConstraint(G::AbstractMatrix, h::AbstractVector)
    T = promote_type(eltype(G), eltype(h), Float64)
    return DenseLinearQPConstraint(zeros(T, 0, size(G, 2)), zeros(T, 0), G, h)
end

function _dense_qp_inputs(c::DenseLinearQPConstraint, zhat::AbstractVector)
    zh = float.(zhat)
    T = eltype(zh)
    Aeq = Matrix(T.(c.Aeq))
    beq = T.(c.beq)
    G = Matrix(T.(c.G))
    h = T.(c.h)
    n = length(zh)
    size(Aeq, 1) == length(beq) || throw(DimensionMismatch("Aeq row count must match length(beq)"))
    size(Aeq, 2) == n || throw(DimensionMismatch("Aeq column count must match length(zhat)"))
    size(G, 1) == length(h) || throw(DimensionMismatch("G row count must match length(h)"))
    size(G, 2) == n || throw(DimensionMismatch("G column count must match length(zhat)"))
    all(isfinite, Aeq) || throw(ArgumentError("Aeq must be finite"))
    all(isfinite, beq) || throw(ArgumentError("beq must be finite"))
    all(isfinite, G) || throw(ArgumentError("G must be finite"))
    all(isfinite, h) || throw(ArgumentError("h must be finite"))
    return Aeq, beq, G, h, zh
end

function _dense_qp_subset(mask::Int, q::Int)
    active = Int[]
    for j in 1:q
        if (mask & (1 << (j - 1))) != 0
            push!(active, j)
        end
    end
    return active
end

function _dense_qp_matrix(Aeq, beq, G, h, active)
    C = vcat(Aeq, G[active, :])
    d = vcat(beq, h[active])
    return C, d
end

function _dense_qp_rank_ok(M, rows, tol_rank)
    rows == 0 && return true
    svals = svdvals(M)
    return length(svals) >= rows && minimum(svals) >= tol_rank
end

function _dense_qp_fail_result(zh, Aeq, beq, G, h, status, iterations)
    T = eltype(zh)
    eq = Aeq * zh - beq
    ineq = max.(G * zh - h, zero(T))
    ProjectionResult(zh, zeros(T, length(beq) + length(h)), T(norm(vcat(eq, ineq))),
                     zero(T), zero(T), iterations, zero(T), T(Inf), status)
end

function _dense_qp_result(zh, Aeq, beq, G, h, zstar, lambda_eq, mu,
                          active, status, iterations, jac_min, cond_est)
    T = eltype(zh)
    eq = Aeq * zstar - beq
    ineq = max.(G * zstar - h, zero(T))
    stationarity = zstar .- zh .+ transpose(Aeq) * lambda_eq .+ transpose(G) * mu
    ProjectionResult(zstar, vcat(lambda_eq, mu), T(norm(vcat(eq, ineq))),
                     T(norm(stationarity)), T(norm(zstar - zh)), iterations,
                     T(jac_min), T(cond_est), status)
end

"""
    project(c::DenseLinearQPConstraint, zhat) -> ProjectionResult

Euclidean projection onto a small dense linear equality and inequality system.
The status is `:success` only when the selected active set is full rank, strict,
and well conditioned.
"""
function project(c::DenseLinearQPConstraint, zhat::AbstractVector;
                 tol_rank = 1e-8, tol_feas = 1e-8, tol_dual = 1e-8,
                 tol_active = 1e-8, cond_max = 1e12, max_ineq = 20)
    Aeq, beq, G, h, zh = _dense_qp_inputs(c, zhat)
    T = eltype(zh)
    m = length(beq)
    q = length(h)
    n = length(zh)

    if q > max_ineq
        throw(ArgumentError("too many inequalities for exhaustive small dense QP projection"))
    end
    if !_dense_qp_rank_ok(Aeq, m, tol_rank)
        return _dense_qp_fail_result(zh, Aeq, beq, G, h, :singular_constraint, 0)
    end

    best = nothing
    best_obj = T(Inf)
    iterations = 0

    for mask in 0:((1 << q) - 1)
        iterations += 1
        active = _dense_qp_subset(mask, q)
        C, d = _dense_qp_matrix(Aeq, beq, G, h, active)
        k = size(C, 1)
        if !_dense_qp_rank_ok(C, k, tol_rank)
            continue
        end

        res = project(AffineConstraint(C, d), zh, tol_rank = tol_rank)
        res.status === :success || continue
        z = res.zstar
        slack = h .- G * z
        active_mu = res.lambda[(m + 1):end]
        primal_ok = all(slack .>= -tol_feas)
        dual_ok = all(active_mu .>= -tol_dual)
        if primal_ok && dual_ok
            obj = T(0.5) * sum(abs2, z - zh)
            if obj < best_obj - T(1e-12)
                best_obj = obj
                best = (active = active, res = res, slack = slack)
            end
        end
    end

    if best === nothing
        return _dense_qp_fail_result(zh, Aeq, beq, G, h, :infeasible_constraint, iterations)
    end

    active = best.active
    res = best.res
    slack = best.slack
    mu = zeros(T, q)
    mu[active] = res.lambda[(m + 1):end]
    inactive = trues(q)
    inactive[active] .= false
    weak_active = any(mu[active] .<= tol_dual)
    tight_inactive = any(slack[inactive] .<= tol_active)
    C, _ = _dense_qp_matrix(Aeq, beq, G, h, active)
    k = size(C, 1)
    jac_min = k == 0 ? T(Inf) : minimum(svdvals(C))
    M = [Matrix(I, n, n) transpose(C)
         C zeros(T, k, k)]
    cond_est = k == 0 ? one(T) : cond(M)
    status = :success
    if cond_est > cond_max
        status = :ill_conditioned
    elseif weak_active || tight_inactive
        status = :nonunique_input
    end
    lambda_eq = res.lambda[1:m]
    return _dense_qp_result(zh, Aeq, beq, G, h, res.zstar, lambda_eq, mu,
                            active, status, iterations, jac_min, cond_est)
end

"""
    vjp(c::DenseLinearQPConstraint, res, gbar) -> Vector

Active-set VJP for a successful dense linear QP projection. The upstream
gradient is projected onto the null space of the equality rows and active
inequality rows.
"""
function vjp(c::DenseLinearQPConstraint, res::ProjectionResult, gbar::AbstractVector;
             tol_active = 1e-8, tol_rank = 1e-8)
    res.status === :success || error("no gradient is claimed for status :$(res.status)")
    Aeq, beq, G, h, gb = _dense_qp_inputs(c, gbar)
    length(res.zstar) == length(gb) || throw(DimensionMismatch("gbar length must match zstar length"))
    m = length(beq)
    mu = res.lambda[(m + 1):end]
    active = findall(mu .> tol_active)
    C, _ = _dense_qp_matrix(Aeq, beq, G, h, active)
    k = size(C, 1)
    if k == 0
        return copy(gb)
    end
    _dense_qp_rank_ok(C, k, tol_rank) || error("no gradient is claimed for a rank-deficient active set")
    return vjp(AffineConstraint(C, zeros(eltype(gb), k)), gb, tol_rank = tol_rank)
end
