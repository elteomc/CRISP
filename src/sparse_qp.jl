"""
    SparseLinearQPActiveSetConstraint(Aeq, beq, G, h)

Sparse projection onto `{z : Aeq * z = beq, G * z <= h}` with Euclidean
objective `0.5 * norm(z - zhat)^2`.

The active-set backend changes a working set of inequalities and solves sparse
equality KKT systems for each working face. On success, the VJP freezes the
strict active set and projects the upstream gradient onto the sparse tangent
space for the equality rows plus active inequality rows.
"""
struct SparseLinearQPActiveSetConstraint{A<:SparseMatrixCSC, B<:AbstractVector, G<:SparseMatrixCSC, H<:AbstractVector}
    Aeq::A
    beq::B
    G::G
    h::H
end

SparseLinearQPActiveSetConstraint(Aeq::AbstractMatrix, beq::AbstractVector,
                                  G::AbstractMatrix, h::AbstractVector) =
    SparseLinearQPActiveSetConstraint(sparse(Aeq), beq, sparse(G), h)

function SparseLinearQPActiveSetConstraint(G::AbstractMatrix, h::AbstractVector)
    T = promote_type(eltype(G), eltype(h), Float64)
    return SparseLinearQPActiveSetConstraint(spzeros(T, 0, size(G, 2)),
                                             zeros(T, 0), sparse(G), h)
end

"""
    SparseLinearQPPrimalDualConstraint(Aeq, beq, G, h)

Sparse projection onto `{z : Aeq * z = beq, G * z <= h}` with Euclidean
objective `0.5 * norm(z - zhat)^2`.

The primal-dual backend runs a sparse interior-point solve, identifies the
active face, then polishes the final point with a sparse equality KKT solve.
The returned VJP is the exact active-set projection for the polished face.
"""
struct SparseLinearQPPrimalDualConstraint{A<:SparseMatrixCSC, B<:AbstractVector, G<:SparseMatrixCSC, H<:AbstractVector}
    Aeq::A
    beq::B
    G::G
    h::H
end

SparseLinearQPPrimalDualConstraint(Aeq::AbstractMatrix, beq::AbstractVector,
                                   G::AbstractMatrix, h::AbstractVector) =
    SparseLinearQPPrimalDualConstraint(sparse(Aeq), beq, sparse(G), h)

function SparseLinearQPPrimalDualConstraint(G::AbstractMatrix, h::AbstractVector)
    T = promote_type(eltype(G), eltype(h), Float64)
    return SparseLinearQPPrimalDualConstraint(spzeros(T, 0, size(G, 2)),
                                              zeros(T, 0), sparse(G), h)
end

function _sparse_qp_inputs(c, zhat::AbstractVector)
    zh = float.(zhat)
    T = eltype(zh)
    Aeq = sparse(T.(c.Aeq))
    beq = T.(c.beq)
    G = sparse(T.(c.G))
    h = T.(c.h)
    n = length(zh)
    m = length(beq)
    q = length(h)
    size(Aeq, 1) == m || throw(DimensionMismatch("Aeq row count must match length(beq)"))
    size(Aeq, 2) == n || throw(DimensionMismatch("Aeq column count must match length(zhat)"))
    size(G, 1) == q || throw(DimensionMismatch("G row count must match length(h)"))
    size(G, 2) == n || throw(DimensionMismatch("G column count must match length(zhat)"))
    all(isfinite, nonzeros(Aeq)) || throw(ArgumentError("Aeq must be finite"))
    all(isfinite, beq) || throw(ArgumentError("beq must be finite"))
    all(isfinite, nonzeros(G)) || throw(ArgumentError("G must be finite"))
    all(isfinite, h) || throw(ArgumentError("h must be finite"))
    return Aeq, beq, G, h, zh, m, q, n
end

function _sparse_qp_active_matrix(Aeq, beq, G, h, active)
    C = [Aeq
         G[active, :]]
    d = vcat(beq, h[active])
    return C, d
end

function _sparse_qp_dense_rank(M, tol_rank)
    isempty(M) && return 0
    svals = svdvals(Matrix(M))
    return count(>=(tol_rank), svals)
end

function _sparse_qp_rank_ok(C, tol_rank, diagnostic_limit)
    rows, cols = size(C)
    rows == 0 && return true
    rows > cols && return false
    max(rows, cols) > diagnostic_limit && return true
    return _sparse_qp_dense_rank(C, tol_rank) >= rows
end

function _sparse_qp_inconsistent(C, d, tol_rank, diagnostic_limit)
    rows, cols = size(C)
    rows == 0 && return false
    max(rows, cols) > diagnostic_limit && return false
    return _sparse_qp_dense_rank([Matrix(C) d], tol_rank) >
           _sparse_qp_dense_rank(C, tol_rank)
end

function _sparse_qp_jac_min(C, tol_rank, diagnostic_limit)
    rows, cols = size(C)
    rows == 0 && return convert(eltype(C), Inf)
    rows > cols && return zero(eltype(C))
    max(rows, cols) > diagnostic_limit && return convert(eltype(C), NaN)
    svals = svdvals(Matrix(C))
    length(svals) < rows && return zero(eltype(C))
    return convert(eltype(C), minimum(svals))
end

function _sparse_qp_fail_result(Aeq, beq, G, h, zh, status, iterations)
    T = eltype(zh)
    eq = Aeq * zh - beq
    ineq = max.(G * zh - h, zero(T))
    return ProjectionResult(zh, zeros(T, length(beq) + length(h)),
                            T(norm(vcat(eq, ineq))), zero(T), zero(T),
                            iterations, zero(T), T(Inf), status)
end

function _sparse_qp_solve_active(Aeq, beq, G, h, zh, active, tol_rank,
                                 tol_res, diagnostic_limit)
    C, d = _sparse_qp_active_matrix(Aeq, beq, G, h, active)
    if _sparse_qp_inconsistent(C, d, tol_rank, diagnostic_limit)
        return :infeasible_constraint, nothing, C, d
    end
    if !_sparse_qp_rank_ok(C, tol_rank, diagnostic_limit)
        return :singular_constraint, nothing, C, d
    end
    res = project(SparseAffineConstraint(C, d), zh, tol_rank = tol_rank,
                  tol_res = tol_res, diagnostic_limit = diagnostic_limit)
    res.status === :success || return res.status, nothing, C, d
    return :success, res, C, d
end

function _sparse_qp_result(Aeq, beq, G, h, zh, zstar, active, active_lambda,
                           iterations, tol_feas, tol_dual, tol_active,
                           tol_rank, cond_max, diagnostic_limit)
    T = eltype(zh)
    m = length(beq)
    q = length(h)
    lambda_eq = active_lambda[1:m]
    mu = zeros(T, q)
    mu[active] = active_lambda[(m + 1):end]
    eq = Aeq * zstar - beq
    slack = h - G * zstar
    ineq = max.(-slack, zero(T))
    stationarity = zstar .- zh .+ sparse(transpose(Aeq)) * lambda_eq .+
                   sparse(transpose(G)) * mu
    C, _ = _sparse_qp_active_matrix(Aeq, beq, G, h, active)
    K = _sparse_kkt_matrix(C, nothing)
    jac_min = _sparse_qp_jac_min(C, tol_rank, diagnostic_limit)
    cond_est = _sparse_kkt_cond_estimate(K, diagnostic_limit)
    inactive = trues(q)
    inactive[active] .= false
    weak_active = any(mu[active] .<= tol_dual)
    tight_inactive = any(slack[inactive] .<= tol_active)
    cres = T(norm(vcat(eq, ineq)))
    sres = T(norm(stationarity))
    status =
        isfinite(cond_est) && cond_est > cond_max ? :ill_conditioned :
        (cres > tol_feas || sres > tol_feas) ? :ill_conditioned :
        (weak_active || tight_inactive) ? :nonunique_input :
        :success
    return ProjectionResult(zstar, vcat(lambda_eq, mu), cres, sres,
                            T(norm(zstar - zh)), iterations, T(jac_min),
                            T(cond_est), status)
end

function _sparse_qp_polish(Aeq, beq, G, h, zh, active, iterations, tol_feas,
                           tol_dual, tol_active, tol_rank, cond_max,
                           diagnostic_limit)
    status, res, _, _ = _sparse_qp_solve_active(Aeq, beq, G, h, zh, active,
                                                tol_rank, tol_feas,
                                                diagnostic_limit)
    status === :success ||
        return _sparse_qp_fail_result(Aeq, beq, G, h, zh, status, iterations)
    return _sparse_qp_result(Aeq, beq, G, h, zh, res.zstar, active, res.lambda,
                             iterations, tol_feas, tol_dual, tol_active,
                             tol_rank, cond_max, diagnostic_limit)
end

function _sparse_qp_try_add(Aeq, beq, G, h, zh, active, candidates,
                            tol_rank, tol_feas, diagnostic_limit)
    for idx in candidates
        next_active = sort!(vcat(active, idx))
        status, _, C, d = _sparse_qp_solve_active(Aeq, beq, G, h, zh,
                                                  next_active, tol_rank,
                                                  tol_feas, diagnostic_limit)
        status === :success && return status, next_active
        status === :infeasible_constraint && return status, next_active
        _sparse_qp_inconsistent(C, d, tol_rank, diagnostic_limit) &&
            return :infeasible_constraint, next_active
    end
    return :singular_constraint, active
end

function _sparse_qp_active_set_project(Aeq, beq, G, h, zh, m, q;
                                       maxiter = 200, tol_feas = 1e-9,
                                       tol_dual = 1e-8, tol_active = 1e-8,
                                       tol_rank = 1e-8, cond_max = 1e12,
                                       diagnostic_limit = 256)
    if !_sparse_qp_rank_ok(Aeq, tol_rank, diagnostic_limit)
        return _sparse_qp_fail_result(Aeq, beq, G, h, zh,
                                      :singular_constraint, 0)
    end
    active = Int[]
    iterations = 0
    for it in 1:maxiter
        iterations = it
        status, res, _, _ = _sparse_qp_solve_active(Aeq, beq, G, h, zh,
                                                    active, tol_rank,
                                                    tol_feas,
                                                    diagnostic_limit)
        status !== :success &&
            return _sparse_qp_fail_result(Aeq, beq, G, h, zh, status, it)
        z = res.zstar
        slack = h - G * z
        violated = findall(slack .< -tol_feas)
        if isempty(violated)
            mu_active = res.lambda[(m + 1):end]
            if isempty(mu_active) || all(mu_active .>= -tol_dual)
                return _sparse_qp_result(Aeq, beq, G, h, zh, z, active,
                                         res.lambda, iterations, tol_feas,
                                         tol_dual, tol_active, tol_rank,
                                         cond_max, diagnostic_limit)
            end
            drop = argmin(mu_active)
            deleteat!(active, drop)
        else
            order = sortperm(slack[violated])
            candidates = violated[order]
            add_status, next_active =
                _sparse_qp_try_add(Aeq, beq, G, h, zh, active, candidates,
                                   tol_rank, tol_feas, diagnostic_limit)
            add_status !== :success &&
                return _sparse_qp_fail_result(Aeq, beq, G, h, zh,
                                              add_status, it)
            active = next_active
        end
    end
    return _sparse_qp_fail_result(Aeq, beq, G, h, zh, :max_iters, iterations)
end

function project(c::SparseLinearQPActiveSetConstraint, zhat::AbstractVector;
                 maxiter = 200, tol_feas = 1e-9, tol_dual = 1e-8,
                 tol_active = 1e-8, tol_rank = 1e-8, cond_max = 1e12,
                 diagnostic_limit = 256)
    Aeq, beq, G, h, zh, m, q, _ = _sparse_qp_inputs(c, zhat)
    return _sparse_qp_active_set_project(Aeq, beq, G, h, zh, m, q,
                                         maxiter = maxiter,
                                         tol_feas = tol_feas,
                                         tol_dual = tol_dual,
                                         tol_active = tol_active,
                                         tol_rank = tol_rank,
                                         cond_max = cond_max,
                                         diagnostic_limit = diagnostic_limit)
end

function _sparse_qp_primal_dual_solve(Aeq, beq, G, h, zh, m, q, n;
                                      maxiter = 100, tol_feas = 1e-9,
                                      tol_gap = 1e-9, centering = 0.1)
    T = eltype(zh)
    eq_res = project(SparseAffineConstraint(Aeq, beq), zh,
                     tol_res = tol_feas)
    eq_res.status === :success || return eq_res.status, nothing
    z = copy(eq_res.zstar)
    lambda = zeros(T, m)
    slack = max.(h - G * z, one(T))
    mu = ones(T, q)
    eye = spdiagm(0 => ones(T, n))

    for it in 1:maxiter
        rdual = z .- zh .+ sparse(transpose(Aeq)) * lambda .+
                sparse(transpose(G)) * mu
        req = Aeq * z - beq
        rineq = G * z .+ slack .- h
        gap = dot(slack, mu) / q
        residual = norm(vcat(rdual, req, rineq))
        if residual <= tol_feas && gap <= tol_gap
            return :success, (z = z, lambda = lambda, mu = mu,
                              slack = h - G * z, iterations = it)
        end

        target = T(centering) * gap
        rcent = slack .* mu .- target
        ratio = mu ./ slack
        H = eye + sparse(transpose(G)) * spdiagm(0 => ratio) * G
        rhs_z = -rdual .+ sparse(transpose(G)) *
                ((rcent .- mu .* rineq) ./ slack)
        K = [H sparse(transpose(Aeq))
             Aeq spzeros(T, m, m)]
        sol = _sparse_kkt_solve(K, vcat(rhs_z, -req))
        sol === nothing && return :singular_constraint, nothing
        dz = sol[1:n]
        dlambda = sol[(n + 1):end]
        dmu = (mu .* (G * dz) .- rcent .+ mu .* rineq) ./ slack
        dslack = -rineq .- G * dz
        alpha = one(T)
        neg_s = findall(dslack .< zero(T))
        neg_mu = findall(dmu .< zero(T))
        if !isempty(neg_s)
            alpha = min(alpha, T(0.99) * minimum(-slack[neg_s] ./ dslack[neg_s]))
        end
        if !isempty(neg_mu)
            alpha = min(alpha, T(0.99) * minimum(-mu[neg_mu] ./ dmu[neg_mu]))
        end
        isfinite(alpha) || return :line_search_fail, nothing
        alpha <= sqrt(eps(T)) && return :line_search_fail, nothing
        z .+= alpha .* dz
        lambda .+= alpha .* dlambda
        slack .+= alpha .* dslack
        mu .+= alpha .* dmu
    end
    return :max_iters, nothing
end

function _sparse_qp_prune_active(Aeq, beq, G, h, active, tol_rank,
                                 diagnostic_limit)
    isempty(active) && return active
    Cbase = Aeq
    rows = size(Cbase, 1)
    if max(size(Aeq, 1) + length(active), size(Aeq, 2)) > diagnostic_limit
        return active
    end
    kept = Int[]
    current = Cbase
    current_rank = _sparse_qp_dense_rank(current, tol_rank)
    for idx in active
        trial = [current
                 G[idx:idx, :]]
        trial_rank = _sparse_qp_dense_rank(trial, tol_rank)
        if trial_rank > current_rank
            push!(kept, idx)
            current = trial
            current_rank = trial_rank
        end
    end
    rows == current_rank || return kept
    return kept
end

function project(c::SparseLinearQPPrimalDualConstraint, zhat::AbstractVector;
                 maxiter = 100, tol_feas = 1e-9, tol_gap = 1e-9,
                 tol_dual = 1e-8, tol_active = 1e-7, tol_rank = 1e-8,
                 cond_max = 1e12, diagnostic_limit = 256,
                 centering = 0.1)
    Aeq, beq, G, h, zh, m, q, n = _sparse_qp_inputs(c, zhat)
    if !_sparse_qp_rank_ok(Aeq, tol_rank, diagnostic_limit)
        return _sparse_qp_fail_result(Aeq, beq, G, h, zh,
                                      :singular_constraint, 0)
    end
    if q == 0
        return _sparse_qp_active_set_project(Aeq, beq, G, h, zh, m, q,
                                             maxiter = maxiter,
                                             tol_feas = tol_feas,
                                             tol_dual = tol_dual,
                                             tol_active = tol_active,
                                             tol_rank = tol_rank,
                                             cond_max = cond_max,
                                             diagnostic_limit = diagnostic_limit)
    end

    status, state = _sparse_qp_primal_dual_solve(Aeq, beq, G, h, zh, m, q, n,
                                                maxiter = maxiter,
                                                tol_feas = tol_feas,
                                                tol_gap = tol_gap,
                                                centering = centering)
    status !== :success &&
        return _sparse_qp_fail_result(Aeq, beq, G, h, zh, status, maxiter)
    active = findall(state.slack .<= tol_active)
    active = _sparse_qp_prune_active(Aeq, beq, G, h, active, tol_rank,
                                     diagnostic_limit)
    return _sparse_qp_polish(Aeq, beq, G, h, zh, active, state.iterations,
                             tol_feas, tol_dual, tol_active, tol_rank,
                             cond_max, diagnostic_limit)
end

function _sparse_qp_vjp(Aeq, beq, G, h, res::ProjectionResult,
                        gbar::AbstractVector, tol_active, tol_rank,
                        diagnostic_limit)
    res.status === :success || error("no gradient is claimed for status :$(res.status)")
    _, _, _, _, gb, m, q, n =
        _sparse_qp_inputs(SparseLinearQPActiveSetConstraint(Aeq, beq, G, h), gbar)
    length(res.zstar) == n || throw(DimensionMismatch("gbar length must match zstar length"))
    mu = res.lambda[(m + 1):end]
    active = findall(mu .> tol_active)
    C, _ = _sparse_qp_active_matrix(Aeq, beq, G, h, active)
    if size(C, 1) == 0
        return copy(gb)
    end
    _sparse_qp_rank_ok(C, tol_rank, diagnostic_limit) ||
        error("no gradient is claimed for a rank-deficient sparse active set")
    tangent = project(SparseAffineConstraint(C, zeros(eltype(gb), size(C, 1))),
                      gb, tol_rank = tol_rank,
                      diagnostic_limit = diagnostic_limit)
    tangent.status === :success ||
        error("no gradient is claimed for a singular sparse active set")
    return tangent.zstar
end

function vjp(c::SparseLinearQPActiveSetConstraint, res::ProjectionResult,
             gbar::AbstractVector; tol_active = 1e-8, tol_rank = 1e-8,
             diagnostic_limit = 256)
    Aeq, beq, G, h, _, _, _, _ = _sparse_qp_inputs(c, gbar)
    return _sparse_qp_vjp(Aeq, beq, G, h, res, gbar, tol_active, tol_rank,
                          diagnostic_limit)
end

function vjp(c::SparseLinearQPPrimalDualConstraint, res::ProjectionResult,
             gbar::AbstractVector; tol_active = 1e-8, tol_rank = 1e-8,
             diagnostic_limit = 256)
    Aeq, beq, G, h, _, _, _, _ = _sparse_qp_inputs(c, gbar)
    return _sparse_qp_vjp(Aeq, beq, G, h, res, gbar, tol_active, tol_rank,
                          diagnostic_limit)
end
