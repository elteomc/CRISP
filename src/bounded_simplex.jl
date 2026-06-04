"""
    BoundedWeightedSimplexConstraint(weights, mass, lower, upper)

The bounded weighted equality set
`{z : lower .<= z .<= upper, dot(weights, z) = mass}` with strictly positive
weights. This M6 layer combines a single linear equality with componentwise
bounds.

The Euclidean projection has the threshold form

    zstar_i = clamp(zhat_i - tau * weights_i, lower_i, upper_i)

where `tau` is chosen so `dot(weights, zstar) = mass`. The backward pass uses
the free set at `zstar`. At active-set kinks, no gradient is claimed.
"""
struct BoundedWeightedSimplexConstraint{W<:AbstractVector, T<:Real, L<:AbstractVector, U<:AbstractVector}
    weights::W
    mass::T
    lower::L
    upper::U
end

function _bounded_simplex_inputs(c::BoundedWeightedSimplexConstraint, zhat::AbstractVector)
    zh = float.(zhat)
    T = eltype(zh)
    w = T.(c.weights)
    lo = T.(c.lower)
    hi = T.(c.upper)
    length(w) == length(zh) || throw(DimensionMismatch("weights length must match zhat length"))
    length(lo) == length(zh) || throw(DimensionMismatch("lower length must match zhat length"))
    length(hi) == length(zh) || throw(DimensionMismatch("upper length must match zhat length"))
    all(isfinite, w) || throw(ArgumentError("weights must be finite"))
    all(wi -> wi > zero(T), w) || throw(ArgumentError("weights must be strictly positive"))
    all(isfinite, lo) || throw(ArgumentError("lower bounds must be finite"))
    all(isfinite, hi) || throw(ArgumentError("upper bounds must be finite"))
    mass = T(c.mass)
    isfinite(mass) || throw(ArgumentError("mass must be finite"))
    return w, mass, lo, hi, zh
end

function _clamp_bounds(x, lo, hi)
    return min.(max.(x, lo), hi)
end

function _bounded_simplex_threshold(zh, w, mass, lo, hi; maxiter = 120)
    T = eltype(zh)
    tau_lo = minimum((zh .- hi) ./ w)
    tau_hi = maximum((zh .- lo) ./ w)
    f(tau) = dot(w, _clamp_bounds(zh .- tau .* w, lo, hi)) - mass

    for _ in 1:maxiter
        tau = (tau_lo + tau_hi) / 2
        if f(tau) > zero(T)
            tau_lo = tau
        else
            tau_hi = tau
        end
    end
    return (tau_lo + tau_hi) / 2
end

function _bounded_simplex_plateau_tau(zh, w, zstar, lo, hi, tol_active)
    T = eltype(zh)
    lower_active = zstar .<= lo .+ tol_active
    upper_active = zstar .>= hi .- tol_active
    lower_thresholds = (zh[lower_active] .- lo[lower_active]) ./ w[lower_active]
    upper_thresholds = (zh[upper_active] .- hi[upper_active]) ./ w[upper_active]
    tau_low = isempty(lower_thresholds) ? -T(Inf) : maximum(lower_thresholds)
    tau_high = isempty(upper_thresholds) ? T(Inf) : minimum(upper_thresholds)

    if isfinite(tau_low) && isfinite(tau_high) && tau_low <= tau_high
        return (tau_low + tau_high) / 2
    elseif isfinite(tau_low)
        return tau_low + one(T)
    elseif isfinite(tau_high)
        return tau_high - one(T)
    else
        return zero(T)
    end
end

function _bounded_simplex_kink(raw, lo, hi, tol_active)
    pinned = abs.(hi .- lo) .<= tol_active
    lower_kink = abs.(raw .- lo) .<= tol_active
    upper_kink = abs.(raw .- hi) .<= tol_active
    return any((lower_kink .| upper_kink) .& .!pinned)
end

function _bounded_simplex_free(zstar, lo, hi, tol_active)
    pinned = abs.(hi .- lo) .<= tol_active
    return (zstar .> lo .+ tol_active) .& (zstar .< hi .- tol_active) .& .!pinned
end

function _bounded_simplex_result(zh, w, mass, lo, hi, zstar, tau, status, tol_active)
    T = eltype(zh)
    n = length(zh)
    lower_active = zstar .<= lo .+ tol_active
    upper_active = zstar .>= hi .- tol_active
    stationarity = zstar .- zh .+ tau .* w
    lower_mu = zeros(T, n)
    upper_mu = zeros(T, n)
    lower_mu[lower_active] = max.(stationarity[lower_active], zero(T))
    upper_mu[upper_active] = max.(-stationarity[upper_active], zero(T))

    cres = norm(vcat(dot(w, zstar) - mass,
                     min.(zstar .- lo, zero(T)),
                     min.(hi .- zstar, zero(T))))
    sres = norm(zstar .- zh .+ tau .* w .- lower_mu .+ upper_mu)
    free = _bounded_simplex_free(zstar, lo, hi, tol_active)
    if any(free)
        wf = w[free]
        k = length(wf)
        M = [Matrix(I, k, k) wf
             transpose(wf) zeros(T, 1, 1)]
        jac_min = norm(wf)
        cond_est = cond(M)
    else
        jac_min = zero(T)
        cond_est = one(T)
    end

    ProjectionResult(zstar, vcat(T(tau), lower_mu, upper_mu), T(cres), T(sres),
                     T(norm(zstar - zh)), 1, T(jac_min), T(cond_est), status)
end

"""
    project(c::BoundedWeightedSimplexConstraint, zhat) -> ProjectionResult

Euclidean projection onto a single weighted equality plus finite bounds. The
status is `:success` only when the active set is strict enough for the returned
VJP to be valid.
"""
function project(c::BoundedWeightedSimplexConstraint, zhat::AbstractVector;
                 tol_mass = 1e-10, tol_active = 1e-8)
    w, mass, lo, hi, zh = _bounded_simplex_inputs(c, zhat)
    T = eltype(zh)
    n = length(zh)

    if any(lo .> hi)
        return ProjectionResult(zh, zeros(T, 1 + 2n), norm(max.(lo .- hi, zero(T))),
                                zero(T), zero(T), 0, zero(T), T(Inf),
                                :infeasible_constraint)
    end

    mass_min = dot(w, lo)
    mass_max = dot(w, hi)
    if mass < mass_min - tol_mass || mass > mass_max + tol_mass
        residual = max(mass_min - mass, mass - mass_max, zero(T))
        zstar = _clamp_bounds(zh, lo, hi)
        return ProjectionResult(zstar, zeros(T, 1 + 2n), T(residual),
                                zero(T), T(norm(zstar - zh)), 0, zero(T),
                                T(Inf), :infeasible_constraint)
    end

    if abs(mass - mass_min) <= tol_mass
        zstar = copy(lo)
        tau = _bounded_simplex_plateau_tau(zh, w, zstar, lo, hi, tol_active)
        return _bounded_simplex_result(zh, w, mass, lo, hi, zstar, tau,
                                       :success, tol_active)
    elseif abs(mass - mass_max) <= tol_mass
        zstar = copy(hi)
        tau = _bounded_simplex_plateau_tau(zh, w, zstar, lo, hi, tol_active)
        return _bounded_simplex_result(zh, w, mass, lo, hi, zstar, tau,
                                       :success, tol_active)
    end

    tau = _bounded_simplex_threshold(zh, w, mass, lo, hi)
    raw = zh .- tau .* w
    zstar = _clamp_bounds(raw, lo, hi)
    free = _bounded_simplex_free(zstar, lo, hi, tol_active)
    if !any(free)
        tau = _bounded_simplex_plateau_tau(zh, w, zstar, lo, hi, tol_active)
        raw = zh .- tau .* w
        zstar = _clamp_bounds(raw, lo, hi)
    end

    status = _bounded_simplex_kink(raw, lo, hi, tol_active) ? :nonunique_input : :success
    return _bounded_simplex_result(zh, w, mass, lo, hi, zstar, tau,
                                   status, tol_active)
end

"""
    vjp(c::BoundedWeightedSimplexConstraint, res, gbar) -> Vector

Active-set VJP for the bounded weighted equality projection. Bound coordinates
receive zero gradient. Free coordinates are projected onto the tangent space
`dot(weights_free, dz_free) = 0`.
"""
function vjp(c::BoundedWeightedSimplexConstraint, res::ProjectionResult,
             gbar::AbstractVector; tol_active = 1e-8)
    res.status === :success || error("no gradient is claimed for status :$(res.status)")
    w, _, lo, hi, gb = _bounded_simplex_inputs(c, gbar)
    length(res.zstar) == length(gb) || throw(DimensionMismatch("gbar length must match zstar length"))
    T = eltype(gb)
    free = _bounded_simplex_free(res.zstar, lo, hi, tol_active)
    out = zeros(T, length(gb))
    if any(free)
        wf = w[free]
        gf = collect(gb)[free]
        out[free] = gf .- wf .* (dot(wf, gf) / dot(wf, wf))
    end
    return out
end
