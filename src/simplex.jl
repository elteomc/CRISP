"""
    WeightedSimplexConstraint(weights, mass)

The positivity plus normalization constraint set
`{z : z .>= 0, dot(weights, z) = mass}` with strictly positive weights. This is
the first M6 inequality layer.

The Euclidean projection has the threshold form

    zstar_i = max(zhat_i - tau * weights_i, 0)

where `tau` is chosen so `dot(weights, zstar) = mass`. The backward pass uses
the active positive set at `zstar`. At active-set kinks, no gradient is claimed.
"""
struct WeightedSimplexConstraint{V<:AbstractVector, T<:Real}
    weights::V
    mass::T
end

function _simplex_validate(c::WeightedSimplexConstraint, n::Int, T)
    w = float.(c.weights)
    length(w) == n || throw(DimensionMismatch("weights length must match zhat length"))
    all(isfinite, w) || throw(ArgumentError("weights must be finite"))
    all(wi -> wi > zero(eltype(w)), w) || throw(ArgumentError("weights must be strictly positive"))
    mass = T(c.mass)
    isfinite(mass) || throw(ArgumentError("mass must be finite"))
    return w, mass
end

function _simplex_threshold(zh, w, mass; maxiter = 120)
    T = eltype(zh)
    hi = maximum(zh ./ w)
    f(tau) = dot(w, max.(zh .- tau .* w, zero(T))) - mass
    span = one(T)
    lo = hi - span
    while f(lo) <= zero(T)
        span *= 2
        lo = hi - span
        isfinite(lo) || error("could not bracket weighted simplex threshold")
    end
    tau = (lo + hi) / 2
    for _ in 1:maxiter
        tau = (lo + hi) / 2
        if f(tau) > zero(T)
            lo = tau
        else
            hi = tau
        end
    end
    return (lo + hi) / 2
end

"""
    project(c::WeightedSimplexConstraint, zhat; kwargs...) -> ProjectionResult

Euclidean projection onto `{z : z .>= 0, dot(weights, z) = mass}`. The status is
`:success` only when the active set is strict, so the derivative is valid.
"""
function project(c::WeightedSimplexConstraint, zhat::AbstractVector;
                 tol_mass = 1e-10, tol_active = 1e-8)
    zh = float.(zhat)
    T = eltype(zh)
    n = length(zh)
    w, mass = _simplex_validate(c, n, T)

    if mass < -tol_mass
        return ProjectionResult(zeros(T, n), zeros(T, n + 1), abs(mass),
                                zero(T), norm(zh), 0, zero(T), T(Inf),
                                :infeasible_constraint)
    end
    if mass <= tol_mass
        zstar = zeros(T, n)
        return ProjectionResult(zstar, zeros(T, n + 1), zero(T), norm(zstar - zh),
                                norm(zstar - zh), 0, zero(T), T(Inf),
                                :singular_constraint)
    end

    tau = _simplex_threshold(zh, w, mass)
    margins = zh .- tau .* w
    zstar = max.(margins, zero(T))
    active = margins .> tol_active
    kink = any(abs(mi) <= tol_active for mi in margins)

    if !any(active)
        return ProjectionResult(zstar, vcat(T(tau), zeros(T, n)),
                                T(abs(dot(w, zstar) - mass)),
                                zero(T), T(norm(zstar - zh)), 0,
                                zero(T), T(Inf), :singular_constraint)
    end

    mu = max.(tau .* w .- zh, zero(T))
    cres = norm(vcat(dot(w, zstar) - mass, min.(zstar, zero(T))))
    sres = norm(zstar - zh .+ tau .* w .- mu)
    wactive = w[active]
    M = [Matrix(I, length(wactive), length(wactive)) wactive;
         transpose(wactive) zeros(T, 1, 1)]
    status = kink ? :nonunique_input : :success
    ProjectionResult(zstar, vcat(T(tau), mu), T(cres), T(sres),
                     T(norm(zstar - zh)), 1, T(norm(wactive)),
                     T(cond(M)), status)
end

"""
    vjp(c::WeightedSimplexConstraint, res, gbar) -> Vector

Active-set VJP for the weighted simplex projection. Inactive entries receive
zero gradient. Active entries are projected onto the tangent space
`dot(weights_active, dz_active) = 0`.
"""
function vjp(c::WeightedSimplexConstraint, res::ProjectionResult, gbar::AbstractVector;
             tol_active = 1e-8)
    res.status === :success || error("no gradient is claimed for status :$(res.status)")
    length(gbar) == length(res.zstar) || throw(DimensionMismatch("gbar length must match zstar length"))
    T = eltype(gbar)
    w, _ = _simplex_validate(c, length(gbar), T)
    active = res.zstar .> tol_active
    any(active) || error("no gradient is claimed for an empty active set")
    wa = w[active]
    ga = collect(gbar)[active]
    out = zeros(T, length(gbar))
    out[active] = ga .- wa .* (dot(wa, ga) / dot(wa, wa))
    return out
end
