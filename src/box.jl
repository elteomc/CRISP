"""
    BoxConstraint(lower, upper)

The componentwise bound constraint `{z : lower .<= z .<= upper}`. This M6 layer
uses Euclidean projection, so the forward map is componentwise clamping.

The derivative is valid only away from active-boundary kinks. At a kink no
gradient is claimed and the status is `:nonunique_input`.
"""
struct BoxConstraint{L<:AbstractVector, U<:AbstractVector}
    lower::L
    upper::U
end

function _box_inputs(c::BoxConstraint, zhat::AbstractVector)
    zh = float.(zhat)
    T = eltype(zh)
    lo = T.(c.lower)
    hi = T.(c.upper)
    length(lo) == length(zh) || throw(DimensionMismatch("lower length must match zhat length"))
    length(hi) == length(zh) || throw(DimensionMismatch("upper length must match zhat length"))
    all(isfinite, lo) || throw(ArgumentError("lower bounds must be finite"))
    all(isfinite, hi) || throw(ArgumentError("upper bounds must be finite"))
    return lo, hi, zh
end

"""
    project(c::BoxConstraint, zhat; tol_active=1e-8) -> ProjectionResult

Euclidean projection onto componentwise bounds. Equal lower and upper bounds are
allowed, but no gradient is claimed for those pinned coordinates.
"""
function project(c::BoxConstraint, zhat::AbstractVector; tol_active = 1e-8)
    lo, hi, zh = _box_inputs(c, zhat)
    T = eltype(zh)
    n = length(zh)

    if any(lo .> hi)
        return ProjectionResult(zh, zeros(T, 2n), norm(max.(lo .- hi, zero(T))),
                                zero(T), zero(T), 0, zero(T), T(Inf),
                                :infeasible_constraint)
    end

    zstar = min.(max.(zh, lo), hi)
    lower_mu = max.(lo .- zh, zero(T))
    upper_mu = max.(zh .- hi, zero(T))
    lower_slack = zstar .- lo
    upper_slack = hi .- zstar
    kink = any(abs.(zh .- lo) .<= tol_active) || any(abs.(zh .- hi) .<= tol_active)
    cres = norm(vcat(min.(lower_slack, zero(T)), min.(upper_slack, zero(T))))
    sres = norm(zstar - zh .- lower_mu .+ upper_mu)
    pinned = count(i -> lo[i] == hi[i], eachindex(lo))
    free = n - pinned
    status = kink ? :nonunique_input : :success
    ProjectionResult(zstar, vcat(lower_mu, upper_mu), T(cres), T(sres),
                     T(norm(zstar - zh)), 1, T(free > 0 ? 1 : 0),
                     one(T), status)
end

"""
    vjp(c::BoxConstraint, res, gbar) -> Vector

Active-set VJP for the box projection. Strictly interior coordinates pass the
upstream gradient through. Coordinates on a bound receive zero, but this method
only runs for `:success`, so active-boundary kinks are excluded.
"""
function vjp(c::BoxConstraint, res::ProjectionResult, gbar::AbstractVector; tol_active = 1e-8)
    res.status === :success || error("no gradient is claimed for status :$(res.status)")
    lo, hi, gb = _box_inputs(c, gbar)
    length(res.zstar) == length(gb) || throw(DimensionMismatch("gbar length must match zstar length"))
    n = length(gb)
    lower_mu = res.lambda[1:n]
    upper_mu = res.lambda[(n + 1):(2n)]
    free = (lower_mu .<= tol_active) .& (upper_mu .<= tol_active)
    return gb .* free
end
