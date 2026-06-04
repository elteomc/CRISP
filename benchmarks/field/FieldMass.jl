"""
    FieldMass

M5 fixed-grid field-correction benchmark. A small supervised model predicts a
1D field on a shared grid from low-dimensional features. The projected baseline
uses the M1 affine layer to enforce each sample's discrete mass exactly before
the loss is evaluated.
"""
module FieldMass

using LinearAlgebra, Random, Statistics, Zygote, StructPINN

export FailureCounter, record!,
       grid, trapezoid_weights, mass, sample_fields,
       init_mlp, field_model, vanilla_loss, soft_loss,
       mass_constraint, weighted_mass_constraint,
       projected_field, projected_loss, weighted_projected_field, weighted_projected_loss,
       box_constraint, box_projected_field, box_projected_loss,
       positive_mass_constraint, positive_projected_field, positive_projected_loss,
       bounded_mass_constraint, bounded_projected_field, bounded_projected_loss,
       train!, field_rmse, mass_violation, evaluate_model

struct FailureCounter
    counts::Dict{Symbol,Int}
end

FailureCounter() = FailureCounter(Dict{Symbol,Int}())

function record!(fc::FailureCounter, status::Symbol)
    fc.counts[status] = get(fc.counts, status, 0) + 1
    return fc
end

grid(K::Int) = collect(range(0.0, 1.0, length = K))

function trapezoid_weights(x::AbstractVector)
    K = length(x)
    K >= 2 || throw(ArgumentError("grid must contain at least two points"))
    w = zeros(float(eltype(x)), K)
    w[1] = (x[2] - x[1]) / 2
    for i in 2:(K - 1)
        w[i] = (x[i + 1] - x[i - 1]) / 2
    end
    w[K] = (x[K] - x[K - 1]) / 2
    return w
end

mass(u::AbstractVector, w::AbstractVector) = dot(w, u)

function target_field(x, theta)
    a, b, c = theta
    baseline = 1.0 + 0.25 * a
    shape = 0.35 * b .* sin.(2pi .* x) .+ 0.20 * c .* cos.(4pi .* x)
    bump = 0.15 * a * b .* sin.(pi .* x)
    return baseline .+ shape .+ bump
end

function sample_fields(N, K; rng = Random.default_rng())
    x = grid(K)
    w = trapezoid_weights(x)
    data = NamedTuple[]
    for _ in 1:N
        theta = 2 .* rand(rng, 3) .- 1
        u = target_field(x, theta)
        push!(data, (theta = theta, u = u, mass0 = mass(u, w)))
    end
    return data, x, w
end

function init_mlp(; nin = 3, nh = 24, nout = 32, seed = 0, scale = 0.1)
    rng = MersenneTwister(seed)
    r(d...) = scale .* randn(rng, d...)
    return (W1 = r(nh, nin), b1 = zeros(nh),
            W2 = r(nh, nh), b2 = zeros(nh),
            W3 = r(nout, nh), b3 = zeros(nout))
end

field_model(p, theta) = p.W3 * tanh.(p.W2 * tanh.(p.W1 * theta .+ p.b1) .+ p.b2) .+ p.b3

function vanilla_loss(p, data)
    s = 0.0
    for d in data
        pred = field_model(p, d.theta)
        s += mean(abs2, pred .- d.u)
    end
    return s / length(data)
end

function soft_loss(p, data, w; beta = 10.0)
    s = 0.0
    for d in data
        pred = field_model(p, d.theta)
        s += mean(abs2, pred .- d.u)
        s += beta * (mass(pred, w) - d.mass0)^2
    end
    return s / length(data)
end

mass_constraint(w, mass0) = AffineConstraint(reshape(collect(w), 1, length(w)), [mass0])
weighted_mass_constraint(w, mass0) = DiagonalWeightedAffineConstraint(reshape(collect(w), 1, length(w)), [mass0], collect(w))
box_constraint(K; lower = 0.0, upper = 2.0) = BoxConstraint(fill(lower, K), fill(upper, K))
positive_mass_constraint(w, mass0) = WeightedSimplexConstraint(collect(w), mass0)
_bound_vector(v, K) = v isa AbstractVector ? collect(v) : fill(v, K)
bounded_mass_constraint(w, mass0; lower = 0.0, upper = 2.0) =
    BoundedWeightedSimplexConstraint(collect(w), mass0,
                                     _bound_vector(lower, length(w)),
                                     _bound_vector(upper, length(w)))

function _record_projection!(c, pred, fc, failure_policy)
    res = project(c, pred)
    fc === nothing || record!(fc, res.status)
    if !StructPINN.issuccess(res)
        if failure_policy === :error
            error("field projection failed with status :$(res.status)")
        elseif failure_policy !== :continue
            error("unknown projection failure policy :$(failure_policy)")
        end
    end
    return nothing
end

function projected_field(p, theta, w, mass0; fc = nothing, failure_policy = :error)
    pred = field_model(p, theta)
    c = mass_constraint(w, mass0)
    Zygote.ignore() do
        _record_projection!(c, pred, fc, failure_policy)
    end
    return correct(c, pred)
end

function projected_loss(p, data, w; fc = nothing, failure_policy = :error)
    s = 0.0
    for d in data
        pred = projected_field(p, d.theta, w, d.mass0;
                               fc = fc, failure_policy = failure_policy)
        s += mean(abs2, pred .- d.u)
    end
    return s / length(data)
end

function weighted_projected_field(p, theta, w, mass0; fc = nothing, failure_policy = :error)
    pred = field_model(p, theta)
    c = weighted_mass_constraint(w, mass0)
    Zygote.ignore() do
        _record_projection!(c, pred, fc, failure_policy)
    end
    return correct(c, pred)
end

function weighted_projected_loss(p, data, w; fc = nothing, failure_policy = :error)
    s = 0.0
    for d in data
        pred = weighted_projected_field(p, d.theta, w, d.mass0;
                                        fc = fc, failure_policy = failure_policy)
        s += mean(abs2, pred .- d.u)
    end
    return s / length(data)
end

function box_projected_field(p, theta, lower, upper; fc = nothing, failure_policy = :error)
    pred = field_model(p, theta)
    c = box_constraint(length(pred); lower = lower, upper = upper)
    Zygote.ignore() do
        _record_projection!(c, pred, fc, failure_policy)
    end
    return correct(c, pred)
end

function box_projected_loss(p, data, lower, upper; fc = nothing, failure_policy = :error)
    s = 0.0
    for d in data
        pred = box_projected_field(p, d.theta, lower, upper;
                                   fc = fc, failure_policy = failure_policy)
        s += mean(abs2, pred .- d.u)
    end
    return s / length(data)
end

function positive_projected_field(p, theta, w, mass0; fc = nothing, failure_policy = :error)
    pred = field_model(p, theta)
    c = positive_mass_constraint(w, mass0)
    Zygote.ignore() do
        _record_projection!(c, pred, fc, failure_policy)
    end
    return correct(c, pred)
end

function positive_projected_loss(p, data, w; fc = nothing, failure_policy = :error)
    s = 0.0
    for d in data
        pred = positive_projected_field(p, d.theta, w, d.mass0;
                                        fc = fc, failure_policy = failure_policy)
        s += mean(abs2, pred .- d.u)
    end
    return s / length(data)
end

function bounded_projected_field(p, theta, w, mass0, lower, upper;
                                 fc = nothing, failure_policy = :error)
    pred = field_model(p, theta)
    c = bounded_mass_constraint(w, mass0, lower = lower, upper = upper)
    Zygote.ignore() do
        _record_projection!(c, pred, fc, failure_policy)
    end
    return correct(c, pred)
end

function bounded_projected_loss(p, data, w, lower, upper;
                                fc = nothing, failure_policy = :error)
    s = 0.0
    for d in data
        pred = bounded_projected_field(p, d.theta, w, d.mass0, lower, upper;
                                       fc = fc, failure_policy = failure_policy)
        s += mean(abs2, pred .- d.u)
    end
    return s / length(data)
end

_ntmap(f, nts...) = NamedTuple{keys(nts[1])}(map(f, map(values, nts)...))

function train!(lossfn, p; steps = 300, lr = 1e-2, beta1 = 0.9, beta2 = 0.999, eps = 1e-8)
    m = _ntmap(zero, p)
    v = _ntmap(zero, p)
    history = Float64[]
    for t in 1:steps
        L, back = Zygote.pullback(lossfn, p)
        g = back(1.0)[1]
        push!(history, L)
        m = _ntmap((mk, gk) -> beta1 .* mk .+ (1 - beta1) .* gk, m, g)
        v = _ntmap((vk, gk) -> beta2 .* vk .+ (1 - beta2) .* (gk .^ 2), v, g)
        mh = _ntmap(mk -> mk ./ (1 - beta1^t), m)
        vh = _ntmap(vk -> vk ./ (1 - beta2^t), v)
        p = _ntmap((pk, a, b) -> pk .- lr .* a ./ (sqrt.(b) .+ eps), p, mh, vh)
    end
    return p, history
end

field_rmse(pred, truth) = sqrt(mean(abs2, pred .- truth))
mass_violation(pred, d, w) = abs(mass(pred, w) - d.mass0)
negative_violation(pred) = maximum(max.(.-pred, 0.0))
upper_violation(pred, upper) = maximum(max.(pred .- upper, 0.0))

function evaluate_model(predict, data, w; upper = 2.0)
    rmse = Float64[]
    mviol = Float64[]
    nviol = Float64[]
    uviol = Float64[]
    for d in data
        pred = predict(d)
        push!(rmse, field_rmse(pred, d.u))
        push!(mviol, mass_violation(pred, d, w))
        push!(nviol, negative_violation(pred))
        push!(uviol, upper_violation(pred, upper))
    end
    return (rmse = mean(rmse), mass_max = maximum(mviol), mass_mean = mean(mviol),
            negative_max = maximum(nviol), negative_mean = mean(nviol),
            upper_max = maximum(uviol), upper_mean = mean(uviol))
end

end # module
