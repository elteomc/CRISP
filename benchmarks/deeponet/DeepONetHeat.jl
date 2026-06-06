"""
    DeepONetHeat

Small DeepONet-style heat-operator benchmark for StructPINN helper work. The
model predicts a fixed-grid field from heat-equation parameters. StructPINN
then corrects the output onto boundary, mass, and box constraints.
"""
module DeepONetHeat

using LinearAlgebra, Random, SparseArrays, Statistics, Zygote, StructPINN

export ProjectionLog, record!,
       grid, trapezoid_weights, mass, heat_operator_field,
       sample_heat_operator, init_deeponet, deeponet_model,
       boundary_box_constraint, full_heat_constraint,
       heat_correction_context, heat_correction_contexts,
       vanilla_loss, soft_loss, hard_projected_field, hard_loss,
       soft_plus_hard_loss, context_projected_field, context_hard_loss,
       context_soft_plus_hard_loss, train!, field_rmse,
       boundary_violation, mass_violation, lower_violation, upper_violation,
       evaluate_model, evaluate_context_model

mutable struct ProjectionLog
    counts::Dict{Symbol,Int}
    correction_norms::Vector{Float64}
end

ProjectionLog() = ProjectionLog(Dict{Symbol,Int}(), Float64[])

function record!(log::ProjectionLog, res)
    log.counts[res.status] = get(log.counts, res.status, 0) + 1
    push!(log.correction_norms, Float64(res.correction_norm))
    return log
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

function heat_operator_field(x, theta; diffusivity = 0.08)
    t, left, right, a, b = theta
    lift = left .+ (right - left) .* x
    mode1 = exp(-diffusivity * pi^2 * t) .* sin.(pi .* x)
    mode2 = exp(-4 * diffusivity * pi^2 * t) .* sin.(2pi .* x)
    return lift .+ 0.22 * a .* mode1 .+ 0.14 * b .* mode2
end

function sample_heat_operator(N, K; rng = Random.default_rng())
    x = grid(K)
    w = trapezoid_weights(x)
    data = NamedTuple[]
    for _ in 1:N
        theta = [0.05 + 0.95 * rand(rng),
                 0.75 + 0.35 * rand(rng),
                 0.75 + 0.35 * rand(rng),
                 2 * rand(rng) - 1,
                 2 * rand(rng) - 1]
        u = heat_operator_field(x, theta)
        push!(data, (theta = theta, u = u, left = theta[2],
                     right = theta[3], mass0 = mass(u, w)))
    end
    return data, x, w
end

function init_deeponet(; nin = 5, width = 24, rank = 12, seed = 0,
                       scale = 0.12)
    rng = MersenneTwister(seed)
    r(d...) = scale .* randn(rng, d...)
    return (Wb1 = r(width, nin), bb1 = zeros(width),
            Wb2 = r(rank, width), bb2 = zeros(rank),
            Wt1 = r(width, 1), bt1 = zeros(width),
            Wt2 = r(rank, width), bt2 = zeros(rank),
            c = zeros(1))
end

function deeponet_model(p, theta, x)
    branch = p.Wb2 * tanh.(p.Wb1 * theta .+ p.bb1) .+ p.bb2
    xrow = reshape(x, 1, length(x))
    trunk_hidden = tanh.(p.Wt1 * xrow .+ reshape(p.bt1, :, 1))
    trunk = p.Wt2 * trunk_hidden .+ reshape(p.bt2, :, 1)
    return vec(transpose(trunk) * branch) .+ p.c[1]
end

function _boundary_matrix(K)
    A = spzeros(Float64, 2, K)
    A[1, 1] = 1.0
    A[2, K] = 1.0
    return A
end

function _full_matrix(w)
    K = length(w)
    A = spzeros(Float64, 3, K)
    A[1, 1] = 1.0
    A[2, K] = 1.0
    for i in 1:K
        A[3, i] = w[i]
    end
    return A
end

function _bound_vector(v, K)
    return v isa AbstractVector ? collect(v) : fill(v, K)
end

function boundary_box_constraint(K, left, right; lower = 0.0, upper = 2.0)
    return SparseBoxAffineConstraint(_boundary_matrix(K), [left, right],
                                     _bound_vector(lower, K),
                                     _bound_vector(upper, K))
end

function full_heat_constraint(w, left, right, mass0; lower = 0.0,
                              upper = 2.0)
    K = length(w)
    return SparseBoxAffineConstraint(_full_matrix(w), [left, right, mass0],
                                     _bound_vector(lower, K),
                                     _bound_vector(upper, K))
end

function heat_correction_context(d, w; lower = 0.0, upper = 2.0,
                                 mode = :full, cached = true)
    base = mode === :boundary_box ?
        boundary_box_constraint(length(w), d.left, d.right,
                                lower = lower, upper = upper) :
        full_heat_constraint(w, d.left, d.right, d.mass0,
                             lower = lower, upper = upper)
    constraint = cached ? CachedSparseBoxAffineConstraint(base) : base
    return (data = d, weights = collect(w), constraint = constraint,
            mode = mode, cached = cached)
end

function heat_correction_contexts(data, w; lower = 0.0, upper = 2.0,
                                  mode = :full, cached = true)
    return [heat_correction_context(d, w, lower = lower, upper = upper,
                                    mode = mode, cached = cached)
            for d in data]
end

function _constraint_penalty(pred, d, w; lower = 0.0, upper = 2.0,
                             beta_boundary = 10.0, beta_mass = 10.0,
                             beta_box = 2.0)
    bpen = (pred[1] - d.left)^2 + (pred[end] - d.right)^2
    mpen = (mass(pred, w) - d.mass0)^2
    lpen = mean(abs2, max.(lower .- pred, 0.0))
    upen = mean(abs2, max.(pred .- upper, 0.0))
    return beta_boundary * bpen + beta_mass * mpen + beta_box * (lpen + upen)
end

function vanilla_loss(p, data, x)
    s = 0.0
    for d in data
        pred = deeponet_model(p, d.theta, x)
        s += mean(abs2, pred .- d.u)
    end
    return s / length(data)
end

function soft_loss(p, data, x, w; lower = 0.0, upper = 2.0,
                   beta_boundary = 10.0, beta_mass = 10.0,
                   beta_box = 2.0)
    s = 0.0
    for d in data
        pred = deeponet_model(p, d.theta, x)
        s += mean(abs2, pred .- d.u)
        s += _constraint_penalty(pred, d, w, lower = lower, upper = upper,
                                 beta_boundary = beta_boundary,
                                 beta_mass = beta_mass,
                                 beta_box = beta_box)
    end
    return s / length(data)
end

function _record_projection!(c, pred, log, failure_policy)
    res = project(c, pred)
    log === nothing || record!(log, res)
    if !StructPINN.issuccess(res)
        if failure_policy === :error
            error("DeepONet projection failed with status :$(res.status)")
        elseif failure_policy !== :continue
            error("unknown projection failure policy :$(failure_policy)")
        end
    end
    return nothing
end

function hard_projected_field(p, d, x, w; lower = 0.0, upper = 2.0,
                              log = nothing, failure_policy = :error,
                              mode = :full)
    pred = deeponet_model(p, d.theta, x)
    c = Zygote.ignore() do
        mode === :boundary_box ?
            boundary_box_constraint(length(x), d.left, d.right,
                                    lower = lower, upper = upper) :
            full_heat_constraint(w, d.left, d.right, d.mass0,
                                 lower = lower, upper = upper)
    end
    Zygote.ignore() do
        _record_projection!(c, pred, log, failure_policy)
    end
    return correct(c, pred)
end

function hard_loss(p, data, x, w; lower = 0.0, upper = 2.0,
                   log = nothing, failure_policy = :error,
                   mode = :full)
    s = 0.0
    for d in data
        pred = hard_projected_field(p, d, x, w, lower = lower,
                                    upper = upper, log = log,
                                    failure_policy = failure_policy,
                                    mode = mode)
        s += mean(abs2, pred .- d.u)
    end
    return s / length(data)
end

function context_projected_field(p, ctx, x; log = nothing,
                                 failure_policy = :error)
    d = ctx.data
    pred = deeponet_model(p, d.theta, x)
    c = ctx.constraint
    Zygote.ignore() do
        _record_projection!(c, pred, log, failure_policy)
    end
    return correct(c, pred)
end

function context_hard_loss(p, contexts, x; log = nothing,
                           failure_policy = :error)
    s = 0.0
    for ctx in contexts
        d = ctx.data
        pred = context_projected_field(p, ctx, x, log = log,
                                       failure_policy = failure_policy)
        s += mean(abs2, pred .- d.u)
    end
    return s / length(contexts)
end

function soft_plus_hard_loss(p, data, x, w; lower = 0.0, upper = 2.0,
                             beta_boundary = 2.0, beta_mass = 2.0,
                             beta_box = 0.5, log = nothing,
                             failure_policy = :error, mode = :full)
    s = 0.0
    for d in data
        raw = deeponet_model(p, d.theta, x)
        c = Zygote.ignore() do
            mode === :boundary_box ?
                boundary_box_constraint(length(x), d.left, d.right,
                                        lower = lower, upper = upper) :
                full_heat_constraint(w, d.left, d.right, d.mass0,
                                     lower = lower, upper = upper)
        end
        Zygote.ignore() do
            _record_projection!(c, raw, log, failure_policy)
        end
        pred = correct(c, raw)
        s += mean(abs2, pred .- d.u)
        s += _constraint_penalty(raw, d, w, lower = lower, upper = upper,
                                 beta_boundary = beta_boundary,
                                 beta_mass = beta_mass,
                                 beta_box = beta_box)
    end
    return s / length(data)
end

function context_soft_plus_hard_loss(p, contexts, x; lower = 0.0,
                                     upper = 2.0, beta_boundary = 2.0,
                                     beta_mass = 2.0, beta_box = 0.5,
                                     log = nothing,
                                     failure_policy = :error)
    s = 0.0
    for ctx in contexts
        d = ctx.data
        raw = deeponet_model(p, d.theta, x)
        c = ctx.constraint
        Zygote.ignore() do
            _record_projection!(c, raw, log, failure_policy)
        end
        pred = correct(c, raw)
        s += mean(abs2, pred .- d.u)
        s += _constraint_penalty(raw, d, ctx.weights, lower = lower,
                                 upper = upper,
                                 beta_boundary = beta_boundary,
                                 beta_mass = beta_mass,
                                 beta_box = beta_box)
    end
    return s / length(contexts)
end

_ntmap(f, nts...) = NamedTuple{keys(nts[1])}(map(f, map(values, nts)...))

function train!(lossfn, p; steps = 200, lr = 1e-2, beta1 = 0.9,
                beta2 = 0.999, eps = 1e-8)
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
        p = _ntmap((pk, a, b) -> pk .- lr .* a ./ (sqrt.(b) .+ eps),
                   p, mh, vh)
    end
    return p, history
end

field_rmse(pred, truth) = sqrt(mean(abs2, pred .- truth))
boundary_violation(pred, d) = max(abs(pred[1] - d.left), abs(pred[end] - d.right))
mass_violation(pred, d, w) = abs(mass(pred, w) - d.mass0)
lower_violation(pred, lower) = maximum(max.(lower .- pred, 0.0))
upper_violation(pred, upper) = maximum(max.(pred .- upper, 0.0))

function evaluate_model(predict, data, w; lower = 0.0, upper = 2.0)
    rmses = Float64[]
    bviol = Float64[]
    mviol = Float64[]
    lviol = Float64[]
    uviol = Float64[]
    for d in data
        pred = predict(d)
        push!(rmses, field_rmse(pred, d.u))
        push!(bviol, boundary_violation(pred, d))
        push!(mviol, mass_violation(pred, d, w))
        push!(lviol, lower_violation(pred, lower))
        push!(uviol, upper_violation(pred, upper))
    end
    return (rmse = mean(rmses), boundary_max = maximum(bviol),
            boundary_mean = mean(bviol), mass_max = maximum(mviol),
            mass_mean = mean(mviol), lower_max = maximum(lviol),
            lower_mean = mean(lviol), upper_max = maximum(uviol),
            upper_mean = mean(uviol))
end

function evaluate_context_model(predict, contexts; lower = 0.0,
                                upper = 2.0)
    rmses = Float64[]
    bviol = Float64[]
    mviol = Float64[]
    lviol = Float64[]
    uviol = Float64[]
    for ctx in contexts
        d = ctx.data
        pred = predict(ctx)
        push!(rmses, field_rmse(pred, d.u))
        push!(bviol, boundary_violation(pred, d))
        push!(mviol, mass_violation(pred, d, ctx.weights))
        push!(lviol, lower_violation(pred, lower))
        push!(uviol, upper_violation(pred, upper))
    end
    return (rmse = mean(rmses), boundary_max = maximum(bviol),
            boundary_mean = mean(bviol), mass_max = maximum(mviol),
            mass_mean = mean(mviol), lower_max = maximum(lviol),
            lower_mean = mean(lviol), upper_max = maximum(uviol),
            upper_mean = mean(uviol))
end

end # module
