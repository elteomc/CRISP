# Projection-frequency ablation for the DeepONet helper benchmark.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/frequency_ablation.jl
include("DeepONetHeat.jl")
using .DeepONetHeat
using Random, Statistics, Printf, Zygote

const K = 32
const SEEDS = 1:5
const STEPS = 80
const OUT = joinpath(@__DIR__, "results")

isdir(OUT) || mkdir(OUT)

const MODEL_NAMES = ["no_correction", "eval_only_full", "every_step",
                     "every_2_steps", "every_5_steps",
                     "every_10_steps"]
const PERIODS = Dict("every_step" => 1, "every_2_steps" => 2,
                     "every_5_steps" => 5, "every_10_steps" => 10)
const METRICS = (:rmse, :boundary_max, :boundary_mean, :mass_max, :mass_mean,
                 :lower_max, :lower_mean, :upper_max, :upper_mean)

function ntmap(f, nts...)
    return NamedTuple{keys(nts[1])}(map(f, map(values, nts)...))
end

function train_periodic!(p, data, x, w, period, log, steps = STEPS,
                         lr = 8e-3, beta1 = 0.9, beta2 = 0.999,
                         eps = 1e-8)
    m = ntmap(zero, p)
    v = ntmap(zero, p)
    history = Float64[]
    for t in 1:steps
        lossfn(pp) = periodic_hard_loss(pp, data, x, w, log = log,
                                        mode = :full, period = period,
                                        step = t)
        L, back = Zygote.pullback(lossfn, p)
        g = back(1.0)[1]
        push!(history, L)
        m = ntmap((mk, gk) -> beta1 .* mk .+ (1 - beta1) .* gk, m, g)
        v = ntmap((vk, gk) -> beta2 .* vk .+ (1 - beta2) .* (gk .^ 2),
                  v, g)
        mh = ntmap(mk -> mk ./ (1 - beta1^t), m)
        vh = ntmap(vk -> vk ./ (1 - beta2^t), v)
        p = ntmap((pk, a, b) -> pk .- lr .* a ./ (sqrt.(b) .+ eps),
                  p, mh, vh)
    end
    return p, history
end

function stat(xs)
    isempty(xs) && return 0.0, 0.0
    return mean(xs), length(xs) == 1 ? 0.0 : std(xs)
end

function merge_log!(dest::ProjectionLog, src::ProjectionLog)
    for (status, count) in src.counts
        dest.counts[status] = get(dest.counts, status, 0) + count
    end
    append!(dest.correction_norms, src.correction_norms)
    return dest
end

function add_eval!(acc, model, ev)
    for k in METRICS
        push!(acc[model][k], getfield(ev, k))
    end
    return nothing
end

function warmup_frequency()
    rng = MersenneTwister(91_000)
    data, x, w = sample_heat_operator(2, K, rng = rng)
    p = init_deeponet(seed = 91_000)
    log = ProjectionLog()
    train_periodic!(p, data, x, w, 2, log, 1)
    train!(pp -> vanilla_loss(pp, data, x), p, steps = 1, lr = 8e-3)
    return nothing
end

warmup_frequency()

acc = Dict(m => Dict(k => Float64[] for k in METRICS) for m in MODEL_NAMES)
times = Dict(m => Float64[] for m in MODEL_NAMES)
train_status = Dict(m => ProjectionLog() for m in MODEL_NAMES)
eval_status = Dict(m => ProjectionLog() for m in MODEL_NAMES)

for seed in SEEDS
    @printf("seed %d: running projection-frequency ablation ...\n", seed)
    rng = MersenneTwister(20_000 + seed)
    train_data, x, w = sample_heat_operator(18, K, rng = rng)
    test_data, _, _ = sample_heat_operator(8, K, rng = rng)
    test_contexts = heat_correction_contexts(test_data, w, mode = :full,
                                             cached = true)
    p0 = init_deeponet(seed = 20_000 + seed)

    tv = @elapsed pv, _ = train!(p -> vanilla_loss(p, train_data, x), p0,
                                 steps = STEPS, lr = 8e-3)
    push!(times["no_correction"], tv)
    push!(times["eval_only_full"], 0.0)

    add_eval!(acc, "no_correction",
              evaluate_model(d -> deeponet_model(pv, d.theta, x),
                             test_data, w))

    eval_only_log = ProjectionLog()
    add_eval!(acc, "eval_only_full",
              evaluate_context_model(ctx ->
                  context_projected_field(pv, ctx, x, log = eval_only_log),
                  test_contexts))
    merge_log!(eval_status["eval_only_full"], eval_only_log)

    for name in keys(PERIODS)
        period = PERIODS[name]
        log_train = ProjectionLog()
        elapsed = @elapsed pp, _ = train_periodic!(p0, train_data, x, w,
                                                   period, log_train)
        push!(times[name], elapsed)
        merge_log!(train_status[name], log_train)

        log_eval = ProjectionLog()
        add_eval!(acc, name,
                  evaluate_context_model(ctx ->
                      context_projected_field(pp, ctx, x, log = log_eval),
                      test_contexts))
        merge_log!(eval_status[name], log_eval)
    end
end

open(joinpath(OUT, "frequency_ablation.csv"), "w") do io
    println(io, "model,rmse_mean,rmse_std,boundarymax_mean,boundarymax_std,boundarymean_mean,boundarymean_std,massmax_mean,massmax_std,massmean_mean,massmean_std,lowermax_mean,lowermax_std,lowermean_mean,lowermean_std,uppermax_mean,uppermax_std,uppermean_mean,uppermean_std,train_seconds_mean,train_seconds_std")
    for model in MODEL_NAMES
        ms = Dict(k => stat(acc[model][k]) for k in METRICS)
        tm = stat(times[model])
        println(io, @sprintf("%s,%.8f,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8f,%.8f",
                             model, ms[:rmse]..., ms[:boundary_max]...,
                             ms[:boundary_mean]..., ms[:mass_max]...,
                             ms[:mass_mean]..., ms[:lower_max]...,
                             ms[:lower_mean]..., ms[:upper_max]...,
                             ms[:upper_mean]..., tm...))
    end
end

open(joinpath(OUT, "frequency_statuses.csv"), "w") do io
    println(io, "phase,model,status,count,correction_mean,correction_max")
    for (phase, table) in [("train", train_status), ("eval", eval_status)]
        for model in MODEL_NAMES
            log = table[model]
            cmean = isempty(log.correction_norms) ? 0.0 :
                mean(log.correction_norms)
            cmax = isempty(log.correction_norms) ? 0.0 :
                maximum(log.correction_norms)
            if isempty(log.counts)
                println(io, @sprintf("%s,%s,none,0,%.8e,%.8e",
                                     phase, model, cmean, cmax))
            else
                for (status, count) in sort(collect(log.counts))
                    println(io, @sprintf("%s,%s,%s,%d,%.8e,%.8e",
                                         phase, model, status, count,
                                         cmean, cmax))
                end
            end
        end
    end
end

println("wrote frequency_ablation.csv and frequency_statuses.csv to ", OUT)
