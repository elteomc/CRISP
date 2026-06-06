# Constraint-family ablation for the DeepONet helper benchmark.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/constraint_ablation.jl
include("DeepONetHeat.jl")
using .DeepONetHeat
using Random, Statistics, Printf

const K = 32
const SEEDS = 1:5
const STEPS = 80
const OUT = joinpath(@__DIR__, "results")

isdir(OUT) || mkdir(OUT)

const TRAIN_MODES = (:boundary_only, :mass_only, :boundary_box, :full)
const EVAL_MODES = (:boundary_only, :box_only, :mass_only, :boundary_box,
                    :full)
const MODEL_NAMES = vcat(["vanilla"], ["hard_$(m)" for m in TRAIN_MODES],
                         ["eval_$(m)" for m in EVAL_MODES])
const METRICS = (:rmse, :boundary_max, :boundary_mean, :mass_max, :mass_mean,
                 :lower_max, :lower_mean, :upper_max, :upper_mean)

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

function warmup_constraints()
    rng = MersenneTwister(92_000)
    data, x, w = sample_heat_operator(2, K, rng = rng)
    p = init_deeponet(seed = 92_000)
    train!(pp -> vanilla_loss(pp, data, x), p, steps = 1, lr = 8e-3)
    for mode in TRAIN_MODES
        train!(pp -> hard_loss(pp, data, x, w, mode = mode), p,
               steps = 1, lr = 8e-3)
    end
    return nothing
end

warmup_constraints()

acc = Dict(m => Dict(k => Float64[] for k in METRICS) for m in MODEL_NAMES)
times = Dict(m => Float64[] for m in MODEL_NAMES)
train_status = Dict(m => ProjectionLog() for m in MODEL_NAMES)
eval_status = Dict(m => ProjectionLog() for m in MODEL_NAMES)

for seed in SEEDS
    @printf("seed %d: running constraint-family ablation ...\n", seed)
    rng = MersenneTwister(30_000 + seed)
    train_data, x, w = sample_heat_operator(18, K, rng = rng)
    test_data, _, _ = sample_heat_operator(8, K, rng = rng)
    p0 = init_deeponet(seed = 30_000 + seed)

    tv = @elapsed pv, _ = train!(p -> vanilla_loss(p, train_data, x), p0,
                                 steps = STEPS, lr = 8e-3)
    push!(times["vanilla"], tv)
    add_eval!(acc, "vanilla",
              evaluate_model(d -> deeponet_model(pv, d.theta, x),
                             test_data, w))

    for mode in TRAIN_MODES
        hard_name = "hard_$(mode)"

        log_train = ProjectionLog()
        elapsed = @elapsed ph, _ = train!(p -> hard_loss(p, train_data, x,
                                                         w, log = log_train,
                                                         mode = mode),
                                          p0, steps = STEPS, lr = 8e-3)
        push!(times[hard_name], elapsed)
        merge_log!(train_status[hard_name], log_train)

        log_hard_eval = ProjectionLog()
        add_eval!(acc, hard_name,
                  evaluate_model(d -> hard_projected_field(ph, d, x, w,
                                                           log =
                                                               log_hard_eval,
                                                           mode = mode),
                                 test_data, w))
        merge_log!(eval_status[hard_name], log_hard_eval)
    end

    for mode in EVAL_MODES
        eval_name = "eval_$(mode)"
        log_eval_only = ProjectionLog()
        add_eval!(acc, eval_name,
                  evaluate_model(d -> hard_projected_field(pv, d, x, w,
                                                           log =
                                                               log_eval_only,
                                                           mode = mode,
                                                           failure_policy =
                                                               :continue),
                                 test_data, w))
        merge_log!(eval_status[eval_name], log_eval_only)
        push!(times[eval_name], 0.0)
    end
end

open(joinpath(OUT, "constraint_ablation.csv"), "w") do io
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

open(joinpath(OUT, "constraint_statuses.csv"), "w") do io
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

println("wrote constraint_ablation.csv and constraint_statuses.csv to ", OUT)
