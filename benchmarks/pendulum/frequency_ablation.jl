# Projection-frequency ablation for the pendulum benchmark: no correction,
# evaluation-only correction, and train-time correction every 1, 2, or 5 rollout
# steps. Every corrected variant is evaluated with full per-step projection.
# Run: julia --project=benchmarks/pendulum benchmarks/pendulum/frequency_ablation.jl
include("Pendulum.jl")
include("PendulumScenarios.jl")
using .Pendulum
using .PendulumScenarios
using Random, Statistics, Printf, DelimitedFiles

const SCENARIO = frequency_ablation_scenario()
const DT = SCENARIO.dt
const NOBS = SCENARIO.nobs
const NLONG = SCENARIO.nlong
const SEEDS = SCENARIO.seeds
const STEPS = SCENARIO.steps
const PERIODS = SCENARIO.periods
const OUT = SCENARIO.out
isdir(OUT) || mkdir(OUT)

period_names = [string(name) for name in keys(PERIODS)]
const MODEL_NAMES = vcat(["no_correction", "eval_only"], period_names)
metrics = (:rmse, :emax, :edrift, :rmse_long)
acc = Dict(m => Dict(k => Float64[] for k in metrics) for m in MODEL_NAMES)
times = Dict(m => Float64[] for m in MODEL_NAMES)
train_status = Dict(m => FailureCounter() for m in MODEL_NAMES)
eval_status = Dict(m => FailureCounter() for m in MODEL_NAMES)

stat(xs) = (mean(xs), length(xs) == 1 ? 0.0 : std(xs))

function merge_counts!(dest::FailureCounter, src::FailureCounter)
    for (s, c) in src.counts
        dest.counts[s] = get(dest.counts, s, 0) + c
    end
    return dest
end

function add_eval!(model, ev)
    for k in metrics
        push!(acc[model][k], getfield(ev, k))
    end
    return nothing
end

for seed in SEEDS
    @printf("seed %d: running pendulum projection-frequency ablation ...\n", seed)
    rng = MersenneTwister(30_000 + seed)
    train_data = sample_band(SCENARIO.train_samples, DT, NOBS; rng = rng)
    test_data  = sample_band(SCENARIO.test_samples, DT, NOBS; rng = rng)
    p0 = init_mlp(seed = 30_000 + seed)

    tv = @elapsed pv, _ = train!(p -> vanilla_loss(p, train_data, DT), p0;
                                 steps = STEPS, lr = 1e-2)
    push!(times["no_correction"], tv)
    push!(times["eval_only"], 0.0)

    add_eval!("no_correction",
              evaluate_model((d, n) -> rollout(pv, d.z0, DT, n),
                             test_data, DT, NOBS, NLONG))

    fc_eval_only = FailureCounter()
    add_eval!("eval_only",
              evaluate_model((d, n) ->
                  projected_rollout_status!(pv, d.z0, DT, n, d.H0, fc_eval_only),
                             test_data, DT, NOBS, NLONG))
    merge_counts!(eval_status["eval_only"], fc_eval_only)

    for name in keys(PERIODS)
        period = PERIODS[name]
        model = string(name)
        fc_train = FailureCounter()
        elapsed = @elapsed pp, _ = train!(p -> projected_loss(p, train_data, DT;
                                                              fc = fc_train,
                                                              project_every = period),
                                          p0; steps = STEPS, lr = 1e-2)
        push!(times[model], elapsed)
        merge_counts!(train_status[model], fc_train)

        fc_eval = FailureCounter()
        add_eval!(model,
                  evaluate_model((d, n) ->
                      projected_rollout_status!(pp, d.z0, DT, n, d.H0, fc_eval),
                                 test_data, DT, NOBS, NLONG))
        merge_counts!(eval_status[model], fc_eval)
    end
end

println("\n=== Pendulum projection-frequency ablation: mean +/- std over $(length(SEEDS)) seeds ===")
for m in MODEL_NAMES
    ms = Dict(k => stat(acc[m][k]) for k in metrics)
    tm = stat(times[m])
    @printf("%-14s  RMSE %.4f+/-%.4f  max|dH| %.4f+/-%.4f  drift_long %.4f+/-%.4f  trajRMSE_long %.4f+/-%.4f  train %.2fs+/-%.2fs\n",
            m, ms[:rmse]..., ms[:emax]..., ms[:edrift]..., ms[:rmse_long]..., tm...)
end

open(joinpath(OUT, "frequency_results.csv"), "w") do io
    println(io, "model,rmse_mean,rmse_std,emax_mean,emax_std,edrift_mean,edrift_std,rmselong_mean,rmselong_std,train_seconds_mean,train_seconds_std")
    for m in MODEL_NAMES
        ms = Dict(k => stat(acc[m][k]) for k in metrics)
        tm = stat(times[m])
        println(io, @sprintf("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                             m, ms[:rmse]..., ms[:emax]..., ms[:edrift]...,
                             ms[:rmse_long]..., tm...))
    end
end

open(joinpath(OUT, "frequency_statuses.csv"), "w") do io
    println(io, "phase,model,status,count")
    for (phase, table) in [("train", train_status), ("eval", eval_status)]
        for m in MODEL_NAMES
            counter = table[m]
            if isempty(counter.counts)
                println(io, string(phase, ",", m, ",none,0"))
            else
                for (status, count) in sort(collect(counter.counts))
                    println(io, string(phase, ",", m, ",", status, ",", count))
                end
            end
        end
    end
end

println("wrote frequency_results.csv and frequency_statuses.csv to ", OUT)
