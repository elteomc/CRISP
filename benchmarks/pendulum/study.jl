# M4 comparison study: vanilla vs a soft-penalty sweep vs projected across seeds,
# with the full metric suite, runtime, and per-status failure rates (I10/I12).
# Defaults are paper scale and overridable through STRUCTPINN_PENDULUM_* settings
# in PendulumScenarios.jl. Writes results to results/ (gitignored, reproducible).
# Run: julia --project=benchmarks/pendulum benchmarks/pendulum/study.jl
include("Pendulum.jl")
include("PendulumScenarios.jl")
using .Pendulum
using .PendulumScenarios
using Random, Statistics, Printf, DelimitedFiles

const SCENARIO = study_scenario()
const DT = SCENARIO.dt
const NOBS = SCENARIO.nobs
const NLONG = SCENARIO.nlong
const SEEDS = SCENARIO.seeds
const STEPS = SCENARIO.steps
const SOFT_CONFIGS = SCENARIO.soft_configs
const OUT = SCENARIO.out
isdir(OUT) || mkdir(OUT)

soft_names = [c.name for c in SOFT_CONFIGS]
betas = Dict(c.name => c.beta for c in SOFT_CONFIGS)
curve_soft = "soft_default" in soft_names ? "soft_default" : first(soft_names)
models = vcat(["vanilla"], soft_names, ["projected"])
metrics = (:rmse, :emax, :edrift, :rmse_long)
acc = Dict(m => Dict(k => Float64[] for k in metrics) for m in models)
times = Dict(m => Float64[] for m in models)
status_total = FailureCounter()
train_status_total = FailureCounter()
curves = nothing
energy_t = nothing

stat(xs) = (mean(xs), length(xs) == 1 ? 0.0 : std(xs))

for seed in SEEDS
    @printf("seed %d: training vanilla, soft sweep, projected ...\n", seed)
    rng = MersenneTwister(seed)
    train_data = sample_band(SCENARIO.train_samples, DT, NOBS; rng = rng)
    test_data  = sample_band(SCENARIO.test_samples, DT, NOBS; rng = rng)
    p0 = init_mlp(seed = seed)

    tv = @elapsed pv, hv = train!(p -> vanilla_loss(p, train_data, DT), p0;
                                  steps = STEPS, lr = 1e-2)
    push!(times["vanilla"], tv)

    soft_params = Dict{String,Any}()
    soft_hist = Dict{String,Any}()
    for c in SOFT_CONFIGS
        ts = @elapsed ps, hs = train!(p -> soft_loss(p, train_data, DT;
                                                     beta = c.beta), p0;
                                      steps = STEPS, lr = 1e-2)
        push!(times[c.name], ts)
        soft_params[c.name] = ps
        soft_hist[c.name] = hs
    end

    fc_train = FailureCounter()
    tp = @elapsed pp, hp = train!(p -> projected_loss(p, train_data, DT;
                                                      fc = fc_train),
                                  p0; steps = STEPS, lr = 1e-2)
    push!(times["projected"], tp)
    for (s, c) in fc_train.counts
        train_status_total.counts[s] = get(train_status_total.counts, s, 0) + c
    end

    fc = FailureCounter()
    evs = Dict{String,Any}(
        "vanilla"   => evaluate_model((d, n) -> rollout(pv, d.z0, DT, n),
                                      test_data, DT, NOBS, NLONG),
        "projected" => evaluate_model((d, n) ->
                           projected_rollout_status!(pp, d.z0, DT, n, d.H0, fc),
                                      test_data, DT, NOBS, NLONG),
    )
    for c in SOFT_CONFIGS
        ps = soft_params[c.name]
        evs[c.name] = evaluate_model((d, n) -> rollout(ps, d.z0, DT, n),
                                     test_data, DT, NOBS, NLONG)
    end
    for (s, c) in fc.counts
        status_total.counts[s] = get(status_total.counts, s, 0) + c
    end
    for m in models, k in metrics
        push!(acc[m][k], getfield(evs[m], k))
    end

    if seed == first(SEEDS)
        global curves = hcat(collect(1:STEPS), hv, soft_hist[curve_soft], hp)
        d = test_data[1]
        ev_of(traj) = [H(traj[:, k]) for k in 1:size(traj, 2)]
        tv_traj = rollout(pv, d.z0, DT, NLONG)
        ts_traj = rollout(soft_params[curve_soft], d.z0, DT, NLONG)
        tp_traj = projected_rollout_status!(pp, d.z0, DT, NLONG, d.H0,
                                            FailureCounter())
        global energy_t = hcat(collect(0:NLONG), fill(d.H0, NLONG + 1),
                               ev_of(tv_traj), ev_of(ts_traj), ev_of(tp_traj))
    end
end

# ---- results table (mean +/- std across seeds) ----
println("\n=== M4 comparison: mean +/- std over $(length(SEEDS)) seeds ===")
for m in models
    ms = Dict(k => stat(acc[m][k]) for k in metrics)
    tm = stat(times[m])
    @printf("%-13s  RMSE %.4f+/-%.4f  max|dH| %.4f+/-%.4f  drift_long %.4f+/-%.4f  trajRMSE_long %.4f+/-%.4f  train %.2fs+/-%.2fs\n",
            m, ms[:rmse]..., ms[:emax]..., ms[:edrift]..., ms[:rmse_long]..., tm...)
end
println("\nprojected-model training projection statuses (all seeds): ", train_status_total.counts)
println("projected-model evaluation projection statuses (all seeds): ", status_total.counts)

# ---- write artifacts ----
open(joinpath(OUT, "results.csv"), "w") do io
    println(io, "model,beta,rmse_mean,rmse_std,emax_mean,emax_std,edrift_mean,edrift_std,rmselong_mean,rmselong_std,train_seconds_mean,train_seconds_std")
    for m in models
        ms = Dict(k => stat(acc[m][k]) for k in metrics)
        tm = stat(times[m])
        beta = haskey(betas, m) ? @sprintf("%.4f", betas[m]) : ""
        println(io, @sprintf("%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                             m, beta, ms[:rmse]..., ms[:emax]..., ms[:edrift]...,
                             ms[:rmse_long]..., tm...))
    end
end

open(joinpath(OUT, "statuses.csv"), "w") do io
    println(io, "phase,model,status,count")
    for (phase, counter) in [("train", train_status_total),
                             ("eval", status_total)]
        if isempty(counter.counts)
            println(io, string(phase, ",projected,none,0"))
        else
            for (status, count) in sort(collect(counter.counts))
                println(io, string(phase, ",projected,", status, ",", count))
            end
        end
    end
end

writedlm(joinpath(OUT, "training_curves.csv"),
         vcat(["step" "vanilla" curve_soft "projected"], curves), ',')
writedlm(joinpath(OUT, "energy_over_time.csv"),
         vcat(["t" "H0" "vanilla" curve_soft "projected"], energy_t), ',')
println("wrote results.csv, statuses.csv, training_curves.csv, energy_over_time.csv to ", OUT)
