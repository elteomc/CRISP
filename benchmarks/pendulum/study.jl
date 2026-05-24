# M4 comparison study: vanilla vs soft vs projected across seeds, with the full
# metric suite and per-status failure rates (PLAN.md M4, I10/I12). Writes a
# results table and plot-data CSVs to results/ (gitignored, reproducible).
# Run: julia --project=benchmarks/pendulum benchmarks/pendulum/study.jl
include("Pendulum.jl")
using .Pendulum
using Random, Statistics, Printf, DelimitedFiles

const DT = 0.1
const NOBS = 20
const NLONG = 200          # long horizon, 10x training
const SEEDS = 1:3
const STEPS = 300

const OUT = joinpath(@__DIR__, "results")
isdir(OUT) || mkdir(OUT)

models = ["vanilla", "soft", "projected"]
metrics = (:rmse, :emax, :edrift, :rmse_long)
acc = Dict(m => Dict(k => Float64[] for k in metrics) for m in models)
status_total = FailureCounter()
curves = nothing
energy_t = nothing

for seed in SEEDS
    @printf("seed %d: training vanilla, soft, projected ...\n", seed)
    rng = MersenneTwister(seed)
    train_data = sample_band(12, DT, NOBS; rng = rng)
    test_data  = sample_band(8, DT, NOBS; rng = rng)
    p0 = init_mlp(seed = seed)

    pv, hv = train!(p -> vanilla_loss(p, train_data, DT), p0; steps = STEPS, lr = 1e-2)
    ps, hs = train!(p -> soft_loss(p, train_data, DT; beta = 5.0), p0; steps = STEPS, lr = 1e-2)
    pp, hp = train!(p -> projected_loss(p, train_data, DT), p0; steps = STEPS, lr = 1e-2)

    fc = FailureCounter()
    evs = Dict(
        "vanilla"   => evaluate_model((d, n) -> rollout(pv, d.z0, DT, n), test_data, DT, NOBS, NLONG),
        "soft"      => evaluate_model((d, n) -> rollout(ps, d.z0, DT, n), test_data, DT, NOBS, NLONG),
        "projected" => evaluate_model((d, n) -> projected_rollout_status!(pp, d.z0, DT, n, d.H0, fc),
                                      test_data, DT, NOBS, NLONG),
    )
    for (s, c) in fc.counts
        status_total.counts[s] = get(status_total.counts, s, 0) + c
    end
    for m in models, k in metrics
        push!(acc[m][k], getfield(evs[m], k))
    end

    if seed == first(SEEDS)
        global curves = hcat(collect(1:STEPS), hv, hs, hp)
        d = test_data[1]
        ev_of(traj) = [H(traj[:, k]) for k in 1:size(traj, 2)]
        tv = rollout(pv, d.z0, DT, NLONG)
        ts = rollout(ps, d.z0, DT, NLONG)
        tp = projected_rollout_status!(pp, d.z0, DT, NLONG, d.H0, FailureCounter())
        global energy_t = hcat(collect(0:NLONG), fill(d.H0, NLONG + 1), ev_of(tv), ev_of(ts), ev_of(tp))
    end
end

# ---- results table (mean +/- std across seeds) ----
println("\n=== M4 comparison: mean +/- std over $(length(SEEDS)) seeds ===")
for m in models
    ms = Dict(k => (mean(acc[m][k]), std(acc[m][k])) for k in metrics)
    @printf("%-10s  RMSE %.4f+/-%.4f  max|dH| %.4f+/-%.4f  drift_long %.4f+/-%.4f  trajRMSE_long %.4f+/-%.4f\n",
            m, ms[:rmse]..., ms[:emax]..., ms[:edrift]..., ms[:rmse_long]...)
end
println("\nprojected-model projection statuses (all seeds): ", status_total.counts)

# ---- write artifacts ----
open(joinpath(OUT, "results.csv"), "w") do io
    println(io, "model,rmse_mean,rmse_std,emax_mean,emax_std,edrift_mean,edrift_std,rmselong_mean,rmselong_std")
    for m in models
        ms = Dict(k => (mean(acc[m][k]), std(acc[m][k])) for k in metrics)
        println(io, @sprintf("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                             m, ms[:rmse]..., ms[:emax]..., ms[:edrift]..., ms[:rmse_long]...))
    end
end
writedlm(joinpath(OUT, "training_curves.csv"),
         vcat(["step" "vanilla" "soft" "projected"], curves), ',')
writedlm(joinpath(OUT, "energy_over_time.csv"),
         vcat(["t" "H0" "vanilla" "soft" "projected"], energy_t), ',')
println("wrote results.csv, training_curves.csv, energy_over_time.csv to ", OUT)
