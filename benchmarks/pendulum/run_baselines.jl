# M2/M3 comparison: vanilla vs soft-penalty vs hard-constraint (projected)
# neural ODE on the pendulum energy band.
# Run: julia --project=benchmarks/pendulum benchmarks/pendulum/run_baselines.jl
include("Pendulum.jl")
using .Pendulum
using Random, Statistics, Printf

const DT = 0.1
const NOBS = 20          # training horizon, t up to 2.0
const SEED = 0

rng = MersenneTwister(SEED)
train_data = sample_band(16, DT, NOBS; rng = rng)
test_data  = sample_band(8, DT, NOBS; rng = rng)

# Fairness (I12): all models share data, seed, architecture, and initial params.
p0 = init_mlp(seed = SEED)

println("training vanilla neural ODE ...")
pv, _ = train!(p -> vanilla_loss(p, train_data, DT), p0; steps = 400, lr = 1e-2)
println("training soft-penalty neural ODE (beta = 5) ...")
ps, _ = train!(p -> soft_loss(p, train_data, DT; beta = 5.0), p0; steps = 400, lr = 1e-2)
println("training projected (hard-constraint) neural ODE ...")
pp, _ = train!(p -> projected_loss(p, train_data, DT), p0; steps = 400, lr = 1e-2)

# `predict(d, nsteps) -> trajectory` lets each model use its own rollout.
function report(name, predict)
    rmse = Float64[]; emax = Float64[]; drift = Float64[]
    for d in test_data
        pred = predict(d, NOBS)
        push!(rmse, traj_rmse(pred, d.traj))
        push!(emax, energy_violation(pred, d.H0).max)
        long = predict(d, NOBS * 10)                       # long horizon, 10x train
        push!(drift, abs(H(long[:, end]) - d.H0))
    end
    @printf("%-16s  test RMSE %.4f   max |dH| %.4f   long-horizon drift %.4f\n",
            name, mean(rmse), mean(emax), mean(drift))
end

fc = FailureCounter()
println("\n=== held-out metrics (mean over $(length(test_data)) test trajectories) ===")
report("vanilla",       (d, n) -> rollout(pv, d.z0, DT, n))
report("soft (beta=5)", (d, n) -> rollout(ps, d.z0, DT, n))
report("projected",     (d, n) -> projected_rollout_status!(pp, d.z0, DT, n, d.H0, fc))
println("\nprojected-model projection statuses: ", fc.counts)
