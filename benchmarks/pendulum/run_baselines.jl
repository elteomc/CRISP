# M2 comparison: vanilla vs soft-penalty neural ODE on the pendulum energy band.
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

# Fairness (I12): both models share data, seed, architecture, and initial params.
p0 = init_mlp(seed = SEED)

println("training vanilla neural ODE ...")
pv, hv = train!(p -> vanilla_loss(p, train_data, DT), p0; steps = 400, lr = 1e-2)
println("training soft-penalty neural ODE (beta = 5) ...")
ps, hs = train!(p -> soft_loss(p, train_data, DT; beta = 5.0), p0; steps = 400, lr = 1e-2)

function report(name, p)
    rmse = Float64[]; emax = Float64[]; drift = Float64[]
    for d in test_data
        pred = rollout(p, d.z0, DT, NOBS)
        push!(rmse, traj_rmse(pred, d.traj))
        push!(emax, energy_violation(pred, d.H0).max)
        push!(drift, energy_drift(p, d, DT, NOBS * 10))   # long horizon, 10x train
    end
    @printf("%-16s  test RMSE %.4f   max |dH| %.4f   long-horizon drift %.4f\n",
            name, mean(rmse), mean(emax), mean(drift))
end

println("\n=== held-out metrics (mean over $(length(test_data)) test trajectories) ===")
report("vanilla", pv)
report("soft (beta=5)", ps)
@printf("\nfinal train loss: vanilla %.4f   soft %.4f\n", hv[end], hs[end])
