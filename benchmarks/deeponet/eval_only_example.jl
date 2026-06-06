# Evaluation-only correction example for a trained DeepONet-style surrogate.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/eval_only_example.jl
include("DeepONetHeat.jl")
include("DeepONetScenarios.jl")
using .DeepONetHeat
using .DeepONetScenarios
using Random, Printf
using Statistics

const SCENARIO = eval_only_scenario()
const OUT = SCENARIO.out
isdir(OUT) || mkdir(OUT)

const K = SCENARIO.grid
const STEPS = SCENARIO.steps
const TRAIN_SAMPLES = SCENARIO.train_samples
const TEST_SAMPLES = SCENARIO.test_samples

rng = MersenneTwister(SCENARIO.seed)
train_data, x, w = sample_heat_operator(TRAIN_SAMPLES, K, rng = rng)
test_data, _, _ = sample_heat_operator(TEST_SAMPLES, K, rng = rng)
p0 = init_deeponet(seed = SCENARIO.seed)

println("training vanilla DeepONet surrogate ...")
p, hist = train!(pp -> vanilla_loss(pp, train_data, x), p0,
                 steps = STEPS, lr = 8e-3)

contexts = heat_correction_contexts(test_data, w, mode = :full,
                                    cached = true)
log = ProjectionLog()

raw = evaluate_model(d -> deeponet_model(p, d.theta, x), test_data, w)
corrected = evaluate_context_model(ctx -> context_projected_field(p, ctx, x,
                                                                  log = log),
                                   contexts)

open(joinpath(OUT, "eval_only_example.csv"), "w") do io
    println(io, "model,rmse,boundary_max,boundary_mean,mass_max,mass_mean,lower_max,lower_mean,upper_max,upper_mean,status_success,correction_mean,correction_max")
    for (name, ev, success, cmean, cmax) in [
        ("raw", raw, 0, 0.0, 0.0),
        ("corrected", corrected, get(log.counts, :success, 0),
         mean(log.correction_norms), maximum(log.correction_norms)),
    ]
        println(io, @sprintf("%s,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%d,%.8e,%.8e",
                             name, ev.rmse, ev.boundary_max,
                             ev.boundary_mean, ev.mass_max, ev.mass_mean,
                             ev.lower_max, ev.lower_mean, ev.upper_max,
                             ev.upper_mean, success, cmean, cmax))
    end
end

@printf("raw RMSE %.5f, boundary %.3e, mass %.3e\n",
        raw.rmse, raw.boundary_max, raw.mass_max)
@printf("corrected RMSE %.5f, boundary %.3e, mass %.3e, statuses %s\n",
        corrected.rmse, corrected.boundary_max, corrected.mass_max,
        string(log.counts))
println("wrote eval_only_example.csv to ", OUT)
