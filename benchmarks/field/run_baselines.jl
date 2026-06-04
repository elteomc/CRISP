# M5 comparison: vanilla vs soft-penalty vs affine-projected fixed-grid fields.
# Run: julia --project=benchmarks/field benchmarks/field/run_baselines.jl
include("FieldMass.jl")
using .FieldMass
using Random, Statistics, Printf

const K = 32
const SEED = 0
const STEPS = 300

rng = MersenneTwister(SEED)
train_data, _, w = sample_fields(32, K; rng = rng)
test_data, _, _ = sample_fields(16, K; rng = rng)
p0 = init_mlp(nout = K, seed = SEED)

println("training vanilla field model ...")
pv, _ = train!(p -> vanilla_loss(p, train_data), p0; steps = STEPS, lr = 1e-2)
println("training soft-penalty field model (beta = 20) ...")
ps, _ = train!(p -> soft_loss(p, train_data, w; beta = 20.0), p0; steps = STEPS, lr = 1e-2)
println("training projected field model ...")
fc_train = FailureCounter()
pp, _ = train!(p -> projected_loss(p, train_data, w; fc = fc_train), p0; steps = STEPS, lr = 1e-2)
println("training weighted projected field model ...")
fc_weighted = FailureCounter()
pw, _ = train!(p -> weighted_projected_loss(p, train_data, w; fc = fc_weighted),
               p0; steps = STEPS, lr = 1e-2)
println("training box projected field model ...")
fc_box = FailureCounter()
pb, _ = train!(p -> box_projected_loss(p, train_data, 0.0, 2.0; fc = fc_box),
               p0; steps = STEPS, lr = 1e-2)
println("training positive projected field model ...")
fc_pos = FailureCounter()
ppos, _ = train!(p -> positive_projected_loss(p, train_data, w; fc = fc_pos),
                 p0; steps = STEPS, lr = 1e-2)
println("training bounded mass projected field model ...")
fc_bounded = FailureCounter()
pbd, _ = train!(p -> bounded_projected_loss(p, train_data, w, 0.0, 2.0, fc = fc_bounded),
                p0, steps = STEPS, lr = 1e-2)

function report(name, predict)
    ev = evaluate_model(predict, test_data, w)
    @printf("%-14s  RMSE %.5f   max |dmass| %.3e   max negativity %.3e   max upper %.3e\n",
            name, ev.rmse, ev.mass_max, ev.negative_max, ev.upper_max)
end

println("\n=== held-out fixed-grid field metrics ===")
report("vanilla", d -> field_model(pv, d.theta))
report("soft", d -> field_model(ps, d.theta))
report("projected", d -> projected_field(pp, d.theta, w, d.mass0))
report("weighted", d -> weighted_projected_field(pw, d.theta, w, d.mass0))
report("box", d -> box_projected_field(pb, d.theta, 0.0, 2.0))
report("positive", d -> positive_projected_field(ppos, d.theta, w, d.mass0))
report("bounded", d -> bounded_projected_field(pbd, d.theta, w, d.mass0, 0.0, 2.0))
println("\nprojected-model training projection statuses: ", fc_train.counts)
println("weighted-projected training projection statuses: ", fc_weighted.counts)
println("box-projected training projection statuses: ", fc_box.counts)
println("positive-projected training projection statuses: ", fc_pos.counts)
println("bounded-projected training projection statuses: ", fc_bounded.counts)
