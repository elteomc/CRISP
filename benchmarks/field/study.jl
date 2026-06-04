# M5 multi-seed fixed-grid field correction study. Writes results to results/.
# Run: julia --project=benchmarks/field benchmarks/field/study.jl
include("FieldMass.jl")
using .FieldMass
using Random, Statistics, Printf, DelimitedFiles

const K = 32
const SEEDS = 1:3
const STEPS = 250
const OUT = joinpath(@__DIR__, "results")

isdir(OUT) || mkdir(OUT)

models = ["vanilla", "soft", "projected", "weighted", "box", "positive", "bounded"]
metrics = (:rmse, :mass_max, :mass_mean, :negative_max, :negative_mean, :upper_max, :upper_mean)
acc = Dict(m => Dict(k => Float64[] for k in metrics) for m in models)
status_total = FailureCounter()
weighted_status_total = FailureCounter()
box_status_total = FailureCounter()
positive_status_total = FailureCounter()
bounded_status_total = FailureCounter()

for seed in SEEDS
    @printf("seed %d: training vanilla, soft, projected ...\n", seed)
    rng = MersenneTwister(seed)
    train_data, _, w = sample_fields(32, K; rng = rng)
    test_data, _, _ = sample_fields(16, K; rng = rng)
    p0 = init_mlp(nout = K, seed = seed)

    pv, _ = train!(p -> vanilla_loss(p, train_data), p0; steps = STEPS, lr = 1e-2)
    ps, _ = train!(p -> soft_loss(p, train_data, w; beta = 20.0), p0; steps = STEPS, lr = 1e-2)
    fc = FailureCounter()
    pp, _ = train!(p -> projected_loss(p, train_data, w; fc = fc), p0; steps = STEPS, lr = 1e-2)
    fcw = FailureCounter()
    pw, _ = train!(p -> weighted_projected_loss(p, train_data, w; fc = fcw),
                   p0; steps = STEPS, lr = 1e-2)
    fcb = FailureCounter()
    pb, _ = train!(p -> box_projected_loss(p, train_data, 0.0, 2.0; fc = fcb),
                   p0; steps = STEPS, lr = 1e-2)
    fcp = FailureCounter()
    ppos, _ = train!(p -> positive_projected_loss(p, train_data, w; fc = fcp),
                     p0; steps = STEPS, lr = 1e-2)
    fcbd = FailureCounter()
    pbd, _ = train!(p -> bounded_projected_loss(p, train_data, w, 0.0, 2.0, fc = fcbd),
                    p0, steps = STEPS, lr = 1e-2)

    for (s, c) in fc.counts
        status_total.counts[s] = get(status_total.counts, s, 0) + c
    end
    for (s, c) in fcw.counts
        weighted_status_total.counts[s] = get(weighted_status_total.counts, s, 0) + c
    end
    for (s, c) in fcb.counts
        box_status_total.counts[s] = get(box_status_total.counts, s, 0) + c
    end
    for (s, c) in fcp.counts
        positive_status_total.counts[s] = get(positive_status_total.counts, s, 0) + c
    end
    for (s, c) in fcbd.counts
        bounded_status_total.counts[s] = get(bounded_status_total.counts, s, 0) + c
    end

    evs = Dict(
        "vanilla" => evaluate_model(d -> field_model(pv, d.theta), test_data, w),
        "soft" => evaluate_model(d -> field_model(ps, d.theta), test_data, w),
        "projected" => evaluate_model(d -> projected_field(pp, d.theta, w, d.mass0), test_data, w),
        "weighted" => evaluate_model(d -> weighted_projected_field(pw, d.theta, w, d.mass0), test_data, w),
        "box" => evaluate_model(d -> box_projected_field(pb, d.theta, 0.0, 2.0), test_data, w),
        "positive" => evaluate_model(d -> positive_projected_field(ppos, d.theta, w, d.mass0), test_data, w),
        "bounded" => evaluate_model(d -> bounded_projected_field(pbd, d.theta, w, d.mass0, 0.0, 2.0), test_data, w),
    )

    for m in models, k in metrics
        push!(acc[m][k], getfield(evs[m], k))
    end
end

println("\n=== M5 fixed-grid field correction: mean +/- std over $(length(SEEDS)) seeds ===")
for m in models
    ms = Dict(k => (mean(acc[m][k]), std(acc[m][k])) for k in metrics)
    @printf("%-10s  RMSE %.5f+/-%.5f  max|dmass| %.3e+/-%.3e  maxneg %.3e+/-%.3e  maxupper %.3e+/-%.3e\n",
            m, ms[:rmse]..., ms[:mass_max]..., ms[:negative_max]..., ms[:upper_max]...)
end
println("\nprojected-model training projection statuses: ", status_total.counts)
println("weighted-projected training projection statuses: ", weighted_status_total.counts)
println("box-projected training projection statuses: ", box_status_total.counts)
println("positive-projected training projection statuses: ", positive_status_total.counts)
println("bounded-projected training projection statuses: ", bounded_status_total.counts)

open(joinpath(OUT, "results.csv"), "w") do io
    println(io, "model,rmse_mean,rmse_std,massmax_mean,massmax_std,massmean_mean,massmean_std,negmax_mean,negmax_std,negmean_mean,negmean_std,uppermax_mean,uppermax_std,uppermean_mean,uppermean_std")
    for m in models
        ms = Dict(k => (mean(acc[m][k]), std(acc[m][k])) for k in metrics)
        println(io, @sprintf("%s,%.8f,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e",
                             m, ms[:rmse]..., ms[:mass_max]..., ms[:mass_mean]...,
                             ms[:negative_max]..., ms[:negative_mean]...,
                             ms[:upper_max]..., ms[:upper_mean]...))
    end
end
println("wrote results.csv to ", OUT)
