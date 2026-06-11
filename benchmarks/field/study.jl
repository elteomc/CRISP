# M5 multi-seed fixed-grid field correction study with a soft-penalty sweep,
# runtime, correction norms, and per-status counts. Defaults are paper scale and
# overridable through STRUCTPINN_FIELD_* settings in FieldScenarios.jl.
# Run: julia --project=benchmarks/field benchmarks/field/study.jl
include("FieldMass.jl")
include("FieldScenarios.jl")
using .FieldMass
using .FieldScenarios
using Random, Statistics, Printf, DelimitedFiles

const SCENARIO = study_scenario()
const K = SCENARIO.grid
const SEEDS = SCENARIO.seeds
const STEPS = SCENARIO.steps
const SOFT_CONFIGS = SCENARIO.soft_configs
const OUT = SCENARIO.out
isdir(OUT) || mkdir(OUT)

const LOWER = 0.0
const UPPER = 2.0

soft_names = [c.name for c in SOFT_CONFIGS]
betas = Dict(c.name => c.beta for c in SOFT_CONFIGS)
projected_models = ["projected", "weighted", "box", "positive", "bounded",
                    "sparse_bounded"]
models = vcat(["vanilla"], soft_names, projected_models)
metrics = (:rmse, :mass_max, :mass_mean, :negative_max, :negative_mean,
           :upper_max, :upper_mean)
acc = Dict(m => Dict(k => Float64[] for k in metrics) for m in models)
times = Dict(m => Float64[] for m in models)
corr_means = Dict(m => Float64[] for m in projected_models)
corr_maxes = Dict(m => Float64[] for m in projected_models)
train_status = Dict(m => FailureCounter() for m in projected_models)
eval_status = Dict(m => FailureCounter() for m in projected_models)

stat(xs) = (mean(xs), length(xs) == 1 ? 0.0 : std(xs))

function merge_counts!(dest::FailureCounter, src::FailureCounter)
    for (s, c) in src.counts
        dest.counts[s] = get(dest.counts, s, 0) + c
    end
    return dest
end

# Mean and max Euclidean distance between the corrected prediction and the raw
# model output over the test set, so large successful corrections stay visible.
function correction_stats(predict, params, test_data)
    norms = Float64[]
    for d in test_data
        corrected = predict(d)
        raw = field_model(params, d.theta)
        push!(norms, sqrt(sum(abs2, corrected .- raw)))
    end
    return mean(norms), maximum(norms)
end

for seed in SEEDS
    @printf("seed %d: training vanilla, soft sweep, projected variants ...\n", seed)
    rng = MersenneTwister(seed)
    train_data, _, w = sample_fields(SCENARIO.train_samples, K; rng = rng)
    test_data, _, _ = sample_fields(SCENARIO.test_samples, K; rng = rng)
    p0 = init_mlp(nout = K, seed = seed)

    tv = @elapsed pv, _ = train!(p -> vanilla_loss(p, train_data), p0;
                                 steps = STEPS, lr = 1e-2)
    push!(times["vanilla"], tv)

    soft_params = Dict{String,Any}()
    for c in SOFT_CONFIGS
        ts = @elapsed ps, _ = train!(p -> soft_loss(p, train_data, w;
                                                    beta = c.beta), p0;
                                     steps = STEPS, lr = 1e-2)
        push!(times[c.name], ts)
        soft_params[c.name] = ps
    end

    fcs = Dict(m => FailureCounter() for m in projected_models)
    trainers = Dict{String,Any}(
        "projected" => p -> projected_loss(p, train_data, w;
                                           fc = fcs["projected"]),
        "weighted" => p -> weighted_projected_loss(p, train_data, w;
                                                   fc = fcs["weighted"]),
        "box" => p -> box_projected_loss(p, train_data, LOWER, UPPER;
                                         fc = fcs["box"]),
        "positive" => p -> positive_projected_loss(p, train_data, w;
                                                   fc = fcs["positive"]),
        "bounded" => p -> bounded_projected_loss(p, train_data, w, LOWER,
                                                 UPPER; fc = fcs["bounded"]),
        "sparse_bounded" => p ->
            sparse_bounded_projected_loss(p, train_data, w, LOWER, UPPER;
                                          fc = fcs["sparse_bounded"]),
    )
    trained = Dict{String,Any}()
    for m in projected_models
        elapsed = @elapsed begin
            trained[m], _ = train!(trainers[m], p0; steps = STEPS, lr = 1e-2)
        end
        push!(times[m], elapsed)
        merge_counts!(train_status[m], fcs[m])
    end

    eval_fcs = Dict(m => FailureCounter() for m in projected_models)
    predictors = Dict{String,Any}(
        "projected" => d -> projected_field(trained["projected"], d.theta, w,
                                            d.mass0;
                                            fc = eval_fcs["projected"]),
        "weighted" => d -> weighted_projected_field(trained["weighted"],
                                                    d.theta, w, d.mass0;
                                                    fc = eval_fcs["weighted"]),
        "box" => d -> box_projected_field(trained["box"], d.theta, LOWER,
                                          UPPER; fc = eval_fcs["box"]),
        "positive" => d -> positive_projected_field(trained["positive"],
                                                    d.theta, w, d.mass0;
                                                    fc = eval_fcs["positive"]),
        "bounded" => d -> bounded_projected_field(trained["bounded"], d.theta,
                                                  w, d.mass0, LOWER, UPPER;
                                                  fc = eval_fcs["bounded"]),
        "sparse_bounded" => d ->
            sparse_bounded_projected_field(trained["sparse_bounded"], d.theta,
                                           w, d.mass0, LOWER, UPPER;
                                           fc = eval_fcs["sparse_bounded"]),
    )

    evs = Dict{String,Any}(
        "vanilla" => evaluate_model(d -> field_model(pv, d.theta), test_data, w),
    )
    for c in SOFT_CONFIGS
        ps = soft_params[c.name]
        evs[c.name] = evaluate_model(d -> field_model(ps, d.theta), test_data, w)
    end
    for m in projected_models
        evs[m] = evaluate_model(predictors[m], test_data, w)
        merge_counts!(eval_status[m], eval_fcs[m])
        cmean, cmax = correction_stats(predictors[m], trained[m], test_data)
        push!(corr_means[m], cmean)
        push!(corr_maxes[m], cmax)
    end

    for m in models, k in metrics
        push!(acc[m][k], getfield(evs[m], k))
    end
end

println("\n=== M5 fixed-grid field correction: mean +/- std over $(length(SEEDS)) seeds ===")
for m in models
    ms = Dict(k => stat(acc[m][k]) for k in metrics)
    tm = stat(times[m])
    @printf("%-14s  RMSE %.5f+/-%.5f  max|dmass| %.3e+/-%.3e  maxneg %.3e+/-%.3e  maxupper %.3e+/-%.3e  train %.2fs+/-%.2fs\n",
            m, ms[:rmse]..., ms[:mass_max]..., ms[:negative_max]...,
            ms[:upper_max]..., tm...)
end
for m in projected_models
    println(m, " training statuses: ", train_status[m].counts,
            "  evaluation statuses: ", eval_status[m].counts)
end

open(joinpath(OUT, "results.csv"), "w") do io
    println(io, "model,beta,rmse_mean,rmse_std,massmax_mean,massmax_std,massmean_mean,massmean_std,negmax_mean,negmax_std,negmean_mean,negmean_std,uppermax_mean,uppermax_std,uppermean_mean,uppermean_std,corrnorm_mean,corrnorm_std,corrmax_mean,corrmax_std,train_seconds_mean,train_seconds_std")
    for m in models
        ms = Dict(k => stat(acc[m][k]) for k in metrics)
        tm = stat(times[m])
        beta = haskey(betas, m) ? @sprintf("%.4f", betas[m]) : ""
        cn = m in projected_models ? stat(corr_means[m]) : (0.0, 0.0)
        cx = m in projected_models ? stat(corr_maxes[m]) : (0.0, 0.0)
        println(io, @sprintf("%s,%s,%.8f,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.6f,%.6f",
                             m, beta, ms[:rmse]..., ms[:mass_max]...,
                             ms[:mass_mean]..., ms[:negative_max]...,
                             ms[:negative_mean]..., ms[:upper_max]...,
                             ms[:upper_mean]..., cn..., cx..., tm...))
    end
end

open(joinpath(OUT, "statuses.csv"), "w") do io
    println(io, "phase,model,status,count")
    for (phase, table) in [("train", train_status), ("eval", eval_status)]
        for m in projected_models
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

println("wrote results.csv and statuses.csv to ", OUT)
