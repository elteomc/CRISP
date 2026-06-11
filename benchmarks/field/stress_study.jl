# Stress study on the harder initial-condition field family: high-frequency
# content, a sharp bump, and targets pressed against the [0, 2] box. Train-time
# correction uses the mass equality projection, which stays regular everywhere.
# Bound correction runs at evaluation only with failure_policy :continue, so
# active-bound kinks are counted as statuses instead of crashing the study and
# no training gradient is ever claimed through a kink.
# Run: julia --project=benchmarks/field benchmarks/field/stress_study.jl
include("FieldMass.jl")
include("FieldScenarios.jl")
using .FieldMass
using .FieldScenarios
using Random, Statistics, Printf, DelimitedFiles

const SCENARIO = stress_study_scenario()
const K = SCENARIO.grid
const SEEDS = SCENARIO.seeds
const STEPS = SCENARIO.steps
const OUT = SCENARIO.out
isdir(OUT) || mkdir(OUT)

const LOWER = 0.0
const UPPER = 2.0

models = ["vanilla", "soft_default", "projected", "eval_bounded",
          "eval_sparse_bounded"]
metrics = (:rmse, :mass_max, :mass_mean, :negative_max, :negative_mean,
           :upper_max, :upper_mean)
acc = Dict(m => Dict(k => Float64[] for k in metrics) for m in models)
times = Dict(m => Float64[] for m in models)
corr_means = Dict(m => Float64[] for m in
                  ["projected", "eval_bounded", "eval_sparse_bounded"])
corr_maxes = Dict(m => Float64[] for m in keys(corr_means))
train_status = FailureCounter()
eval_status = Dict(m => FailureCounter() for m in
                   ["projected", "eval_bounded", "eval_sparse_bounded"])

stat(xs) = (mean(xs), length(xs) == 1 ? 0.0 : std(xs))

function merge_counts!(dest::FailureCounter, src::FailureCounter)
    for (s, c) in src.counts
        dest.counts[s] = get(dest.counts, s, 0) + c
    end
    return dest
end

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
    @printf("seed %d: running stress-family field study ...\n", seed)
    rng = MersenneTwister(40_000 + seed)
    train_data, _, w = sample_stress_fields(SCENARIO.train_samples, K; rng = rng)
    test_data, _, _ = sample_stress_fields(SCENARIO.test_samples, K; rng = rng)
    p0 = init_mlp(nout = K, seed = 40_000 + seed)

    tv = @elapsed pv, _ = train!(p -> vanilla_loss(p, train_data), p0;
                                 steps = STEPS, lr = 1e-2)
    push!(times["vanilla"], tv)

    ts = @elapsed ps, _ = train!(p -> soft_loss(p, train_data, w;
                                                beta = SCENARIO.soft_beta), p0;
                                 steps = STEPS, lr = 1e-2)
    push!(times["soft_default"], ts)

    fc_train = FailureCounter()
    tp = @elapsed pp, _ = train!(p -> projected_loss(p, train_data, w;
                                                     fc = fc_train),
                                 p0; steps = STEPS, lr = 1e-2)
    push!(times["projected"], tp)
    merge_counts!(train_status, fc_train)
    push!(times["eval_bounded"], 0.0)
    push!(times["eval_sparse_bounded"], 0.0)

    eval_fcs = Dict(m => FailureCounter() for m in keys(eval_status))
    predictors = Dict{String,Any}(
        "projected" => d -> projected_field(pp, d.theta, w, d.mass0;
                                            fc = eval_fcs["projected"]),
        "eval_bounded" => d ->
            bounded_projected_field(pv, d.theta, w, d.mass0, LOWER, UPPER;
                                    fc = eval_fcs["eval_bounded"],
                                    failure_policy = :continue),
        "eval_sparse_bounded" => d ->
            sparse_bounded_projected_field(pv, d.theta, w, d.mass0, LOWER,
                                           UPPER;
                                           fc = eval_fcs["eval_sparse_bounded"],
                                           failure_policy = :continue),
    )
    raw_params = Dict("projected" => pp, "eval_bounded" => pv,
                      "eval_sparse_bounded" => pv)

    evs = Dict{String,Any}(
        "vanilla" => evaluate_model(d -> field_model(pv, d.theta), test_data, w),
        "soft_default" => evaluate_model(d -> field_model(ps, d.theta),
                                         test_data, w),
    )
    for m in keys(eval_status)
        evs[m] = evaluate_model(predictors[m], test_data, w)
        merge_counts!(eval_status[m], eval_fcs[m])
        cmean, cmax = correction_stats(predictors[m], raw_params[m], test_data)
        push!(corr_means[m], cmean)
        push!(corr_maxes[m], cmax)
    end

    for m in models, k in metrics
        push!(acc[m][k], getfield(evs[m], k))
    end
end

println("\n=== Stress-family field study: mean +/- std over $(length(SEEDS)) seeds ===")
for m in models
    ms = Dict(k => stat(acc[m][k]) for k in metrics)
    tm = stat(times[m])
    @printf("%-20s  RMSE %.5f+/-%.5f  max|dmass| %.3e+/-%.3e  maxneg %.3e+/-%.3e  maxupper %.3e+/-%.3e  train %.2fs+/-%.2fs\n",
            m, ms[:rmse]..., ms[:mass_max]..., ms[:negative_max]...,
            ms[:upper_max]..., tm...)
end
println("projected training statuses: ", train_status.counts)
for m in sort(collect(keys(eval_status)))
    println(m, " evaluation statuses: ", eval_status[m].counts)
end

open(joinpath(OUT, "stress_results.csv"), "w") do io
    println(io, "model,rmse_mean,rmse_std,massmax_mean,massmax_std,massmean_mean,massmean_std,negmax_mean,negmax_std,negmean_mean,negmean_std,uppermax_mean,uppermax_std,uppermean_mean,uppermean_std,corrnorm_mean,corrnorm_std,corrmax_mean,corrmax_std,train_seconds_mean,train_seconds_std")
    for m in models
        ms = Dict(k => stat(acc[m][k]) for k in metrics)
        tm = stat(times[m])
        cn = haskey(corr_means, m) ? stat(corr_means[m]) : (0.0, 0.0)
        cx = haskey(corr_maxes, m) ? stat(corr_maxes[m]) : (0.0, 0.0)
        println(io, @sprintf("%s,%.8f,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.6f,%.6f",
                             m, ms[:rmse]..., ms[:mass_max]...,
                             ms[:mass_mean]..., ms[:negative_max]...,
                             ms[:negative_mean]..., ms[:upper_max]...,
                             ms[:upper_mean]..., cn..., cx..., tm...))
    end
end

open(joinpath(OUT, "stress_statuses.csv"), "w") do io
    println(io, "phase,model,status,count")
    if isempty(train_status.counts)
        println(io, "train,projected,none,0")
    else
        for (status, count) in sort(collect(train_status.counts))
            println(io, string("train,projected,", status, ",", count))
        end
    end
    for m in sort(collect(keys(eval_status)))
        counter = eval_status[m]
        if isempty(counter.counts)
            println(io, string("eval,", m, ",none,0"))
        else
            for (status, count) in sort(collect(counter.counts))
                println(io, string("eval,", m, ",", status, ",", count))
            end
        end
    end
end

println("wrote stress_results.csv and stress_statuses.csv to ", OUT)
