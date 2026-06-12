# Mechanism study for the headline result (headline_result.md). Three
# predictions, three parts:
#   1. Non-expansiveness: with feasible targets, evaluation-only full
#      correction must reduce the error of every individual test sample.
#   2. Low data: the train-time hard-correction advantage over vanilla should
#      grow as the training set shrinks.
#   3. Constraint noise: perturbing the constraint values away from the data
#      (the realistic deployment case) should erode the correction gain, with
#      evaluation-only correction degrading first.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/mechanism_study.jl
include("DeepONetHeat.jl")
include("DeepONetScenarios.jl")
using .DeepONetHeat
using .DeepONetScenarios
using Random, Statistics, Printf, LinearAlgebra

const SCENARIO = mechanism_scenario()
const K = SCENARIO.grid
const SEEDS = SCENARIO.seeds
const STEPS = SCENARIO.steps
const OUT = SCENARIO.out
const LR = 8e-3
isdir(OUT) || mkdir(OUT)

stat(xs) = (mean(xs), length(xs) == 1 ? 0.0 : std(xs))

function status_rows!(rows, part, log::ProjectionLog)
    cmean = isempty(log.correction_norms) ? 0.0 : mean(log.correction_norms)
    cmax = isempty(log.correction_norms) ? 0.0 :
        maximum(log.correction_norms)
    if isempty(log.counts)
        push!(rows, (part = part, status = "none", count = 0,
                     correction_mean = cmean, correction_max = cmax))
    else
        for (status, count) in sort(collect(log.counts))
            push!(rows, (part = part, status = string(status),
                         count = count, correction_mean = cmean,
                         correction_max = cmax))
        end
    end
    return rows
end

# ---- Part 1: per-sample non-expansiveness of evaluation-only correction ----
function run_nonexpansive()
    rows = NamedTuple[]
    log = ProjectionLog()
    for seed in SEEDS
        @printf("nonexpansive: seed %d ...\n", seed)
        rng = MersenneTwister(50_000 + seed)
        train_data, x, w = sample_heat_operator(SCENARIO.train_samples, K,
                                                rng = rng)
        test_data, _, _ = sample_heat_operator(SCENARIO.test_samples, K,
                                               rng = rng)
        p0 = init_deeponet(seed = 50_000 + seed)
        pv, _ = train!(p -> vanilla_loss(p, train_data, x), p0,
                       steps = STEPS, lr = LR)
        for (i, d) in enumerate(test_data)
            ctx = heat_correction_context(d, w, mode = :full)
            raw = deeponet_model(pv, d.theta, x)
            corrected = corrected_output(raw, ctx, log = log)
            raw_error = norm(raw .- d.u)
            corrected_error = norm(corrected .- d.u)
            push!(rows, (seed = seed, sample = i, raw_error = raw_error,
                         corrected_error = corrected_error,
                         delta = corrected_error - raw_error,
                         target_boundary = boundary_violation(d.u, d),
                         target_mass = mass_violation(d.u, d, w)))
        end
    end
    return rows, log
end

# ---- Part 2: low-data sweep ----
# One full data draw per seed, subset for each training size, so the size
# effect is isolated from sampling noise across sizes.
function run_low_data()
    models = ["vanilla", "eval_only", "hard"]
    acc = Dict((n, m) => Float64[] for n in SCENARIO.sample_grid,
               m in models)
    train_log = ProjectionLog()
    eval_log = ProjectionLog()
    nmax = maximum(SCENARIO.sample_grid)
    for seed in SEEDS
        @printf("low data: seed %d ...\n", seed)
        rng = MersenneTwister(60_000 + seed)
        full_train, x, w = sample_heat_operator(nmax, K, rng = rng)
        test_data, _, _ = sample_heat_operator(SCENARIO.test_samples, K,
                                               rng = rng)
        test_contexts = heat_correction_contexts(test_data, w,
                                                 mode = :full,
                                                 cached = true)
        p0 = init_deeponet(seed = 60_000 + seed)
        for n in SCENARIO.sample_grid
            train_data = full_train[1:n]
            train_contexts = heat_correction_contexts(train_data, w,
                                                      mode = :full,
                                                      cached = true)
            pv, _ = train!(p -> vanilla_loss(p, train_data, x), p0,
                           steps = STEPS, lr = LR)
            ph, _ = train!(p -> context_hard_loss(p, train_contexts, x,
                                                  log = train_log), p0,
                           steps = STEPS, lr = LR)
            push!(acc[(n, "vanilla")],
                  evaluate_model(d -> deeponet_model(pv, d.theta, x),
                                 test_data, w).rmse)
            push!(acc[(n, "eval_only")],
                  evaluate_context_model(ctx ->
                      context_projected_field(pv, ctx, x, log = eval_log),
                      test_contexts).rmse)
            push!(acc[(n, "hard")],
                  evaluate_context_model(ctx ->
                      context_projected_field(ph, ctx, x, log = eval_log),
                      test_contexts).rmse)
        end
    end
    rows = NamedTuple[]
    for n in SCENARIO.sample_grid, m in models
        rm = stat(acc[(n, m)])
        ratios = acc[(n, "vanilla")] ./ acc[(n, m)]
        rr = stat(ratios)
        push!(rows, (train_samples = n, model = m, rmse_mean = rm[1],
                     rmse_std = rm[2], advantage_mean = rr[1],
                     advantage_std = rr[2]))
    end
    return rows, train_log, eval_log
end

# ---- Part 3: feasibility-breaking constraint noise ----
# Perturb the constraint values (boundary and mass) away from the data while
# the targets stay clean, the realistic case where physics metadata disagrees
# with the labels. Clamps keep the perturbed constraint set nonempty.
function perturb_sample(d, eps, rng)
    return merge(d, (left = clamp(d.left + eps * randn(rng), 0.1, 1.9),
                     right = clamp(d.right + eps * randn(rng), 0.1, 1.9),
                     mass0 = clamp(d.mass0 + eps * randn(rng), 0.2, 1.8)))
end

function run_noise()
    models = ["vanilla", "eval_only", "hard"]
    acc = Dict((eps, m) => Float64[] for eps in SCENARIO.noise_levels,
               m in models)
    gaps = Dict(eps => Float64[] for eps in SCENARIO.noise_levels)
    train_log = ProjectionLog()
    eval_log = ProjectionLog()
    for seed in SEEDS
        @printf("constraint noise: seed %d ...\n", seed)
        rng = MersenneTwister(70_000 + seed)
        train_data, x, w = sample_heat_operator(SCENARIO.train_samples, K,
                                                rng = rng)
        test_data, _, _ = sample_heat_operator(SCENARIO.test_samples, K,
                                               rng = rng)
        p0 = init_deeponet(seed = 70_000 + seed)
        pv, _ = train!(p -> vanilla_loss(p, train_data, x), p0,
                       steps = STEPS, lr = LR)
        vanilla_rmse = mean(field_rmse(deeponet_model(pv, d.theta, x), d.u)
                            for d in test_data)
        for eps in SCENARIO.noise_levels
            nrng = MersenneTwister(80_000 + 1000 * seed +
                                   round(Int, 10_000 * eps))
            ptrain = [perturb_sample(d, eps, nrng) for d in train_data]
            ptest = [perturb_sample(d, eps, nrng) for d in test_data]
            train_contexts = heat_correction_contexts(ptrain, w,
                                                      mode = :full,
                                                      cached = true)
            test_contexts = heat_correction_contexts(ptest, w,
                                                     mode = :full,
                                                     cached = true)
            push!(acc[(eps, "vanilla")], vanilla_rmse)

            eo_preds = [context_projected_field(pv, ctx, x,
                                                log = eval_log,
                                                failure_policy = :continue)
                        for ctx in test_contexts]
            push!(acc[(eps, "eval_only")],
                  mean(field_rmse(pred, d.u)
                       for (pred, d) in zip(eo_preds, test_data)))
            push!(gaps[eps],
                  mean(boundary_violation(pred, d)
                       for (pred, d) in zip(eo_preds, test_data)))

            ph, _ = train!(p -> context_hard_loss(p, train_contexts, x,
                                                  log = train_log), p0,
                           steps = STEPS, lr = LR)
            h_preds = [context_projected_field(ph, ctx, x,
                                               log = eval_log,
                                               failure_policy = :continue)
                       for ctx in test_contexts]
            push!(acc[(eps, "hard")],
                  mean(field_rmse(pred, d.u)
                       for (pred, d) in zip(h_preds, test_data)))
        end
    end
    rows = NamedTuple[]
    for eps in SCENARIO.noise_levels, m in models
        rm = stat(acc[(eps, m)])
        gap = stat(gaps[eps])
        push!(rows, (eps = eps, model = m, rmse_mean = rm[1],
                     rmse_std = rm[2],
                     true_boundary_gap_mean = m == "eval_only" ? gap[1] : 0.0))
    end
    crossover = Dict{String,Any}()
    for m in ["eval_only", "hard"]
        hit = findfirst(eps -> mean(acc[(eps, m)]) >
                            mean(acc[(eps, "vanilla")]),
                        collect(SCENARIO.noise_levels))
        crossover[m] = hit === nothing ? "none" :
            collect(SCENARIO.noise_levels)[hit]
    end
    return rows, crossover, train_log, eval_log
end

# ---- run all parts ----
ne_rows, ne_log = run_nonexpansive()
ld_rows, ld_train_log, ld_eval_log = run_low_data()
nz_rows, crossover, nz_train_log, nz_eval_log = run_noise()

ne_tol = 1e-10
violations = count(r -> r.delta > ne_tol, ne_rows)
max_delta = maximum(r -> r.delta, ne_rows)
max_target_infeas = maximum(r -> max(r.target_boundary, r.target_mass),
                            ne_rows)

println("\n=== Mechanism study summary ===")
@printf("nonexpansive: %d samples, %d violations (tol %.1e), max delta %.3e, max target infeasibility %.3e\n",
        length(ne_rows), violations, ne_tol, max_delta, max_target_infeas)
for row in ld_rows
    row.model == "hard" &&
        @printf("low data n=%-3d  vanilla/hard advantage %.3f+/-%.3f\n",
                row.train_samples, row.advantage_mean, row.advantage_std)
end
for row in nz_rows
    row.model != "vanilla" &&
        @printf("noise eps=%.2f  %-10s RMSE %.4f+/-%.4f\n",
                row.eps, row.model, row.rmse_mean, row.rmse_std)
end
println("crossover eps where correction stops beating vanilla: eval_only = ",
        crossover["eval_only"], ", hard = ", crossover["hard"])

open(joinpath(OUT, "mechanism_nonexpansive.csv"), "w") do io
    println(io, "seed,sample,raw_error,corrected_error,delta,target_boundary,target_mass")
    for r in ne_rows
        println(io, @sprintf("%d,%d,%.10e,%.10e,%.10e,%.3e,%.3e",
                             r.seed, r.sample, r.raw_error,
                             r.corrected_error, r.delta, r.target_boundary,
                             r.target_mass))
    end
end

open(joinpath(OUT, "mechanism_low_data.csv"), "w") do io
    println(io, "train_samples,model,rmse_mean,rmse_std,advantage_mean,advantage_std")
    for r in ld_rows
        println(io, @sprintf("%d,%s,%.8f,%.8f,%.6f,%.6f",
                             r.train_samples, r.model, r.rmse_mean,
                             r.rmse_std, r.advantage_mean, r.advantage_std))
    end
end

open(joinpath(OUT, "mechanism_noise.csv"), "w") do io
    println(io, "eps,model,rmse_mean,rmse_std,true_boundary_gap_mean")
    for r in nz_rows
        println(io, @sprintf("%.4f,%s,%.8f,%.8f,%.8e",
                             r.eps, r.model, r.rmse_mean, r.rmse_std,
                             r.true_boundary_gap_mean))
    end
end

open(joinpath(OUT, "mechanism_summary.csv"), "w") do io
    println(io, "part,metric,value")
    println(io, "nonexpansive,samples,", length(ne_rows))
    println(io, "nonexpansive,violations,", violations)
    println(io, @sprintf("nonexpansive,max_delta,%.6e", max_delta))
    println(io, @sprintf("nonexpansive,max_target_infeasibility,%.6e",
                         max_target_infeas))
    println(io, "nonexpansive,prediction_holds,", violations == 0)
    for r in ld_rows
        r.model == "hard" &&
            println(io, @sprintf("low_data,advantage_at_%d,%.6f",
                                 r.train_samples, r.advantage_mean))
    end
    println(io, "noise,crossover_eval_only,", crossover["eval_only"])
    println(io, "noise,crossover_hard,", crossover["hard"])
end

status_rows = NamedTuple[]
status_rows!(status_rows, "nonexpansive_eval", ne_log)
status_rows!(status_rows, "low_data_train", ld_train_log)
status_rows!(status_rows, "low_data_eval", ld_eval_log)
status_rows!(status_rows, "noise_train", nz_train_log)
status_rows!(status_rows, "noise_eval", nz_eval_log)
open(joinpath(OUT, "mechanism_statuses.csv"), "w") do io
    println(io, "part,status,count,correction_mean,correction_max")
    for r in status_rows
        println(io, @sprintf("%s,%s,%d,%.8e,%.8e", r.part, r.status,
                             r.count, r.correction_mean, r.correction_max))
    end
end

println("wrote mechanism_nonexpansive.csv, mechanism_low_data.csv, mechanism_noise.csv, mechanism_summary.csv, mechanism_statuses.csv to ", OUT)
