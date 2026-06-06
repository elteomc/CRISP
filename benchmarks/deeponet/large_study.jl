# Larger-grid DeepONet helper study. Writes CSV artifacts to results/.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/large_study.jl
include("DeepONetHeat.jl")
using .DeepONetHeat
using Random, Statistics, Printf

const GRIDS = (64, 96)
const SEEDS = 1:5
const STEPS = 70
const OUT = joinpath(@__DIR__, "results")

isdir(OUT) || mkdir(OUT)

const MODEL_NAMES = ["vanilla", "soft_weak", "eval_only_full",
                     "hard_full", "hard_full_cached",
                     "soft_plus_hard_full_cached"]
const METRICS = (:rmse, :boundary_max, :boundary_mean, :mass_max, :mass_mean,
                 :lower_max, :lower_mean, :upper_max, :upper_mean)

function stat(xs)
    isempty(xs) && return 0.0, 0.0
    return mean(xs), length(xs) == 1 ? 0.0 : std(xs)
end

function merge_log!(dest::ProjectionLog, src::ProjectionLog)
    for (status, count) in src.counts
        dest.counts[status] = get(dest.counts, status, 0) + count
    end
    append!(dest.correction_norms, src.correction_norms)
    return dest
end

function add_eval!(acc, model, ev)
    for k in METRICS
        push!(acc[model][k], getfield(ev, k))
    end
    return nothing
end

open(joinpath(OUT, "large_results.csv"), "w") do rio
    println(rio, "grid,model,rmse_mean,rmse_std,boundarymax_mean,boundarymax_std,boundarymean_mean,boundarymean_std,massmax_mean,massmax_std,massmean_mean,massmean_std,lowermax_mean,lowermax_std,lowermean_mean,lowermean_std,uppermax_mean,uppermax_std,uppermean_mean,uppermean_std,train_seconds_mean,train_seconds_std")

    open(joinpath(OUT, "large_statuses.csv"), "w") do sio
        println(sio, "grid,phase,model,status,count,correction_mean,correction_max")

        for K in GRIDS
            warm_rng = MersenneTwister(99_000 + K)
            warm_data, warm_x, warm_w = sample_heat_operator(2, K,
                                                             rng = warm_rng)
            warm_contexts = heat_correction_contexts(warm_data, warm_w,
                                                     mode = :full,
                                                     cached = true)
            warm_p = init_deeponet(seed = 99_000 + K)
            train!(p -> vanilla_loss(p, warm_data, warm_x), warm_p,
                   steps = 1, lr = 8e-3)
            train!(p -> soft_loss(p, warm_data, warm_x, warm_w,
                                  beta_boundary = 1.0,
                                  beta_mass = 1.0,
                                  beta_box = 0.2), warm_p,
                   steps = 1, lr = 8e-3)
            train!(p -> hard_loss(p, warm_data, warm_x, warm_w,
                                  mode = :full), warm_p,
                   steps = 1, lr = 8e-3)
            train!(p -> context_hard_loss(p, warm_contexts, warm_x),
                   warm_p, steps = 1, lr = 8e-3)
            train!(p -> context_soft_plus_hard_loss(p, warm_contexts,
                                                    warm_x,
                                                    beta_boundary = 2.0,
                                                    beta_mass = 2.0,
                                                    beta_box = 0.5),
                   warm_p, steps = 1, lr = 8e-3)

            acc = Dict(m => Dict(k => Float64[] for k in METRICS)
                       for m in MODEL_NAMES)
            times = Dict(m => Float64[] for m in MODEL_NAMES)
            train_status = Dict(m => ProjectionLog() for m in MODEL_NAMES)
            eval_status = Dict(m => ProjectionLog() for m in MODEL_NAMES)

            for seed in SEEDS
                @printf("grid %d seed %d: running larger DeepONet helper rows ...\n",
                        K, seed)
                rng = MersenneTwister(10_000 + 100K + seed)
                train_data, x, w = sample_heat_operator(18, K, rng = rng)
                test_data, _, _ = sample_heat_operator(8, K, rng = rng)
                train_contexts = heat_correction_contexts(train_data, w,
                                                          mode = :full,
                                                          cached = true)
                test_contexts = heat_correction_contexts(test_data, w,
                                                         mode = :full,
                                                         cached = true)
                p0 = init_deeponet(seed = 10_000 + K + seed)

                tv = @elapsed pv, _ = train!(p -> vanilla_loss(p, train_data, x),
                                             p0, steps = STEPS, lr = 8e-3)
                push!(times["vanilla"], tv)

                ts = @elapsed ps, _ = train!(p -> soft_loss(p, train_data,
                                                            x, w,
                                                            beta_boundary = 1.0,
                                                            beta_mass = 1.0,
                                                            beta_box = 0.2),
                                             p0, steps = STEPS, lr = 8e-3)
                push!(times["soft_weak"], ts)

                log_h = ProjectionLog()
                th = @elapsed ph, _ = train!(p -> hard_loss(p, train_data,
                                                            x, w,
                                                            log = log_h,
                                                            mode = :full),
                                             p0, steps = STEPS, lr = 8e-3)
                push!(times["hard_full"], th)
                merge_log!(train_status["hard_full"], log_h)

                log_hc = ProjectionLog()
                thc = @elapsed phc, _ =
                    train!(p -> context_hard_loss(p, train_contexts, x,
                                                  log = log_hc),
                           p0, steps = STEPS, lr = 8e-3)
                push!(times["hard_full_cached"], thc)
                merge_log!(train_status["hard_full_cached"], log_hc)

                log_sc = ProjectionLog()
                tsc = @elapsed psc, _ =
                    train!(p -> context_soft_plus_hard_loss(p,
                                                            train_contexts,
                                                            x,
                                                            beta_boundary = 2.0,
                                                            beta_mass = 2.0,
                                                            beta_box = 0.5,
                                                            log = log_sc),
                           p0, steps = STEPS, lr = 8e-3)
                push!(times["soft_plus_hard_full_cached"], tsc)
                merge_log!(train_status["soft_plus_hard_full_cached"], log_sc)

                push!(times["eval_only_full"], 0.0)

                add_eval!(acc, "vanilla",
                          evaluate_model(d -> deeponet_model(pv, d.theta, x),
                                         test_data, w))
                add_eval!(acc, "soft_weak",
                          evaluate_model(d -> deeponet_model(ps, d.theta, x),
                                         test_data, w))

                log_eval_only = ProjectionLog()
                add_eval!(acc, "eval_only_full",
                          evaluate_context_model(ctx ->
                              context_projected_field(pv, ctx, x,
                                                      log = log_eval_only),
                              test_contexts))
                merge_log!(eval_status["eval_only_full"], log_eval_only)

                log_eval_h = ProjectionLog()
                add_eval!(acc, "hard_full",
                          evaluate_model(d -> hard_projected_field(ph, d, x,
                                                                   w,
                                                                   log = log_eval_h,
                                                                   mode = :full),
                                         test_data, w))
                merge_log!(eval_status["hard_full"], log_eval_h)

                log_eval_hc = ProjectionLog()
                add_eval!(acc, "hard_full_cached",
                          evaluate_context_model(ctx ->
                              context_projected_field(phc, ctx, x,
                                                      log = log_eval_hc),
                              test_contexts))
                merge_log!(eval_status["hard_full_cached"], log_eval_hc)

                log_eval_sc = ProjectionLog()
                add_eval!(acc, "soft_plus_hard_full_cached",
                          evaluate_context_model(ctx ->
                              context_projected_field(psc, ctx, x,
                                                      log = log_eval_sc),
                              test_contexts))
                merge_log!(eval_status["soft_plus_hard_full_cached"],
                           log_eval_sc)
            end

            for model in MODEL_NAMES
                ms = Dict(k => stat(acc[model][k]) for k in METRICS)
                tm = stat(times[model])
                println(rio, @sprintf("%d,%s,%.8f,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8f,%.8f",
                                      K, model, ms[:rmse]...,
                                      ms[:boundary_max]...,
                                      ms[:boundary_mean]..., ms[:mass_max]...,
                                      ms[:mass_mean]..., ms[:lower_max]...,
                                      ms[:lower_mean]..., ms[:upper_max]...,
                                      ms[:upper_mean]..., tm...))
            end

            for (phase, table) in [("train", train_status),
                                   ("eval", eval_status)]
                for model in MODEL_NAMES
                    log = table[model]
                    cmean = isempty(log.correction_norms) ? 0.0 :
                        mean(log.correction_norms)
                    cmax = isempty(log.correction_norms) ? 0.0 :
                        maximum(log.correction_norms)
                    if isempty(log.counts)
                        println(sio, @sprintf("%d,%s,%s,none,0,%.8e,%.8e",
                                             K, phase, model, cmean, cmax))
                    else
                        for (status, count) in sort(collect(log.counts))
                            println(sio, @sprintf("%d,%s,%s,%s,%d,%.8e,%.8e",
                                                 K, phase, model, status,
                                                 count, cmean, cmax))
                        end
                    end
                end
            end
        end
    end
end

println("wrote large_results.csv and large_statuses.csv to ", OUT)
