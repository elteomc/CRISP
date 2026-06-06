# DeepONet helper study. Writes CSV artifacts to results/.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/study.jl
include("DeepONetHeat.jl")
using .DeepONetHeat
using Random, Statistics, Printf

const K = 32
const SEEDS = 1:10
const STEPS = 90
const OUT = joinpath(@__DIR__, "results")

isdir(OUT) || mkdir(OUT)

const SOFT_CONFIGS = [
    (name = "soft_weak", beta_boundary = 1.0, beta_mass = 1.0,
     beta_box = 0.2),
    (name = "soft_default", beta_boundary = 10.0, beta_mass = 10.0,
     beta_box = 2.0),
    (name = "soft_strong", beta_boundary = 50.0, beta_mass = 50.0,
     beta_box = 10.0),
    (name = "soft_boundary_heavy", beta_boundary = 80.0,
     beta_mass = 10.0, beta_box = 2.0),
]

const MODEL_NAMES = vcat(
    ["vanilla"],
    [cfg.name for cfg in SOFT_CONFIGS],
    ["eval_only_full", "hard_boundary_box", "hard_full",
     "hard_full_cached", "soft_plus_hard_boundary_box",
     "soft_plus_hard_full", "soft_plus_hard_full_cached"],
)

const METRICS = (:rmse, :boundary_max, :boundary_mean, :mass_max, :mass_mean,
                 :lower_max, :lower_mean, :upper_max, :upper_mean)

acc = Dict(m => Dict(k => Float64[] for k in METRICS) for m in MODEL_NAMES)
times = Dict(m => Float64[] for m in MODEL_NAMES)
train_status = Dict{String,ProjectionLog}()
eval_status = Dict{String,ProjectionLog}()
histories = Dict{String,Vector{Float64}}()

for m in MODEL_NAMES
    train_status[m] = ProjectionLog()
    eval_status[m] = ProjectionLog()
end

function merge_log!(dest::ProjectionLog, src::ProjectionLog)
    for (status, count) in src.counts
        dest.counts[status] = get(dest.counts, status, 0) + count
    end
    append!(dest.correction_norms, src.correction_norms)
    return dest
end

function stat(xs)
    isempty(xs) && return 0.0, 0.0
    return mean(xs), length(xs) == 1 ? 0.0 : std(xs)
end

function run_soft(cfg, p0, train_data, x, w)
    return train!(p -> soft_loss(p, train_data, x, w,
                                 beta_boundary = cfg.beta_boundary,
                                 beta_mass = cfg.beta_mass,
                                 beta_box = cfg.beta_box),
                  p0, steps = STEPS, lr = 8e-3)
end

function run_hard(mode, p0, train_data, x, w, log)
    return train!(p -> hard_loss(p, train_data, x, w,
                                 log = log, mode = mode),
                  p0, steps = STEPS, lr = 8e-3)
end

function run_soft_plus_hard(mode, p0, train_data, x, w, log)
    return train!(p -> soft_plus_hard_loss(p, train_data, x, w,
                                           beta_boundary = 2.0,
                                           beta_mass = 2.0,
                                           beta_box = 0.5,
                                           log = log,
                                           mode = mode),
                  p0, steps = STEPS, lr = 8e-3)
end

function run_context_hard(p0, contexts, x, log)
    return train!(p -> context_hard_loss(p, contexts, x, log = log),
                  p0, steps = STEPS, lr = 8e-3)
end

function run_context_soft_plus_hard(p0, contexts, x, log)
    return train!(p -> context_soft_plus_hard_loss(p, contexts, x,
                                                   beta_boundary = 2.0,
                                                   beta_mass = 2.0,
                                                   beta_box = 0.5,
                                                   log = log),
                  p0, steps = STEPS, lr = 8e-3)
end

function add_eval!(name, ev)
    for k in METRICS
        push!(acc[name][k], getfield(ev, k))
    end
    return nothing
end

for seed in SEEDS
    @printf("seed %d: training DeepONet helper baselines and ablations ...\n",
            seed)
    rng = MersenneTwister(seed)
    train_data, x, w = sample_heat_operator(18, K, rng = rng)
    test_data, _, _ = sample_heat_operator(8, K, rng = rng)
    train_contexts_full = heat_correction_contexts(train_data, w,
                                                   mode = :full,
                                                   cached = true)
    test_contexts_full = heat_correction_contexts(test_data, w,
                                                  mode = :full,
                                                  cached = true)
    p0 = init_deeponet(seed = seed)

    tv = @elapsed pv, hv = train!(p -> vanilla_loss(p, train_data, x), p0,
                                  steps = STEPS, lr = 8e-3)
    push!(times["vanilla"], tv)
    seed == first(SEEDS) && (histories["vanilla"] = hv)

    soft_params = Dict{String,Any}()
    for cfg in SOFT_CONFIGS
        ts = @elapsed ps, hs = run_soft(cfg, p0, train_data, x, w)
        soft_params[cfg.name] = ps
        push!(times[cfg.name], ts)
        seed == first(SEEDS) && (histories[cfg.name] = hs)
    end

    log_hbb = ProjectionLog()
    thbb = @elapsed phbb, hhbb = run_hard(:boundary_box, p0, train_data,
                                          x, w, log_hbb)
    push!(times["hard_boundary_box"], thbb)
    merge_log!(train_status["hard_boundary_box"], log_hbb)
    seed == first(SEEDS) && (histories["hard_boundary_box"] = hhbb)

    log_hfull = ProjectionLog()
    thfull = @elapsed phfull, hhfull = run_hard(:full, p0, train_data, x, w,
                                                log_hfull)
    push!(times["hard_full"], thfull)
    merge_log!(train_status["hard_full"], log_hfull)
    seed == first(SEEDS) && (histories["hard_full"] = hhfull)

    log_hfull_cached = ProjectionLog()
    thfull_cached = @elapsed begin
        phfull_cached, hhfull_cached =
            run_context_hard(p0, train_contexts_full, x,
                             log_hfull_cached)
    end
    push!(times["hard_full_cached"], thfull_cached)
    merge_log!(train_status["hard_full_cached"], log_hfull_cached)
    seed == first(SEEDS) &&
        (histories["hard_full_cached"] = hhfull_cached)

    log_cbb = ProjectionLog()
    tcbb = @elapsed pcbb, hcbb = run_soft_plus_hard(:boundary_box, p0,
                                                    train_data, x, w,
                                                    log_cbb)
    push!(times["soft_plus_hard_boundary_box"], tcbb)
    merge_log!(train_status["soft_plus_hard_boundary_box"], log_cbb)
    seed == first(SEEDS) && (histories["soft_plus_hard_boundary_box"] = hcbb)

    log_cfull = ProjectionLog()
    tcfull = @elapsed pcfull, hcfull = run_soft_plus_hard(:full, p0,
                                                         train_data, x, w,
                                                         log_cfull)
    push!(times["soft_plus_hard_full"], tcfull)
    merge_log!(train_status["soft_plus_hard_full"], log_cfull)
    seed == first(SEEDS) && (histories["soft_plus_hard_full"] = hcfull)

    log_cfull_cached = ProjectionLog()
    tcfull_cached = @elapsed begin
        pcfull_cached, hcfull_cached =
            run_context_soft_plus_hard(p0, train_contexts_full, x,
                                       log_cfull_cached)
    end
    push!(times["soft_plus_hard_full_cached"], tcfull_cached)
    merge_log!(train_status["soft_plus_hard_full_cached"],
               log_cfull_cached)
    seed == first(SEEDS) &&
        (histories["soft_plus_hard_full_cached"] = hcfull_cached)

    push!(times["eval_only_full"], 0.0)

    add_eval!("vanilla",
              evaluate_model(d -> deeponet_model(pv, d.theta, x),
                             test_data, w))
    for cfg in SOFT_CONFIGS
        ps = soft_params[cfg.name]
        add_eval!(cfg.name,
                  evaluate_model(d -> deeponet_model(ps, d.theta, x),
                                 test_data, w))
    end

    eval_only_log = ProjectionLog()
    add_eval!("eval_only_full",
              evaluate_model(d -> hard_projected_field(pv, d, x, w,
                                                       log = eval_only_log,
                                                       mode = :full),
                             test_data, w))
    merge_log!(eval_status["eval_only_full"], eval_only_log)

    eval_hbb_log = ProjectionLog()
    add_eval!("hard_boundary_box",
              evaluate_model(d -> hard_projected_field(phbb, d, x, w,
                                                       log = eval_hbb_log,
                                                       mode = :boundary_box),
                             test_data, w))
    merge_log!(eval_status["hard_boundary_box"], eval_hbb_log)

    eval_hfull_log = ProjectionLog()
    add_eval!("hard_full",
              evaluate_model(d -> hard_projected_field(phfull, d, x, w,
                                                       log = eval_hfull_log,
                                                       mode = :full),
                             test_data, w))
    merge_log!(eval_status["hard_full"], eval_hfull_log)

    eval_hfull_cached_log = ProjectionLog()
    add_eval!("hard_full_cached",
              evaluate_context_model(ctx -> context_projected_field(phfull_cached,
                                                                    ctx, x,
                                                                    log = eval_hfull_cached_log),
                                     test_contexts_full))
    merge_log!(eval_status["hard_full_cached"], eval_hfull_cached_log)

    eval_cbb_log = ProjectionLog()
    add_eval!("soft_plus_hard_boundary_box",
              evaluate_model(d -> hard_projected_field(pcbb, d, x, w,
                                                       log = eval_cbb_log,
                                                       mode = :boundary_box),
                             test_data, w))
    merge_log!(eval_status["soft_plus_hard_boundary_box"], eval_cbb_log)

    eval_cfull_log = ProjectionLog()
    add_eval!("soft_plus_hard_full",
              evaluate_model(d -> hard_projected_field(pcfull, d, x, w,
                                                       log = eval_cfull_log,
                                                       mode = :full),
                             test_data, w))
    merge_log!(eval_status["soft_plus_hard_full"], eval_cfull_log)

    eval_cfull_cached_log = ProjectionLog()
    add_eval!("soft_plus_hard_full_cached",
              evaluate_context_model(ctx -> context_projected_field(pcfull_cached,
                                                                    ctx, x,
                                                                    log = eval_cfull_cached_log),
                                     test_contexts_full))
    merge_log!(eval_status["soft_plus_hard_full_cached"],
               eval_cfull_cached_log)
end

println("\n=== DeepONet helper study: mean +/- std over $(length(SEEDS)) seeds ===")
for m in MODEL_NAMES
    rm = stat(acc[m][:rmse])
    bm = stat(acc[m][:boundary_max])
    mm = stat(acc[m][:mass_max])
    lm = stat(acc[m][:lower_max])
    um = stat(acc[m][:upper_max])
    tm = stat(times[m])
    @printf("%-29s RMSE %.5f+/-%.5f  boundary %.3e+/-%.3e  mass %.3e+/-%.3e  lower %.3e+/-%.3e  upper %.3e+/-%.3e  train %.2fs+/-%.2fs\n",
            m, rm..., bm..., mm..., lm..., um..., tm...)
end

function best_model_by(metric)
    vals = [(m, mean(acc[m][metric])) for m in MODEL_NAMES]
    return sort(vals, by = x -> x[2])[1]
end

best_rmse = best_model_by(:rmse)
best_boundary = best_model_by(:boundary_max)
best_mass = best_model_by(:mass_max)
println("best RMSE row: ", best_rmse)
println("best boundary row: ", best_boundary)
println("best mass row: ", best_mass)

open(joinpath(OUT, "results.csv"), "w") do io
    println(io, "model,rmse_mean,rmse_std,boundarymax_mean,boundarymax_std,boundarymean_mean,boundarymean_std,massmax_mean,massmax_std,massmean_mean,massmean_std,lowermax_mean,lowermax_std,lowermean_mean,lowermean_std,uppermax_mean,uppermax_std,uppermean_mean,uppermean_std,train_seconds_mean,train_seconds_std")
    for m in MODEL_NAMES
        ms = Dict(k => stat(acc[m][k]) for k in METRICS)
        tm = stat(times[m])
        println(io, @sprintf("%s,%.8f,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8f,%.8f",
                             m, ms[:rmse]..., ms[:boundary_max]...,
                             ms[:boundary_mean]..., ms[:mass_max]...,
                             ms[:mass_mean]..., ms[:lower_max]...,
                             ms[:lower_mean]..., ms[:upper_max]...,
                             ms[:upper_mean]..., tm...))
    end
end

open(joinpath(OUT, "statuses.csv"), "w") do io
    println(io, "phase,model,status,count,correction_mean,correction_max")
    for (phase, table) in [("train", train_status), ("eval", eval_status)]
        for m in MODEL_NAMES
            log = table[m]
            cmean = isempty(log.correction_norms) ? 0.0 :
                mean(log.correction_norms)
            cmax = isempty(log.correction_norms) ? 0.0 :
                maximum(log.correction_norms)
            if isempty(log.counts)
                println(io, @sprintf("%s,%s,none,0,%.8e,%.8e",
                                     phase, m, cmean, cmax))
            else
                for (status, count) in sort(collect(log.counts))
                    println(io, @sprintf("%s,%s,%s,%d,%.8e,%.8e",
                                         phase, m, status, count, cmean,
                                         cmax))
                end
            end
        end
    end
end

open(joinpath(OUT, "training_curves.csv"), "w") do io
    names = collect(keys(histories))
    maxlen = maximum(length.(values(histories)))
    println(io, join(vcat(["step"], names), ","))
    for i in 1:maxlen
        row = String[string(i)]
        for name in names
            hist = histories[name]
            push!(row, i <= length(hist) ? @sprintf("%.12e", hist[i]) : "")
        end
        println(io, join(row, ","))
    end
end

println("wrote results.csv, statuses.csv, and training_curves.csv to ", OUT)
