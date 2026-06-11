if !isdefined(@__MODULE__, :read_exported_batch)
    include("summer_batch_eval.jl")
end

using Printf, Random, Statistics

function batch_grid(samples)
    for sample in samples
        sample.x === nothing || return sample.x
    end
    throw(ArgumentError("training batches need a grid column"))
end

function require_trainable(samples)
    isempty(samples) && throw(ArgumentError("training batch is empty"))
    feature_dim = nothing
    K = nothing
    for sample in samples
        sample.features === nothing &&
            throw(ArgumentError("sample $(sample.case_id) needs features"))
        sample.target === nothing &&
            throw(ArgumentError("sample $(sample.case_id) needs target_output"))
        feature_dim === nothing && (feature_dim = length(sample.features))
        K === nothing && (K = length(sample.target))
        length(sample.features) == feature_dim ||
            throw(ArgumentError("all samples need the same feature length"))
        length(sample.target) == K ||
            throw(ArgumentError("all samples need the same target length"))
    end
    return feature_dim, K
end

function split_samples(samples)
    tagged = any(sample -> sample.split !== nothing, samples)
    if tagged
        train = [s for s in samples if s.split in ("train", "training")]
        test = [s for s in samples if s.split in
                ("test", "eval", "evaluation", "val", "validation",
                 "holdout")]
        isempty(train) &&
            throw(ArgumentError("split column did not mark any training rows"))
        isempty(test) &&
            throw(ArgumentError("split column did not mark any evaluation rows"))
        return train, test
    end
    n = length(samples)
    if n == 1
        return samples, samples
    end
    ntrain = max(1, floor(Int, 0.7n))
    ntrain = min(ntrain, n - 1)
    return samples[1:ntrain], samples[(ntrain + 1):end]
end

function batch_vanilla_loss(p, samples, x)
    s = 0.0
    for sample in samples
        pred = deeponet_model(p, sample.features, x)
        s += mean(abs2, pred .- sample.target)
    end
    return s / length(samples)
end

function batch_context_hard_loss(p, contexts, x; log = nothing,
                                 failure_policy = :error)
    s = 0.0
    for ctx in contexts
        sample = ctx.data
        raw = deeponet_model(p, sample.features, x)
        pred = corrected_output(raw, ctx, log = log,
                                failure_policy = failure_policy)
        s += mean(abs2, pred .- sample.target)
    end
    return s / length(contexts)
end

function finite_mean(xs)
    ys = [x for x in xs if !isnan(x)]
    isempty(ys) && return NaN
    return mean(ys)
end

function finite_max(xs)
    ys = [x for x in xs if !isnan(x)]
    isempty(ys) && return NaN
    return maximum(ys)
end

function evaluate_batch_predictions(samples, preds)
    rmses = Float64[]
    boundaries = Float64[]
    balances = Float64[]
    boxes = Float64[]
    for (sample, pred) in zip(samples, preds)
        push!(rmses, safe_rmse(pred, sample.target))
        push!(boundaries, boundary_error(pred, sample))
        push!(balances, balance_error(pred, sample))
        push!(boxes, box_error(pred, sample))
    end
    return (rmse = finite_mean(rmses),
            boundary_max = finite_max(boundaries),
            balance_max = finite_max(balances),
            box_max = finite_max(boxes))
end

function raw_model_predictions(p, samples, x)
    return [deeponet_model(p, sample.features, x) for sample in samples]
end

function corrected_model_predictions(p, contexts, x, log)
    preds = Vector{Vector{Float64}}()
    for ctx in contexts
        sample = ctx.data
        raw = deeponet_model(p, sample.features, x)
        res = project(ctx.constraint, raw)
        record!(log, res)
        push!(preds, collect(res.zstar))
    end
    return preds
end

function log_total(log::ProjectionLog)
    return sum(values(log.counts), init = 0)
end

function success_rate(log::ProjectionLog)
    total = log_total(log)
    total == 0 && return 0.0
    return get(log.counts, :success, 0) / total
end

function correction_mean(log::ProjectionLog)
    isempty(log.correction_norms) && return 0.0
    return mean(log.correction_norms)
end

function correction_max(log::ProjectionLog)
    isempty(log.correction_norms) && return 0.0
    return maximum(log.correction_norms)
end

function metric_row(model, metrics; train_seconds = 0.0, log = nothing,
                    gate_passed = true, trained = true)
    success = log === nothing ? 0 : get(log.counts, :success, 0)
    total = log === nothing ? 0 : log_total(log)
    rate = log === nothing ? NaN : success_rate(log)
    cmean = log === nothing ? 0.0 : correction_mean(log)
    cmax = log === nothing ? 0.0 : correction_max(log)
    return (model = model, rmse = metrics.rmse,
            boundary_max = metrics.boundary_max,
            balance_max = metrics.balance_max,
            box_max = metrics.box_max,
            status_success = success,
            status_total = total,
            success_rate = rate,
            correction_mean = cmean,
            correction_max = cmax,
            train_seconds = train_seconds,
            gate_passed = gate_passed,
            trained = trained)
end

function write_training_outputs(rows, out; source_path = "",
                                source_kind = :unknown)
    isdir(out) || mkpath(out)
    open(joinpath(out, "summer_batch_training_meta.csv"), "w") do io
        println(io, "metric,value")
        println(io, "source_path,", replace(string(source_path), "," => " "))
        println(io, "source_kind,", source_kind)
    end
    open(joinpath(out, "summer_batch_training.csv"), "w") do io
        println(io, "model,rmse,boundary_max,balance_max,box_max,status_success,status_total,success_rate,correction_mean,correction_max,train_seconds,gate_passed,trained")
        for row in rows
            println(io, join(fmt.((row.model, row.rmse, row.boundary_max,
                                   row.balance_max, row.box_max,
                                   row.status_success, row.status_total,
                                   row.success_rate, row.correction_mean,
                                   row.correction_max, row.train_seconds,
                                   row.gate_passed, row.trained)), ","))
        end
    end
    open(joinpath(out, "summer_batch_training.md"), "w") do io
        println(io, "# Summer Batch Training Gate")
        println(io)
        println(io, "| model | RMSE | boundary | balance | box | success rate | trained |")
        println(io, "| --- | --- | --- | --- | --- | --- | --- |")
        for row in rows
            println(io, "| ", row.model, " | ", fmt(row.rmse), " | ",
                    fmt(row.boundary_max), " | ", fmt(row.balance_max),
                    " | ", fmt(row.box_max), " | ", fmt(row.success_rate),
                    " | ", row.trained, " |")
        end
        println(io)
        println(io, "Train-time correction is run only when the evaluation-only status gate passes.")
    end
end

function run_exported_batch_training(samples; mode = :auto,
                                     out = DEFAULT_OUT, steps = 50,
                                     lr = 8e-3, seed = 1234,
                                     success_threshold = 0.95,
                                     source_path = "",
                                     source_kind = :unknown)
    feature_dim, _ = require_trainable(samples)
    x = batch_grid(samples)
    train_samples, test_samples = split_samples(samples)
    train_contexts = [context_for_sample(sample, mode = mode, cached = true)
                      for sample in train_samples]
    test_contexts = [context_for_sample(sample, mode = mode, cached = true)
                     for sample in test_samples]

    p0 = init_deeponet(nin = feature_dim, seed = seed)
    vanilla_seconds = @elapsed pv, _ =
        train!(p -> batch_vanilla_loss(p, train_samples, x), p0,
               steps = steps, lr = lr)

    raw_metrics =
        evaluate_batch_predictions(test_samples,
                                   raw_model_predictions(pv, test_samples,
                                                         x))
    eval_log = ProjectionLog()
    eval_metrics =
        evaluate_batch_predictions(test_samples,
                                   corrected_model_predictions(pv,
                                                               test_contexts,
                                                               x,
                                                               eval_log))
    gate = success_rate(eval_log) >= success_threshold

    rows = NamedTuple[]
    push!(rows, metric_row("vanilla", raw_metrics,
                           train_seconds = vanilla_seconds))
    push!(rows, metric_row("eval_only_corrected", eval_metrics,
                           log = eval_log, gate_passed = gate))

    if gate
        train_log = ProjectionLog()
        hard_seconds = @elapsed ph, _ =
            train!(p -> batch_context_hard_loss(p, train_contexts, x,
                                                log = train_log),
                   p0, steps = steps, lr = lr)
        hard_eval_log = ProjectionLog()
        hard_metrics =
            evaluate_batch_predictions(test_samples,
                                       corrected_model_predictions(ph,
                                                                   test_contexts,
                                                                   x,
                                                                   hard_eval_log))
        push!(rows, metric_row("train_time_corrected", hard_metrics,
                               train_seconds = hard_seconds,
                               log = hard_eval_log,
                               gate_passed = gate))
    else
        skipped = (rmse = NaN, boundary_max = NaN, balance_max = NaN,
                   box_max = NaN)
        push!(rows, metric_row("train_time_corrected", skipped,
                               gate_passed = gate, trained = false))
    end

    write_training_outputs(rows, out, source_path = source_path,
                           source_kind = source_kind)
    return rows
end

function parse_env_int(name, default)
    value = get(ENV, name, "")
    isempty(value) && return default
    return parse(Int, value)
end

function parse_env_float(name, default)
    value = get(ENV, name, "")
    isempty(value) && return default
    return parse(Float64, value)
end

function detect_source_kind(path)
    value = get(ENV, "STRUCTPINN_SUMMER_BATCH_SOURCE_KIND", "")
    if !isempty(value)
        return Symbol(lowercase(strip(value)))
    end
    name = lowercase(basename(path))
    occursin("fixture", name) && return :fixture
    return :exported
end

function main(args = ARGS)
    isempty(args) &&
        throw(ArgumentError("usage: julia --project=benchmarks/deeponet benchmarks/deeponet/summer_batch_train.jl path/to/batch.csv [mode]"))
    input = args[1]
    mode = length(args) >= 2 ? Symbol(args[2]) : :auto
    rows = run_exported_batch_training(read_exported_batch(input),
                                       mode = mode,
                                       steps =
                                           parse_env_int("STRUCTPINN_SUMMER_BATCH_STEPS",
                                                         50),
                                       success_threshold =
                                           parse_env_float("STRUCTPINN_SUMMER_BATCH_SUCCESS_THRESHOLD",
                                                           0.95),
                                       source_path = input,
                                       source_kind = detect_source_kind(input))
    println("wrote summer_batch_training.csv and summer_batch_training.md to ",
            DEFAULT_OUT)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
