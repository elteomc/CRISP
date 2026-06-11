# Inspect summer batch training artifacts and decide integration readiness.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/summer_training_decision.jl
using DelimitedFiles, Printf

const OUT = joinpath(@__DIR__, "results")

function table(path)
    data, header = readdlm(path, ',', header = true)
    names = vec(String.(header))
    if data isa AbstractVector
        data = reshape(data, 1, :)
    end
    return data, names
end

function col(names, name)
    idx = findfirst(==(name), names)
    idx === nothing && error("missing column $(name)")
    return idx
end

function row_by(data, names, column, value)
    idx = findfirst(==(value), String.(data[:, col(names, column)]))
    idx === nothing && error("missing row $(value)")
    return data[idx, :]
end

function cell(row, names, name)
    return row[col(names, name)]
end

function bool_value(x)
    return lowercase(strip(string(x))) in ("true", "1", "yes", "y")
end

function float_value(x)
    text = lowercase(strip(string(x)))
    text == "nan" && return NaN
    return parse(Float64, text)
end

function metric_value(path, key)
    data, names = table(path)
    row = row_by(data, names, "metric", key)
    return string(cell(row, names, "value"))
end

function review_status(path, key)
    data, names = table(path)
    row = row_by(data, names, "check", key)
    return string(cell(row, names, "status"))
end

function model_row(path, model)
    data, names = table(path)
    return row_by(data, names, "model", model), names
end

function model_float(path, model, key)
    row, names = model_row(path, model)
    return float_value(cell(row, names, key))
end

function model_bool(path, model, key)
    row, names = model_row(path, model)
    return bool_value(cell(row, names, key))
end

function acceptable_source(kind)
    return lowercase(kind) in ("exported", "real", "summer")
end

function decision_for(source_kind, units_status, eval_gate, train_gate,
                      eval_rmse, train_rmse)
    if !acceptable_source(source_kind)
        return "wait_for_real_export"
    elseif units_status != "pass"
        return "review_units_before_integration"
    elseif !eval_gate
        return "do_not_integrate_eval_gate_failed"
    elseif !train_gate
        return "do_not_integrate_train_gate_failed"
    elseif train_rmse > eval_rmse
        return "defer_train_time_eval_only_better"
    end
    return "integrate_train_time_correction_candidate"
end

function fmt(x)
    x isa AbstractFloat && isnan(x) && return "nan"
    x isa AbstractFloat && return @sprintf("%.10g", x)
    return string(x)
end

training_path = joinpath(OUT, "summer_batch_training.csv")
meta_path = joinpath(OUT, "summer_batch_training_meta.csv")
review_path = joinpath(OUT, "summer_batch_review.csv")

isfile(training_path) || error("missing summer_batch_training.csv")
isfile(meta_path) || error("missing summer_batch_training_meta.csv")
isfile(review_path) || error("missing summer_batch_review.csv")

source_kind = metric_value(meta_path, "source_kind")
source_path = metric_value(meta_path, "source_path")
units_status = review_status(review_path, "units_metadata")
mode_status = review_status(review_path, "recommended_mode")

eval_success_rate = model_float(training_path, "eval_only_corrected",
                                "success_rate")
train_success_rate = model_float(training_path, "train_time_corrected",
                                 "success_rate")
eval_gate = model_bool(training_path, "eval_only_corrected",
                       "gate_passed") && eval_success_rate >= 0.95
train_gate = model_bool(training_path, "train_time_corrected",
                        "trained") && train_success_rate >= 0.95

vanilla_rmse = model_float(training_path, "vanilla", "rmse")
eval_rmse = model_float(training_path, "eval_only_corrected", "rmse")
train_rmse = model_float(training_path, "train_time_corrected", "rmse")
eval_cmean = model_float(training_path, "eval_only_corrected",
                         "correction_mean")
eval_cmax = model_float(training_path, "eval_only_corrected",
                        "correction_max")
train_cmean = model_float(training_path, "train_time_corrected",
                          "correction_mean")
train_cmax = model_float(training_path, "train_time_corrected",
                         "correction_max")
eval_improvement =
    iszero(vanilla_rmse) ? NaN : (vanilla_rmse - eval_rmse) / vanilla_rmse
train_improvement =
    iszero(eval_rmse) ? NaN : (eval_rmse - train_rmse) / eval_rmse

decision = decision_for(source_kind, units_status, eval_gate, train_gate,
                        eval_rmse, train_rmse)

rows = [
    ("decision", decision),
    ("source_kind", source_kind),
    ("source_path", source_path),
    ("recommended_mode_status", mode_status),
    ("units_status", units_status),
    ("eval_gate_passed", eval_gate),
    ("train_gate_passed", train_gate),
    ("vanilla_rmse", vanilla_rmse),
    ("eval_only_rmse", eval_rmse),
    ("train_time_rmse", train_rmse),
    ("eval_only_rmse_improvement_vs_vanilla", eval_improvement),
    ("train_time_rmse_improvement_vs_eval_only", train_improvement),
    ("eval_correction_mean", eval_cmean),
    ("eval_correction_max", eval_cmax),
    ("train_correction_mean", train_cmean),
    ("train_correction_max", train_cmax),
]

open(joinpath(OUT, "summer_training_decision.csv"), "w") do io
    println(io, "metric,value")
    for (key, value) in rows
        println(io, key, ",", replace(fmt(value), "," => " "))
    end
end

open(joinpath(OUT, "summer_training_decision.md"), "w") do io
    println(io, "# Summer Training Integration Decision")
    println(io)
    println(io, "Decision: `", decision, "`")
    println(io)
    println(io, "Source kind: `", source_kind, "`")
    println(io, "Source path: `", source_path, "`")
    println(io)
    println(io, "| metric | value |")
    println(io, "| --- | --- |")
    for (key, value) in rows[4:end]
        println(io, "| ", key, " | ", fmt(value), " |")
    end
    println(io)
    if decision == "wait_for_real_export"
        println(io, "Train-time correction is promising on the fixture, but it should not be integrated into the real training loop until an exported summer batch passes the same gate.")
    elseif decision == "integrate_train_time_correction_candidate"
        println(io, "Train-time correction is a candidate for integration. Keep status logging and correction-norm reporting in the real loop.")
    else
        println(io, "Train-time correction is not ready for integration under the current gate.")
    end
end

println("wrote summer_training_decision.csv and summer_training_decision.md to ",
        OUT)
