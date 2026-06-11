# Gate expanded DeepONet seed runs.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/expanded_seed_gate.jl
include("DeepONetScenarios.jl")
using .DeepONetScenarios
using DelimitedFiles, Printf

const SCENARIO = expansion_gate_scenario()
const OUT = SCENARIO.out
isdir(OUT) || mkdir(OUT)

function read_table(path)
    data, header = readdlm(path, ',', header = true)
    names = vec(String.(header))
    if data isa AbstractVector
        data = reshape(data, 1, :)
    end
    return data, names
end

function find_col(names, name)
    idx = findfirst(==(name), names)
    idx === nothing && error("missing column $(name)")
    return idx
end

function metric_value(path, key)
    data, names = read_table(path)
    kcol = find_col(names, "metric")
    vcol = find_col(names, "value")
    for i in axes(data, 1)
        string(data[i, kcol]) == key && return string(data[i, vcol])
    end
    return ""
end

function row_value(path, model, column)
    data, names = read_table(path)
    mcol = find_col(names, "model")
    ccol = find_col(names, column)
    for i in axes(data, 1)
        string(data[i, mcol]) == model && return string(data[i, ccol])
    end
    return ""
end

function metric_value_if_exists(path, key)
    existing(path) || return ""
    return metric_value(path, key)
end

function bool_text(value)
    return lowercase(strip(string(value))) in ("true", "1", "yes", "y")
end

function existing(path)
    return isfile(path)
end

function acceptable_source_kind(path)
    kind = lowercase(metric_value_if_exists(path, "source_kind"))
    return kind in ("exported", "real", "summer")
end

function summer_training_stable(path)
    existing(path) || return false
    eval_gate = bool_text(row_value(path, "eval_only_corrected",
                                    "gate_passed"))
    train_ran = bool_text(row_value(path, "train_time_corrected",
                                    "trained"))
    meta_path = joinpath(dirname(path), "summer_batch_training_meta.csv")
    return eval_gate && train_ran && acceptable_source_kind(meta_path)
end

function passfail(flag)
    return flag ? "pass" : "fail"
end

function command_for(kind)
    if kind == :main
        return "STRUCTPINN_DEEPONET_STUDY_SEEDS=1:$(SCENARIO.main_seed_target) julia --project=benchmarks/deeponet benchmarks/deeponet/study.jl"
    elseif kind == :large
        return "STRUCTPINN_DEEPONET_LARGE_SEEDS=1:$(SCENARIO.large_seed_target) julia --project=benchmarks/deeponet benchmarks/deeponet/large_study.jl"
    end
    error("unknown command kind")
end

results_path = joinpath(OUT, "results.csv")
large_path = joinpath(OUT, "large_results.csv")
sparse_path = joinpath(OUT, "sparse_decision.csv")
larger_profile_path = joinpath(OUT, "larger_projection_profile.csv")
summer_training_path = joinpath(OUT, "summer_batch_training.csv")
summer_training_meta_path = joinpath(OUT, "summer_batch_training_meta.csv")

core_ready = existing(results_path) && existing(large_path)
sparse_ready = existing(sparse_path) &&
               bool_text(metric_value(sparse_path,
                                      "larger_profile_used"))
summer_ready = summer_training_stable(summer_training_path)
summer_source_kind =
    metric_value_if_exists(summer_training_meta_path, "source_kind")
interface_gate = SCENARIO.require_summer ? summer_ready :
                 (summer_ready || SCENARIO.allow_synthetic)
if SCENARIO.allow_synthetic
    interface_gate = true
end

main_allowed = core_ready && sparse_ready && interface_gate
large_allowed = core_ready && sparse_ready && interface_gate
expansion_scope = summer_ready ? "real" : "synthetic"

checks = [
    (check = "core_results_exist", status = passfail(core_ready),
     detail = "results.csv and large_results.csv are present"),
    (check = "larger_sparse_profile_used", status = passfail(sparse_ready),
     detail = "sparse_decision.csv consumed larger_projection_profile.csv"),
    (check = "summer_training_stable", status = passfail(summer_ready),
     detail = "summer_batch_training.csv passed eval and train gates for an exported source"),
    (check = "summer_training_source", status =
         acceptable_source_kind(summer_training_meta_path) ? "pass" :
         "fail",
     detail = isempty(summer_source_kind) ? "missing source metadata" :
         "source_kind $(summer_source_kind)"),
    (check = "synthetic_expansion_override", status =
         SCENARIO.allow_synthetic ? "pass" : "not_used",
     detail = "synthetic seed expansion is allowed by default and set STRUCTPINN_DEEPONET_EXPANSION_REQUIRES_SUMMER=true to re-couple it to the summer gate"),
    (check = "summer_claims_allowed", status = passfail(summer_ready),
     detail = "geothermal and summer claims require a real exported batch that passes the training gate"),
    (check = "expansion_scope", status = expansion_scope,
     detail = "expanded seed results without a real export are synthetic helper results and must be labeled as synthetic"),
    (check = "expanded_main_allowed", status = passfail(main_allowed),
     detail = command_for(:main)),
    (check = "expanded_large_allowed", status = passfail(large_allowed),
     detail = command_for(:large)),
]

open(joinpath(OUT, "expanded_seed_gate.csv"), "w") do io
    println(io, "check,status,detail")
    for row in checks
        println(io, row.check, ",", row.status, ",",
                replace(row.detail, "," => " "))
    end
end

open(joinpath(OUT, "expanded_seed_gate.md"), "w") do io
    println(io, "# Expanded Seed Gate")
    println(io)
    println(io, "Main seed target: ", SCENARIO.main_seed_target)
    println(io, "Large-grid seed target: ", SCENARIO.large_seed_target)
    println(io, "Require summer gate: ", SCENARIO.require_summer)
    println(io, "Synthetic override: ", SCENARIO.allow_synthetic)
    println(io)
    println(io, "| check | status | detail |")
    println(io, "| --- | --- | --- |")
    for row in checks
        println(io, "| ", row.check, " | ", row.status, " | ",
                row.detail, " |")
    end
    println(io)
    if main_allowed && large_allowed
        println(io, "Expanded seed runs are allowed by the current gates.")
        println(io, "Expansion scope: ", expansion_scope, ".")
        if expansion_scope == "synthetic"
            println(io, "These runs are synthetic helper results. They support the paper scaffold and must not be presented as geothermal or summer results.")
        end
        println(io)
        println(io, "Run:")
        println(io)
        println(io, "```powershell")
        println(io, command_for(:main))
        println(io, command_for(:large))
        println(io, "```")
    else
        println(io, "Expanded seed runs are not yet allowed by the current gates.")
    end
end

println("wrote expanded_seed_gate.csv and expanded_seed_gate.md to ", OUT)
