# Decide whether sparse solver internals should be worked on now.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/sparse_internal_work_decision.jl
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

function metric_value(path, key)
    data, names = table(path)
    idx = findfirst(==(key), String.(data[:, col(names, "metric")]))
    idx === nothing && error("missing sparse metric $(key)")
    return string(data[idx, col(names, "value")])
end

function float_metric(path, key)
    return parse(Float64, metric_value(path, key))
end

function action_for(decision)
    if decision in ("defer_sparse_internals",
                    "defer_sparse_internals_until_real_bottleneck")
        return "defer_solver_internal_work"
    elseif decision == "profile_larger_outputs_before_deciding"
        return "run_larger_output_profile_first"
    elseif decision == "measure_solver_internals_on_summer_scale"
        return "measure_solver_internals_on_real_scale"
    end
    return "review_unknown_sparse_decision"
end

function detail_for(action)
    if action == "defer_solver_internal_work"
        return "cached contexts remain the default and projection is not the measured bottleneck"
    elseif action == "run_larger_output_profile_first"
        return "larger-output profile evidence is needed before solver work"
    elseif action == "measure_solver_internals_on_real_scale"
        return "summer-scale profiling shows enough overhead to time solver internals"
    end
    return "unknown sparse decision needs manual review"
end

sparse_path = joinpath(OUT, "sparse_decision.csv")
isfile(sparse_path) || error("missing sparse_decision.csv")

decision = metric_value(sparse_path, "decision")
action = action_for(decision)
larger_seconds = float_metric(sparse_path, "larger_context_seconds")
larger_over_raw = float_metric(sparse_path, "larger_context_over_raw")
max_grid = Int(round(float_metric(sparse_path, "max_profile_grid")))

rows = [
    ("action", action),
    ("sparse_decision", decision),
    ("max_profile_grid", max_grid),
    ("larger_context_seconds", larger_seconds),
    ("larger_context_over_raw", larger_over_raw),
    ("detail", detail_for(action)),
]

open(joinpath(OUT, "sparse_internal_work_decision.csv"), "w") do io
    println(io, "metric,value")
    for (key, value) in rows
        println(io, key, ",", replace(string(value), "," => " "))
    end
end

open(joinpath(OUT, "sparse_internal_work_decision.md"), "w") do io
    println(io, "# Sparse Internal Work Decision")
    println(io)
    println(io, "Action: `", action, "`")
    println(io)
    println(io, "| metric | value |")
    println(io, "| --- | --- |")
    for (key, value) in rows[2:end]
        println(io, "| ", key, " | ", value, " |")
    end
    println(io)
    println(io, detail_for(action), ".")
end

println("wrote sparse_internal_work_decision.csv and sparse_internal_work_decision.md to ",
        OUT)
