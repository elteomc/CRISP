# Soft-penalty sweep summary for the DeepONet helper benchmark.
# Run after study.jl:
# julia --project=benchmarks/deeponet benchmarks/deeponet/soft_sweep_summary.jl
include("DeepONetScenarios.jl")
using .DeepONetScenarios
using DelimitedFiles, Printf

const OUT = deeponet_results_dir()

function table(path)
    data, header = readdlm(path, ',', header = true)
    names = vec(String.(header))
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

function fmt(x)
    return @sprintf("%.4g", Float64(x))
end

function csv_text(value)
    return replace(string(value), "," => " ")
end

function violation_score(row, names)
    return maximum(Float64[
        row[col(names, "boundarymax_mean")],
        row[col(names, "massmax_mean")],
        row[col(names, "lowermax_mean")],
        row[col(names, "uppermax_mean")],
    ])
end

function is_pareto(rows)
    flags = trues(length(rows))
    objectives =
        [[row.rmse, row.violation, row.train_seconds] for row in rows]
    for i in eachindex(rows)
        for j in eachindex(rows)
            i == j && continue
            dominates =
                all(objectives[j] .<= objectives[i]) &&
                any(objectives[j] .< objectives[i])
            if dominates
                flags[i] = false
                break
            end
        end
    end
    return flags
end

results_path = joinpath(OUT, "results.csv")
isfile(results_path) || error("missing results.csv")

data, names = table(results_path)
scenario = study_scenario()

rows = NamedTuple[]
for cfg in scenario.soft_configs
    row = row_by(data, names, "model", cfg.name)
    push!(rows,
          (model = string(cfg.name),
           beta_boundary = cfg.beta_boundary,
           beta_mass = cfg.beta_mass,
           beta_box = cfg.beta_box,
           rmse = Float64(row[col(names, "rmse_mean")]),
           boundary = Float64(row[col(names, "boundarymax_mean")]),
           mass = Float64(row[col(names, "massmax_mean")]),
           lower = Float64(row[col(names, "lowermax_mean")]),
           upper = Float64(row[col(names, "uppermax_mean")]),
           violation = violation_score(row, names),
           train_seconds = Float64(row[col(names, "train_seconds_mean")])))
end

pareto = is_pareto(rows)
best_rmse_idx = argmin([row.rmse for row in rows])
best_violation_idx = argmin([row.violation for row in rows])
best_runtime_idx = argmin([row.train_seconds for row in rows])

open(joinpath(OUT, "soft_sweep_pareto.csv"), "w") do io
    println(io, "model,beta_boundary,beta_mass,beta_box,rmse_mean,boundarymax_mean,massmax_mean,lowermax_mean,uppermax_mean,violation_score,train_seconds_mean,pareto_efficient,best_rmse,best_violation,best_runtime")
    for i in eachindex(rows)
        row = rows[i]
        println(io, join(csv_text.([
            row.model, row.beta_boundary, row.beta_mass, row.beta_box,
            @sprintf("%.8f", row.rmse),
            @sprintf("%.8e", row.boundary),
            @sprintf("%.8e", row.mass),
            @sprintf("%.8e", row.lower),
            @sprintf("%.8e", row.upper),
            @sprintf("%.8e", row.violation),
            @sprintf("%.8f", row.train_seconds),
            pareto[i], i == best_rmse_idx, i == best_violation_idx,
            i == best_runtime_idx,
        ]), ","))
    end
end

references = ["vanilla", "eval_only_full", "hard_full_cached",
              "soft_plus_hard_full_cached"]

open(joinpath(OUT, "soft_sweep_pareto.md"), "w") do io
    println(io, "# DeepONet Soft Sweep Pareto Summary")
    println(io)
    println(io, "Soft rows are compared on RMSE, violation score, and train time.")
    println(io)
    println(io, "| model | beta boundary | beta mass | beta box | RMSE | violation | train seconds | Pareto | flags |")
    println(io, "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i in eachindex(rows)
        row = rows[i]
        flags = String[]
        i == best_rmse_idx && push!(flags, "best RMSE")
        i == best_violation_idx && push!(flags, "best violation")
        i == best_runtime_idx && push!(flags, "best runtime")
        isempty(flags) && push!(flags, "")
        println(io, "| ", row.model, " | ", row.beta_boundary,
                " | ", row.beta_mass, " | ", row.beta_box,
                " | ", fmt(row.rmse), " | ", fmt(row.violation),
                " | ", fmt(row.train_seconds), " | ", pareto[i],
                " | ", join(flags, " "), " |")
    end
    println(io)
    println(io, "## Reference Rows")
    println(io)
    println(io, "| model | RMSE | violation | train seconds |")
    println(io, "| --- | --- | --- | --- |")
    for model in references
        row = row_by(data, names, "model", model)
        println(io, "| ", model, " | ",
                fmt(row[col(names, "rmse_mean")]), " | ",
                fmt(violation_score(row, names)), " | ",
                fmt(row[col(names, "train_seconds_mean")]), " |")
    end
    println(io)
    println(io, "The soft sweep is a penalty-tuning diagnostic. It does not claim exact feasibility.")
end

println("wrote soft_sweep_pareto.csv and soft_sweep_pareto.md to ", OUT)
