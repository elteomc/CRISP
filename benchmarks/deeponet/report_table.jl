# Compact report tables for DeepONet helper artifacts.
# Run after study, large_study, profile_projection, and eval_only_example.
# julia --project=benchmarks/deeponet benchmarks/deeponet/report_table.jl
using DelimitedFiles, Printf

const OUT = joinpath(@__DIR__, "results")

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

function rows_by(data, names, column, value)
    mask = String.(data[:, col(names, column)]) .== value
    return data[mask, :]
end

function row_by(data, names, column, value)
    rows = rows_by(data, names, column, value)
    size(rows, 1) > 0 || error("missing row $(value)")
    return rows[1, :]
end

function fmt(x)
    return @sprintf("%.4g", Float64(x))
end

function csv_line(io, cells)
    fields = string.(cells)
    if length(fields) < 7
        fields = vcat(fields, fill("", 7 - length(fields)))
    end
    println(io, join(fields[1:7], ","))
end

function md_header(io, cells)
    println(io, "| ", join(cells, " | "), " |")
    println(io, "| ", join(fill("---", length(cells)), " | "), " |")
end

function md_line(io, cells)
    println(io, "| ", join(string.(cells), " | "), " |")
end

function model_summary_row(data, names, model)
    row = row_by(data, names, "model", model)
    return [model,
            fmt(row[col(names, "rmse_mean")]),
            fmt(row[col(names, "boundarymax_mean")]),
            fmt(row[col(names, "massmax_mean")]),
            fmt(row[col(names, "train_seconds_mean")])]
end

function write_main_table(io, csvio)
    data, names = table(joinpath(OUT, "results.csv"))
    models = ["vanilla", "soft_weak", "eval_only_full",
              "hard_full_cached", "soft_plus_hard_full_cached"]
    println(io, "## Ten Seed Helper Study")
    println(io)
    md_header(io, ["model", "RMSE", "boundary max", "mass max",
                   "train seconds"])
    for model in models
        row = model_summary_row(data, names, model)
        md_line(io, row)
        csv_line(csvio, vcat(["main"], row))
    end
    println(io)
end

function write_large_table(io, csvio)
    data, names = table(joinpath(OUT, "large_results.csv"))
    println(io, "## Larger Grid Study")
    println(io)
    md_header(io, ["K", "model", "RMSE", "mass max", "train seconds"])
    keep = ("vanilla", "eval_only_full", "hard_full_cached",
            "soft_plus_hard_full_cached")
    for i in axes(data, 1)
        model = String(data[i, col(names, "model")])
        model in keep || continue
        row = [Int(data[i, col(names, "grid")]), model,
               fmt(data[i, col(names, "rmse_mean")]),
               fmt(data[i, col(names, "massmax_mean")]),
               fmt(data[i, col(names, "train_seconds_mean")])]
        md_line(io, row)
        csv_line(csvio, vcat(["large"], row))
    end
    println(io)
end

function write_profile_table(io, csvio)
    data, names = table(joinpath(OUT, "projection_profile.csv"))
    println(io, "## Projection Profile")
    println(io)
    md_header(io, ["K", "raw seconds", "uncached seconds",
                   "cached seconds", "context seconds", "cache speedup"])
    for i in axes(data, 1)
        raw = Float64(data[i, col(names, "raw_seconds")])
        uncached = Float64(data[i, col(names,
                                      "uncached_project_seconds")])
        cached = Float64(data[i, col(names,
                                    "cached_project_seconds")])
        context = Float64(data[i, col(names,
                                     "context_project_seconds")])
        speedup = uncached / cached
        row = [Int(data[i, col(names, "grid")]), fmt(raw),
               fmt(uncached), fmt(cached), fmt(context), fmt(speedup)]
        md_line(io, row)
        csv_line(csvio, vcat(["profile"], row))
    end
    println(io)
end

function write_eval_table(io, csvio)
    data, names = table(joinpath(OUT, "eval_only_example.csv"))
    println(io, "## Evaluation Only Example")
    println(io)
    md_header(io, ["model", "RMSE", "boundary max", "mass max"])
    for model in ["raw", "corrected"]
        row = row_by(data, names, "model", model)
        out = [model, fmt(row[col(names, "rmse")]),
               fmt(row[col(names, "boundary_max")]),
               fmt(row[col(names, "mass_max")])]
        md_line(io, out)
        csv_line(csvio, vcat(["eval_only"], out))
    end
end

function write_status_table(io, csvio)
    path = joinpath(OUT, "statuses.csv")
    isfile(path) || return nothing
    data, names = table(path)
    println(io)
    println(io, "## Status And Correction Norms")
    println(io)
    md_header(io, ["phase", "model", "status", "count",
                   "correction mean", "correction max"])
    keep = ("eval_only_full", "hard_full_cached",
            "soft_plus_hard_full_cached")
    for i in axes(data, 1)
        model = String(data[i, col(names, "model")])
        model in keep || continue
        row = [String(data[i, col(names, "phase")]), model,
               String(data[i, col(names, "status")]),
               Int(data[i, col(names, "count")]),
               fmt(data[i, col(names, "correction_mean")]),
               fmt(data[i, col(names, "correction_max")])]
        md_line(io, row)
        csv_line(csvio, vcat(["status"], row))
    end
    println(io)
    return nothing
end

function write_frequency_table(io, csvio)
    path = joinpath(OUT, "frequency_ablation.csv")
    isfile(path) || return nothing
    data, names = table(path)
    println(io, "## Projection Frequency Ablation")
    println(io)
    md_header(io, ["model", "RMSE", "mass max", "train seconds"])
    for model in ["no_correction", "eval_only_full", "every_step",
                  "every_2_steps", "every_5_steps", "every_10_steps"]
        row = row_by(data, names, "model", model)
        out = [model, fmt(row[col(names, "rmse_mean")]),
               fmt(row[col(names, "massmax_mean")]),
               fmt(row[col(names, "train_seconds_mean")])]
        md_line(io, out)
        csv_line(csvio, vcat(["frequency"], out))
    end
    println(io)
    return nothing
end

function write_constraint_table(io, csvio)
    path = joinpath(OUT, "constraint_ablation.csv")
    isfile(path) || return nothing
    data, names = table(path)
    println(io, "## Constraint Family Ablation")
    println(io)
    md_header(io, ["model", "RMSE", "boundary max", "mass max"])
    for model in ["eval_boundary_only", "eval_box_only",
                  "eval_mass_only", "eval_boundary_box", "eval_full",
                  "hard_boundary_only", "hard_mass_only",
                  "hard_boundary_box", "hard_full"]
        row = row_by(data, names, "model", model)
        out = [model, fmt(row[col(names, "rmse_mean")]),
               fmt(row[col(names, "boundarymax_mean")]),
               fmt(row[col(names, "massmax_mean")])]
        md_line(io, out)
        csv_line(csvio, vcat(["constraint"], out))
    end
    println(io)
    return nothing
end

open(joinpath(OUT, "report_table.csv"), "w") do csvio
    csv_line(csvio, ["section", "field1", "field2", "field3", "field4",
                     "field5", "field6"])
    open(joinpath(OUT, "report_table.md"), "w") do io
        println(io, "# DeepONet Helper Report Tables")
        println(io)
        write_main_table(io, csvio)
        write_large_table(io, csvio)
        write_profile_table(io, csvio)
        write_eval_table(io, csvio)
        write_status_table(io, csvio)
        write_frequency_table(io, csvio)
        write_constraint_table(io, csvio)
    end
end

println("wrote report_table.md and report_table.csv to ", OUT)
