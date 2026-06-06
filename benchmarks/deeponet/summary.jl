# Report-facing summary for DeepONet helper artifacts.
# Run after study, large_study, profile_projection, and eval_only_example.
# julia --project=benchmarks/deeponet benchmarks/deeponet/summary.jl
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

function row_by(data, names, column, value)
    idx = findfirst(==(value), String.(data[:, col(names, column)]))
    idx === nothing && error("missing row $(value)")
    return data[idx, :]
end

function cell(row, names, name)
    return row[col(names, name)]
end

function fmt(x)
    return @sprintf("%.4g", Float64(x))
end

open(joinpath(OUT, "summary.txt"), "w") do io
    println(io, "StructPINN DeepONet helper summary")
    println(io)

    results_path = joinpath(OUT, "results.csv")
    if isfile(results_path)
        data, names = table(results_path)
        for model in ["vanilla", "soft_weak", "eval_only_full",
                      "hard_full", "hard_full_cached",
                      "soft_plus_hard_full_cached"]
            row = row_by(data, names, "model", model)
            println(io, model)
            println(io, "  RMSE: ", fmt(cell(row, names, "rmse_mean")))
            println(io, "  boundary max: ",
                    fmt(cell(row, names, "boundarymax_mean")))
            println(io, "  mass max: ", fmt(cell(row, names, "massmax_mean")))
            println(io, "  train seconds: ",
                    fmt(cell(row, names, "train_seconds_mean")))
        end
        println(io)
    end

    profile_path = joinpath(OUT, "projection_profile.csv")
    if isfile(profile_path)
        data, names = table(profile_path)
        println(io, "Projection profile")
        for i in axes(data, 1)
            grid = Int(data[i, col(names, "grid")])
            raw = Float64(data[i, col(names, "raw_seconds")])
            uncached = Float64(data[i, col(names,
                                          "uncached_project_seconds")])
            cached = Float64(data[i, col(names,
                                        "cached_project_seconds")])
            context = Float64(data[i, col(names,
                                         "context_project_seconds")])
            println(io, @sprintf("  K=%d raw %.3e, uncached %.3e, cached %.3e, context %.3e",
                                 grid, raw, uncached, cached, context))
        end
        println(io)
    end

    large_path = joinpath(OUT, "large_results.csv")
    if isfile(large_path)
        data, names = table(large_path)
        println(io, "Large-grid rows")
        for i in axes(data, 1)
            model = String(data[i, col(names, "model")])
            model in ("hard_full_cached", "soft_plus_hard_full_cached",
                      "eval_only_full") || continue
            println(io, @sprintf("  K=%d %s RMSE %s mass %s train %s",
                                 Int(data[i, col(names, "grid")]), model,
                                 fmt(data[i, col(names, "rmse_mean")]),
                                 fmt(data[i, col(names, "massmax_mean")]),
                                 fmt(data[i, col(names,
                                                "train_seconds_mean")])))
        end
        println(io)
    end

    eval_path = joinpath(OUT, "eval_only_example.csv")
    if isfile(eval_path)
        data, names = table(eval_path)
        raw = row_by(data, names, "model", "raw")
        corrected = row_by(data, names, "model", "corrected")
        println(io, "Evaluation-only example")
        println(io, "  raw RMSE: ", fmt(cell(raw, names, "rmse")))
        println(io, "  corrected RMSE: ",
                fmt(cell(corrected, names, "rmse")))
        println(io, "  corrected mass max: ",
                fmt(cell(corrected, names, "mass_max")))
        println(io, "  corrected boundary max: ",
                fmt(cell(corrected, names, "boundary_max")))
    end

    frequency_path = joinpath(OUT, "frequency_ablation.csv")
    if isfile(frequency_path)
        data, names = table(frequency_path)
        println(io)
        println(io, "Projection-frequency ablation")
        for model in ["no_correction", "eval_only_full", "every_step",
                      "every_2_steps", "every_5_steps", "every_10_steps"]
            row = row_by(data, names, "model", model)
            println(io, "  ", model, " RMSE ",
                    fmt(cell(row, names, "rmse_mean")), " mass ",
                    fmt(cell(row, names, "massmax_mean")), " train ",
                    fmt(cell(row, names, "train_seconds_mean")))
        end
    end

    constraint_path = joinpath(OUT, "constraint_ablation.csv")
    if isfile(constraint_path)
        data, names = table(constraint_path)
        println(io)
        println(io, "Constraint-family ablation")
        for model in ["eval_boundary_only", "eval_box_only",
                      "eval_mass_only", "eval_boundary_box", "eval_full"]
            row = row_by(data, names, "model", model)
            println(io, "  ", model, " RMSE ",
                    fmt(cell(row, names, "rmse_mean")), " boundary ",
                    fmt(cell(row, names, "boundarymax_mean")), " mass ",
                    fmt(cell(row, names, "massmax_mean")))
        end
    end

    sparse_path = joinpath(OUT, "sparse_decision.csv")
    if isfile(sparse_path)
        data, names = table(sparse_path)
        println(io)
        println(io, "Sparse optimization decision")
        for i in axes(data, 1)
            println(io, "  ", string(data[i, col(names, "metric")]),
                    ": ", string(data[i, col(names, "value")]))
        end
    end
end

println("wrote summary.txt to ", OUT)
