# Sparse optimization decision gate for DeepONet helper artifacts.
# Run after profile_projection.jl and large_study.jl:
# julia --project=benchmarks/deeponet benchmarks/deeponet/sparse_decision.jl
using DelimitedFiles, Printf, Statistics

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

profile_path = joinpath(OUT, "projection_profile.csv")
larger_profile_path = joinpath(OUT, "larger_projection_profile.csv")
large_path = joinpath(OUT, "large_results.csv")
isfile(profile_path) || error("missing projection_profile.csv")
isfile(large_path) || error("missing large_results.csv")

profile, pnames = table(profile_path)
larger_profile_used = isfile(larger_profile_path)
if larger_profile_used
    larger_profile, larger_names = table(larger_profile_path)
    larger_names == pnames ||
        error("larger projection profile columns do not match base profile")
    profile = vcat(profile, larger_profile)
end
large, lnames = table(large_path)

grids = Int.(profile[:, col(pnames, "grid")])
raw = Float64.(profile[:, col(pnames, "raw_seconds")])
uncached = Float64.(profile[:, col(pnames, "uncached_project_seconds")])
cached = Float64.(profile[:, col(pnames, "cached_project_seconds")])
context = Float64.(profile[:, col(pnames, "context_project_seconds")])
construct = Float64.(profile[:, col(pnames, "construct_context_seconds")])

cached_speedup = uncached ./ cached
context_over_raw = context ./ raw
construct_over_context = construct ./ context

train_speedups = Float64[]
for K in sort(unique(Int.(large[:, col(lnames, "grid")])))
    rows = large[Int.(large[:, col(lnames, "grid")]) .== K, :]
    names = lnames
    uncached_row = row_by(rows, names, "model", "hard_full")
    cached_row = row_by(rows, names, "model", "hard_full_cached")
    push!(train_speedups,
          Float64(uncached_row[col(names, "train_seconds_mean")]) /
          Float64(cached_row[col(names, "train_seconds_mean")]))
end

max_context_seconds = maximum(context)
max_context_over_raw = maximum(context_over_raw)
min_cached_speedup = minimum(cached_speedup)
mean_train_speedup = mean(train_speedups)
max_profile_grid = maximum(grids)
profile_grid_count = length(unique(grids))
larger_mask = grids .> 256
larger_context_seconds =
    any(larger_mask) ? maximum(context[larger_mask]) : max_context_seconds
larger_context_over_raw =
    any(larger_mask) ? maximum(context_over_raw[larger_mask]) :
    max_context_over_raw

decision =
    larger_profile_used && larger_context_seconds < 1e-3 &&
    larger_context_over_raw < 5.0 ?
    "defer_sparse_internals_until_real_bottleneck" :
    !larger_profile_used ?
    "profile_larger_outputs_before_deciding" :
    max_context_seconds < 5e-4 && max_context_over_raw < 5.0 ?
    "defer_sparse_internals" :
    "measure_solver_internals_on_summer_scale"

rationale =
    decision == "defer_sparse_internals" ||
    decision == "defer_sparse_internals_until_real_bottleneck" ?
    "Cached context projection is sub-millisecond at the profiled grids and the remaining overhead is not yet the dominant DeepONet result bottleneck." :
    decision == "profile_larger_outputs_before_deciding" ?
    "Projection overhead is large enough that larger-grid profiling should happen before adding more experiments." :
    "Larger-output profiling shows enough projection overhead to justify solver-internal sparse timing on the summer-sized interface before more result expansion."

next_action =
    decision == "defer_sparse_internals" ||
    decision == "defer_sparse_internals_until_real_bottleneck" ?
    "Keep the cached context path as the default and defer solver-internal sparse work until a summer-sized interface or larger output grid makes projection the measured bottleneck." :
    larger_profile_used ?
    "Keep cached contexts as the default and measure solver-internal timing on the summer-sized interface before expanding result runs." :
    "Keep cached contexts as the default for current experiments, but profile larger outputs before committing to more result runs or solver-internal work."

open(joinpath(OUT, "sparse_decision.csv"), "w") do io
    println(io, "metric,value")
    println(io, "decision,$decision")
    println(io, "max_context_seconds,$(@sprintf("%.8e", max_context_seconds))")
    println(io, "max_context_over_raw,$(@sprintf("%.8f", max_context_over_raw))")
    println(io, "min_cached_speedup,$(@sprintf("%.8f", min_cached_speedup))")
    println(io, "mean_train_speedup,$(@sprintf("%.8f", mean_train_speedup))")
    println(io, "profile_grid_count,$profile_grid_count")
    println(io, "max_profile_grid,$max_profile_grid")
    println(io, "larger_profile_used,$larger_profile_used")
    println(io, "larger_context_seconds,$(@sprintf("%.8e", larger_context_seconds))")
    println(io, "larger_context_over_raw,$(@sprintf("%.8f", larger_context_over_raw))")
end

open(joinpath(OUT, "sparse_decision.md"), "w") do io
    println(io, "# Sparse Optimization Decision")
    println(io)
    println(io, "Decision: `$(decision)`")
    println(io)
    println(io, rationale)
    println(io)
    println(io, "## Profile Evidence")
    println(io)
    println(io, "Profiled grids: ", profile_grid_count,
            ". Maximum K: ", max_profile_grid, ".")
    println(io, "Larger-output profile used: ", larger_profile_used, ".")
    println(io)
    println(io, "| K | raw seconds | cached seconds | context seconds | cache speedup | context over raw |")
    println(io, "| --- | --- | --- | --- | --- | --- |")
    for i in eachindex(grids)
        println(io, "| ", grids[i], " | ", fmt(raw[i]), " | ",
                fmt(cached[i]), " | ", fmt(context[i]), " | ",
                fmt(cached_speedup[i]), " | ",
                fmt(context_over_raw[i]), " |")
    end
    println(io)
    println(io, "## Training Evidence")
    println(io)
    println(io, "Mean uncached-to-cached hard-training speedup: ",
            fmt(mean_train_speedup), ".")
    println(io)
    println(io, "Next action: ", next_action)
end

println("wrote sparse_decision.md and sparse_decision.csv to ", OUT)
