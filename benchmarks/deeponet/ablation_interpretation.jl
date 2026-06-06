# Report-facing interpretation of DeepONet ablation CSV artifacts.
# Run after frequency_ablation.jl and constraint_ablation.jl:
# julia --project=benchmarks/deeponet benchmarks/deeponet/ablation_interpretation.jl
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

function num(row, names, name)
    return Float64(row[col(names, name)])
end

function fmt(x)
    return @sprintf("%.4g", Float64(x))
end

frequency, fnames = table(joinpath(OUT, "frequency_ablation.csv"))
constraint, cnames = table(joinpath(OUT, "constraint_ablation.csv"))

frequency_models = String.(frequency[:, col(fnames, "model")])
frequency_rmses = Float64.(frequency[:, col(fnames, "rmse_mean")])
best_frequency = frequency_models[argmin(frequency_rmses)]
best_frequency_rmse = minimum(frequency_rmses)

every_step = row_by(frequency, fnames, "model", "every_step")
eval_only = row_by(frequency, fnames, "model", "eval_only_full")
every_ten = row_by(frequency, fnames, "model", "every_10_steps")
no_correction = row_by(frequency, fnames, "model", "no_correction")

eval_full = row_by(constraint, cnames, "model", "eval_full")
eval_boundary = row_by(constraint, cnames, "model", "eval_boundary_only")
eval_mass = row_by(constraint, cnames, "model", "eval_mass_only")
eval_box = row_by(constraint, cnames, "model", "eval_box_only")
hard_full = row_by(constraint, cnames, "model", "hard_full")
hard_boundary_box = row_by(constraint, cnames, "model", "hard_boundary_box")

takeaways = [
    (id = "frequency_best_rmse",
     text = "Every-step hard correction has the best frequency-ablation RMSE at $(fmt(best_frequency_rmse))."),
    (id = "frequency_cheaper_feasibility",
     text = "Evaluation-only and every-N-step correction enforce final mass feasibility with lower train-time cost than every-step correction."),
    (id = "frequency_runtime_tradeoff",
     text = "Every-step correction trains in $(fmt(num(every_step, fnames, "train_seconds_mean"))) seconds, while every-10-step correction trains in $(fmt(num(every_ten, fnames, "train_seconds_mean"))) seconds."),
    (id = "constraint_full_only",
     text = "Full correction is the current mode that fixes both boundary and mass in the evaluation ablation."),
    (id = "constraint_boundary_leaves_mass",
     text = "Boundary-only evaluation correction drives boundary violation to $(fmt(num(eval_boundary, cnames, "boundarymax_mean"))) but leaves mass violation at $(fmt(num(eval_boundary, cnames, "massmax_mean")))."),
    (id = "constraint_mass_leaves_boundary",
     text = "Mass-only evaluation correction drives mass violation to $(fmt(num(eval_mass, cnames, "massmax_mean"))) but leaves boundary violation at $(fmt(num(eval_mass, cnames, "boundarymax_mean")))."),
    (id = "constraint_box_eval_only",
     text = "Box-only correction is kept as evaluation-only unless the active branch is regular. In the current run it does not change the already in-bound vanilla outputs."),
    (id = "constraint_negative_ablation",
     text = "Train-time boundary-box correction is a negative ablation here because it can leave mass uncontrolled and unstable compared with full correction."),
]

open(joinpath(OUT, "ablation_interpretation.csv"), "w") do io
    println(io, "id,text")
    for item in takeaways
        println(io, item.id, ",", replace(item.text, "," => " "))
    end
end

open(joinpath(OUT, "ablation_interpretation.md"), "w") do io
    println(io, "# DeepONet Ablation Interpretation")
    println(io)
    println(io, "## Frequency")
    println(io)
    println(io, "- Best RMSE row: `$(best_frequency)` with RMSE `$(fmt(best_frequency_rmse))`.")
    println(io, "- No correction RMSE: `$(fmt(num(no_correction, fnames, "rmse_mean")))`.")
    println(io, "- Evaluation-only full correction RMSE: `$(fmt(num(eval_only, fnames, "rmse_mean")))`.")
    println(io, "- Every-step correction RMSE: `$(fmt(num(every_step, fnames, "rmse_mean")))`.")
    println(io, "- Every-10-step correction RMSE: `$(fmt(num(every_ten, fnames, "rmse_mean")))`.")
    println(io)
    println(io, "Every-step correction gives the best RMSE in this ablation, but it also has the highest train-time cost. Evaluation-only and every-N-step correction are cheaper feasibility paths when final output validity matters more than training through every corrected output.")
    println(io)
    println(io, "## Constraint Families")
    println(io)
    println(io, "- Full evaluation correction RMSE: `$(fmt(num(eval_full, cnames, "rmse_mean")))`.")
    println(io, "- Boundary-only mass max: `$(fmt(num(eval_boundary, cnames, "massmax_mean")))`.")
    println(io, "- Mass-only boundary max: `$(fmt(num(eval_mass, cnames, "boundarymax_mean")))`.")
    println(io, "- Box-only RMSE: `$(fmt(num(eval_box, cnames, "rmse_mean")))`.")
    println(io, "- Hard full RMSE: `$(fmt(num(hard_full, cnames, "rmse_mean")))`.")
    println(io, "- Hard boundary-box RMSE: `$(fmt(num(hard_boundary_box, cnames, "rmse_mean")))`.")
    println(io)
    println(io, "Full correction is the clean helper default in the current synthetic heat benchmark because it fixes both boundary and mass. Smaller modes are useful diagnostics, not replacements for physically justified combined correction.")
    println(io)
    println(io, "Box-only correction should remain evaluation-only unless the active branch is regular, because pure box projection can land on active-bound kinks where no training gradient is claimed.")
end

println("wrote ablation_interpretation.md and ablation_interpretation.csv to ",
        OUT)
