# Verification gate for DeepONet report-facing artifacts.
# Run after artifact generation:
# julia --project=benchmarks/deeponet benchmarks/deeponet/verify_report_artifacts.jl
using DelimitedFiles, Printf

const ROOT = @__DIR__
const OUT = joinpath(ROOT, "results")
const SECTION = joinpath(ROOT, "deeponet_results_section.md")
const TOL = 1e-9

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

function require_file(path)
    isfile(path) || error("missing artifact $(path)")
    return path
end

function require_nonempty(path)
    require_file(path)
    filesize(path) > 0 || error("empty artifact $(path)")
    return path
end

function require_text(path, snippets)
    text = read(path, String)
    for snippet in snippets
        occursin(snippet, text) || error("missing text $(snippet) in $(path)")
    end
    return text
end

function require_close_zero(value, label)
    abs(Float64(value)) <= TOL || error("$(label) expected near zero")
    return nothing
end

function require_positive(value, label)
    Float64(value) > TOL || error("$(label) expected positive")
    return nothing
end

function verify_main_results()
    data, names = table(require_file(joinpath(OUT, "results.csv")))
    hard = row_by(data, names, "model", "hard_full_cached")
    eval_only = row_by(data, names, "model", "eval_only_full")
    soft = row_by(data, names, "model", "soft_weak")
    vanilla = row_by(data, names, "model", "vanilla")
    sph = row_by(data, names, "model", "soft_plus_hard_full_cached")

    for (label, row) in [("hard_full_cached", hard),
                         ("eval_only_full", eval_only),
                         ("soft_plus_hard_full_cached", sph)]
        require_close_zero(cell(row, names, "boundarymax_mean"),
                           "$(label) boundary")
        require_close_zero(cell(row, names, "massmax_mean"),
                           "$(label) mass")
    end
    require_positive(cell(soft, names, "boundarymax_mean"),
                     "soft boundary violation")
    require_positive(cell(vanilla, names, "boundarymax_mean"),
                     "vanilla boundary violation")
    return nothing
end

function verify_statuses()
    for filename in ["statuses.csv", "large_statuses.csv",
                     "frequency_statuses.csv", "constraint_statuses.csv"]
        path = joinpath(OUT, filename)
        data, names = table(require_file(path))
        statuses = String.(data[:, col(names, "status")])
        for status in statuses
            status in ("none", "success") ||
                error("unexpected report status $(status) in $(filename)")
        end
    end
    return nothing
end

function verify_stress_outputs()
    for filename in ["stress_diagnostics.csv", "failure_mode_summary.csv"]
        data, names = table(require_file(joinpath(OUT, filename)))
        passes = string.(data[:, col(names, "pass")])
        all(==("true"), passes) || error("failed stress row in $(filename)")
    end
    return nothing
end

function verify_soft_sweep()
    data, names = table(require_file(joinpath(OUT, "soft_sweep_pareto.csv")))
    models = String.(data[:, col(names, "model")])
    "soft_weak" in models || error("missing soft_weak row")
    any(string.(data[:, col(names, "best_rmse")]) .== "true") ||
        error("missing best RMSE flag")
    any(string.(data[:, col(names, "best_violation")]) .== "true") ||
        error("missing best violation flag")
    return nothing
end

function verify_sparse_decision()
    data, names = table(require_file(joinpath(OUT, "sparse_decision.csv")))
    decision = row_by(data, names, "metric", "decision")
    value = string(cell(decision, names, "value"))
    value in ("defer_sparse_internals",
              "profile_larger_outputs_before_deciding") ||
        error("unexpected sparse decision $(value)")
    max_grid = row_by(data, names, "metric", "max_profile_grid")
    Int(parse(Float64, string(cell(max_grid, names, "value")))) >= 256 ||
        error("profile grid ceiling is too small")
    return nothing
end

function verify_section()
    require_file(SECTION)
    require_text(SECTION, [
        "Table placeholder: `deeponet_main_results`.",
        "Figure placeholder: `deeponet_report_bundle`.",
        "Figure placeholder: `failure_mode_diagnostics`.",
        "Do not claim hard correction always improves RMSE.",
        "Do not claim geothermal conservation from this synthetic heat helper.",
        "Treat the current results as pilot-scale helper evidence",
    ])
    return nothing
end

function verify_required_artifacts()
    for filename in ["summary.txt", "report_table.csv", "report_table.md",
                     "run_scenarios.csv", "result_run_plan.csv",
                     "soft_sweep_pareto.csv", "ablation_interpretation.csv",
                     "sparse_decision.csv", "stress_diagnostics.csv",
                     "failure_mode_summary.csv"]
        require_nonempty(joinpath(OUT, filename))
    end
    for filename in ["deeponet_report_bundle.svg",
                     "deeponet_soft_sweep_pareto.svg",
                     "failure_mode_diagnostics.svg"]
        require_nonempty(joinpath(OUT, filename))
    end
    return nothing
end

verify_required_artifacts()
verify_main_results()
verify_statuses()
verify_stress_outputs()
verify_soft_sweep()
verify_sparse_decision()
verify_section()

println("DeepONet report artifact verification passed.")
