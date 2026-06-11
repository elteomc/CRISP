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
              "profile_larger_outputs_before_deciding",
              "defer_sparse_internals_until_real_bottleneck",
              "measure_solver_internals_on_summer_scale") ||
        error("unexpected sparse decision $(value)")
    max_grid = row_by(data, names, "metric", "max_profile_grid")
    Int(parse(Float64, string(cell(max_grid, names, "value")))) >= 256 ||
        error("profile grid ceiling is too small")
    larger_used = row_by(data, names, "metric", "larger_profile_used")
    string(cell(larger_used, names, "value")) == "true" ||
        error("larger-output profile was not consumed")
    return nothing
end

function verify_expanded_seed_gate()
    data, names = table(require_file(joinpath(OUT,
                                              "expanded_seed_gate.csv")))
    sparse = row_by(data, names, "check", "larger_sparse_profile_used")
    string(cell(sparse, names, "status")) == "pass" ||
        error("expanded seed gate did not use larger sparse profile")
    main = row_by(data, names, "check", "expanded_main_allowed")
    status = string(cell(main, names, "status"))
    status in ("pass", "fail") ||
        error("unexpected expanded main gate status $(status)")
    claims = row_by(data, names, "check", "summer_claims_allowed")
    claims_status = string(cell(claims, names, "status"))
    claims_status in ("pass", "fail") ||
        error("unexpected summer claims gate status $(claims_status)")
    scope = row_by(data, names, "check", "expansion_scope")
    scope_value = string(cell(scope, names, "status"))
    scope_value in ("synthetic", "real") ||
        error("unexpected expansion scope $(scope_value)")
    scope_value == "synthetic" && claims_status == "pass" &&
        error("synthetic expansion scope must keep summer claims closed")
    return nothing
end

function verify_guarded_actions()
    run_data, run_names = table(require_file(joinpath(OUT,
                                                      "expanded_seed_run.csv")))
    statuses = String.(run_data[:, col(run_names, "status")])
    all(status -> status in ("blocked", "complete"), statuses) ||
        error("unexpected expanded seed run status")
    scopes = String.(run_data[:, col(run_names, "scope")])
    all(scope -> scope in ("synthetic", "real"), scopes) ||
        error("unexpected expanded seed run scope")

    sparse, snames = table(require_file(joinpath(OUT,
                                                 "sparse_internal_work_decision.csv")))
    action = row_by(sparse, snames, "metric", "action")
    string(cell(action, snames, "value")) in
        ("defer_solver_internal_work",
         "run_larger_output_profile_first",
         "measure_solver_internals_on_real_scale",
         "review_unknown_sparse_decision") ||
        error("unexpected sparse internal work action")
    return nothing
end

function verify_summer_batch_eval()
    require_file(joinpath(OUT, "summer_batch_fixture.csv"))
    review, rnames = table(require_file(joinpath(OUT,
                                                 "summer_batch_review.csv")))
    source = row_by(review, rnames, "check", "source_kind")
    string(cell(source, rnames, "status")) == "fixture" ||
        error("summer batch fixture source was not marked")
    data, names = table(require_file(joinpath(OUT, "summer_batch_eval.csv")))
    statuses = String.(data[:, col(names, "status")])
    all(==("success"), statuses) ||
        error("summer batch eval contains non-success status")
    modes = String.(data[:, col(names, "mode")])
    all(==("full_boundary_balance_box"), modes) ||
        error("summer batch eval selected unexpected mode")
    train, tnames = table(require_file(joinpath(OUT,
                                                "summer_batch_training.csv")))
    eval_row = row_by(train, tnames, "model", "eval_only_corrected")
    string(cell(eval_row, tnames, "gate_passed")) == "true" ||
        error("fixture evaluation gate did not pass")
    hard_row = row_by(train, tnames, "model", "train_time_corrected")
    string(cell(hard_row, tnames, "trained")) == "true" ||
        error("fixture train-time correction did not run")
    meta, mnames = table(require_file(joinpath(OUT,
                                               "summer_batch_training_meta.csv")))
    kind = row_by(meta, mnames, "metric", "source_kind")
    string(cell(kind, mnames, "value")) == "fixture" ||
        error("fixture training source kind was not recorded")
    decision, dnames = table(require_file(joinpath(OUT,
                                                   "summer_training_decision.csv")))
    drow = row_by(decision, dnames, "metric", "decision")
    string(cell(drow, dnames, "value")) == "wait_for_real_export" ||
        error("fixture training decision should wait for real export")
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
                     "sparse_decision.csv", "larger_projection_profile.csv",
                     "expanded_seed_gate.csv", "summer_batch_fixture.csv",
                     "summer_batch_review.csv", "summer_batch_eval.csv",
                     "summer_batch_training.csv",
                     "summer_batch_training_meta.csv",
                     "summer_training_decision.csv",
                     "expanded_seed_run.csv",
                     "sparse_internal_work_decision.csv",
                     "stress_diagnostics.csv",
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
verify_expanded_seed_gate()
verify_guarded_actions()
verify_summer_batch_eval()
verify_section()

println("DeepONet report artifact verification passed.")
