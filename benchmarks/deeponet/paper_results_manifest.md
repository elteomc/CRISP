# DeepONet Paper Results Manifest

This manifest maps current DeepONet helper artifacts to commands and claims. It is a traceability file for report and paper drafting, not a final paper outline.

## Core Study

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/study.jl
```

Artifacts:

- `results.csv`
- `statuses.csv`
- `training_curves.csv`

Supported claims:

- Full hard correction enforces boundary and mass constraints to numerical precision when status is `:success`.
- Cached full correction matches uncached hard-correction metrics in the small helper run.
- Soft penalties reduce some violations but do not enforce exact feasibility in this pilot.
- Correction norms expose whether the raw model is far from the feasible set.

## Larger Grid Study

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/large_study.jl
```

Artifacts:

- `large_results.csv`
- `large_statuses.csv`

Supported claims:

- Cached hard correction remains feasible at K=64 and K=96.
- Runtime grows with output dimension but remains tractable for the current helper scale.
- Larger-grid status counts remain explicit.

## Projection Profile

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/profile_projection.jl
```

Artifacts:

- `projection_profile.csv`

Supported claims:

- Cached context projection is substantially faster than uncached projection.
- Current cached context projection is sub-millisecond at the profiled grids.
- Default profiling now includes larger output grids up to K=256.
- Larger-output profiling should happen before choosing more result runs or solver-internal sparse work.

## Larger Output Projection Profile

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/profile_larger_outputs.jl
```

Artifacts:

- `larger_projection_profile.csv`

Supported claims:

- The sparse decision gate includes larger-output projection timings beyond K=256.
- The current larger-output profile reaches K=512.
- Cached context projection remains sub-millisecond in the current larger-output profile.

## Run Scenario Manifest

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/scenario_manifest.jl
```

Artifacts:

- `run_scenarios.csv`
- `run_scenarios.md`

Supported claims:

- The DeepONet result-suite grids, seeds, steps, and sample counts are auditable before longer runs.
- Environment overrides can change result-suite size without editing benchmark source files.

## Summer Exported Batch Adapter

Fixture command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/export_summer_batch_fixture.jl
```

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/summer_batch_review.jl path\to\batch.csv
julia --project=benchmarks/deeponet benchmarks/deeponet/summer_batch_eval.jl path\to\batch.csv
```

Artifacts:

- `summer_batch_fixture.csv`
- `summer_batch_review.csv`
- `summer_batch_review.md`
- `summer_batch_eval.csv`
- `summer_batch_eval.md`

Supported claims:

- A file-backed summer batch can be corrected without depending on the synthetic heat sample type.
- The adapter reports status counts, correction norms, RMSE, boundary violation, balance violation, and box violation when the relevant metadata is present.
- The metadata review reports grid ordering, endpoint boundary consistency, target balance consistency, target bounds, source kind, and missing unit metadata.
- The automatically selected mode is a starting point for physical review, not a final constraint decision.
- The deterministic fixture is an adapter smoke test, not a real summer or geothermal result.

## Summer Batch Training Gate

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/summer_batch_train.jl path\to\batch.csv
julia --project=benchmarks/deeponet benchmarks/deeponet/summer_training_decision.jl
```

Artifacts:

- `summer_batch_training.csv`
- `summer_batch_training.md`
- `summer_batch_training_meta.csv`
- `summer_training_decision.csv`
- `summer_training_decision.md`

Supported claims:

- Batches with `features`, `target_output`, and `grid` can run the same evaluation-first workflow as the synthetic helper.
- Train-time correction is attempted only after the held-out evaluation-only correction status gate passes.
- The generated artifacts record raw model metrics, evaluation-only corrected metrics, train-time corrected metrics when run, status counts, correction norms, and the gate decision.
- The metadata artifact records whether the source was a fixture or an exported batch, so fixture runs cannot unlock summer or geothermal claims.
- The integration decision artifact blocks real-loop integration when the current evidence is only a fixture, when units need review, or when gates fail.

## Expanded Seed Gate

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/expanded_seed_gate.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/run_expanded_seed_jobs.jl
```

Artifacts:

- `expanded_seed_gate.csv`
- `expanded_seed_gate.md`
- `expanded_seed_run.csv`
- `expanded_seed_run.md`

Supported claims:

- Expanded main and larger-grid seed runs are gated rather than automatic.
- Synthetic seed expansion needs core helper artifacts and larger-output sparse profiling. It does not wait for the summer batch, which is expected August 30 at the earliest.
- The gate reports a separate `summer_claims_allowed` check. That check stays closed until a real exported batch passes the evaluation and train-time correction gates.
- Every expanded run row carries an explicit scope label, `synthetic` or `real`, so synthetic helper results cannot be presented as geothermal or summer results.
- The guarded runner writes blocked rows instead of launching long jobs when the gate fails.

## Sparse Internal Work Decision

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/sparse_internal_work_decision.jl
```

Artifacts:

- `sparse_internal_work_decision.csv`
- `sparse_internal_work_decision.md`

Supported claims:

- Sparse solver-internal work is guarded by measured projection overhead.
- The current action is to defer solver-internal work because cached contexts are not the measured bottleneck.

## Result Run Plan

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/result_run_plan.jl
```

Artifacts:

- `result_run_plan.csv`
- `result_run_plan.md`

Supported claims:

- Broader DeepONet result runs have explicit priority, gates, grids, seeds, and purpose before compute is spent.
- Current 10 seed and 5 seed helper runs are separated from guarded expansion runs.

## Soft Sweep Summary

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/soft_sweep_summary.jl
```

Artifacts:

- `soft_sweep_pareto.csv`
- `soft_sweep_pareto.md`
- `deeponet_soft_sweep_pareto.svg`

Supported claims:

- Soft-penalty rows are evaluated as a Pareto tradeoff over RMSE, violation score, and runtime.
- Best-soft selection is explicit and tied to the study beta grid.

## Results Section Draft

Artifact:

- `deeponet_results_section.md`

Supported claims:

- The first DeepONet report section is separated from generated artifacts.
- The section uses only claims supported by the current artifact manifest.
- The section carries guardrails for pilot scale, synthetic-helper evidence, and correction status validity.

## Evaluation-Only Correction

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/eval_only_example.jl
```

Artifacts:

- `eval_only_example.csv`

Supported claims:

- A trained raw surrogate can be corrected at evaluation time without retraining.
- Evaluation-only correction enforces selected constraints exactly on successful projections.
- Evaluation-only correction is a practical first integration path for the summer project.

## Projection Frequency Ablation

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/frequency_ablation.jl
```

Artifacts:

- `frequency_ablation.csv`
- `frequency_statuses.csv`

Supported claims:

- Every-step correction gives the best RMSE in the current frequency ablation.
- Evaluation-only and every-N-step correction are cheaper feasibility paths.
- Frequency choices should be reported as a runtime and accuracy tradeoff.

## Constraint Family Ablation

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/constraint_ablation.jl
```

Artifacts:

- `constraint_ablation.csv`
- `constraint_statuses.csv`

Supported claims:

- Boundary-only correction fixes boundary violations but leaves mass uncontrolled.
- Mass-only correction fixes mass but leaves boundary uncontrolled.
- Full correction is the current clean default when boundary, balance, and bounds are all physically justified.
- Box-only correction should remain evaluation-only unless the active branch is regular.

## Summer Adapter Example

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/summer_adapter_example.jl
```

Artifacts:

- `summer_adapter_example.csv`
- `summer_adapter_example.md`

Supported claims:

- Generic correction contexts can wrap a sample shape that is not the synthetic heat sample type.
- The summer project can pass boundary metadata, balance metadata, bounds, and fixed-grid outputs through the same adapter.

## Constraint Audit

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/constraint_audit.jl
```

Artifacts:

- `constraint_audit.csv`
- `constraint_audit.md`

Supported claims:

- Constraint selection can be audited before connecting to a model.
- Missing metadata leads to a conservative `none` recommendation.
- Recommended modes are starting points that still require physical review.

## Stress Diagnostics

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/stress_diagnostics.jl
```

Artifacts:

- `stress_diagnostics.csv`
- `stress_diagnostics.md`

Supported claims:

- Infeasible balance under bounds is detected as `:infeasible_constraint`.
- Active-bound box kinks are detected as `:nonunique_input`.
- Malformed adapter rows fail early.
- Large correction norms remain visible even on successful projections.

## Failure-Mode Figure

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/failure_mode_plots.jl
```

Artifacts:

- `failure_mode_summary.csv`
- `failure_mode_diagnostics.svg`

Supported claims:

- Failure and stress cases are reportable as a compact status-certified diagnostic figure.
- Status and correction norm should be interpreted together.

## Ablation Interpretation

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/ablation_interpretation.jl
```

Artifacts:

- `ablation_interpretation.csv`
- `ablation_interpretation.md`

Supported claims:

- The frequency and constraint ablations have explicit, reproducible takeaways.
- The soft sweep has explicit best-RMSE and best-violation rows.
- The interpretation avoids claiming hard correction always wins.

## Sparse Decision

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/sparse_decision.jl
```

Artifacts:

- `sparse_decision.csv`
- `sparse_decision.md`

Supported claims:

- Current profiling includes larger-output rows through K=512 and supports deferring solver-internal sparse work until projection becomes a measured bottleneck.
- Cached contexts should remain the default helper path.

## Report Tables And Figures

Commands:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/summary.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/report_table.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/plots.jl
```

Artifacts:

- `summary.txt`
- `report_table.csv`
- `report_table.md`
- `deeponet_rmse.png`
- `deeponet_feasibility.png`
- `deeponet_runtime.svg`
- `deeponet_correction_norm.svg`
- `deeponet_soft_sweep_pareto.svg`
- `deeponet_report_bundle.svg`

Supported claims:

- The DeepONet helper result set is reproducible from scripts.
- Report-facing tables and figures can be regenerated from CSV artifacts.
- Status counts and correction-norm warning bands are part of the result surface.

## Report Verification

Command:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/verify_report_artifacts.jl
```

Supported claims:

- Required DeepONet report artifacts exist before report use.
- Main exact-feasibility claims are checked against CSV artifacts.
- Stress diagnostics, guardrail text, and tracked section placeholders are checked.

## Current Claim Guardrails

- Do not claim hard correction always improves RMSE.
- Do not claim geothermal conservation until the summer numerical model supplies a defensible balance.
- Do not hide large correction norms behind successful statuses.
- Do not train through non-success projection branches.
- Do not treat synthetic heat-helper results as final geothermal evidence.
