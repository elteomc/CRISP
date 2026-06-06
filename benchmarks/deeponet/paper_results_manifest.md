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

- Current profiling says to profile larger outputs before choosing more result runs or solver-internal sparse work.
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

## Current Claim Guardrails

- Do not claim hard correction always improves RMSE.
- Do not claim geothermal conservation until the summer numerical model supplies a defensible balance.
- Do not hide large correction norms behind successful statuses.
- Do not train through non-success projection branches.
- Do not treat synthetic heat-helper results as final geothermal evidence.
