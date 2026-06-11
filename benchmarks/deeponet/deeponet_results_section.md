# DeepONet Helper Results Section Draft

This section is a report-facing draft for the synthetic DeepONet heat helper benchmark. It is not a geothermal result. The current claims are limited to the artifacts listed in `paper_results_manifest.md`.

## Setup

The helper benchmark trains a small branch-trunk DeepONet-style surrogate on fixed-grid heat-operator samples. The model predicts a vector field on the output grid. StructPINN then optionally corrects that field onto selected constraints: boundary values, a sample-specific integral balance, and loose box bounds.

The main 20 seed study compares vanilla training, soft penalties, evaluation-only hard correction, train-time hard correction, cached train-time hard correction, and soft-plus-hard correction. The larger-grid study repeats the main helper rows for K=64 and K=96 with 10 seeds. Both expanded runs carry an explicit synthetic scope label from the expanded-seed gate. The projection profile measures raw model evaluation, uncached projection, cached projection, context projection, and context construction through K=256, with a larger-output extension through K=512.

## Main Results

Table placeholder: `deeponet_main_results`.

The current 20 seed run shows that hard correction enforces the selected boundary and mass constraints to numerical precision on successful projections. Cached full hard correction has RMSE `0.02452`, boundary maximum `0`, and mass maximum `5.662e-16`. The uncached full hard row has the same RMSE and feasibility values, while cached training is faster in the current run, `0.654` seconds against `0.871` seconds mean training time.

The vanilla row has RMSE `0.06015`, boundary maximum `0.1972`, and mass maximum `0.05817`. Evaluation-only full correction improves the vanilla prediction to RMSE `0.04308` while enforcing boundary and mass constraints to numerical precision. This supports evaluation-time correction as a practical first integration path when retraining a group DeepONet is not yet available.

The soft-penalty sweep is a tradeoff, not exact enforcement. The best soft row by RMSE is `soft_weak` with RMSE `0.06091` and violation score `0.1708`. The best soft row by violation is `soft_boundary_heavy` with violation score `0.1559`, but its RMSE is `0.07090`. No current soft-only row enforces boundary or mass exactly.

## Scaling And Runtime

Table placeholder: `deeponet_large_results`.

At K=64, cached full hard correction has RMSE `0.02866`, exact boundary and mass feasibility to numerical precision, and mean training time `0.622` seconds in the current 10 seed run. At K=96, cached full hard correction has RMSE `0.02735`, exact boundary and mass feasibility to numerical precision, and mean training time `2.096` seconds. These are pilot-scale timing measurements, not final scaling claims.

The projection profile now covers K=32 through K=256, and the larger-output profile extends it through K=384 and K=512. Cached projection is faster than uncached projection for every profiled grid, and cached context projection remains sub-millisecond at the largest profiled outputs. The sparse decision gate is `defer_sparse_internals_until_real_bottleneck`: cached contexts stay the default, and solver-internal sparse work waits until a real interface makes projection the measured bottleneck.

## Ablations

Figure placeholder: `deeponet_report_bundle`.

Projection-frequency ablations show that every-step correction gives the best current ablation RMSE, `0.02874`, while evaluation-only and every-N-step correction give cheaper feasibility paths. Every-step correction trains in `2.621` seconds in the current ablation, while every-10-step correction trains in `0.2669` seconds.

Constraint-family ablations separate which constraints matter. Boundary-only evaluation correction drives boundary violation to zero but leaves mass violation at `0.06204`. Mass-only evaluation correction drives mass violation to numerical precision but leaves boundary violation at `0.1708`. Full evaluation correction fixes both boundary and mass in the current synthetic helper.

## Failure Modes

Figure placeholder: `failure_mode_diagnostics`.

Stress diagnostics check infeasible balance under bounds, large feasible corrections, box-only active-bound kinks, and malformed adapter rows. These cases are included to keep the status contract visible. Successful projection is the only branch where a valid gradient is claimed. Large correction norms remain reportable even when the projection status is `:success`.

## Guardrails

- Do not claim hard correction always improves RMSE.
- Do not claim geothermal conservation from this synthetic heat helper.
- Do not hide large correction norms behind successful statuses.
- Do not train through non-success projection branches.
- Treat the current results as pilot-scale helper evidence until broader seed counts and the summer interface are available.
