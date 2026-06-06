# StructPINN

StructPINN is a prototype for differentiable hard-constraint layers for scientific machine learning. The current implementation is in Julia, but the project goal is broader: test whether projection-style constraint layers can help PINN and SciML workflows enforce physical structure more reliably than soft penalties alone.

The central idea is simple. A model emits a raw vector `zhat`. A constraint layer corrects it to `zstar`, returns a `ProjectionResult` with feasibility and status diagnostics, and differentiates the KKT or active-set optimality conditions rather than the solver iterations.

## Goal

The project is trying to answer:

> When does replacing a soft penalty with a differentiable correction step improve physical fidelity, training stability, and long-horizon prediction quality?

The current scope is not limited to the Julia ecosystem. Julia is the working vehicle because it gives direct access to differentiable programming, sparse linear algebra, and SciML-style benchmarks.

## Current Status

The package currently implements:

- Affine equality projection.
- Diagonal weighted affine projection.
- Sparse affine KKT projection with reusable factorization caches.
- Nonlinear equality projection through local KKT solves.
- Box projection with active-set VJP.
- Weighted simplex projection for positivity plus normalization.
- Bounded weighted simplex projection for equality plus box constraints.
- Small dense linear-QP projection by active-set enumeration.
- Sparse box-affine projection using Dykstra iterations.
- General sparse linear-QP projection with active-set and primal-dual forward backends.
- ChainRules integration through `correct`.

The benchmark suite currently includes:

- A fixed-step pendulum neural ODE benchmark with vanilla, soft-penalty, and projected rollouts.
- A fixed-grid field correction benchmark with mass, positivity, and bound constraints.
- Larger fixed-grid heat, Burgers, and Allen-Cahn integrations using sparse projection paths.
- A DeepONet-style heat-operator helper benchmark with vanilla, soft-penalty, hard-corrected, and soft-plus-hard output paths.
- Generic DeepONet output-correction adapter helpers for fixed-grid operator surrogates.
- Projection-frequency and constraint-family ablations for the DeepONet helper path.

The active project direction is to make StructPINN a helper layer for physics-informed DeepONet and operator-surrogate work. The first self-contained helper benchmark now lives under `benchmarks/deeponet`.

## Results Snapshot

Current reproducible pilot artifacts are generated under `benchmarks/*/results`.

Pendulum, 3 seeds:

- Projected rollout has zero printed energy max error and zero printed long-horizon drift.
- Long-horizon RMSE is `0.1360` projected, `0.1730` soft, and `0.2261` vanilla.

Fixed-grid field correction, 3 seeds:

- Vanilla RMSE is `0.0290`.
- Projected mass-correction RMSE is `0.0116`.
- Constrained models drive mass error to numerical precision.

Larger PDE integrations, 3 seeds:

- Heat sparse-bounded RMSE is `0.0170`, with mass and bounds at numerical precision.
- Burgers sparse-QP-bounded RMSE is `0.0243`, with mass and bounds at numerical precision.
- Allen-Cahn sparse-QP-box RMSE is `0.0182`, with box bounds at numerical precision.

DeepONet-style heat operator, 10 seeds:

- Full hard-corrected RMSE is `0.0248`, with boundary and mass violations at numerical precision.
- Cached full hard correction matches the uncached full hard metrics and reduces mean training time from `1.09` to `0.80` seconds in this small run.
- Full soft-plus-hard RMSE is `0.0456`, also with boundary and mass violations at numerical precision.
- Evaluation-only full correction improves vanilla RMSE to `0.0447` while enforcing boundary and mass constraints.
- Vanilla RMSE is `0.0573`, with maximum boundary error `0.1879` and maximum mass error `0.0458`.
- The best soft-penalty sweep row by RMSE is `0.0610`, and none of the soft rows enforces boundary or mass constraints exactly.

DeepONet larger-grid helper, 5 seeds:

- At K=64, cached full hard correction has RMSE `0.0272`, exact boundary and mass feasibility, and mean training time `1.15` seconds.
- At K=96, cached full hard correction has RMSE `0.0247`, exact boundary and mass feasibility, and mean training time `2.12` seconds.
- Evaluation-only full correction improves vanilla predictions while enforcing constraints without retraining.
- The projection profile reports raw, uncached, cached, context-based, and context-construction timings for default grids from K=32 through K=256.
- Projection-frequency and constraint-family ablations now write separate CSVs. Box-only correction is kept as an evaluation ablation in this helper because pure box projection can hit active-bound kinks where no training gradient is claimed.
- A shared DeepONet scenario layer records default grids, seeds, steps, and sample counts, with `STRUCTPINN_DEEPONET_*` overrides for larger or smaller runs.
- A mock summer adapter example shows how geothermal-style sample metadata can feed generic correction contexts without using the synthetic heat sample type.
- A tracked constraint-selection guide explains when boundary, box, positivity, and balance correction are physically justified.
- A tracked summer integration checklist records the output shape, grid, metadata, units, bounds, balance quantities, hooks, and logging needed from the group DeepONet code.
- A constraint audit script checks mock or CSV sample metadata and recommends a starting correction mode.
- Stress diagnostics exercise infeasible balance, large correction norms, box-only kink status, and malformed adapter rows.
- A failure-mode figure generator turns stress diagnostics into compact CSV and SVG artifacts.
- An ablation interpretation generator writes concise report takeaways from the frequency and constraint-family CSVs.
- A tracked paper-results manifest maps every DeepONet command to artifacts and supported claims.
- A sparse optimization decision script reads the current DeepONet projection profile and larger-grid timings. The current decision is to profile larger outputs before choosing more result runs or solver-internal sparse work.
- The report bundle SVG now combines main helper rows, frequency ablation, constraint-family ablation, sparse decision metrics, and status/correction-norm diagnostics.
- Report diagnostics now classify correction norms into warning bands so successful projections with large raw-output corrections stay visible.
- A report-table generator combines the 10 seed study, larger-grid rows, projection profile, evaluation-only example, statuses, correction norms, and ablations into markdown and CSV tables.

These are pilot-scale results, not final paper claims. The next result step is broader seed counts, harder PDE families, and a polished paper results section.

## Repository Map

- `src/`: constraint layer implementations.
- `test/`: package-level invariant and gradient tests.
- `benchmarks/pendulum/`: pendulum neural ODE benchmark.
- `benchmarks/field/`: fixed-grid field and larger PDE benchmarks.
- `benchmarks/deeponet/`: DeepONet-style heat-operator helper benchmark.
- `quick.md`: compact current project state.
- `deep.md`: detailed local project state and interpretation notes.
- `PLAN.md`: active forward plan, centered on DeepONet output correction and conference-oriented alternatives.
- `PAST_PLAN.md`: archived original specification, invariants, and milestone history.
- `pinn_proposal.tex`: local proposal narrative.
- `refs.bib`: verified bibliography for the proposal references.

`deep.md`, `PLAN.md`, `PAST_PLAN.md`, and `pinn_proposal.tex` are local project documents and are ignored by git. `README.md`, `quick.md`, and `refs.bib` are tracked.

## Run Tests

From the repository root:

```powershell
julia --project=. -e "using Pkg" -e "Pkg.instantiate()" -e "Pkg.test()"
julia --project=benchmarks/pendulum benchmarks/pendulum/test.jl
julia --project=benchmarks/field benchmarks/field/test.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/test.jl
```

## Generate Result Artifacts

Pendulum:

```powershell
julia --project=benchmarks/pendulum benchmarks/pendulum/study.jl
julia --project=benchmarks/pendulum benchmarks/pendulum/plots.jl
```

Field and PDE:

```powershell
julia --project=benchmarks/field benchmarks/field/study.jl
julia --project=benchmarks/field benchmarks/field/pde_study.jl
julia --project=benchmarks/field benchmarks/field/plots.jl
```

DeepONet helper:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/study.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/large_study.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/scenario_manifest.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/profile_projection.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/eval_only_example.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/frequency_ablation.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/constraint_ablation.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/summer_adapter_example.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/constraint_audit.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/stress_diagnostics.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/failure_mode_plots.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/ablation_interpretation.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/sparse_decision.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/summary.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/report_table.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/plots.jl
```

Sparse projection profiling:

```powershell
julia --project=benchmarks/field benchmarks/field/profile_sparse_projection.jl
```

## Build Proposal

```powershell
pdflatex -interaction=nonstopmode pinn_proposal.tex
bibtex pinn_proposal
pdflatex -interaction=nonstopmode pinn_proposal.tex
pdflatex -interaction=nonstopmode pinn_proposal.tex
```

## Design Contract

The layer contract is status-first:

- `:success` is the only status that claims a valid gradient.
- Rank failure, ill-conditioning, infeasibility, active-set kinks, nonunique inputs, and solver failures are distinct statuses.
- Benchmarks record projection statuses during training and evaluation.
- Gradients are checked by finite differences before a layer is used in a benchmark.

This is the main engineering guardrail behind the current implementation.
