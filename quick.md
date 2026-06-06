# quick.md

## Summary

StructPINN builds differentiable hard-constraint layers for PINNs, operator surrogates, and related scientific ML systems. Julia is the current implementation vehicle. A layer corrects a raw model output `zhat` onto a constraint set, returns a `ProjectionResult` status object, and differentiates the KKT or active-set optimality conditions rather than solver iterations. The current code implements affine projections, sparse KKT affine projections, diagonal weighted affine projections, sparse weighted affine projections, box projections, nonlinear equality projections, weighted simplex projections for positivity plus normalization, bounded weighted simplex projections for equality plus box constraints, sparse box-affine inequality projection, general sparse linear-QP inequality projection with active-set and primal-dual backends, small dense linear-QP projections, ChainRules integration through `correct`, a fixed-step pendulum neural ODE benchmark, fixed-grid PDE correction benchmarks including heat, viscous Burgers, and Allen-Cahn integrations, and a DeepONet-style heat-operator helper benchmark. Projected benchmark paths record statuses during training and evaluation. The active strategic direction is now to make StructPINN the helper project for physics-informed DeepONet work by correcting DeepONet output fields onto physical constraints.

## Current Status

- M0 is complete: package skeleton, tests, CI, verified `refs.bib`, and proposal build path are in place. The proposal builds after the standard `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` sequence.
- M0.5, M1, M2, M3, M4, M5, and the current M6 inequality slices are implemented and tested at prototype scale.
- M1 now handles overconstrained affine systems by returning `:singular_constraint` instead of crashing.
- I10 training status logging is implemented for projected pendulum training. Non-success projection statuses are counted when a `FailureCounter` is supplied and flagged before invalid gradients are used.
- M5 fixed-grid field correction uses the M1 affine layer to enforce discrete mass on supervised 1D grid fields.
- M6 now includes `DiagonalWeightedAffineConstraint`, which handles diagonal weighted Euclidean projection onto affine constraints.
- M6 now includes `BoxConstraint`, which handles componentwise lower and upper bounds with active-set VJPs and kink status handling.
- M6 also includes `WeightedSimplexConstraint`, which enforces nonnegativity and weighted normalization with active-set VJPs and kink status handling.
- M6 now includes `BoundedWeightedSimplexConstraint`, which enforces one weighted equality together with lower and upper bounds.
- M6 now includes `DenseLinearQPConstraint`, which projects onto small dense linear equality and inequality systems by exhaustive active-set enumeration.
- M6 now includes sparse equality KKT systems through `SparseAffineConstraint` and `SparseDiagonalWeightedAffineConstraint`.
- M6 now includes `SparseBoxAffineConstraint`, an iterative sparse backend for `{A z = b, lower <= z <= upper}`.
- M6 now includes `SparseAffineProjectionCache`, `CachedSparseBoxAffineConstraint`, and `WarmStartedSparseBoxAffineConstraint` for repeated sparse projection performance.
- M6 now includes `SparseLinearQPActiveSetConstraint` and `SparseLinearQPPrimalDualConstraint`, general sparse backends for `{Aeq * z = beq, G * z <= h}`.
- The field benchmark now includes heat, viscous Burgers, and Allen-Cahn fixed-grid integrations that train through sparse projection layers at larger grid sizes.
- Paper-quality pilot artifacts now exist: multi-seed pendulum, fixed-grid field, and larger PDE CSVs plus plots.
- The old project plan has been archived as `PAST_PLAN.md`.
- The active `PLAN.md` now prioritizes DeepONet output correction as the summer helper path, while preserving the hard-vs-soft study and sparse scaling paths for a six-month submission-ready target.
- The `deeponet-helper` branch exists for helper implementation, and `archive/m6-pilot` preserves the pre-helper code snapshot.
- `benchmarks/deeponet` now implements a small DeepONet-style heat-operator benchmark with vanilla, soft-penalty, hard-corrected, and soft-plus-hard paths.
- The DeepONet helper test passes, including finite-difference checks through the hard-corrected loss.
- The first 10 seed DeepONet helper study CSVs and plots exist under `benchmarks/deeponet/results`. Full hard correction drives boundary and mass violations to numerical precision with RMSE `0.0248`. Cached full hard correction matches the uncached metrics and reduces mean training time from `1.09` to `0.80` seconds in this small run. Full soft-plus-hard also enforces feasibility with RMSE `0.0456`. Evaluation-only full correction improves vanilla RMSE to `0.0447` while enforcing feasibility. Boundary-box-only correction is a useful negative ablation because it leaves mass uncontrolled.
- The DeepONet helper now exposes generic adapter helpers so a fixed-grid operator surrogate can supply linear equality rows, per-sample equality values, bounds, and raw output vectors without using the synthetic heat sample type.
- The DeepONet helper now has projection-frequency and constraint-family ablation scripts. The constraint-family script treats box-only correction as evaluation-only in this helper because pure box projection can hit active-bound kinks where no training gradient is claimed.
- Remaining M6 work: connect these adapter helpers to the summer data interface, deeper performance studies, broader seed counts, harder PDE families, and paper writeup polishing.

## Open Questions

- Should the first DeepONet helper result use the current synthetic heat operator as the report-facing scaffold, or should it move immediately toward the summer geothermal interface once that interface exists?
- Which constraints are physically justified for the summer geothermal DeepONet case, boundary values, bounds, positivity, integral balance, or a smaller subset?
- Should sparse projection optimization wait until the DeepONet helper benchmark exposes a real bottleneck?

## Latest Change

- Archived the original milestone plan as `PAST_PLAN.md`.
- Replaced `PLAN.md` with a new forward plan that locks StructPINN in as the DeepONet helper project and ranks the remaining conference-oriented alternatives.
- Added the immediate next direction: create a `benchmarks/deeponet` helper benchmark with vanilla, soft, hard, and soft-plus-hard output correction paths.
- Created the `archive/m6-pilot` and `deeponet-helper` git branches.
- Added `benchmarks/deeponet/DeepONetHeat.jl`, `Project.toml`, `test.jl`, and `study.jl`.
- Expanded the DeepONet helper study to 10 seeds.
- Added four soft-penalty sweep rows, evaluation-only full correction, boundary-box-only hard correction, full boundary-mass-box hard correction, and soft-plus-hard variants for both correction modes.
- Added prebuilt cached correction contexts for DeepONet helper samples, plus context-based hard and soft-plus-hard losses.
- Added DeepONet helper plots for RMSE, feasibility, runtime, and correction norm.
- Added `benchmarks/deeponet/helper_note.txt` with the generic integration pattern for the summer DeepONet project.
- Generated DeepONet helper study CSVs. In the current 10 seed run, full hard correction has RMSE `0.0248` with exact boundary and mass feasibility, cached full hard correction matches those metrics with mean training time `0.80` seconds, full soft-plus-hard has RMSE `0.0456` with exact feasibility, evaluation-only full correction has RMSE `0.0447`, vanilla has RMSE `0.0573`, and the best soft row by RMSE has RMSE `0.0610`.
- Added `benchmarks/deeponet/large_study.jl` for K=64 and K=96 helper scaling runs.
- Added `benchmarks/deeponet/profile_projection.jl` for raw, uncached, cached, context-based, and context-construction timing.
- Added `benchmarks/deeponet/eval_only_example.jl` for correcting a trained vanilla DeepONet surrogate at evaluation time.
- Added `benchmarks/deeponet/summary.jl`, which writes a report-facing `summary.txt` from existing DeepONet helper artifacts.
- Generated the large-grid helper artifacts. In the current 5 seed run, K=64 cached full hard correction has RMSE `0.0272` and mean training time `1.15` seconds, while K=96 cached full hard correction has RMSE `0.0247` and mean training time `2.12` seconds. Both enforce boundary and mass constraints to numerical precision.
- Added `benchmarks/deeponet/report_table.jl`, which writes compact markdown and CSV tables for the helper study, scaling rows, projection profile, and evaluation-only example.
- Added generic DeepONet output-correction adapter helpers: `boundary_mass_rows`, `operator_correction_context`, `operator_correction_contexts`, and `corrected_output`.
- Extended DeepONet heat correction modes to support boundary-only, mass-only, box-only, boundary-box, and full boundary-mass-box correction.
- Added `benchmarks/deeponet/frequency_ablation.jl`, which compares no correction, evaluation-only correction, every-step training correction, and every-N-step training correction.
- Added `benchmarks/deeponet/constraint_ablation.jl`, which compares boundary-only, box-only, mass-only, boundary-box, and full correction families.
- Added a mock summer-sample adapter fixture to the DeepONet helper tests so the generic context path is checked without depending on unfinished group code.
- Updated the DeepONet plots and report tables with ablation rows, status counts, correction norms, and a compact SVG report bundle.
- Updated the helper note with the summer adapter pattern.
- Added `DiagonalWeightedAffineConstraint` for weighted affine projection.
- Integrated weighted affine projection into the field benchmark as a physically weighted mass-correction baseline.
- Added `BoxConstraint` for componentwise bounds and integrated it into the field benchmark as a bounded-output baseline.
- Added `WeightedSimplexConstraint` for positivity plus weighted normalization.
- Added `BoundedWeightedSimplexConstraint` for a weighted equality plus finite lower and upper bounds.
- Added `DenseLinearQPConstraint` for small dense linear-QP projections.
- Added `SparseAffineConstraint` and `SparseDiagonalWeightedAffineConstraint` for sparse equality KKT solves and VJPs.
- Added `SparseBoxAffineConstraint` for iterative sparse projection onto affine constraints with box inequalities.
- Added cached sparse affine projections, cached sparse box-affine projections, and warm active-face starts for repeated Dykstra projections.
- Added `SparseLinearQPActiveSetConstraint` and `SparseLinearQPPrimalDualConstraint` for general sparse linear equality and inequality projections.
- Added M6 tests for feasibility, active-set VJP correctness, Zygote integration, active-set kink status, and infeasible constraints.
- Integrated the sparse bounded projection into the fixed-grid field benchmark and added a heat-style field test.
- Added periodic viscous Burgers and Allen-Cahn fixed-grid PDE integrations using the general sparse QP projection paths.
- Verified the current bibliography against primary sources and official package documentation.
- Added field and PDE plotting scripts, plus a multi-seed larger PDE study.
- Generated current multi-seed result artifacts. Pendulum projected rollout has zero measured energy drift and lower long-horizon RMSE than vanilla or soft in the current 3 seed run. Fixed-grid mass projection improves RMSE from 0.0290 vanilla to 0.0116 projected while driving mass error to numerical precision. Heat and Burgers PDE studies preserve enforced mass and bounds to numerical precision, and Allen-Cahn enforces bounds to numerical precision.
- Added a root `README.md` summarizing the goal, current status, result artifacts, and reproduction commands.

## Codebase Map

Package source (`src/`)
- `src/StructPINN.jl`: top-level module, exports the public API and includes the layer files.
- `src/projection_result.jl`: `ProjectionResult`, the per-call status object that records projection outcomes.
- `src/affine.jl`: `AffineConstraint` and `DiagonalWeightedAffineConstraint`, with rank checks and VJPs.
- `src/sparse_kkt.jl`: sparse KKT equality projection, cached sparse affine projection, and weighted sparse KKT projection.
- `src/sparse_box_affine.jl`: iterative sparse box-affine projection, cached Dykstra affine steps, warm active-face starts, and active-set VJP.
- `src/sparse_qp.jl`: general sparse linear-QP projections with active-set and primal-dual backends.
- `src/box.jl`: `BoxConstraint`, componentwise clamp projection, and active-set VJP.
- `src/nonlinear.jl`: `NonlinearConstraint`, toy constraints, Newton KKT solve, and implicit VJP.
- `src/simplex.jl`: `WeightedSimplexConstraint`, projection onto `{z >= 0, weights' z = mass}`, and active-set VJP.
- `src/bounded_simplex.jl`: `BoundedWeightedSimplexConstraint`, projection onto `{lower <= z <= upper, weights' z = mass}`, and active-set VJP.
- `src/dense_qp.jl`: `DenseLinearQPConstraint`, exhaustive active-set projection onto `{Aeq * z = beq, G * z <= h}`, and active-set VJP.
- `src/ad.jl`: `correct` and its `ChainRulesCore.rrule`.

Tests (`test/`)
- `test/runtests.jl`: test entry point, finite-difference helper, and unrolled-Newton oracle.
- `test/test_affine.jl`: affine projection correctness, VJP checks, and rank-failure regressions.
- `test/test_weighted_affine.jl`: weighted affine projection, VJP, Zygote, and rank-failure checks.
- `test/test_sparse_kkt.jl`: sparse KKT projection, cached sparse affine projection, weighted sparse KKT projection, VJP, Zygote, and rank-failure checks.
- `test/test_sparse_box_affine.jl`: sparse box-affine projection, cached projection, warm-started projection, Dykstra convergence, VJP, Zygote, and infeasible checks.
- `test/test_sparse_qp.jl`: general sparse linear-QP projection, active-set and primal-dual agreement, finite-difference VJP checks, Zygote, kink, infeasible, rank-failure, and larger sparse checks.
- `test/test_box.jl`: box projection, outside-clamp VJP, exact-bound kink, and infeasible-bound checks.
- `test/test_nonlinear.jl`: nonlinear KKT regular and failure-status cases.
- `test/test_m3.jl`: nonlinear implicit-diff VJP checked against a ForwardDiff oracle.
- `test/test_simplex.jl`: M6 weighted simplex projection, active-set VJP, Zygote, kink, and infeasible cases.
- `test/test_bounded_simplex.jl`: M6 bounded weighted simplex projection, active-set VJP, Zygote, kink, vertex, and infeasible cases.
- `test/test_dense_qp.jl`: M6 dense linear-QP projection, finite-difference VJP checks, Zygote, kink, infeasible, and rank-failure cases.
- `test/test_ad.jl`: Zygote gradients through `correct`.

Pendulum benchmark (`benchmarks/pendulum/`)
- `benchmarks/pendulum/Pendulum.jl`: pendulum physics, neural ODE field, RK4 rollout, losses, projected rollout, training, status logging, and metrics.
- `benchmarks/pendulum/run_baselines.jl`: single comparison of vanilla, soft, and projected models.
- `benchmarks/pendulum/study.jl`: multi-seed study writing CSV artifacts.
- `benchmarks/pendulum/plots.jl`: figures from study CSVs.
- `benchmarks/pendulum/test.jl`: benchmark harness and projected-gradient tests.

Field benchmark (`benchmarks/field/`)
- `benchmarks/field/FieldMass.jl`: fixed-grid supervised, heat, Burgers, and Allen-Cahn field data, mass weights, MLP, vanilla and soft losses, unweighted affine, weighted affine, box, nonnegative-normalized, bounded-mass, sparse bounded, and sparse QP field correction, training, status logging, and metrics.
- `benchmarks/field/run_baselines.jl`: single comparison of vanilla, soft, affine-projected, weighted-projected, box-projected, positive-projected, bounded-projected, and sparse-bounded field models.
- `benchmarks/field/run_pde_integrations.jl`: heat, Burgers, and Allen-Cahn projected PDE smoke benchmark.
- `benchmarks/field/pde_study.jl`: multi-seed heat, Burgers, and Allen-Cahn sparse projection study writing PDE result CSVs.
- `benchmarks/field/plots.jl`: field and PDE result plots from study CSVs.
- `benchmarks/field/profile_sparse_projection.jl`: uncached, cached, and warm-started sparse bounded projection timing and iteration profile.
- `benchmarks/field/study.jl`: multi-seed field study writing result CSVs.
- `benchmarks/field/test.jl`: field harness, projection, PDE integration, training, metric, and gradient tests.

DeepONet helper benchmark (`benchmarks/deeponet/`)
- `benchmarks/deeponet/DeepONetHeat.jl`: DeepONet-style branch/trunk heat-operator model, synthetic heat data, boundary and full boundary-mass-box constraints, prebuilt cached correction contexts, vanilla, soft, hard, and soft-plus-hard losses, training, status logging, correction norms, and metrics.
- `benchmarks/deeponet/test.jl`: helper benchmark harness, projection feasibility checks, Zygote gradient checks, training checks, and finite-difference checks through the hard-corrected loss.
- `benchmarks/deeponet/study.jl`: 10 seed helper study writing `results.csv`, `statuses.csv`, and `training_curves.csv`.
- `benchmarks/deeponet/large_study.jl`: larger-grid helper study for K=64 and K=96.
- `benchmarks/deeponet/frequency_ablation.jl`: projection-frequency ablation for no correction, evaluation-only correction, every-step correction, and every-N-step correction.
- `benchmarks/deeponet/constraint_ablation.jl`: constraint-family ablation for boundary-only, box-only, mass-only, boundary-box, and full correction modes.
- `benchmarks/deeponet/profile_projection.jl`: projection runtime profile for raw, uncached, cached, context-based, and construction paths.
- `benchmarks/deeponet/eval_only_example.jl`: example of applying full correction only at evaluation time.
- `benchmarks/deeponet/summary.jl`: text summary generator for DeepONet helper artifacts.
- `benchmarks/deeponet/report_table.jl`: markdown and CSV report table generator for DeepONet helper artifacts.
- `benchmarks/deeponet/plots.jl`: plots helper RMSE, feasibility, runtime, and correction-norm artifacts from study CSVs.
- `benchmarks/deeponet/helper_note.txt`: short integration note for applying StructPINN to summer DeepONet outputs.

Spec and docs
- `README.md`: public project overview, current status, result snapshot, and reproduction commands.
- `PLAN.md`: active strategy, now centered on DeepONet output correction plus conference-oriented alternatives.
- `PAST_PLAN.md`: archived original living specification, invariants, milestones, and scope.
- `pinn_proposal.tex`: updated proposal narrative aligned with StructPINN.
- `refs.bib`: verified bibliography for the current proposal references. Future additions should pass the same source gate.
