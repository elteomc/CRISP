# quick.md

## Summary

StructPINN builds differentiable hard-constraint layers for PINNs and related scientific ML systems. Julia is the current implementation vehicle. A layer corrects a raw model output `zhat` onto a constraint set, returns a `ProjectionResult` status object, and differentiates the KKT or active-set optimality conditions rather than solver iterations. The current code implements affine projections, sparse KKT affine projections, diagonal weighted affine projections, sparse weighted affine projections, box projections, nonlinear equality projections, weighted simplex projections for positivity plus normalization, bounded weighted simplex projections for equality plus box constraints, sparse box-affine inequality projection, general sparse linear-QP inequality projection with active-set and primal-dual backends, small dense linear-QP projections, ChainRules integration through `correct`, a fixed-step pendulum neural ODE benchmark, and fixed-grid PDE correction benchmarks including heat, viscous Burgers, and Allen-Cahn integrations. Projected benchmark paths record statuses during training and evaluation.

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
- Remaining M6 work: deeper performance studies, broader seed counts, harder PDE families, and paper writeup polishing.

## Open Questions

- Should the next PDE result focus on deeper multi-seed studies, harder initial-condition families, or solver-level optimization beyond the Dykstra path?
- Should the next paper-style writeup include workflow automation, or focus only on scientific ML constraint layers?
- Should the project coordinate with the PCFM group after the first complete M4 or M5 result?

## Latest Change

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

Spec and docs
- `README.md`: public project overview, current status, result snapshot, and reproduction commands.
- `PLAN.md`: living specification, invariants, milestones, and scope.
- `pinn_proposal.tex`: updated proposal narrative aligned with StructPINN.
- `refs.bib`: verified bibliography for the current proposal references. Future additions should pass the same source gate.
