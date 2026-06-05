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

These are pilot-scale results, not final paper claims. The next result step is broader seed counts, harder PDE families, and a polished paper results section.

## Repository Map

- `src/`: constraint layer implementations.
- `test/`: package-level invariant and gradient tests.
- `benchmarks/pendulum/`: pendulum neural ODE benchmark.
- `benchmarks/field/`: fixed-grid field and larger PDE benchmarks.
- `quick.md`: compact current project state.
- `deep.md`: detailed local project state and interpretation notes.
- `PLAN.md`: local living specification and milestone plan.
- `pinn_proposal.tex`: local proposal narrative.
- `refs.bib`: verified bibliography for the proposal references.

`deep.md`, `PLAN.md`, and `pinn_proposal.tex` are local project documents and are ignored by git. `README.md`, `quick.md`, and `refs.bib` are tracked.

## Run Tests

From the repository root:

```powershell
julia --project=. -e "using Pkg" -e "Pkg.instantiate()" -e "Pkg.test()"
julia --project=benchmarks/pendulum benchmarks/pendulum/test.jl
julia --project=benchmarks/field benchmarks/field/test.jl
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
