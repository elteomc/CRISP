# quick.md

## Summary

StructPINN builds differentiable hard-constraint layers for SciML in Julia that correct a network output zhat onto a constraint set c(z)=0 by differentiating the KKT optimality conditions rather than unrolling the solver, to test when hard enforcement beats soft penalties on energy and mass conservation. The plan is now hardened into 12 invariants and gated milestones (M0 infra, M0.5 AD/KKT spike, M1 affine layer, M2 soft baseline, M3 nonlinear KKT layer, M4 comparison, M5 required fixed-grid field correction), with the anchor benchmark a Hamiltonian neural ODE on a degeneracy-safe pendulum libration orbit using a fixed-step rollout. Current status: no code on disk yet, refs.bib unwritten so the proposal does not compile, and the nonlinear indefinite-KKT adjoint is the hardest upcoming build, the immediate next step is implementing M0, M0.5, and M1 with invariant-proving tests. Main open issues are logistical (timeline, team, inequality scope).

## Open questions

- Timeline: class deliverable still due, already submitted, or post-semester research phase?
- Team: solo or with teammates, and any coordination with the PCFM group under Utkarsh?
- Are inequality constraints (positivity, normalization) in scope for the class timeline, or is equality enough for first results?
- Keep or drop the optional agentic layer?

## Status

Round 5, last verdict approve, gpt spend $0.32 / $4.00

## Latest change

- round 5 (claude): Rewrote PLAN.md to resolve every Round 2 and Round 4 finding via 12 stated invariants, a ProjectionResult API, a gated M0.5 spike, a precise degeneracy-safe pendulum benchmark, an honest PINN scope decision, and a findings-to-resolution map, fixed a typo and removed prose semicolons per repo rules.

## Codebase map

A curated index so a new session can find the right file fast without scanning the tree. Regenerate with /zoom-out when it drifts. The Julia package is `StructPINN`, loaded with `using StructPINN`.

Package source (`src/`)
- `src/StructPINN.jl`: top-level module, exports the public API and includes the layer files.
- `src/projection_result.jl`: `ProjectionResult`, the per-call status object that records every projection outcome so no failure is silent (I10).
- `src/affine.jl`: `AffineConstraint` and its projection onto `{z : A z = b}` via thin QR with no explicit inverse, milestone M1.
- `src/nonlinear.jl`: `NonlinearConstraint` plus the `circle_constraint` and `tangential_constraint` toys, the nonlinear KKT projection spike M0.5 with analytic derivatives.
- `src/ad.jl`: `correct`, the differentiable forward map, and its `ChainRulesCore.rrule` so a layer trains end to end under reverse-mode AD.

Tests (`test/`)
- `test/runtests.jl`: test entry point, finite-difference and unrolled-Newton oracles, includes the suites below.
- `test/test_affine.jl`: M1 affine projection correctness and VJP checks.
- `test/test_nonlinear.jl`: M0.5 nonlinear KKT spike, regular, nonunique-input, and failure-status cases.
- `test/test_m3.jl`: M3 implicit-diff VJP checked against the unrolled-Newton oracle.
- `test/test_ad.jl`: Zygote gradients through `correct` matched to finite differences.

Pendulum benchmark (`benchmarks/pendulum/`)
- `benchmarks/pendulum/Pendulum.jl`: M2 harness module, pendulum physics, neural-ODE field, RK4 rollout, vanilla and soft losses, training, and metrics.
- `benchmarks/pendulum/run_baselines.jl`: M2/M3 single comparison of vanilla, soft, and projected models on the energy band.
- `benchmarks/pendulum/study.jl`: M4 multi-seed study writing the results table and plot-data CSVs.
- `benchmarks/pendulum/plots.jl`: M4 figures (training curves, energy over time) from the study CSVs.
- `benchmarks/pendulum/test.jl`: checks the M2 harness (data band, energy conservation, no angle wrap).

Spec and docs
- `PLAN.md`: full specification, invariants I1..I13 and gated milestones M0..M5.
- `quick.md`, `deep.md`: project summary and detailed notes.