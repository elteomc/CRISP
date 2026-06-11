# Summer DeepONet Integration Checklist

Use this checklist before connecting StructPINN correction to the group DeepONet code. The goal is to make the adapter explicit, physically justified, and easy to audit.

## Output Shape

- [ ] Confirm the model output is a fixed-length vector for each sample.
- [ ] Record the output length.
- [ ] Record whether the vector is spatial, temporal, or flattened space-time.
- [ ] Record the ordering convention for flattened outputs.
- [ ] Confirm whether boundary grid points are included in the output.
- [ ] Save a small example raw output and target output for adapter tests.
- [ ] Export a CSV batch with `raw_output`, optional `target_output`, and `case_id`.

## Grid Metadata

- [ ] Save the coordinate grid used by the output vector.
- [ ] Save quadrature weights if any integral balance will be used.
- [ ] Confirm the weights match the output ordering.
- [ ] Confirm whether the grid changes between samples.
- [ ] Record any masking, padding, or inactive cells.

## Boundary Metadata

- [ ] Identify whether boundary values are Dirichlet, Neumann, Robin, periodic, or mixed.
- [ ] Use endpoint equality rows only for Dirichlet values represented in the output vector.
- [ ] Record per-sample boundary values when they vary by case.
- [ ] Record fixed boundary values when they are shared across all cases.
- [ ] Do not use endpoint value correction for boundary conditions that are not endpoint values.

## Units And Variables

- [ ] Name the corrected output variable.
- [ ] Record the units before normalization.
- [ ] Record the normalization transform if one is used.
- [ ] Confirm whether negative values are physically valid.
- [ ] Confirm whether positivity is valid in the modeled units.
- [ ] Confirm whether temperature is represented in Kelvin, Celsius, centered units, or normalized units.

## Bounds

- [ ] Record lower bounds from physics, safety, or simulation validity.
- [ ] Record upper bounds from physics, safety, or simulation validity.
- [ ] Confirm bounds are not chosen only from model errors.
- [ ] Confirm bounds are loose enough to reveal large raw-model mistakes.
- [ ] Log correction norms whenever bounds are used.

## Balance Quantities

- [ ] Identify whether the numerical model supplies a scalar balance target.
- [ ] Record whether the balance is mass, energy, integral temperature, or another quantity.
- [ ] Confirm source terms, sinks, and boundary fluxes are accounted for.
- [ ] Confirm the balance is meaningful on the chosen output window.
- [ ] Do not use a balance correction if the balance target is only a rough trend.

## Train And Eval Hooks

- [ ] Locate the evaluation function that returns raw model outputs.
- [ ] Add evaluation-time correction first.
- [ ] Add status and correction-norm logging to evaluation.
- [ ] Run `summer_batch_train.jl` only after the evaluation-only status gate passes.
- [ ] Locate the training loss hook if train-time correction will be tested.
- [ ] Add train-time correction only after statuses are mostly `:success`.
- [ ] Keep correction contexts cached when grid and sample metadata repeat.

## Logging

- [ ] Log projection status counts.
- [ ] Log correction norm mean and max.
- [ ] Log RMSE before and after correction.
- [ ] Log boundary violation before and after correction.
- [ ] Log bound violation before and after correction.
- [ ] Log balance violation before and after correction when a balance is used.
- [ ] Flag large correction norms as raw-model feasibility warnings.

## Minimal First Integration

- [ ] Run `constraint_audit.jl` on a mock or exported batch.
- [ ] Run `summer_batch_review.jl` on the exported batch.
- [ ] Run `summer_batch_eval.jl` on the exported batch in `:auto` mode.
- [ ] Choose the simplest physically justified correction mode.
- [ ] Run evaluation-only correction on a trained model.
- [ ] Compare raw and corrected outputs on held-out samples.
- [ ] Run `summer_batch_train.jl` on a batch with `features` and `target_output`.
- [ ] Run `summer_training_decision.jl` after the training gate artifact exists.
- [ ] Run `expanded_seed_gate.jl` and `run_expanded_seed_jobs.jl` only after the training decision is ready.
- [ ] Run `sparse_internal_work_decision.jl` before starting sparse solver-internal work.
- [ ] Add train-time correction only after the evaluation path is stable.

## Exported CSV Adapter

Use this command once a batch exists:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/summer_batch_eval.jl path\to\batch.csv
```

Run the metadata review before train-time correction:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/summer_batch_review.jl path\to\batch.csv
```

Accepted columns are `case_id`, `raw_output`, `target_output`, `features`, `grid`, `weights`, `left_bc`, `right_bc`, `balance`, `energy_balance`, `lower_bound`, and `upper_bound`. The adapter writes `summer_batch_eval.csv` and `summer_batch_eval.md`.

For gated train-time correction, use:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/summer_batch_train.jl path\to\batch.csv
```

This command expects `features`, `target_output`, and `grid` columns. It trains vanilla first, applies evaluation-only correction on held-out rows, and runs train-time correction only when the evaluation status gate passes.

Inspect the integration decision with:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/summer_training_decision.jl
```

Guard expanded result runs with:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/expanded_seed_gate.jl
julia --project=benchmarks/deeponet benchmarks/deeponet/run_expanded_seed_jobs.jl
```

Guard sparse solver work with:

```powershell
julia --project=benchmarks/deeponet benchmarks/deeponet/sparse_internal_work_decision.jl
```
