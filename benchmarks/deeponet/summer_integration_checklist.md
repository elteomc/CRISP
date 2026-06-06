# Summer DeepONet Integration Checklist

Use this checklist before connecting StructPINN correction to the group DeepONet code. The goal is to make the adapter explicit, physically justified, and easy to audit.

## Output Shape

- [ ] Confirm the model output is a fixed-length vector for each sample.
- [ ] Record the output length.
- [ ] Record whether the vector is spatial, temporal, or flattened space-time.
- [ ] Record the ordering convention for flattened outputs.
- [ ] Confirm whether boundary grid points are included in the output.
- [ ] Save a small example raw output and target output for adapter tests.

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
- [ ] Choose the simplest physically justified correction mode.
- [ ] Run evaluation-only correction on a trained model.
- [ ] Compare raw and corrected outputs on held-out samples.
- [ ] Add train-time correction only after the evaluation path is stable.
