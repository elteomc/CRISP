# DeepONet Constraint Selection Guide

This guide is for choosing StructPINN correction constraints for fixed-grid DeepONet or operator-surrogate outputs. The rule is simple: only correct to constraints that are physically justified by the modeled variable, grid, units, and dataset metadata.

## Default Workflow

1. Start with evaluation-time correction on a trained model.
2. Log status, correction norm, RMSE, and each constraint violation.
3. Move to train-time correction only when projection statuses are mostly `:success`.
4. Use cached contexts when the same grid and sample metadata repeat.
5. Treat large correction norms as a model diagnostic, not as a success by itself.

## Boundary Values

Use boundary equality rows when:

- The PDE case has known Dirichlet boundary values.
- The output vector includes the boundary grid points.
- The boundary values are available per sample or fixed across samples.

Do not use boundary correction when:

- The boundary condition is Neumann, Robin, periodic, or implicit in a way not represented by endpoint values.
- The output vector omits the boundary points.
- The boundary metadata is inferred loosely rather than given by the numerical problem.

Boundary correction is easy to explain and useful for reports, but it is usually not enough for a strong result by itself.

## Integral Balance

Use an integral or energy-balance equality when:

- The discretization supplies a meaningful target balance.
- Quadrature weights are known and match the output grid.
- The balance quantity is a true physical or numerical invariant for the chosen output.

Do not use an integral balance when:

- Source terms, sinks, boundary fluxes, or time-window choices make the scalar balance ambiguous.
- The summer model does not expose the balance target.
- The balance is only a rough trend.

For the current synthetic heat helper, the mass-like integral is sample metadata. For the geothermal project, use a balance only if the numerical model gives a defensible target.

## Box Bounds

Use lower and upper bounds when:

- Bounds come from physical limits, safety limits, normalization limits, or simulation validity limits.
- The bounds are loose enough to flag bad predictions rather than hide them.
- Correction norms are reported with RMSE and violation metrics.

Do not use bounds when:

- The bounds are chosen only to make metrics look better.
- The variable can physically exceed the proposed range.
- The corrected value would hide important out-of-domain model behavior.

Pure box correction can hit active-bound kinks. In this project, box-only correction should usually be evaluation-only unless the active branch is known to be regular.

## Positivity

Use positivity when:

- The output is concentration, density, pressure-like positive quantity, probability, or a normalized nonnegative field.
- The units and normalization make negative values invalid.

Do not use positivity when:

- The output is temperature in a shifted scale where negative values are valid.
- The variable is signed by definition.
- The target data contains legitimate negative values.

Temperature fields need special care. Positivity may be valid in Kelvin or after a specific normalization, but not automatically in Celsius or centered variables.

## Combined Correction

Use combined correction when several constraints are simultaneously true:

- Boundary plus box.
- Boundary plus balance.
- Boundary plus balance plus box.
- Positivity plus normalization.

Combined correction is the current helper default because it makes the reported output feasible with respect to all selected constraints. Smaller modes are useful ablations and diagnostics.

## Status Rules

- `:success` is the only status that claims a valid gradient.
- `:infeasible_constraint` means the requested physics and bounds are incompatible.
- `:nonunique_input` often means an active-set kink. Use it as a warning for train-time correction.
- Large correction norms mean the raw model is far from feasible even if correction succeeds.

## Recommended Summer Path

1. Start with boundary correction if the final DeepONet output includes Dirichlet boundary points.
2. Add bounds only if they come from physical or simulation-validity limits.
3. Add balance correction only if the numerical model supplies a clear balance target.
4. Run evaluation-only correction first.
5. Add train-time correction after status and correction-norm diagnostics look stable.
