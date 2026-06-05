# Larger fixed-grid PDE integrations for the sparse projection backends.
# Run: julia --project=benchmarks/field benchmarks/field/run_pde_integrations.jl
include("FieldMass.jl")
using .FieldMass
using Random, Printf

const SEED = 21
const STEPS = 80

function report_bounded(name, predict, data, w, lower, upper)
    ev = evaluate_bounded_model(predict, data, w, lower = lower, upper = upper)
    @printf("%-18s  RMSE %.5f   max |dmass| %.3e   max lower %.3e   max upper %.3e\n",
            name, ev.rmse, ev.mass_max, ev.lower_max, ev.upper_max)
end

function run_heat(rng)
    K = 96
    train, _, w = sample_heat_fields(16, K, rng = rng)
    test, _, _ = sample_heat_fields(8, K, rng = rng)
    p0 = init_mlp(nout = K, seed = SEED)
    fc = FailureCounter()
    p, _ = train!(pp -> sparse_bounded_projected_loss(pp, train, w,
                                                      0.0, 2.0,
                                                      fc = fc),
                  p0, steps = STEPS, lr = 8e-3)
    println("\n=== heat field, sparse bounded projection ===")
    report_bounded("sparse-bounded",
                   d -> sparse_bounded_projected_field(p, d.theta, w,
                                                       d.mass0, 0.0, 2.0),
                   test, w, 0.0, 2.0)
    println("statuses: ", fc.counts)
end

function run_burgers(rng)
    K = 128
    train, _, w = sample_burgers_fields(16, K, rng = rng)
    test, _, _ = sample_burgers_fields(8, K, rng = rng)
    p0 = init_mlp(nout = K, seed = SEED + 1)
    fc = FailureCounter()
    p, _ = train!(pp -> sparse_qp_bounded_projected_loss(pp, train, w,
                                                         -1.25, 1.25,
                                                         backend = :active_set,
                                                         fc = fc),
                  p0, steps = STEPS, lr = 8e-3)
    println("\n=== viscous Burgers field, sparse QP mass and bounds ===")
    report_bounded("sparse-qp-active",
                   d -> sparse_qp_bounded_projected_field(p, d.theta, w,
                                                          d.mass0,
                                                          -1.25, 1.25,
                                                          backend = :active_set),
                   test, w, -1.25, 1.25)
    println("statuses: ", fc.counts)
end

function run_allen_cahn(rng)
    K = 96
    train, _, w = sample_allen_cahn_fields(16, K, rng = rng)
    test, _, _ = sample_allen_cahn_fields(8, K, rng = rng)
    p0 = init_mlp(nout = K, seed = SEED + 2)
    fc = FailureCounter()
    p, _ = train!(pp -> sparse_qp_box_projected_loss(pp, train,
                                                     -1.0, 1.0,
                                                     backend = :active_set,
                                                     fc = fc),
                  p0, steps = STEPS, lr = 8e-3)
    println("\n=== Allen-Cahn field, sparse QP box bounds ===")
    report_bounded("sparse-qp-active",
                   d -> sparse_qp_box_projected_field(p, d.theta,
                                                      -1.0, 1.0,
                                                      backend = :active_set),
                   test, w, -1.0, 1.0)
    println("statuses: ", fc.counts)
end

rng = MersenneTwister(SEED)
run_heat(rng)
run_burgers(rng)
run_allen_cahn(rng)
