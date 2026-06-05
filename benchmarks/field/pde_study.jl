# Multi-seed larger fixed-grid PDE study for sparse projection paths.
# Run: julia --project=benchmarks/field benchmarks/field/pde_study.jl
include("FieldMass.jl")
using .FieldMass
using Random, Statistics, Printf, DelimitedFiles

const SEEDS = 1:3
const STEPS = 100
const OUT = joinpath(@__DIR__, "results")

isdir(OUT) || mkdir(OUT)

function _merge!(dest::FailureCounter, src::FailureCounter)
    for (s, c) in src.counts
        dest.counts[s] = get(dest.counts, s, 0) + c
    end
    return dest
end

function _stat(xs)
    return mean(xs), std(xs)
end

function run_heat(seed)
    rng = MersenneTwister(seed)
    K = 96
    train, _, w = sample_heat_fields(16, K, rng = rng)
    test, _, _ = sample_heat_fields(8, K, rng = rng)
    p0 = init_mlp(nout = K, seed = 100 + seed)
    fc = FailureCounter()
    p, hist = train!(pp -> sparse_bounded_projected_loss(pp, train, w,
                                                         0.0, 2.0,
                                                         fc = fc),
                     p0, steps = STEPS, lr = 8e-3)
    ev = evaluate_bounded_model(d -> sparse_bounded_projected_field(p, d.theta, w,
                                                                    d.mass0,
                                                                    0.0, 2.0),
                                test, w, lower = 0.0, upper = 2.0)
    return ev, hist, fc
end

function run_burgers(seed)
    rng = MersenneTwister(1000 + seed)
    K = 128
    train, _, w = sample_burgers_fields(16, K, rng = rng)
    test, _, _ = sample_burgers_fields(8, K, rng = rng)
    p0 = init_mlp(nout = K, seed = 200 + seed)
    fc = FailureCounter()
    p, hist = train!(pp -> sparse_qp_bounded_projected_loss(pp, train, w,
                                                            -1.25, 1.25,
                                                            backend = :active_set,
                                                            fc = fc),
                     p0, steps = STEPS, lr = 8e-3)
    ev = evaluate_bounded_model(d -> sparse_qp_bounded_projected_field(p, d.theta, w,
                                                                       d.mass0,
                                                                       -1.25, 1.25,
                                                                       backend = :active_set),
                                test, w, lower = -1.25, upper = 1.25)
    return ev, hist, fc
end

function run_allen_cahn(seed)
    rng = MersenneTwister(2000 + seed)
    K = 96
    train, _, w = sample_allen_cahn_fields(16, K, rng = rng)
    test, _, _ = sample_allen_cahn_fields(8, K, rng = rng)
    p0 = init_mlp(nout = K, seed = 300 + seed)
    fc = FailureCounter()
    p, hist = train!(pp -> sparse_qp_box_projected_loss(pp, train,
                                                        -1.0, 1.0,
                                                        backend = :active_set,
                                                        fc = fc),
                     p0, steps = STEPS, lr = 8e-3)
    ev = evaluate_bounded_model(d -> sparse_qp_box_projected_field(p, d.theta,
                                                                   -1.0, 1.0,
                                                                   backend = :active_set),
                                test, w, lower = -1.0, upper = 1.0)
    return ev, hist, fc
end

const runners = [
    ("heat", "sparse_bounded", true, run_heat),
    ("burgers", "sparse_qp_bounded", true, run_burgers),
    ("allen_cahn", "sparse_qp_box", false, run_allen_cahn),
]

metrics = (:rmse, :mass_max, :mass_mean, :lower_max, :lower_mean,
           :upper_max, :upper_mean)
acc = Dict(name => Dict(k => Float64[] for k in metrics) for (name, _, _, _) in runners)
losses = Dict(name => Float64[] for (name, _, _, _) in runners)
status_totals = Dict(name => FailureCounter() for (name, _, _, _) in runners)

for seed in SEEDS
    @printf("seed %d: running larger PDE integrations ...\n", seed)
    for (name, _, _, runner) in runners
        ev, hist, fc = runner(seed)
        for k in metrics
            push!(acc[name][k], getfield(ev, k))
        end
        push!(losses[name], hist[end] / hist[1])
        _merge!(status_totals[name], fc)
    end
end

println("\n=== Larger PDE sparse projection study: mean +/- std over $(length(SEEDS)) seeds ===")
for (name, backend, _, _) in runners
    rm = _stat(acc[name][:rmse])
    mm = _stat(acc[name][:mass_max])
    lm = _stat(acc[name][:lower_max])
    um = _stat(acc[name][:upper_max])
    lr = _stat(losses[name])
    @printf("%-10s %-18s RMSE %.5f+/-%.5f  max|dmass| %.3e+/-%.3e  lower %.3e+/-%.3e  upper %.3e+/-%.3e  loss ratio %.3f+/-%.3f\n",
            name, backend, rm..., mm..., lm..., um..., lr...)
    println("statuses for ", name, ": ", status_totals[name].counts)
end

open(joinpath(OUT, "pde_results.csv"), "w") do io
    println(io, "problem,backend,mass_enforced,rmse_mean,rmse_std,massmax_mean,massmax_std,massmean_mean,massmean_std,lowermax_mean,lowermax_std,lowermean_mean,lowermean_std,uppermax_mean,uppermax_std,uppermean_mean,uppermean_std,lossratio_mean,lossratio_std")
    for (name, backend, mass_enforced, _) in runners
        ms = Dict(k => _stat(acc[name][k]) for k in metrics)
        lr = _stat(losses[name])
        println(io, @sprintf("%s,%s,%s,%.8f,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8f,%.8f",
                             name, backend, mass_enforced, ms[:rmse]..., ms[:mass_max]...,
                             ms[:mass_mean]..., ms[:lower_max]..., ms[:lower_mean]...,
                             ms[:upper_max]..., ms[:upper_mean]..., lr...))
    end
end

open(joinpath(OUT, "pde_statuses.csv"), "w") do io
    println(io, "problem,status,count")
    for (name, _, _, _) in runners
        for (status, count) in sort(collect(status_totals[name].counts))
            println(io, string(name, ",", status, ",", count))
        end
    end
end

println("wrote pde_results.csv and pde_statuses.csv to ", OUT)
