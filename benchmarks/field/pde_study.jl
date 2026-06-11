# Multi-seed larger fixed-grid PDE study for sparse projection paths, with
# runtime and correction-norm reporting. Defaults are paper oriented and
# overridable through STRUCTPINN_FIELD_PDE_* settings in FieldScenarios.jl.
# Run: julia --project=benchmarks/field benchmarks/field/pde_study.jl
include("FieldMass.jl")
include("FieldScenarios.jl")
using .FieldMass
using .FieldScenarios
using Random, Statistics, Printf, DelimitedFiles

const SCENARIO = pde_study_scenario()
const SEEDS = SCENARIO.seeds
const STEPS = SCENARIO.steps
const NTRAIN = SCENARIO.train_samples
const NTEST = SCENARIO.test_samples
const OUT = SCENARIO.out
isdir(OUT) || mkdir(OUT)

function _merge!(dest::FailureCounter, src::FailureCounter)
    for (s, c) in src.counts
        dest.counts[s] = get(dest.counts, s, 0) + c
    end
    return dest
end

function _stat(xs)
    return mean(xs), length(xs) == 1 ? 0.0 : std(xs)
end

# Mean and max Euclidean distance between the corrected prediction and the raw
# model output over the test set, so large successful corrections stay visible.
function correction_stats(predict, p, test_data)
    norms = Float64[]
    for d in test_data
        corrected = predict(d)
        raw = field_model(p, d.theta)
        push!(norms, sqrt(sum(abs2, corrected .- raw)))
    end
    return mean(norms), maximum(norms)
end

function run_heat(seed)
    rng = MersenneTwister(seed)
    K = 96
    train, _, w = sample_heat_fields(NTRAIN, K, rng = rng)
    test, _, _ = sample_heat_fields(NTEST, K, rng = rng)
    p0 = init_mlp(nout = K, seed = 100 + seed)
    fc = FailureCounter()
    local p, hist
    secs = @elapsed begin
        p, hist = train!(pp -> sparse_bounded_projected_loss(pp, train, w,
                                                             0.0, 2.0,
                                                             fc = fc),
                         p0, steps = STEPS, lr = 8e-3)
    end
    predict = d -> sparse_bounded_projected_field(p, d.theta, w, d.mass0,
                                                  0.0, 2.0)
    ev = evaluate_bounded_model(predict, test, w, lower = 0.0, upper = 2.0)
    cmean, cmax = correction_stats(predict, p, test)
    return ev, hist, fc, secs, cmean, cmax
end

function run_burgers(seed)
    rng = MersenneTwister(1000 + seed)
    K = 128
    train, _, w = sample_burgers_fields(NTRAIN, K, rng = rng)
    test, _, _ = sample_burgers_fields(NTEST, K, rng = rng)
    p0 = init_mlp(nout = K, seed = 200 + seed)
    fc = FailureCounter()
    local p, hist
    secs = @elapsed begin
        p, hist = train!(pp -> sparse_qp_bounded_projected_loss(pp, train, w,
                                                                -1.25, 1.25,
                                                                backend = :active_set,
                                                                fc = fc),
                         p0, steps = STEPS, lr = 8e-3)
    end
    predict = d -> sparse_qp_bounded_projected_field(p, d.theta, w, d.mass0,
                                                     -1.25, 1.25,
                                                     backend = :active_set)
    ev = evaluate_bounded_model(predict, test, w, lower = -1.25, upper = 1.25)
    cmean, cmax = correction_stats(predict, p, test)
    return ev, hist, fc, secs, cmean, cmax
end

function run_allen_cahn(seed)
    rng = MersenneTwister(2000 + seed)
    K = 96
    train, _, w = sample_allen_cahn_fields(NTRAIN, K, rng = rng)
    test, _, _ = sample_allen_cahn_fields(NTEST, K, rng = rng)
    p0 = init_mlp(nout = K, seed = 300 + seed)
    fc = FailureCounter()
    local p, hist
    secs = @elapsed begin
        p, hist = train!(pp -> sparse_qp_box_projected_loss(pp, train,
                                                            -1.0, 1.0,
                                                            backend = :active_set,
                                                            fc = fc),
                         p0, steps = STEPS, lr = 8e-3)
    end
    predict = d -> sparse_qp_box_projected_field(p, d.theta, -1.0, 1.0,
                                                 backend = :active_set)
    ev = evaluate_bounded_model(predict, test, w, lower = -1.0, upper = 1.0)
    cmean, cmax = correction_stats(predict, p, test)
    return ev, hist, fc, secs, cmean, cmax
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
times = Dict(name => Float64[] for (name, _, _, _) in runners)
corr_means = Dict(name => Float64[] for (name, _, _, _) in runners)
corr_maxes = Dict(name => Float64[] for (name, _, _, _) in runners)
status_totals = Dict(name => FailureCounter() for (name, _, _, _) in runners)

for seed in SEEDS
    @printf("seed %d: running larger PDE integrations ...\n", seed)
    for (name, _, _, runner) in runners
        ev, hist, fc, secs, cmean, cmax = runner(seed)
        for k in metrics
            push!(acc[name][k], getfield(ev, k))
        end
        push!(losses[name], hist[end] / hist[1])
        push!(times[name], secs)
        push!(corr_means[name], cmean)
        push!(corr_maxes[name], cmax)
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
    tm = _stat(times[name])
    cn = _stat(corr_means[name])
    @printf("%-10s %-18s RMSE %.5f+/-%.5f  max|dmass| %.3e+/-%.3e  lower %.3e+/-%.3e  upper %.3e+/-%.3e  loss ratio %.3f+/-%.3f  train %.1fs+/-%.1fs  corrnorm %.3e+/-%.3e\n",
            name, backend, rm..., mm..., lm..., um..., lr..., tm..., cn...)
    println("statuses for ", name, ": ", status_totals[name].counts)
end

open(joinpath(OUT, "pde_results.csv"), "w") do io
    println(io, "problem,backend,mass_enforced,rmse_mean,rmse_std,massmax_mean,massmax_std,massmean_mean,massmean_std,lowermax_mean,lowermax_std,lowermean_mean,lowermean_std,uppermax_mean,uppermax_std,uppermean_mean,uppermean_std,lossratio_mean,lossratio_std,corrnorm_mean,corrnorm_std,corrmax_mean,corrmax_std,train_seconds_mean,train_seconds_std")
    for (name, backend, mass_enforced, _) in runners
        ms = Dict(k => _stat(acc[name][k]) for k in metrics)
        lr = _stat(losses[name])
        tm = _stat(times[name])
        cn = _stat(corr_means[name])
        cx = _stat(corr_maxes[name])
        println(io, @sprintf("%s,%s,%s,%.8f,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8f,%.8f,%.8e,%.8e,%.8e,%.8e,%.6f,%.6f",
                             name, backend, mass_enforced, ms[:rmse]..., ms[:mass_max]...,
                             ms[:mass_mean]..., ms[:lower_max]..., ms[:lower_mean]...,
                             ms[:upper_max]..., ms[:upper_mean]..., lr...,
                             cn..., cx..., tm...))
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
