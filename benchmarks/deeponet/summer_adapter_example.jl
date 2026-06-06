# Mock summer-interface adapter example for fixed-grid operator outputs.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/summer_adapter_example.jl
include("DeepONetHeat.jl")
using .DeepONetHeat
using Random, Statistics, Printf

const OUT = joinpath(@__DIR__, "results")
isdir(OUT) || mkdir(OUT)

const K = 32
const N = 6

x = grid(K)
w = trapezoid_weights(x)
rows = boundary_mass_rows(w)
rng = MersenneTwister(41)

function make_sample(i)
    left = 0.78 + 0.03 * i
    right = 1.08 - 0.02 * i
    features = [0.18 + 0.05 * i, left, right,
                0.6 * sin(0.7 * i), 0.4 * cos(0.5 * i)]
    target = heat_operator_field(x, features)
    return (case_id = "mock_geothermal_$(i)",
            features = features,
            target = target,
            left_bc = left,
            right_bc = right,
            energy_balance = mass(target, w),
            lower_bound = fill(0.0, K),
            upper_bound = fill(2.0, K))
end

samples = [make_sample(i) for i in 1:N]
contexts = operator_correction_contexts(
    samples,
    K,
    equality_rows = rows,
    equality_values = sample -> [sample.left_bc, sample.right_bc,
                                 sample.energy_balance],
    lower = sample -> sample.lower_bound,
    upper = sample -> sample.upper_bound,
    weights = w,
    mode = :summer_adapter_example,
    metadata = sample -> (case_id = sample.case_id,))

function raw_surrogate(sample)
    noise = 0.08 .* randn(rng, K)
    drift = 0.10 .* sin.(2pi .* x)
    return sample.target .+ noise .+ drift
end

function boundary_error(pred, sample)
    return max(abs(pred[1] - sample.left_bc),
               abs(pred[end] - sample.right_bc))
end

function balance_error(pred, sample)
    return abs(mass(pred, w) - sample.energy_balance)
end

function box_error(pred, sample)
    low = maximum(max.(sample.lower_bound .- pred, 0.0))
    high = maximum(max.(pred .- sample.upper_bound, 0.0))
    return max(low, high)
end

raw_rmse = Float64[]
corrected_rmse = Float64[]
raw_boundary = Float64[]
corrected_boundary = Float64[]
raw_balance = Float64[]
corrected_balance = Float64[]
raw_box = Float64[]
corrected_box = Float64[]
status_log = ProjectionLog()

open(joinpath(OUT, "summer_adapter_example.csv"), "w") do io
    println(io, "case_id,raw_rmse,corrected_rmse,raw_boundary,corrected_boundary,raw_balance,corrected_balance,raw_box,corrected_box,status,correction_norm")
    for (sample, ctx) in zip(samples, contexts)
        raw = raw_surrogate(sample)
        before_rmse = field_rmse(raw, sample.target)
        before_boundary = boundary_error(raw, sample)
        before_balance = balance_error(raw, sample)
        before_box = box_error(raw, sample)
        corrected = corrected_output(raw, ctx, log = status_log)
        after_rmse = field_rmse(corrected, sample.target)
        after_boundary = boundary_error(corrected, sample)
        after_balance = balance_error(corrected, sample)
        after_box = box_error(corrected, sample)
        push!(raw_rmse, before_rmse)
        push!(corrected_rmse, after_rmse)
        push!(raw_boundary, before_boundary)
        push!(corrected_boundary, after_boundary)
        push!(raw_balance, before_balance)
        push!(corrected_balance, after_balance)
        push!(raw_box, before_box)
        push!(corrected_box, after_box)
        println(io, @sprintf("%s,%.8f,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%s,%.8e",
                             sample.case_id, before_rmse, after_rmse,
                             before_boundary, after_boundary,
                             before_balance, after_balance,
                             before_box, after_box, :success,
                             status_log.correction_norms[end]))
    end
end

open(joinpath(OUT, "summer_adapter_example.md"), "w") do io
    println(io, "# Summer Adapter Example")
    println(io)
    println(io, "This mock fixture uses a geothermal-style sample shape with fields for features, target output, boundary metadata, balance metadata, and per-sample bounds.")
    println(io)
    println(io, "Mean raw RMSE: ", @sprintf("%.4g", mean(raw_rmse)))
    println(io, "Mean corrected RMSE: ",
            @sprintf("%.4g", mean(corrected_rmse)))
    println(io, "Max corrected boundary violation: ",
            @sprintf("%.4g", maximum(corrected_boundary)))
    println(io, "Max corrected balance violation: ",
            @sprintf("%.4g", maximum(corrected_balance)))
    println(io, "Max corrected box violation: ",
            @sprintf("%.4g", maximum(corrected_box)))
    println(io, "Projection statuses: ", status_log.counts)
end

println("wrote summer_adapter_example.csv and summer_adapter_example.md to ",
        OUT)
