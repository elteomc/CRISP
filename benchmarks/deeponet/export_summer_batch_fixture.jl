# Export a deterministic summer-style CSV batch fixture.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/export_summer_batch_fixture.jl
include("DeepONetHeat.jl")
using .DeepONetHeat
using Printf, Random

const OUT = joinpath(@__DIR__, "results")
isdir(OUT) || mkdir(OUT)

const K = 32
const TRAIN_ROWS = 8
const TEST_ROWS = 4

function vector_cell(values)
    return "\"" * join((@sprintf("%.12g", Float64(v)) for v in values),
                       "|") * "\""
end

function row_split(i)
    return i <= TRAIN_ROWS ? "train" : "test"
end

function fixture_row(i, x, w, rng)
    left = 0.76 + 0.025 * i
    right = 1.10 - 0.018 * i
    features = [0.10 + 0.045 * i, left, right,
                0.55 * sin(0.6 * i), 0.35 * cos(0.4 * i)]
    target = heat_operator_field(x, features)
    smooth_bias = 0.035 .* sin.(2pi .* x .+ 0.2 * i)
    noise = 0.018 .* randn(rng, length(x))
    raw = target .+ smooth_bias .+ noise
    return (case_id = "fixture_case_$(i)",
            split = row_split(i),
            features = features,
            raw = raw,
            target = target,
            grid = x,
            left_bc = left,
            right_bc = right,
            balance = mass(target, w),
            lower = 0.0,
            upper = 2.0)
end

x = grid(K)
w = trapezoid_weights(x)
rng = MersenneTwister(55)
rows = [fixture_row(i, x, w, rng) for i in 1:(TRAIN_ROWS + TEST_ROWS)]
path = joinpath(OUT, "summer_batch_fixture.csv")

open(path, "w") do io
    println(io, "case_id,split,features,raw_output,target_output,grid,left_bc,right_bc,energy_balance,lower_bound,upper_bound")
    for row in rows
        println(io, join((row.case_id,
                          row.split,
                          vector_cell(row.features),
                          vector_cell(row.raw),
                          vector_cell(row.target),
                          vector_cell(row.grid),
                          @sprintf("%.12g", row.left_bc),
                          @sprintf("%.12g", row.right_bc),
                          @sprintf("%.12g", row.balance),
                          @sprintf("%.12g", row.lower),
                          @sprintf("%.12g", row.upper)), ","))
    end
end

println("wrote summer_batch_fixture.csv to ", OUT)
