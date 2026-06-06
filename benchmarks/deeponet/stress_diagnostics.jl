# Stress diagnostics for DeepONet output-correction adapter paths.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/stress_diagnostics.jl
include("DeepONetHeat.jl")
using .DeepONetHeat
using StructPINN: project
using Printf

const OUT = joinpath(@__DIR__, "results")
isdir(OUT) || mkdir(OUT)

const K = 20
x = grid(K)
w = trapezoid_weights(x)
rows = boundary_mass_rows(w)
base_theta = [0.4, 0.85, 1.05, 0.5, -0.3]
target = heat_operator_field(x, base_theta)
sample = (theta = base_theta, u = target, left = base_theta[2],
          right = base_theta[3], mass0 = mass(target, w))

function status_case(name, res, expected, note)
    pass = res.status === expected
    return (case = name, status = res.status, expected = expected,
            pass = pass, correction_norm = Float64(res.correction_norm),
            note = note)
end

cases = NamedTuple[]

impossible_ctx = operator_correction_context(
    K,
    equality_rows = rows,
    equality_values = [sample.left, sample.right, 10.0],
    lower = 0.0,
    upper = 1.0,
    weights = w,
    mode = :stress_infeasible_mass)
push!(cases, status_case("infeasible_mass_under_bounds",
                         project(impossible_ctx.constraint, target),
                         :infeasible_constraint,
                         "Requested balance exceeds box capacity."))

large_ctx = heat_correction_context(sample, w, mode = :full, cached = true)
large_raw = fill(0.0, K)
large_res = project(large_ctx.constraint, large_raw)
push!(cases, status_case("large_correction_norm",
                         large_res,
                         :success,
                         "Raw output is far from feasible but constraints are compatible."))

box_ctx = operator_correction_context(
    K,
    equality_rows = [],
    equality_values = Float64[],
    lower = 0.0,
    upper = 1.0,
    mode = :stress_box_kink)
box_raw = vcat([0.0], fill(0.5, K - 1))
push!(cases, status_case("box_only_kink",
                         project(box_ctx.constraint, box_raw),
                         :nonunique_input,
                         "Raw output lies exactly on an active bound."))

function malformed_row_case()
    try
        operator_correction_context(K,
                                    equality_rows = [ones(K - 1)],
                                    equality_values = [1.0],
                                    lower = nothing,
                                    upper = nothing,
                                    mode = :stress_bad_rows)
    catch err
        return err isa ArgumentError, sprint(showerror, err)
    end
    return false, ""
end

malformed_pass, malformed_message = malformed_row_case()
push!(cases, (case = "malformed_adapter_rows",
              status = malformed_pass ? :argument_error : :unexpected,
              expected = :argument_error,
              pass = malformed_pass,
              correction_norm = 0.0,
              note = malformed_message))

open(joinpath(OUT, "stress_diagnostics.csv"), "w") do io
    println(io, "case,status,expected,pass,correction_norm,note")
    for c in cases
        println(io, @sprintf("%s,%s,%s,%s,%.8e,%s",
                             c.case, c.status, c.expected, c.pass,
                             c.correction_norm, c.note))
    end
end

open(joinpath(OUT, "stress_diagnostics.md"), "w") do io
    println(io, "# DeepONet Stress Diagnostics")
    println(io)
    println(io, "| case | status | expected | pass | correction norm |")
    println(io, "| --- | --- | --- | --- | --- |")
    for c in cases
        println(io, "| ", c.case, " | ", c.status, " | ",
                c.expected, " | ", c.pass, " | ",
                @sprintf("%.4g", c.correction_norm), " |")
    end
end

all(c -> c.pass, cases) || error("one or more stress diagnostics failed")
println("wrote stress_diagnostics.csv and stress_diagnostics.md to ", OUT)
