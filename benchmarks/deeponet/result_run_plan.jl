# DeepONet helper broader result-run plan.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/result_run_plan.jl
include("DeepONetScenarios.jl")
using .DeepONetScenarios

const OUT = deeponet_results_dir()
isdir(OUT) || mkdir(OUT)

const COLUMNS = ["priority", "batch", "command", "env", "grids",
                 "seeds", "steps", "train_samples", "test_samples",
                 "gate", "purpose"]

function clean_cell(value)
    return replace(string(value), "," => " ")
end

function csv_row(row)
    return join([clean_cell(scenario_field(row, Symbol(col)))
                 for col in COLUMNS], ",")
end

function md_row(row)
    return "| " *
           join([string(scenario_field(row, Symbol(col)))
                 for col in COLUMNS], " | ") *
           " |"
end

rows = final_run_plan_rows()

open(joinpath(OUT, "result_run_plan.csv"), "w") do io
    println(io, join(COLUMNS, ","))
    for row in rows
        println(io, csv_row(row))
    end
end

open(joinpath(OUT, "result_run_plan.md"), "w") do io
    println(io, "# DeepONet Result Run Plan")
    println(io)
    println(io, "This plan separates committed helper runs from guarded expansion runs.")
    println(io)
    println(io, "| ", join(COLUMNS, " | "), " |")
    println(io, "| ", join(fill("---", length(COLUMNS)), " | "), " |")
    for row in rows
        println(io, md_row(row))
    end
    println(io)
    println(io, "Run priority 1 and 2 before treating any DeepONet helper result as paper-facing.")
    println(io, "Run priority 3 only if larger-output train-time behavior is still unclear after profiling.")
    println(io, "Run priority 4 and 5 only after the result tables stabilize.")
    println(io, "For rows with an `env` value, set those variables before running the command.")
end

println("wrote result_run_plan.csv and result_run_plan.md to ", OUT)
