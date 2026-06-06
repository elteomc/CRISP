# DeepONet helper run-scenario manifest.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/scenario_manifest.jl
include("DeepONetScenarios.jl")
using .DeepONetScenarios

const OUT = deeponet_results_dir()
isdir(OUT) || mkdir(OUT)

const COLUMNS = ["script", "scenario", "grids", "seeds", "steps",
                 "train_samples", "test_samples"]

function csv_row(row)
    return join([replace(string(scenario_field(row, Symbol(col))), "," => " ")
                 for col in COLUMNS], ",")
end

function md_row(row)
    return "| " *
           join([string(scenario_field(row, Symbol(col)))
                 for col in COLUMNS], " | ") *
           " |"
end

rows = scenario_rows()

open(joinpath(OUT, "run_scenarios.csv"), "w") do io
    println(io, join(COLUMNS, ","))
    for row in rows
        println(io, csv_row(row))
    end
end

open(joinpath(OUT, "run_scenarios.md"), "w") do io
    println(io, "# DeepONet Run Scenarios")
    println(io)
    println(io, "| ", join(COLUMNS, " | "), " |")
    println(io, "| ", join(fill("---", length(COLUMNS)), " | "), " |")
    for row in rows
        println(io, md_row(row))
    end
    println(io)
    println(io, "Environment overrides use the ",
            "`STRUCTPINN_DEEPONET_*` prefix.")
end

println("wrote run_scenarios.csv and run_scenarios.md to ", OUT)
