# Guarded runner for expanded DeepONet seed jobs.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/run_expanded_seed_jobs.jl
include("DeepONetScenarios.jl")
using .DeepONetScenarios
using DelimitedFiles, Printf

const SCENARIO = expansion_gate_scenario()
const OUT = SCENARIO.out
isdir(OUT) || mkdir(OUT)

function table(path)
    data, header = readdlm(path, ',', header = true)
    names = vec(String.(header))
    if data isa AbstractVector
        data = reshape(data, 1, :)
    end
    return data, names
end

function col(names, name)
    idx = findfirst(==(name), names)
    idx === nothing && error("missing column $(name)")
    return idx
end

function row_status(path, check)
    data, names = table(path)
    idx = findfirst(==(check), String.(data[:, col(names, "check")]))
    idx === nothing && error("missing gate check $(check)")
    return string(data[idx, col(names, "status")])
end

function allowed(path, check)
    return row_status(path, check) == "pass"
end

function julia_script(script)
    return `$(Base.julia_cmd()) --project=$(@__DIR__) $(joinpath(@__DIR__, script))`
end

function run_with_env(script, pair)
    cmd = julia_script(script)
    run(setenv(cmd, pair))
end

function write_rows(rows)
    open(joinpath(OUT, "expanded_seed_run.csv"), "w") do io
        println(io, "job,status,detail")
        for row in rows
            println(io, row.job, ",", row.status, ",",
                    replace(row.detail, "," => " "))
        end
    end
    open(joinpath(OUT, "expanded_seed_run.md"), "w") do io
        println(io, "# Expanded Seed Run")
        println(io)
        println(io, "| job | status | detail |")
        println(io, "| --- | --- | --- |")
        for row in rows
            println(io, "| ", row.job, " | ", row.status, " | ",
                    row.detail, " |")
        end
    end
end

gate_path = joinpath(OUT, "expanded_seed_gate.csv")
isfile(gate_path) || error("missing expanded_seed_gate.csv")

main_ok = allowed(gate_path, "expanded_main_allowed")
large_ok = allowed(gate_path, "expanded_large_allowed")
rows = NamedTuple[]

if main_ok && large_ok
    push!(rows, (job = "expanded_main_$(SCENARIO.main_seed_target)_seed",
                 status = "running",
                 detail = "study.jl with seeds 1:$(SCENARIO.main_seed_target)"))
    write_rows(rows)
    run_with_env("study.jl",
                 "STRUCTPINN_DEEPONET_STUDY_SEEDS" =>
                     "1:$(SCENARIO.main_seed_target)")
    rows[end] = (job = rows[end].job, status = "complete",
                 detail = rows[end].detail)

    push!(rows, (job = "expanded_large_$(SCENARIO.large_seed_target)_seed",
                 status = "running",
                 detail = "large_study.jl with seeds 1:$(SCENARIO.large_seed_target)"))
    write_rows(rows)
    run_with_env("large_study.jl",
                 "STRUCTPINN_DEEPONET_LARGE_SEEDS" =>
                     "1:$(SCENARIO.large_seed_target)")
    rows[end] = (job = rows[end].job, status = "complete",
                 detail = rows[end].detail)
else
    push!(rows, (job = "expanded_main_$(SCENARIO.main_seed_target)_seed",
                 status = "blocked",
                 detail = "expanded_main_allowed gate is $(row_status(gate_path, "expanded_main_allowed"))"))
    push!(rows, (job = "expanded_large_$(SCENARIO.large_seed_target)_seed",
                 status = "blocked",
                 detail = "expanded_large_allowed gate is $(row_status(gate_path, "expanded_large_allowed"))"))
end

write_rows(rows)
println("wrote expanded_seed_run.csv and expanded_seed_run.md to ", OUT)
