# Constraint audit for summer-style DeepONet sample metadata.
# Run with the built-in mock batch:
# julia --project=benchmarks/deeponet benchmarks/deeponet/constraint_audit.jl
# Run with a simple CSV batch:
# julia --project=benchmarks/deeponet benchmarks/deeponet/constraint_audit.jl path\to\batch.csv
using DelimitedFiles, Printf

const OUT = joinpath(@__DIR__, "results")
isdir(OUT) || mkdir(OUT)

function has_field(sample, name)
    return name in propertynames(sample) && getproperty(sample, name) !== nothing
end

function has_boundary(sample)
    return has_field(sample, :left_bc) && has_field(sample, :right_bc)
end

function has_bounds(sample)
    return has_field(sample, :lower_bound) && has_field(sample, :upper_bound)
end

function has_balance(sample)
    return has_field(sample, :balance) || has_field(sample, :energy_balance)
end

function positivity_declared(sample)
    return has_field(sample, :positivity_valid)
end

function positivity_valid(sample)
    return positivity_declared(sample) && Bool(getproperty(sample,
                                                           :positivity_valid))
end

function cell_present(x)
    s = strip(string(x))
    return !(isempty(s) || lowercase(s) in ("missing", "nothing", "na"))
end

function parse_bool(x)
    s = lowercase(strip(string(x)))
    return s in ("true", "t", "1", "yes", "y")
end

function mock_samples()
    return [
        (case_id = "dirichlet_balance_bounds",
         left_bc = 0.8, right_bc = 1.1, balance = 0.94,
         lower_bound = 0.0, upper_bound = 2.0,
         positivity_valid = false),
        (case_id = "boundary_bounds_only",
         left_bc = 0.9, right_bc = 1.0, balance = nothing,
         lower_bound = 0.0, upper_bound = 2.0,
         positivity_valid = false),
        (case_id = "balance_only_positive",
         left_bc = nothing, right_bc = nothing, balance = 1.0,
         lower_bound = nothing, upper_bound = nothing,
         positivity_valid = true),
        (case_id = "metadata_missing",
         left_bc = nothing, right_bc = nothing, balance = nothing,
         lower_bound = nothing, upper_bound = nothing,
         positivity_valid = nothing),
    ]
end

function read_csv_samples(path)
    data, header = readdlm(path, ',', header = true)
    names = Symbol.(vec(String.(header)))
    samples = NamedTuple[]
    for i in axes(data, 1)
        values = Any[]
        for name in names
            value = data[i, findfirst(==(name), names)]
            if name === :positivity_valid
                push!(values, cell_present(value) ? parse_bool(value) :
                      nothing)
            else
                push!(values, cell_present(value) ? value : nothing)
            end
        end
        push!(samples, NamedTuple{Tuple(names)}(Tuple(values)))
    end
    return samples
end

function recommended_mode(sample)
    boundary = has_boundary(sample)
    bounds = has_bounds(sample)
    balance = has_balance(sample)
    positive = positivity_valid(sample)
    if boundary && bounds && balance
        return :full_boundary_balance_box
    elseif boundary && balance
        return :boundary_balance
    elseif boundary && bounds
        return :boundary_box
    elseif balance && bounds
        return :balance_box
    elseif boundary
        return :boundary_only
    elseif balance
        return positive ? :positive_balance : :balance_only
    elseif bounds
        return :box_only_eval
    elseif positive
        return :positivity_only
    else
        return :none
    end
end

function sample_id(sample, idx)
    return has_field(sample, :case_id) ? string(sample.case_id) :
        "sample_$(idx)"
end

function audit_sample(sample, idx)
    return (case_id = sample_id(sample, idx),
            boundary = has_boundary(sample) ? :available : :missing,
            bounds = has_bounds(sample) ? :available : :missing,
            balance = has_balance(sample) ? :available : :missing,
            positivity = positivity_declared(sample) ?
                (positivity_valid(sample) ? :valid : :declared_false) :
                :not_declared,
            recommended_mode = recommended_mode(sample))
end

input_path = isempty(ARGS) ? nothing : ARGS[1]
samples = input_path === nothing ? mock_samples() : read_csv_samples(input_path)
audits = [audit_sample(sample, i) for (i, sample) in enumerate(samples)]

open(joinpath(OUT, "constraint_audit.csv"), "w") do io
    println(io, "case_id,boundary,bounds,balance,positivity,recommended_mode")
    for row in audits
        println(io, @sprintf("%s,%s,%s,%s,%s,%s",
                             row.case_id, row.boundary, row.bounds,
                             row.balance, row.positivity,
                             row.recommended_mode))
    end
end

open(joinpath(OUT, "constraint_audit.md"), "w") do io
    println(io, "# Constraint Audit")
    println(io)
    source = input_path === nothing ? "built-in mock batch" : input_path
    println(io, "Source: `", source, "`")
    println(io)
    println(io, "| case | boundary | bounds | balance | positivity | recommended mode |")
    println(io, "| --- | --- | --- | --- | --- | --- |")
    for row in audits
        println(io, "| ", row.case_id, " | ", row.boundary, " | ",
                row.bounds, " | ", row.balance, " | ", row.positivity,
                " | ", row.recommended_mode, " |")
    end
    println(io)
    println(io, "Use the recommended mode as a starting point only. Final correction choices still need physical review against units, grid ordering, and dataset metadata.")
end

println("wrote constraint_audit.csv and constraint_audit.md to ", OUT)
