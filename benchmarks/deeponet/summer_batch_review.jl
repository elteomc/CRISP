# Review an exported summer-style batch before train-time correction.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/summer_batch_review.jl path/to/batch.csv
if !isdefined(@__MODULE__, :read_exported_batch)
    include("summer_batch_eval.jl")
end

using Printf, Statistics

const REVIEW_OUT = joinpath(@__DIR__, "results")

function passfail(flag)
    return flag ? "pass" : "fail"
end

function finite_values(xs)
    return [x for x in xs if !isnan(x)]
end

function max_or_nan(xs)
    ys = finite_values(xs)
    isempty(ys) && return NaN
    return maximum(ys)
end

function all_have(samples, field)
    return all(sample -> getproperty(sample, field) !== nothing, samples)
end

function shared_length(samples, field)
    lengths = Int[]
    for sample in samples
        value = getproperty(sample, field)
        value === nothing || push!(lengths, length(value))
    end
    isempty(lengths) && return false
    return length(unique(lengths)) == 1
end

function strictly_increasing(x)
    return all(diff(x) .> 0)
end

function grid_ok(samples)
    all_have(samples, :x) || return false
    shared_length(samples, :x) || return false
    return all(sample -> strictly_increasing(sample.x), samples)
end

function boundary_target_error(sample)
    sample.target === nothing && return NaN
    has_boundary(sample) || return NaN
    return max(abs(sample.target[1] - sample.left_bc),
               abs(sample.target[end] - sample.right_bc))
end

function target_balance_error(sample)
    sample.target === nothing && return NaN
    has_balance(sample) || return NaN
    sample.weights === nothing && return NaN
    return abs(mass(sample.target, sample.weights) - sample.balance)
end

function target_box_error(sample)
    sample.target === nothing && return NaN
    has_bounds(sample) || return NaN
    low = maximum(max.(sample.lower_bound .- sample.target, 0.0))
    high = maximum(max.(sample.target .- sample.upper_bound, 0.0))
    return max(low, high)
end

function bounds_order_ok(samples)
    all_have(samples, :lower_bound) || return false
    all_have(samples, :upper_bound) || return false
    return all(sample -> all(sample.lower_bound .<= sample.upper_bound),
               samples)
end

function mode_set(samples)
    return sort(unique(string.(recommended_mode.(samples))))
end

function source_kind(path)
    name = lowercase(basename(path))
    occursin("fixture", name) && return "fixture"
    return "exported"
end

function review_rows(samples, path)
    btarget = max_or_nan(boundary_target_error.(samples))
    balance = max_or_nan(target_balance_error.(samples))
    box = max_or_nan(target_box_error.(samples))
    modes = mode_set(samples)
    return [
        (check = "source_kind", status = source_kind(path),
         detail = "fixture means adapter smoke test only"),
        (check = "recommended_mode", status =
             modes == ["full_boundary_balance_box"] ? "pass" : "review",
         detail = join(modes, " ")),
        (check = "grid_ordering", status = passfail(grid_ok(samples)),
         detail = "grid present with shared length and strict increase"),
        (check = "boundary_semantics", status =
             btarget <= 1e-9 ? "pass" : "review",
         detail = "max target endpoint mismatch $(fmt(btarget))"),
        (check = "balance_metadata", status =
             balance <= 1e-9 ? "pass" : "review",
         detail = "max target balance mismatch $(fmt(balance))"),
        (check = "bounds_metadata", status =
             bounds_order_ok(samples) && box <= 1e-9 ? "pass" : "review",
         detail = "max target bound violation $(fmt(box))"),
        (check = "units_metadata", status = "review",
         detail = "units are not represented in this CSV schema"),
    ]
end

function write_review(rows, samples, path, out)
    isdir(out) || mkpath(out)
    open(joinpath(out, "summer_batch_review.csv"), "w") do io
        println(io, "check,status,detail")
        for row in rows
            println(io, row.check, ",", row.status, ",",
                    replace(row.detail, "," => " "))
        end
    end
    open(joinpath(out, "summer_batch_review.md"), "w") do io
        println(io, "# Summer Batch Metadata Review")
        println(io)
        println(io, "Source: `", path, "`")
        println(io, "Samples: ", length(samples))
        println(io)
        println(io, "| check | status | detail |")
        println(io, "| --- | --- | --- |")
        for row in rows
            println(io, "| ", row.check, " | ", row.status, " | ",
                    row.detail, " |")
        end
        println(io)
        println(io, "The recommended mode remains a starting point. Units, boundary type, and balance semantics still require physical review on a real export.")
    end
end

function main(args = ARGS)
    isempty(args) &&
        throw(ArgumentError("usage: julia --project=benchmarks/deeponet benchmarks/deeponet/summer_batch_review.jl path/to/batch.csv"))
    path = args[1]
    samples = read_exported_batch(path)
    rows = review_rows(samples, path)
    write_review(rows, samples, path, REVIEW_OUT)
    println("wrote summer_batch_review.csv and summer_batch_review.md to ",
            REVIEW_OUT)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
