if !isdefined(@__MODULE__, :DeepONetHeat)
    include("DeepONetHeat.jl")
end
using .DeepONetHeat
using Printf, Statistics
using StructPINN: project

const DEFAULT_OUT = joinpath(@__DIR__, "results")

missing_cell(x) = begin
    s = strip(string(x))
    isempty(s) || lowercase(s) in ("missing", "nothing", "na", "nan")
end

function split_csv_line(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    quoted = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            if quoted && i < lastindex(line) && line[nextind(line, i)] == '"'
                write(buf, '"')
                i = nextind(line, i)
            else
                quoted = !quoted
            end
        elseif c == ',' && !quoted
            push!(fields, String(take!(buf)))
        else
            write(buf, c)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(buf)))
    return fields
end

function read_csv_rows(path)
    lines = filter(!isempty, strip.(readlines(path)))
    isempty(lines) && throw(ArgumentError("batch CSV is empty"))
    header = Symbol.(strip.(split_csv_line(lines[1])))
    rows = NamedTuple[]
    for line in lines[2:end]
        cells = strip.(split_csv_line(line))
        length(cells) == length(header) ||
            throw(ArgumentError("row has $(length(cells)) cells but header has $(length(header))"))
        values = map(c -> missing_cell(c) ? nothing : c, cells)
        push!(rows, NamedTuple{Tuple(header)}(Tuple(values)))
    end
    return rows
end

function first_present(row, names::Symbol...)
    for name in names
        if name in propertynames(row)
            value = getproperty(row, name)
            value === nothing || return value
        end
    end
    return nothing
end

function parse_scalar(value, name)
    value === nothing && return nothing
    try
        return parse(Float64, strip(string(value)))
    catch err
        throw(ArgumentError("could not parse scalar `$(name)` from `$(value)`"))
    end
end

function parse_vector(value, name; required = false)
    if value === nothing
        required && throw(ArgumentError("missing required vector `$(name)`"))
        return nothing
    end
    clean = replace(strip(string(value)),
                    '[' => ' ', ']' => ' ', '(' => ' ', ')' => ' ',
                    '|' => ' ', ',' => ' ')
    parts = split(clean)
    isempty(parts) && required &&
        throw(ArgumentError("missing required vector `$(name)`"))
    try
        return parse.(Float64, parts)
    catch err
        throw(ArgumentError("could not parse vector `$(name)` from `$(value)`"))
    end
end

function parse_bound(value, K, name)
    value === nothing && return nothing
    v = parse_vector(value, name, required = true)
    if length(v) == 1
        return fill(v[1], K)
    elseif length(v) == K
        return v
    end
    throw(ArgumentError("bound `$(name)` must be scalar or length $(K)"))
end

function endpoint_row(K, idx)
    row = zeros(Float64, K)
    row[idx] = 1.0
    return row
end

function exported_sample(row, idx)
    raw = parse_vector(first_present(row, :raw_output, :raw, :prediction),
                       "raw_output")
    target = parse_vector(first_present(row, :target_output, :target, :truth),
                          "target_output")
    raw === nothing && target === nothing &&
        throw(ArgumentError("row needs raw_output or target_output"))
    K = raw === nothing ? length(target) : length(raw)
    if raw !== nothing && length(raw) != K
        throw(ArgumentError("raw_output must have length $(K)"))
    end
    if target !== nothing && length(target) != K
        throw(ArgumentError("target_output must have length $(K)"))
    end
    features = parse_vector(first_present(row, :features, :input_features,
                                          :theta, :parameters),
                            "features")
    x = parse_vector(first_present(row, :grid, :x, :coordinates), "grid")
    if x !== nothing && length(x) != K
        throw(ArgumentError("grid must have length $(K)"))
    end
    weights = parse_vector(first_present(row, :weights, :quadrature_weights),
                           "weights")
    if weights !== nothing && length(weights) != K
        throw(ArgumentError("weights must have length $(K)"))
    elseif weights === nothing && x !== nothing
        weights = trapezoid_weights(x)
    end
    left = parse_scalar(first_present(row, :left_bc, :left, :left_boundary),
                        "left_bc")
    right = parse_scalar(first_present(row, :right_bc, :right,
                                       :right_boundary), "right_bc")
    balance = parse_scalar(first_present(row, :balance, :energy_balance,
                                         :mass, :integral), "balance")
    lower = parse_bound(first_present(row, :lower_bound, :lower), K,
                        "lower_bound")
    upper = parse_bound(first_present(row, :upper_bound, :upper), K,
                        "upper_bound")
    case_id = first_present(row, :case_id, :id, :sample_id)
    split = first_present(row, :split, :fold, :phase)
    id = case_id === nothing ? "sample_$(idx)" : string(case_id)
    return (case_id = id, raw = raw === nothing ? zeros(K) : raw,
            has_raw = raw !== nothing, target = target, features = features,
            split = split === nothing ? nothing : lowercase(string(split)),
            x = x,
            weights = weights, left_bc = left, right_bc = right,
            balance = balance, lower_bound = lower, upper_bound = upper)
end

read_exported_batch(path) =
    [exported_sample(row, i) for (i, row) in enumerate(read_csv_rows(path))]

has_boundary(sample) = sample.left_bc !== nothing && sample.right_bc !== nothing
has_balance(sample) = sample.balance !== nothing
has_bounds(sample) = sample.lower_bound !== nothing && sample.upper_bound !== nothing

function recommended_mode(sample)
    boundary = has_boundary(sample)
    balance = has_balance(sample)
    bounds = has_bounds(sample)
    if boundary && balance && bounds
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
        return :balance_only
    elseif bounds
        return :box_only_eval
    end
    return :none
end

function resolve_mode(sample, requested)
    requested === :auto || return requested
    mode = recommended_mode(sample)
    mode === :none &&
        throw(ArgumentError("sample $(sample.case_id) has no usable correction metadata"))
    return mode
end

function context_for_sample(sample; mode = :auto, cached = true)
    K = length(sample.raw)
    actual = resolve_mode(sample, mode)
    rows = Vector{Vector{Float64}}()
    values = Float64[]
    if actual in (:full_boundary_balance_box, :boundary_balance,
                  :boundary_box, :boundary_only)
        has_boundary(sample) ||
            throw(ArgumentError("mode $(actual) needs left and right boundary values"))
        push!(rows, endpoint_row(K, 1))
        push!(rows, endpoint_row(K, K))
        push!(values, sample.left_bc)
        push!(values, sample.right_bc)
    end
    if actual in (:full_boundary_balance_box, :boundary_balance,
                  :balance_box, :balance_only)
        has_balance(sample) ||
            throw(ArgumentError("mode $(actual) needs a balance value"))
        sample.weights !== nothing ||
            throw(ArgumentError("mode $(actual) needs weights or a grid"))
        push!(rows, collect(sample.weights))
        push!(values, sample.balance)
    end
    use_bounds = actual in (:full_boundary_balance_box, :boundary_box,
                            :balance_box, :box_only_eval)
    lower = use_bounds ? sample.lower_bound : nothing
    upper = use_bounds ? sample.upper_bound : nothing
    if use_bounds && !has_bounds(sample)
        throw(ArgumentError("mode $(actual) needs lower and upper bounds"))
    end
    ctx = operator_correction_context(K, equality_rows = rows,
                                      equality_values = values,
                                      lower = lower, upper = upper,
                                      data = sample,
                                      weights = sample.weights,
                                      cached = cached, mode = actual,
                                      metadata =
                                          (case_id = sample.case_id,))
    return ctx
end

safe_rmse(pred, target) =
    target === nothing ? NaN : field_rmse(pred, target)

function boundary_error(pred, sample)
    has_boundary(sample) || return NaN
    return max(abs(pred[1] - sample.left_bc),
               abs(pred[end] - sample.right_bc))
end

function balance_error(pred, sample)
    has_balance(sample) || return NaN
    sample.weights === nothing && return NaN
    return abs(mass(pred, sample.weights) - sample.balance)
end

function box_error(pred, sample)
    has_bounds(sample) || return NaN
    low = maximum(max.(sample.lower_bound .- pred, 0.0))
    high = maximum(max.(pred .- sample.upper_bound, 0.0))
    return max(low, high)
end

function evaluate_exported_batch(samples; mode = :auto, out = DEFAULT_OUT,
                                 cached = true)
    isdir(out) || mkpath(out)
    rows = NamedTuple[]
    for sample in samples
        sample.has_raw ||
            throw(ArgumentError("sample $(sample.case_id) needs raw_output for exported-batch evaluation"))
        ctx = context_for_sample(sample, mode = mode, cached = cached)
        res = project(ctx.constraint, sample.raw)
        corrected = res.zstar
        push!(rows,
              (case_id = sample.case_id,
               mode = ctx.mode,
               status = res.status,
               correction_norm = Float64(res.correction_norm),
               raw_rmse = safe_rmse(sample.raw, sample.target),
               corrected_rmse = safe_rmse(corrected, sample.target),
               raw_boundary = boundary_error(sample.raw, sample),
               corrected_boundary = boundary_error(corrected, sample),
               raw_balance = balance_error(sample.raw, sample),
               corrected_balance = balance_error(corrected, sample),
               raw_box = box_error(sample.raw, sample),
               corrected_box = box_error(corrected, sample)))
    end
    write_batch_outputs(rows, out)
    return rows
end

function fmt(x)
    x isa Symbol && return string(x)
    x isa AbstractString && return x
    x isa Bool && return string(x)
    x isa Integer && return string(x)
    x isa AbstractFloat && isnan(x) && return "nan"
    return @sprintf("%.10g", Float64(x))
end

function write_batch_outputs(rows, out)
    open(joinpath(out, "summer_batch_eval.csv"), "w") do io
        println(io, "case_id,mode,status,correction_norm,raw_rmse,corrected_rmse,raw_boundary,corrected_boundary,raw_balance,corrected_balance,raw_box,corrected_box")
        for row in rows
            println(io, join(fmt.((row.case_id, row.mode, row.status,
                                   row.correction_norm, row.raw_rmse,
                                   row.corrected_rmse, row.raw_boundary,
                                   row.corrected_boundary, row.raw_balance,
                                   row.corrected_balance, row.raw_box,
                                   row.corrected_box)), ","))
        end
    end
    statuses = Dict{Symbol,Int}()
    for row in rows
        statuses[row.status] = get(statuses, row.status, 0) + 1
    end
    open(joinpath(out, "summer_batch_eval.md"), "w") do io
        println(io, "# Summer Batch Evaluation")
        println(io)
        println(io, "Samples: ", length(rows))
        println(io, "Projection statuses: ", statuses)
        println(io)
        println(io, "| case | mode | status | raw RMSE | corrected RMSE | corrected boundary | corrected balance | corrected box |")
        println(io, "| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in rows
            println(io, "| ", row.case_id, " | ", row.mode, " | ",
                    row.status, " | ", fmt(row.raw_rmse), " | ",
                    fmt(row.corrected_rmse), " | ",
                    fmt(row.corrected_boundary), " | ",
                    fmt(row.corrected_balance), " | ",
                    fmt(row.corrected_box), " |")
        end
        println(io)
        println(io, "Treat the selected mode as an adapter recommendation. Final use still depends on physical review of units, grid ordering, and metadata.")
    end
end

function main(args = ARGS)
    isempty(args) &&
        throw(ArgumentError("usage: julia --project=benchmarks/deeponet benchmarks/deeponet/summer_batch_eval.jl path/to/batch.csv [mode]"))
    input = args[1]
    mode = length(args) >= 2 ? Symbol(args[2]) : :auto
    rows = evaluate_exported_batch(read_exported_batch(input), mode = mode)
    println("wrote summer_batch_eval.csv and summer_batch_eval.md to ",
            DEFAULT_OUT)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
