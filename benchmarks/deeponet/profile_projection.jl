# Projection runtime profile for the DeepONet helper benchmark.
# Run: julia --project=benchmarks/deeponet benchmarks/deeponet/profile_projection.jl
include("DeepONetHeat.jl")
include("DeepONetScenarios.jl")
using .DeepONetHeat
using .DeepONetScenarios
using Random, Printf
using StructPINN

const SCENARIO = profile_projection_scenario()
const OUT = SCENARIO.out
const GRIDS = SCENARIO.grids
const NSAMPLES = SCENARIO.samples
const REPEATS = SCENARIO.repeats
const TRIALS = SCENARIO.trials

isdir(OUT) || mkdir(OUT)

function seconds_per_call(elapsed, calls)
    return elapsed / calls
end

function best_time(f)
    best = Inf
    for _ in 1:TRIALS
        GC.gc()
        elapsed = @elapsed f()
        best = min(best, elapsed)
    end
    return best
end

open(joinpath(OUT, "projection_profile.csv"), "w") do io
    println(io, "grid,samples,repeats,raw_seconds,uncached_project_seconds,cached_project_seconds,context_project_seconds,construct_context_seconds,cached_speedup,context_over_raw,construct_over_context")
    for K in GRIDS
        rng = MersenneTwister(20_000 + K)
        data, x, w = sample_heat_operator(NSAMPLES, K, rng = rng)
        p = init_deeponet(seed = 30_000 + K)
        calls = NSAMPLES * REPEATS

        warm = data[1]
        warm_pred = deeponet_model(p, warm.theta, x)
        warm_constraint = full_heat_constraint(w, warm.left, warm.right,
                                               warm.mass0)
        project(warm_constraint, warm_pred)
        warm_context = heat_correction_context(warm, w, mode = :full,
                                               cached = true)
        context_projected_field(p, warm_context, x)
        heat_correction_contexts(data[1:2], w, mode = :full, cached = true)

        raw_time = best_time() do
            for _ in 1:REPEATS, d in data
                deeponet_model(p, d.theta, x)
            end
        end

        uncached_time = best_time() do
            for _ in 1:REPEATS, d in data
                pred = deeponet_model(p, d.theta, x)
                c = full_heat_constraint(w, d.left, d.right, d.mass0)
                res = project(c, pred)
                StructPINN.issuccess(res) || error("uncached profile failed")
            end
        end

        construct_time = best_time() do
            contexts = heat_correction_contexts(data, w, mode = :full,
                                                cached = true)
            length(contexts) == NSAMPLES || error("context count mismatch")
        end

        contexts = heat_correction_contexts(data, w, mode = :full,
                                            cached = true)
        cached_constraints = [ctx.constraint for ctx in contexts]

        cached_time = best_time() do
            for _ in 1:REPEATS, (d, c) in zip(data, cached_constraints)
                pred = deeponet_model(p, d.theta, x)
                res = project(c, pred)
                StructPINN.issuccess(res) || error("cached profile failed")
            end
        end

        context_time = best_time() do
            for _ in 1:REPEATS, ctx in contexts
                context_projected_field(p, ctx, x)
            end
        end

        raw_seconds = seconds_per_call(raw_time, calls)
        uncached_seconds = seconds_per_call(uncached_time, calls)
        cached_seconds = seconds_per_call(cached_time, calls)
        context_seconds = seconds_per_call(context_time, calls)
        construct_seconds = construct_time / NSAMPLES
        cached_speedup = uncached_seconds / cached_seconds
        context_over_raw = context_seconds / raw_seconds
        construct_over_context = construct_seconds / context_seconds

        println(io, @sprintf("%d,%d,%d,%.8e,%.8e,%.8e,%.8e,%.8e,%.8f,%.8f,%.8f",
                             K, NSAMPLES, REPEATS,
                             raw_seconds, uncached_seconds,
                             cached_seconds, context_seconds,
                             construct_seconds, cached_speedup,
                             context_over_raw, construct_over_context))
    end
end

println("wrote projection_profile.csv to ", OUT)
