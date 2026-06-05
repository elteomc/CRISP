# Profile sparse bounded projection variants without extra dependencies.
# Run: julia --project=benchmarks/field benchmarks/field/profile_sparse_projection.jl
using StructPINN
using SparseArrays, Statistics, Printf

const K = 256
const REPEATS = 48

w = fill(1.0 / K, K)
A = sparse(reshape(w, 1, K))
b = [1.0]
lo = zeros(K)
hi = fill(2.0, K)
c = SparseBoxAffineConstraint(A, b, lo, hi)
cached = CachedSparseBoxAffineConstraint(c, diagnostic_limit = 32)
warm = WarmStartedSparseBoxAffineConstraint(c, diagnostic_limit = 32)

function raw_field(j)
    z = fill(0.8, K)
    z[1:24] .= 2.8 .+ 0.01 * sin(j)
    z[25:48] .= -0.6 .+ 0.01 * cos(j)
    z .+= 0.05 .* sin.(range(0.0, 6pi, length = K) .+ 0.03 * j)
    return z
end

inputs = [raw_field(j) for j in 1:REPEATS]

function run_variant(name, con)
    statuses = Symbol[]
    iterations = Int[]
    elapsed = @elapsed begin
        for z in inputs
            res = project(con, z, tol_feas = 1e-10,
                          tol_step = 1e-11,
                          diagnostic_limit = 32)
            push!(statuses, res.status)
            push!(iterations, res.iterations)
        end
    end
    ok = count(==(:success), statuses)
    @printf("%-12s  time %.4f s   mean iters %.2f   max iters %d   success %d/%d\n",
            name, elapsed, mean(iterations), maximum(iterations), ok, length(statuses))
end

run_variant("uncached", c)
run_variant("cached", cached)
run_variant("warm", warm)
