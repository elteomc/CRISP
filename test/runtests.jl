using StructPINN
using Test
using LinearAlgebra
using Zygote
using ForwardDiff

# Central-difference gradient of a scalar function phi: R^n -> R, used for the
# finite-difference VJP checks (PLAN.md I11).
function fd_grad(phi, x; h = 1e-6)
    g = zeros(float(eltype(x)), length(x))
    for i in eachindex(x)
        xp = collect(float.(x)); xp[i] += h
        xm = collect(float.(x)); xm[i] -= h
        g[i] = (phi(xp) - phi(xm)) / (2h)
    end
    return g
end

# Independent oracle: a minimal fixed-iteration Newton on the KKT system,
# differentiated by ForwardDiff (unrolled forward-mode). This is a different
# mechanism from the package's implicit-diff adjoint, so agreement validates the
# VJP (PLAN.md I11, second independent path for the nonlinear layer).
function oracle_project(con, zhat; iters = 50)
    n = length(zhat)
    m = length(con.c(zhat))
    T = eltype(zhat)
    s = vcat(zhat, zeros(T, m))
    for _ in 1:iters
        z = s[1:n]
        l = s[n+1:end]
        F = vcat(z .- zhat .+ transpose(con.J(z)) * l, con.c(z))
        M = [Matrix(I, n, n) .+ con.Hlam(z, l)  transpose(con.J(z));
             con.J(z)                            zeros(T, m, m)]
        s = s .- (M \ F)
    end
    return s[1:n]
end

@testset "StructPINN" begin
    include("test_affine.jl")
    include("test_nonlinear.jl")
    include("test_ad.jl")
    include("test_m3.jl")
end
