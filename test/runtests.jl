using StructPINN
using Test
using LinearAlgebra

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

@testset "StructPINN" begin
    include("test_affine.jl")
    include("test_nonlinear.jl")
end
