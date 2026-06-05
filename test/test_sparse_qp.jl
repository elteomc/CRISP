@testset "M6 sparse linear QP projection" begin
    Aeq = sparse(reshape([1.0, 1.0, 1.0], 1, 3))
    beq = [1.0]
    G = sparse(-Matrix(I, 3, 3))
    h = zeros(3)
    zhat = [1.2, 0.3, -0.5]
    target = [0.95, 0.05, 0.0]

    active = SparseLinearQPActiveSetConstraint(Aeq, beq, G, h)
    primal = SparseLinearQPPrimalDualConstraint(Aeq, beq, G, h)
    dense = DenseLinearQPConstraint(Matrix(Aeq), beq, Matrix(G), h)

    for c in (active, primal)
        res = project(c, zhat)
        @test res.status === :success
        @test res.constraint_residual < 1e-8
        @test res.stationarity_residual < 1e-8
        @test isapprox(res.zstar, target, atol = 1e-7)
        @test isapprox(res.zstar, project(dense, zhat).zstar, atol = 1e-7)

        gbar = [0.7, -1.3, 0.4]
        got = vjp(c, res, gbar)
        fd = fd_grad(zh -> dot(gbar, project(c, zh).zstar), zhat)
        @test isapprox(got, fd, rtol = 1e-5, atol = 1e-7)

        loss(zh) = sum(abs2, correct(c, zh) .- [0.7, 0.2, 0.1])
        gz = Zygote.gradient(loss, zhat)[1]
        @test isapprox(gz, fd_grad(loss, zhat), rtol = 1e-5, atol = 1e-7)
    end

    triangle_G = sparse(reshape([1.0, -1.0, 0.0, 1.0, 0.0, -1.0], 3, 2))
    triangle_h = [1.0, 0.0, 0.0]
    triangle_z = [1.0, 1.0]
    triangle_expected = [0.5, 0.5]
    for c in (SparseLinearQPActiveSetConstraint(triangle_G, triangle_h),
              SparseLinearQPPrimalDualConstraint(triangle_G, triangle_h))
        res = project(c, triangle_z)
        @test res.status === :success
        @test isapprox(res.zstar, triangle_expected, atol = 1e-7)
        gbar = [2.0, -1.0]
        @test isapprox(vjp(c, res, gbar),
                       fd_grad(zh -> dot(gbar, project(c, zh).zstar),
                               triangle_z),
                       rtol = 1e-5, atol = 1e-7)
    end

    halfspace = SparseLinearQPActiveSetConstraint(sparse(reshape([1.0, 1.0], 1, 2)),
                                                  [1.0])
    boundary = project(halfspace, [0.5, 0.5])
    @test boundary.status === :nonunique_input
    @test_throws ErrorException vjp(halfspace, boundary, [1.0, 2.0])

    infeasible = SparseLinearQPActiveSetConstraint(spzeros(0, 1), zeros(0),
                                                   sparse(reshape([1.0, -1.0], 2, 1)),
                                                   [0.0, -1.0])
    rinfeasible = project(infeasible, [0.2])
    @test rinfeasible.status === :infeasible_constraint

    rank_bad = SparseLinearQPActiveSetConstraint(sparse(reshape([1.0, 2.0, 1.0, 2.0], 2, 2)),
                                                 [1.0, 2.0],
                                                 spzeros(0, 2), zeros(0))
    rbad = project(rank_bad, [0.2, 0.3])
    @test rbad.status === :singular_constraint

    n = 80
    weights = fill(1.0 / n, n)
    large_A = sparse(reshape(weights, 1, n))
    large_G = sparse(-Matrix(I, n, n))
    large_h = zeros(n)
    large_zhat = collect(range(-0.8, 1.2, length = n))
    mass = [0.35]
    for c in (SparseLinearQPActiveSetConstraint(large_A, mass, large_G, large_h),
              SparseLinearQPPrimalDualConstraint(large_A, mass, large_G, large_h))
        res = project(c, large_zhat, diagnostic_limit = 32, maxiter = 200)
        @test res.status === :success
        @test res.constraint_residual < 1e-7
        @test minimum(res.zstar) >= -1e-8
        @test isapprox(dot(weights, res.zstar), mass[1], atol = 1e-8)
        gbar = sin.(range(0.0, 2pi, length = n))
        got = vjp(c, res, gbar, diagnostic_limit = 32)
        @test length(got) == n
        @test all(isfinite, got)
    end
end
