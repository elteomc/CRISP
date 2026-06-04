@testset "M6 bounded weighted simplex projection" begin
    c = BoundedWeightedSimplexConstraint(fill(1.0, 4), 1.6, zeros(4), ones(4))
    zhat = [2.0, 0.4, 0.5, -1.0]
    res = project(c, zhat)

    @test res.status === :success
    @test res.constraint_residual < 1e-12
    @test res.stationarity_residual < 1e-12
    @test isapprox(res.zstar, [1.0, 0.25, 0.35, 0.0], atol = 1e-10)

    gbar = [0.7, -1.3, 0.4, 1.2]
    got = vjp(c, res, gbar)
    fd = fd_grad(zh -> dot(gbar, project(c, zh).zstar), zhat)
    @test isapprox(got, fd, rtol = 1e-5, atol = 1e-8)

    loss(zh) = sum(abs2, correct(c, zh) .- [0.9, 0.2, 0.4, 0.1])
    gz = Zygote.gradient(loss, zhat)[1]
    @test isapprox(gz, fd_grad(loss, zhat), rtol = 1e-5, atol = 1e-8)

    weighted = BoundedWeightedSimplexConstraint([1.0, 2.0, 0.5], 1.1,
                                                [0.0, 0.0, -1.0],
                                                [2.0, 1.0, 1.0])
    rweighted = project(weighted, [0.3, 0.7, -0.4])
    @test rweighted.status === :success
    @test abs(dot(weighted.weights, rweighted.zstar) - weighted.mass) < 1e-10
    @test all(weighted.lower .<= rweighted.zstar .<= weighted.upper)
    gw = [-0.2, 0.5, 1.1]
    @test isapprox(vjp(weighted, rweighted, gw),
                   fd_grad(zh -> dot(gw, project(weighted, zh).zstar), [0.3, 0.7, -0.4]),
                   rtol = 1e-5, atol = 1e-8)

    vertex = BoundedWeightedSimplexConstraint(fill(1.0, 4), 2.0, zeros(4), ones(4))
    rvertex = project(vertex, [2.0, -2.0, 2.0, -2.0])
    @test rvertex.status === :success
    @test isapprox(rvertex.zstar, [1.0, 0.0, 1.0, 0.0], atol = 1e-12)
    @test isapprox(vjp(vertex, rvertex, gbar), zeros(4), atol = 1e-12)
    fd_vertex = fd_grad(zh -> dot(gbar, project(vertex, zh).zstar),
                        [2.0, -2.0, 2.0, -2.0])
    @test isapprox(vjp(vertex, rvertex, gbar), fd_vertex, rtol = 1e-5, atol = 1e-8)

    kink = project(BoundedWeightedSimplexConstraint([1.0, 1.0], 1.0,
                                                    [0.0, 0.0], [2.0, 2.0]),
                   [1.0, 0.0])
    @test kink.status === :nonunique_input
    @test_throws ErrorException vjp(BoundedWeightedSimplexConstraint([1.0, 1.0], 1.0,
                                                                     [0.0, 0.0],
                                                                     [2.0, 2.0]),
                                    kink, [1.0, 1.0])

    bad_mass = project(BoundedWeightedSimplexConstraint([1.0, 1.0], 3.0,
                                                        [0.0, 0.0], [1.0, 1.0]),
                       [0.5, 0.5])
    @test bad_mass.status === :infeasible_constraint

    bad_bounds = project(BoundedWeightedSimplexConstraint([1.0, 1.0], 1.0,
                                                          [0.0, 2.0], [1.0, 1.0]),
                         [0.5, 0.5])
    @test bad_bounds.status === :infeasible_constraint
end
