@testset "M6 weighted simplex projection" begin
    w = [1.0, 1.0, 1.0]
    c = WeightedSimplexConstraint(w, 1.0)
    zhat = [0.8, 0.6, -0.2]
    res = project(c, zhat)

    @test res.status === :success
    @test all(res.zstar .>= -1e-12)
    @test isapprox(dot(w, res.zstar), 1.0; atol = 1e-10)
    @test isapprox(res.zstar, [0.6, 0.4, 0.0]; atol = 1e-8)

    gbar = [0.7, -0.3, 2.0]
    got = vjp(c, res, gbar)
    expected = [0.5, -0.5, 0.0]
    @test isapprox(got, expected; atol = 1e-8)

    fd = fd_grad(zh -> dot(gbar, project(c, zh).zstar), zhat)
    @test isapprox(got, fd; rtol = 1e-5, atol = 1e-7)

    loss(zh) = sum(abs2, correct(c, zh) .- [0.3, 0.7, 0.0])
    gz = Zygote.gradient(loss, zhat)[1]
    @test isapprox(gz, fd_grad(loss, zhat); rtol = 1e-5, atol = 1e-7)

    wc = WeightedSimplexConstraint([0.5, 1.0, 2.0], 1.1)
    zh2 = [0.9, 0.8, 0.4]
    r2 = project(wc, zh2)
    @test r2.status === :success
    @test isapprox(dot(wc.weights, r2.zstar), 1.1; atol = 1e-10)
    @test all(r2.zstar .>= -1e-12)
    fd2 = fd_grad(zh -> dot([0.2, -0.4, 0.9], project(wc, zh).zstar), zh2)
    @test isapprox(vjp(wc, r2, [0.2, -0.4, 0.9]), fd2; rtol = 1e-5, atol = 1e-7)

    kink = project(c, [0.8, 0.2, 0.0])
    @test kink.status === :nonunique_input
    @test_throws ErrorException vjp(c, kink, gbar)

    infeasible = project(WeightedSimplexConstraint(w, -0.1), zhat)
    @test infeasible.status === :infeasible_constraint

    zero_mass = project(WeightedSimplexConstraint(w, 0.0), zhat)
    @test zero_mass.status === :singular_constraint
end
