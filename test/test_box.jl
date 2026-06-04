@testset "M6 box projection" begin
    c = BoxConstraint([-1.0, -0.5, 0.0], [1.0, 0.5, 2.0])
    zhat = [0.2, 0.1, 1.4]
    res = project(c, zhat)

    @test res.status === :success
    @test isapprox(res.zstar, zhat; atol = 1e-12)
    @test res.constraint_residual < 1e-12

    gbar = [0.3, -0.7, 1.2]
    got = vjp(c, res, gbar)
    @test isapprox(got, gbar; atol = 1e-12)
    fd = fd_grad(zh -> dot(gbar, project(c, zh).zstar), zhat)
    @test isapprox(got, fd; rtol = 1e-5, atol = 1e-8)

    loss(zh) = sum(abs2, correct(c, zh) .- [0.1, 0.2, 0.3])
    gz = Zygote.gradient(loss, zhat)[1]
    @test isapprox(gz, fd_grad(loss, zhat); rtol = 1e-5, atol = 1e-8)

    clamped = project(c, [2.0, -2.0, 3.0])
    @test clamped.status === :success
    @test isapprox(clamped.zstar, [1.0, -0.5, 2.0]; atol = 1e-12)
    @test isapprox(vjp(c, clamped, gbar), zeros(3); atol = 1e-12)
    fd_clamped = fd_grad(zh -> dot(gbar, project(c, zh).zstar), [2.0, -2.0, 3.0])
    @test isapprox(vjp(c, clamped, gbar), fd_clamped; rtol = 1e-5, atol = 1e-8)

    lower_kink = project(c, [-1.0, 0.0, 1.0])
    @test lower_kink.status === :nonunique_input
    @test_throws ErrorException vjp(c, lower_kink, gbar)

    bad = project(BoxConstraint([1.0, 0.0], [0.0, 1.0]), [0.5, 0.5])
    @test bad.status === :infeasible_constraint
end
