@testset "M6 dense linear QP projection" begin
    Aeq = reshape([1.0, 1.0, 1.0], 1, 3)
    beq = [1.0]
    G = -Matrix(I, 3, 3)
    h = zeros(3)
    c = DenseLinearQPConstraint(Aeq, beq, G, h)
    zhat = [1.2, 0.3, -0.5]
    res = project(c, zhat)

    @test res.status === :success
    @test res.constraint_residual < 1e-12
    @test res.stationarity_residual < 1e-12
    @test isapprox(res.zstar, [0.95, 0.05, 0.0], atol = 1e-10)
    @test res.lambda[4] > 0.0

    gbar = [0.7, -1.3, 0.4]
    got = vjp(c, res, gbar)
    fd = fd_grad(zh -> dot(gbar, project(c, zh).zstar), zhat)
    @test isapprox(got, fd, rtol = 1e-5, atol = 1e-8)

    loss(zh) = sum(abs2, correct(c, zh) .- [0.7, 0.2, 0.1])
    gz = Zygote.gradient(loss, zhat)[1]
    @test isapprox(gz, fd_grad(loss, zhat), rtol = 1e-5, atol = 1e-8)

    halfspace = DenseLinearQPConstraint(reshape([1.0, 1.0], 1, 2), [1.0])
    rhalf = project(halfspace, [1.0, 1.0])
    @test rhalf.status === :success
    @test isapprox(rhalf.zstar, [0.5, 0.5], atol = 1e-12)
    gh = [2.0, -1.0]
    @test isapprox(vjp(halfspace, rhalf, gh),
                   fd_grad(zh -> dot(gh, project(halfspace, zh).zstar), [1.0, 1.0]),
                   rtol = 1e-5, atol = 1e-8)

    boxG = [Matrix(I, 2, 2); -Matrix(I, 2, 2)]
    boxh = [1.0, 1.0, 0.0, 0.0]
    boxqp = DenseLinearQPConstraint(boxG, boxh)
    rbox = project(boxqp, [2.0, -1.0])
    @test rbox.status === :success
    @test isapprox(rbox.zstar, [1.0, 0.0], atol = 1e-12)
    @test isapprox(vjp(boxqp, rbox, [0.3, -0.8]), zeros(2), atol = 1e-12)

    identity = DenseLinearQPConstraint(zeros(0, 2), zeros(0))
    rid = project(identity, [0.2, -0.7])
    @test rid.status === :success
    @test isapprox(rid.zstar, [0.2, -0.7], atol = 1e-12)
    @test isapprox(vjp(identity, rid, [1.0, 2.0]), [1.0, 2.0], atol = 1e-12)

    boundary = project(halfspace, [0.5, 0.5])
    @test boundary.status === :nonunique_input
    @test_throws ErrorException vjp(halfspace, boundary, gh)

    infeasible = DenseLinearQPConstraint(reshape([1.0, 1.0], 1, 2), [3.0],
                                         Matrix(I, 2, 2), [1.0, 1.0])
    rinfeasible = project(infeasible, [0.2, 0.3])
    @test rinfeasible.status === :infeasible_constraint

    rank_bad = DenseLinearQPConstraint([1.0 1.0; 2.0 2.0], [1.0, 2.0],
                                       zeros(0, 2), zeros(0))
    rbad = project(rank_bad, [0.2, 0.3])
    @test rbad.status === :singular_constraint
end
