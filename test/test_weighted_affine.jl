@testset "M6 weighted affine projection" begin
    A = [1.0 1.0]
    b = [1.0]
    weights = [2.0, 1.0]
    c = DiagonalWeightedAffineConstraint(A, b, weights)
    zhat = [2.0, 0.0]
    res = project(c, zhat)

    @test res.status === :success
    @test res.constraint_residual < 1e-12
    @test res.stationarity_residual < 1e-12
    @test isapprox(res.zstar, [5 / 3, -2 / 3]; atol = 1e-10)

    gbar = [0.7, -1.3]
    got = vjp(c, res, gbar)
    expected = gbar .- transpose(A) * (((A .* transpose(1.0 ./ weights)) * transpose(A)) \ (A * ((1.0 ./ weights) .* gbar)))
    @test isapprox(got, expected; atol = 1e-10)
    fd = fd_grad(zh -> dot(gbar, project(c, zh).zstar), zhat)
    @test isapprox(got, fd; rtol = 1e-5, atol = 1e-8)

    loss(zh) = sum(abs2, correct(c, zh) .- [0.4, 0.6])
    gz = Zygote.gradient(loss, zhat)[1]
    @test isapprox(gz, fd_grad(loss, zhat); rtol = 1e-5, atol = 1e-8)

    Abad = [1.0 1.0; 2.0 2.0]
    rbad = project(DiagonalWeightedAffineConstraint(Abad, [1.0, 2.0], weights), zhat)
    @test rbad.status === :singular_constraint

    Aover = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    rover = project(DiagonalWeightedAffineConstraint(Aover, [1.0, 2.0, 3.0], weights), zhat)
    @test rover.status === :singular_constraint
end
