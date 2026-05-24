@testset "M1 affine projection" begin
    A = [1.0 1.0]
    b = [1.0]
    c = AffineConstraint(A, b)
    zhat = [2.0, 0.0]

    res = project(c, zhat)
    @test res.status === :success
    @test res.constraint_residual < 1e-12
    @test isapprox(res.zstar, [1.5, -0.5]; atol = 1e-10)
    @test norm(A * res.zstar - b) < 1e-12          # I2

    # VJP matches the analytic projector P and central finite differences (I11).
    # inv here is a reference cross-check in the test only, not in the layer (I6).
    P = I - transpose(A) * inv(A * transpose(A)) * A
    gbar = [0.7, -1.3]
    w = vjp(c, gbar)
    @test isapprox(w, P * gbar; atol = 1e-8)
    fd = fd_grad(zh -> dot(gbar, project(c, zh).zstar), zhat)
    @test isapprox(w, fd; rtol = 1e-5)

    # Rank-deficient A is flagged, not silently wrong (I4).
    Abad = [1.0 1.0; 2.0 2.0]
    bbad = [1.0, 2.0]
    rbad = project(AffineConstraint(Abad, bbad), [3.0, 1.0])
    @test rbad.status === :singular_constraint
end
