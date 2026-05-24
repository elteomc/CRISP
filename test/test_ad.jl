@testset "AD integration (Zygote)" begin
    # Affine layer: Zygote gradient through `correct` matches central finite
    # differences (the rrule is wired correctly).
    A = [1.0 1.0]
    b = [1.0]
    ca = AffineConstraint(A, b)
    loss_a(zh) = sum(abs2, correct(ca, zh))
    za = [2.0, 0.3]
    ga = Zygote.gradient(loss_a, za)[1]
    @test isapprox(ga, fd_grad(loss_a, za); rtol = 1e-5, atol = 1e-8)

    # Nonlinear circle layer through Zygote.
    cc = circle_constraint(1.0)
    target = [0.5, 0.5]
    loss_c(zh) = sum(abs2, correct(cc, zh) .- target)
    zc = [2.0, 0.5]
    gc = Zygote.gradient(loss_c, zc)[1]
    @test isapprox(gc, fd_grad(loss_c, zc); rtol = 1e-4, atol = 1e-8)

    # The layer composes inside a tiny model: gradient flows through an upstream
    # linear map and into the projection.
    W = [0.5 -0.2; 0.1 0.3]
    model_loss(zh) = sum(abs2, correct(cc, W * zh) .- target)
    zm = [1.5, 0.7]
    gm = Zygote.gradient(model_loss, zm)[1]
    @test isapprox(gm, fd_grad(model_loss, zm); rtol = 1e-4, atol = 1e-8)

    # No gradient is claimed at a nonunique input (I10): the pullback errors.
    @test_throws Exception Zygote.gradient(zh -> sum(correct(cc, zh)), [0.0, 0.0])
end
