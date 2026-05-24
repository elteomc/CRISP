@testset "M3 nonlinear KKT layer" begin
    # Independent adjoint check (I11 second path): the implicit-diff VJP must
    # match an unrolled-Newton oracle differentiated by ForwardDiff.
    c = circle_constraint(1.0)
    zhat = [2.0, 0.3]
    res = project(c, zhat)
    @test res.status === :success
    Jor = ForwardDiff.jacobian(zh -> oracle_project(c, zh), zhat)
    for gbar in ([1.0, 0.0], [0.0, 1.0], [0.6, -0.9])
        @test isapprox(vjp(c, res, gbar), transpose(Jor) * gbar; rtol = 1e-6, atol = 1e-9)
    end

    # The same agreement should hold at another regular input.
    zhat2 = [-1.5, 1.2]
    res2 = project(c, zhat2)
    @test res2.status === :success
    Jor2 = ForwardDiff.jacobian(zh -> oracle_project(c, zh), zhat2)
    @test isapprox(vjp(c, res2, [0.3, 0.8]), transpose(Jor2) * [0.3, 0.8]; rtol = 1e-6)

    # I5: an ill-conditioned KKT matrix at a regular solution is flagged
    # :ill_conditioned, distinct from :success and :singular_constraint.
    # Projecting from well inside the circle makes cond(M) moderate; a tight
    # cond_max trips I5 while J stays full rank at the solution.
    ric = project(circle_constraint(1.0), [0.5, 0.0]; cond_max = 3.0)
    @test ric.status === :ill_conditioned
    @test ric.jac_min_singular > 1e-6
end
