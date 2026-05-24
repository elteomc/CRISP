@testset "M0.5 nonlinear KKT spike" begin
    # Regular case: circle, regular input -> success with an exact gradient.
    c = circle_constraint(1.0)
    zhat = [2.0, 0.0]
    res = project(c, zhat)
    @test res.status === :success
    @test res.constraint_residual < 1e-8           # I2
    @test res.stationarity_residual < 1e-7         # I3
    @test isapprox(res.zstar, [1.0, 0.0]; atol = 1e-7)

    # VJP matches the analytic projection Jacobian and central finite differences.
    # For the circle the closest point is zstar = r * zhat / norm(zhat), whose
    # Jacobian is (r/norm(zhat)) (I - zu zu') with zu = zhat / norm(zhat).
    gbar = [0.9, -0.4]
    w = vjp(c, res, gbar)
    r = 1.0
    nz = norm(zhat)
    zu = zhat / nz
    Jproj = (r / nz) * (I - zu * zu')
    @test isapprox(w, transpose(Jproj) * gbar; atol = 1e-6)   # I11
    fd = fd_grad(zh -> dot(gbar, project(c, zh).zstar), zhat)
    @test isapprox(w, fd; rtol = 1e-4)

    # Nonunique / nondifferentiable input: the centre of the circle (I13).
    rc = project(c, [0.0, 0.0])
    @test rc.status === :nonunique_input

    # Genuine LICQ failure at the solved point: tangential constraint z_1^2 (I4).
    ct = tangential_constraint(2)
    rt = project(ct, [1.0, 2.0])
    @test rt.status === :singular_constraint

    # No gradient is claimed for a non-success status (I10).
    @test_throws ErrorException vjp(c, rc, gbar)
end
