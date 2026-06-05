@testset "M6 sparse box-affine projection" begin
    using SparseArrays

    w = [1.0, 1.0, 1.0, 1.0]
    A = sparse(reshape(w, 1, 4))
    b = [1.6]
    lo = zeros(4)
    hi = ones(4)
    c = SparseBoxAffineConstraint(A, b, lo, hi)
    oracle = BoundedWeightedSimplexConstraint(w, 1.6, lo, hi)
    zhat = [1.8, 0.6, 0.4, -0.7]
    res = project(c, zhat, tol_feas = 1e-11, tol_step = 1e-12)
    ores = project(oracle, zhat)

    @test res.status === :success
    @test res.iterations > 1
    @test res.constraint_residual < 1e-9
    @test res.stationarity_residual < 1e-8
    @test isapprox(res.zstar, ores.zstar, atol = 1e-8)
    @test all(lo .- 1e-10 .<= res.zstar .<= hi .+ 1e-10)
    @test isapprox(dot(w, res.zstar), b[1], atol = 1e-9)

    cached_c = CachedSparseBoxAffineConstraint(c)
    cached_res = project(cached_c, zhat, tol_feas = 1e-11,
                         tol_step = 1e-12)
    @test cached_res.status === :success
    @test isapprox(cached_res.zstar, res.zstar, atol = 1e-10)

    ws = SparseBoxAffineWorkspace(c)
    first_ws = project(cached_c, zhat, tol_feas = 1e-11,
                       tol_step = 1e-12, workspace = ws)
    @test first_ws.status === :success
    @test ws.initialized
    warm_zhat = zhat .+ [0.02, -0.01, 0.01, -0.02]
    warm = project(cached_c, warm_zhat, tol_feas = 1e-11,
                   tol_step = 1e-12, workspace = ws)
    direct_warm = project(cached_c, warm_zhat, tol_feas = 1e-11,
                          tol_step = 1e-12)
    @test warm.status === :success
    @test warm.iterations <= direct_warm.iterations
    @test isapprox(warm.zstar, direct_warm.zstar, atol = 1e-9)

    warm_c = WarmStartedSparseBoxAffineConstraint(c)
    warm_c_res = project(warm_c, zhat, tol_feas = 1e-11,
                         tol_step = 1e-12)
    warm_c_res2 = project(warm_c, warm_zhat, tol_feas = 1e-11,
                          tol_step = 1e-12)
    @test warm_c_res.status === :success
    @test warm_c_res2.status === :success
    @test isapprox(warm_c_res2.zstar, direct_warm.zstar, atol = 1e-9)

    gbar = [0.7, -1.3, 0.4, 0.2]
    got = vjp(c, res, gbar)
    fd = fd_grad(zh -> dot(gbar, project(c, zh, tol_feas = 1e-11,
                                          tol_step = 1e-12).zstar), zhat)
    @test isapprox(got, fd, rtol = 1e-4, atol = 1e-7)

    loss(zh) = sum(abs2, correct(c, zh) .- [0.4, 0.4, 0.4, 0.4])
    gz = Zygote.gradient(loss, zhat)[1]
    @test isapprox(gz, fd_grad(loss, zhat), rtol = 1e-4, atol = 1e-7)

    cached_loss(zh) = sum(abs2, correct(cached_c, zh) .- [0.4, 0.4, 0.4, 0.4])
    cached_gz = Zygote.gradient(cached_loss, zhat)[1]
    @test isapprox(cached_gz, fd_grad(cached_loss, zhat), rtol = 1e-4, atol = 1e-7)

    interior = project(c, [0.2, 0.3, 0.4, 0.5], tol_feas = 1e-11,
                       tol_step = 1e-12)
    affine = project(SparseAffineConstraint(A, b), [0.2, 0.3, 0.4, 0.5])
    @test interior.status === :success
    @test isapprox(interior.zstar, affine.zstar, atol = 1e-9)

    large_n = 96
    large_w = fill(1.0 / large_n, large_n)
    large_c = SparseBoxAffineConstraint(sparse(reshape(large_w, 1, large_n)),
                                        [1.0], zeros(large_n), fill(2.0, large_n))
    large_zhat = fill(0.8, large_n)
    large_zhat[1:10] .= 2.7
    large_res = project(large_c, large_zhat, tol_feas = 1e-10,
                        tol_step = 1e-11, diagnostic_limit = 16)
    @test large_res.status === :success
    @test abs(dot(large_w, large_res.zstar) - 1.0) < 1e-9
    @test all(large_res.zstar .>= -1e-10)
    @test all(large_res.zstar .<= 2.0 + 1e-10)

    bad_bounds = project(SparseBoxAffineConstraint(A, b, ones(4), zeros(4)), zhat)
    @test bad_bounds.status === :infeasible_constraint

    bad_mass = project(SparseBoxAffineConstraint(A, [5.0], lo, hi), zhat)
    @test bad_mass.status === :infeasible_constraint

    box_kink = SparseBoxAffineConstraint(spzeros(0, 2), zeros(0),
                                         zeros(2), ones(2))
    kink = project(box_kink, [0.0, 0.5])
    @test kink.status === :nonunique_input
    @test_throws ErrorException vjp(box_kink, kink, [0.1, 0.2])
end
