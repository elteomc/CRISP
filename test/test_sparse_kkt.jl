@testset "M6 sparse KKT affine projection" begin
    using SparseArrays

    A = sparse([1.0 1.0 0.0; 0.0 1.0 1.0])
    b = [1.0, 0.5]
    c = SparseAffineConstraint(A, b)
    dense_c = AffineConstraint(Matrix(A), b)
    zhat = [2.0, 0.0, -1.0]
    res = project(c, zhat)
    dense_res = project(dense_c, zhat)

    @test res.status === :success
    @test res.constraint_residual < 1e-12
    @test res.stationarity_residual < 1e-12
    @test isapprox(res.zstar, dense_res.zstar; atol = 1e-10)

    cache = SparseAffineProjectionCache(c)
    cached_res = project(cache, zhat)
    @test cached_res.status === :success
    @test isapprox(cached_res.zstar, res.zstar; atol = 1e-12)
    cached_res2 = project(cache, [-1.0, 0.5, 2.0])
    direct_res2 = project(c, [-1.0, 0.5, 2.0])
    @test cached_res2.status === :success
    @test isapprox(cached_res2.zstar, direct_res2.zstar; atol = 1e-12)

    gbar = [0.7, -1.3, 0.4]
    got = vjp(c, res, gbar)
    @test isapprox(got, vjp(dense_c, dense_res, gbar); atol = 1e-10)
    fd = fd_grad(zh -> dot(gbar, project(c, zh).zstar), zhat)
    @test isapprox(got, fd; rtol = 1e-5, atol = 1e-8)

    loss(zh) = sum(abs2, correct(c, zh) .- [0.2, 0.4, 0.6])
    gz = Zygote.gradient(loss, zhat)[1]
    @test isapprox(gz, fd_grad(loss, zhat); rtol = 1e-5, atol = 1e-8)

    weights = [2.0, 1.0, 3.0]
    wc = SparseDiagonalWeightedAffineConstraint(A, b, weights)
    dense_wc = DiagonalWeightedAffineConstraint(Matrix(A), b, weights)
    wres = project(wc, zhat)
    dense_wres = project(dense_wc, zhat)

    @test wres.status === :success
    @test wres.constraint_residual < 1e-12
    @test wres.stationarity_residual < 1e-12
    @test isapprox(wres.zstar, dense_wres.zstar; atol = 1e-10)

    wgot = vjp(wc, wres, gbar)
    @test isapprox(wgot, vjp(dense_wc, dense_wres, gbar); atol = 1e-10)
    wfd = fd_grad(zh -> dot(gbar, project(wc, zh).zstar), zhat)
    @test isapprox(wgot, wfd; rtol = 1e-5, atol = 1e-8)

    wloss(zh) = sum(abs2, correct(wc, zh) .- [0.2, 0.4, 0.6])
    wgz = Zygote.gradient(wloss, zhat)[1]
    @test isapprox(wgz, fd_grad(wloss, zhat); rtol = 1e-5, atol = 1e-8)

    n = 40
    identity = spdiagm(0 => ones(n))
    target = collect(1.0:n)
    large = project(SparseAffineConstraint(identity, target), zeros(n);
                    diagnostic_limit = 16)
    @test large.status === :success
    @test isapprox(large.zstar, target; atol = 1e-12)
    @test isnan(large.jac_min_singular)
    @test isnan(large.kkt_cond_estimate)

    rank_bad = project(SparseAffineConstraint(sparse([1.0 1.0; 2.0 2.0]),
                                              [1.0, 2.0]), [0.2, 0.3])
    @test rank_bad.status === :singular_constraint
    rank_bad_cache = SparseAffineProjectionCache(sparse([1.0 1.0; 2.0 2.0]),
                                                 [1.0, 2.0])
    @test project(rank_bad_cache, [0.2, 0.3]).status === :singular_constraint
    rank_bad_lu = project(SparseAffineConstraint(sparse([1.0 1.0; 2.0 2.0]),
                                                 [1.0, 2.0]), [0.2, 0.3];
                          diagnostic_limit = 1)
    @test rank_bad_lu.status === :singular_constraint
    @test_throws ErrorException vjp(SparseAffineConstraint(sparse([1.0 1.0; 2.0 2.0]),
                                                           [1.0, 2.0]),
                                   rank_bad, [0.1, 0.2])

    over = project(SparseAffineConstraint(sparse([1.0 0.0; 0.0 1.0; 1.0 1.0]),
                                          [1.0, 2.0, 3.0]), [0.0, 0.0])
    @test over.status === :singular_constraint

    bad_weights = SparseDiagonalWeightedAffineConstraint(sparse([1.0 1.0]),
                                                         [1.0], [1.0, 0.0])
    @test_throws ArgumentError project(bad_weights, [0.0, 0.0])
end
