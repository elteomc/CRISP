include("FieldMass.jl")
using .FieldMass
using StructPINN: project
using Test, Random, Zygote

@testset "M5 fixed-grid field correction" begin
    K = 32
    rng = MersenneTwister(1)
    data, x, w = sample_fields(8, K; rng = rng)

    @test length(x) == K
    @test isapprox(sum(w), 1.0; atol = 1e-12)
    for d in data
        @test isapprox(mass(d.u, w), d.mass0; atol = 1e-12)
        @test all(isfinite, d.u)
    end

    p = init_mlp(nout = K, seed = 2)
    pred = field_model(p, data[1].theta)
    @test length(pred) == K
    @test all(isfinite, pred)

    c = mass_constraint(w, data[1].mass0)
    res = project(c, pred)
    @test res.status === :success
    @test abs(mass(res.zstar, w) - data[1].mass0) < 1e-10

    gp = Zygote.gradient(pp -> projected_loss(pp, data, w), p)[1]
    @test all(isfinite, gp.W1)
    @test sum(abs, gp.W1) > 0

    train_fc = FailureCounter()
    _, hist = train!(pp -> projected_loss(pp, data, w; fc = train_fc),
                     p; steps = 50, lr = 1e-2)
    @test hist[end] < hist[1]
    @test get(train_fc.counts, :success, 0) == length(data) * 50
    @test collect(keys(train_fc.counts)) == [:success]

    weighted_fc = FailureCounter()
    _, hist_weighted = train!(pp -> weighted_projected_loss(pp, data, w; fc = weighted_fc),
                              p; steps = 50, lr = 1e-2)
    @test hist_weighted[end] < hist_weighted[1]
    @test get(weighted_fc.counts, :success, 0) == length(data) * 50
    weighted_pred = weighted_projected_field(p, data[1].theta, w, data[1].mass0)
    @test abs(mass(weighted_pred, w) - data[1].mass0) < 1e-10

    box_fc = FailureCounter()
    _, hist_box = train!(pp -> box_projected_loss(pp, data, 0.0, 2.0; fc = box_fc),
                         p; steps = 50, lr = 1e-2)
    @test hist_box[end] < hist_box[1]
    @test get(box_fc.counts, :success, 0) == length(data) * 50
    box_pred = box_projected_field(p, data[1].theta, 0.0, 2.0)
    @test all(0.0 .<= box_pred .<= 2.0)

    pos_fc = FailureCounter()
    _, hist_pos = train!(pp -> positive_projected_loss(pp, data, w; fc = pos_fc),
                         p; steps = 50, lr = 1e-2)
    @test hist_pos[end] < hist_pos[1]
    @test get(pos_fc.counts, :success, 0) == length(data) * 50
    pos_pred = positive_projected_field(p, data[1].theta, w, data[1].mass0)
    @test all(pos_pred .>= -1e-12)
    @test abs(mass(pos_pred, w) - data[1].mass0) < 1e-10

    bounded_fc = FailureCounter()
    _, hist_bounded = train!(pp -> bounded_projected_loss(pp, data, w, 0.0, 2.0, fc = bounded_fc),
                             p, steps = 50, lr = 1e-2)
    @test hist_bounded[end] < hist_bounded[1]
    @test get(bounded_fc.counts, :success, 0) == length(data) * 50
    bounded_pred = bounded_projected_field(p, data[1].theta, w, data[1].mass0, 0.0, 2.0)
    @test all(0.0 .<= bounded_pred .<= 2.0)
    @test abs(mass(bounded_pred, w) - data[1].mass0) < 1e-10

    sparse_bounded_fc = FailureCounter()
    _, hist_sparse_bounded =
        train!(pp -> sparse_bounded_projected_loss(pp, data, w, 0.0, 2.0,
                                                   fc = sparse_bounded_fc),
               p, steps = 50, lr = 1e-2)
    @test hist_sparse_bounded[end] < hist_sparse_bounded[1]
    @test get(sparse_bounded_fc.counts, :success, 0) == length(data) * 50
    sparse_bounded_pred =
        sparse_bounded_projected_field(p, data[1].theta, w, data[1].mass0, 0.0, 2.0)
    @test all(0.0 .<= sparse_bounded_pred .<= 2.0)
    @test abs(mass(sparse_bounded_pred, w) - data[1].mass0) < 1e-9
end

@testset "M5 comparison metrics" begin
    K = 32
    rng = MersenneTwister(3)
    train, _, w = sample_fields(12, K; rng = rng)
    test, _, _ = sample_fields(6, K; rng = rng)
    p0 = init_mlp(nout = K, seed = 4)

    pv, _ = train!(p -> vanilla_loss(p, train), p0; steps = 60, lr = 1e-2)
    ps, _ = train!(p -> soft_loss(p, train, w; beta = 20.0), p0; steps = 60, lr = 1e-2)
    fc = FailureCounter()
    pp, _ = train!(p -> projected_loss(p, train, w; fc = fc), p0; steps = 60, lr = 1e-2)
    fcw = FailureCounter()
    pw, _ = train!(p -> weighted_projected_loss(p, train, w; fc = fcw), p0; steps = 60, lr = 1e-2)
    fcb = FailureCounter()
    pb, _ = train!(p -> box_projected_loss(p, train, 0.0, 2.0; fc = fcb), p0; steps = 60, lr = 1e-2)
    fcp = FailureCounter()
    ppos, _ = train!(p -> positive_projected_loss(p, train, w; fc = fcp), p0; steps = 60, lr = 1e-2)
    fcbd = FailureCounter()
    pbd, _ = train!(p -> bounded_projected_loss(p, train, w, 0.0, 2.0, fc = fcbd),
                    p0, steps = 60, lr = 1e-2)
    fcsbd = FailureCounter()
    psbd, _ = train!(p -> sparse_bounded_projected_loss(p, train, w, 0.0, 2.0,
                                                        fc = fcsbd),
                     p0, steps = 60, lr = 1e-2)

    ev = evaluate_model(d -> field_model(pv, d.theta), test, w)
    es = evaluate_model(d -> field_model(ps, d.theta), test, w)
    ep = evaluate_model(d -> projected_field(pp, d.theta, w, d.mass0), test, w)
    ew = evaluate_model(d -> weighted_projected_field(pw, d.theta, w, d.mass0), test, w)
    eb = evaluate_model(d -> box_projected_field(pb, d.theta, 0.0, 2.0), test, w)
    epos = evaluate_model(d -> positive_projected_field(ppos, d.theta, w, d.mass0), test, w)
    ebd = evaluate_model(d -> bounded_projected_field(pbd, d.theta, w, d.mass0, 0.0, 2.0), test, w)
    esbd = evaluate_model(d -> sparse_bounded_projected_field(psbd, d.theta, w, d.mass0, 0.0, 2.0), test, w)

    @test all(isfinite, (ev.rmse, ev.mass_max, ev.mass_mean, ev.negative_max))
    @test all(isfinite, (es.rmse, es.mass_max, es.mass_mean, es.negative_max))
    @test all(isfinite, (ep.rmse, ep.mass_max, ep.mass_mean, ep.negative_max))
    @test all(isfinite, (ew.rmse, ew.mass_max, ew.mass_mean, ew.negative_max))
    @test all(isfinite, (eb.rmse, eb.mass_max, eb.mass_mean, eb.negative_max, eb.upper_max))
    @test all(isfinite, (epos.rmse, epos.mass_max, epos.mass_mean, epos.negative_max))
    @test all(isfinite, (ebd.rmse, ebd.mass_max, ebd.mass_mean, ebd.negative_max, ebd.upper_max))
    @test all(isfinite, (esbd.rmse, esbd.mass_max, esbd.mass_mean, esbd.negative_max, esbd.upper_max))
    @test ep.mass_max < 1e-10
    @test ew.mass_max < 1e-10
    @test epos.mass_max < 1e-10
    @test ebd.mass_max < 1e-10
    @test esbd.mass_max < 1e-9
    @test eb.negative_max < 1e-12
    @test eb.upper_max < 1e-12
    @test epos.negative_max < 1e-12
    @test ebd.negative_max < 1e-12
    @test ebd.upper_max < 1e-12
    @test esbd.negative_max < 1e-12
    @test esbd.upper_max < 1e-12
    @test get(fc.counts, :success, 0) == length(train) * 60
    @test get(fcw.counts, :success, 0) == length(train) * 60
    @test get(fcb.counts, :success, 0) == length(train) * 60
    @test get(fcp.counts, :success, 0) == length(train) * 60
    @test get(fcbd.counts, :success, 0) == length(train) * 60
    @test get(fcsbd.counts, :success, 0) == length(train) * 60
end

@testset "M5 projected gradient vs finite differences" begin
    K = 16
    rng = MersenneTwister(5)
    data, _, w = sample_fields(4, K; rng = rng)
    p = init_mlp(nout = K, seed = 6)
    L(pp) = projected_loss(pp, data, w)
    g = Zygote.gradient(L, p)[1]

    function perturb(q, f, ci, h)
        r = deepcopy(q)
        getfield(r, f)[ci] += h
        return r
    end

    h = 1e-6
    checks = [(:W1, CartesianIndex(2, 1)),
              (:W3, CartesianIndex(3, 4)),
              (:b2, CartesianIndex(5))]
    for (f, ci) in checks
        fd = (L(perturb(p, f, ci, h)) - L(perturb(p, f, ci, -h))) / (2h)
        @test isapprox(getfield(g, f)[ci], fd; rtol = 1e-4, atol = 1e-8)
    end

    Lw(pp) = weighted_projected_loss(pp, data, w)
    gw = Zygote.gradient(Lw, p)[1]
    for (f, ci) in checks
        fd = (Lw(perturb(p, f, ci, h)) - Lw(perturb(p, f, ci, -h))) / (2h)
        @test isapprox(getfield(gw, f)[ci], fd; rtol = 1e-4, atol = 1e-8)
    end

    Lbox(pp) = box_projected_loss(pp, data, 0.0, 2.0)
    gbox = Zygote.gradient(Lbox, p)[1]
    for (f, ci) in checks
        fd = (Lbox(perturb(p, f, ci, h)) - Lbox(perturb(p, f, ci, -h))) / (2h)
        @test isapprox(getfield(gbox, f)[ci], fd; rtol = 1e-4, atol = 1e-8)
    end

    Lpos(pp) = positive_projected_loss(pp, data, w)
    gpos = Zygote.gradient(Lpos, p)[1]
    for (f, ci) in checks
        fd = (Lpos(perturb(p, f, ci, h)) - Lpos(perturb(p, f, ci, -h))) / (2h)
        @test isapprox(getfield(gpos, f)[ci], fd; rtol = 1e-4, atol = 1e-8)
    end

    Lbounded(pp) = bounded_projected_loss(pp, data, w, 0.0, 2.0)
    gbounded = Zygote.gradient(Lbounded, p)[1]
    for (f, ci) in checks
        fd = (Lbounded(perturb(p, f, ci, h)) - Lbounded(perturb(p, f, ci, -h))) / (2h)
        @test isapprox(getfield(gbounded, f)[ci], fd, rtol = 1e-4, atol = 1e-8)
    end

    Lsparse(pp) = sparse_bounded_projected_loss(pp, data, w, 0.0, 2.0)
    gsparse = Zygote.gradient(Lsparse, p)[1]
    for (f, ci) in checks
        fd = (Lsparse(perturb(p, f, ci, h)) - Lsparse(perturb(p, f, ci, -h))) / (2h)
        @test isapprox(getfield(gsparse, f)[ci], fd, rtol = 1e-4, atol = 1e-8)
    end
end

@testset "M6 heat-style sparse inequality integration" begin
    K = 96
    rng = MersenneTwister(7)
    train, x, w = sample_heat_fields(10, K; rng = rng)
    test, _, _ = sample_heat_fields(5, K; rng = rng)
    @test length(x) == K
    @test all(d -> all(0.0 .<= d.u .<= 2.0), train)
    @test all(d -> isapprox(mass(d.u, w), d.mass0, atol = 1e-12), train)

    p0 = init_mlp(nout = K, seed = 8)
    fc = FailureCounter()
    p, hist = train!(pp -> sparse_bounded_projected_loss(pp, train, w, 0.0, 2.0,
                                                         fc = fc),
                     p0, steps = 35, lr = 8e-3)
    @test hist[end] < hist[1]
    @test get(fc.counts, :success, 0) == length(train) * 35

    ev = evaluate_model(d -> sparse_bounded_projected_field(p, d.theta, w, d.mass0,
                                                            0.0, 2.0),
                        test, w)
    @test all(isfinite, (ev.rmse, ev.mass_max, ev.negative_max, ev.upper_max))
    @test ev.mass_max < 1e-9
    @test ev.negative_max < 1e-12
    @test ev.upper_max < 1e-12
end
