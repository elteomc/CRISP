include("DeepONetHeat.jl")
using .DeepONetHeat
using StructPINN: project
using Test, Random, Zygote

@testset "DeepONet helper heat benchmark" begin
    K = 24
    rng = MersenneTwister(21)
    data, x, w = sample_heat_operator(6, K, rng = rng)

    @test length(x) == K
    @test isapprox(sum(w), 1.0, atol = 1e-12)
    @test all(d -> isapprox(d.u[1], d.left, atol = 1e-12), data)
    @test all(d -> isapprox(d.u[end], d.right, atol = 1e-12), data)
    @test all(d -> isapprox(mass(d.u, w), d.mass0, atol = 1e-12), data)
    @test all(d -> all(0.0 .<= d.u .<= 2.0), data)

    p = init_deeponet(seed = 22)
    pred = deeponet_model(p, data[1].theta, x)
    @test length(pred) == K
    @test all(isfinite, pred)

    cbox = boundary_box_constraint(K, data[1].left, data[1].right)
    rbox = project(cbox, pred)
    @test rbox.status === :success
    @test boundary_violation(rbox.zstar, data[1]) < 1e-9
    @test lower_violation(rbox.zstar, 0.0) < 1e-12
    @test upper_violation(rbox.zstar, 2.0) < 1e-12

    cfull = full_heat_constraint(w, data[1].left, data[1].right,
                                 data[1].mass0)
    rfull = project(cfull, pred)
    @test rfull.status === :success
    @test boundary_violation(rfull.zstar, data[1]) < 1e-9
    @test mass_violation(rfull.zstar, data[1], w) < 1e-9
    @test lower_violation(rfull.zstar, 0.0) < 1e-12
    @test upper_violation(rfull.zstar, 2.0) < 1e-12

    log = ProjectionLog()
    hp = hard_projected_field(p, data[1], x, w, log = log)
    @test boundary_violation(hp, data[1]) < 1e-9
    @test mass_violation(hp, data[1], w) < 1e-9
    @test get(log.counts, :success, 0) == 1
    @test length(log.correction_norms) == 1

    boundary_only_log = ProjectionLog()
    hb = hard_projected_field(p, data[1], x, w, log = boundary_only_log,
                              mode = :boundary_box)
    @test boundary_violation(hb, data[1]) < 1e-9
    @test lower_violation(hb, 0.0) < 1e-12
    @test upper_violation(hb, 2.0) < 1e-12
    @test get(boundary_only_log.counts, :success, 0) == 1

    ctx = heat_correction_context(data[1], w, mode = :full,
                                  cached = true)
    ctx_log = ProjectionLog()
    hp_ctx = context_projected_field(p, ctx, x, log = ctx_log)
    @test isapprox(hp_ctx, hp, rtol = 1e-10, atol = 1e-10)
    @test get(ctx_log.counts, :success, 0) == 1

    rows = boundary_mass_rows(w)
    adapter_ctx = operator_correction_context(K,
                                              equality_rows = rows,
                                              equality_values =
                                                  [data[1].left,
                                                   data[1].right,
                                                   data[1].mass0],
                                              lower = 0.0, upper = 2.0,
                                              data = data[1], weights = w,
                                              mode = :summer_adapter)
    adapter_log = ProjectionLog()
    adapter_pred = corrected_output(pred, adapter_ctx, log = adapter_log)
    @test isapprox(adapter_pred, hp, rtol = 1e-10, atol = 1e-10)
    @test boundary_violation(adapter_pred, data[1]) < 1e-9
    @test mass_violation(adapter_pred, data[1], w) < 1e-9
    @test get(adapter_log.counts, :success, 0) == 1

    adapter_contexts =
        operator_correction_contexts(data, K,
                                     equality_rows = rows,
                                     equality_values =
                                         d -> [d.left, d.right, d.mass0],
                                     lower = 0.0, upper = 2.0,
                                     weights = w,
                                     mode = :summer_adapter)
    @test length(adapter_contexts) == length(data)
    @test all(ctx -> ctx.mode === :summer_adapter, adapter_contexts)

    g = Zygote.gradient(pp -> hard_loss(pp, data, x, w), p)[1]
    @test all(isfinite, g.Wb1)
    @test all(isfinite, g.Wt1)
    @test sum(abs, g.Wb1) > 0

    p0 = init_deeponet(seed = 23)
    pv, hv = train!(pp -> vanilla_loss(pp, data, x), p0,
                    steps = 25, lr = 8e-3)
    ps, hs = train!(pp -> soft_loss(pp, data, x, w), p0,
                    steps = 25, lr = 8e-3)
    train_log = ProjectionLog()
    ph, hh = train!(pp -> hard_loss(pp, data, x, w, log = train_log), p0,
                    steps = 25, lr = 8e-3)
    contexts = heat_correction_contexts(data, w, mode = :full,
                                        cached = true)
    context_log = ProjectionLog()
    phc, hhc = train!(pp -> context_hard_loss(pp, contexts, x,
                                              log = context_log),
                      p0, steps = 25, lr = 8e-3)
    combo_log = ProjectionLog()
    pc, hc = train!(pp -> soft_plus_hard_loss(pp, data, x, w,
                                              log = combo_log),
                    p0, steps = 25, lr = 8e-3)

    @test hv[end] < hv[1]
    @test hs[end] < hs[1]
    @test hh[end] < hh[1]
    @test hhc[end] < hhc[1]
    @test hc[end] < hc[1]
    @test get(train_log.counts, :success, 0) == length(data) * 25
    @test get(context_log.counts, :success, 0) == length(data) * 25
    @test get(combo_log.counts, :success, 0) == length(data) * 25

    ev = evaluate_model(d -> deeponet_model(pv, d.theta, x), data, w)
    es = evaluate_model(d -> deeponet_model(ps, d.theta, x), data, w)
    eh = evaluate_model(d -> hard_projected_field(ph, d, x, w), data, w)
    ehc = evaluate_context_model(ctx -> context_projected_field(phc, ctx, x),
                                 contexts)
    ec = evaluate_model(d -> hard_projected_field(pc, d, x, w), data, w)
    @test all(isfinite, (ev.rmse, es.rmse, eh.rmse, ehc.rmse, ec.rmse))
    @test eh.boundary_max < 1e-9
    @test eh.mass_max < 1e-9
    @test eh.lower_max < 1e-12
    @test eh.upper_max < 1e-12
    @test ehc.boundary_max < 1e-9
    @test ehc.mass_max < 1e-9
    @test ec.boundary_max < 1e-9
    @test ec.mass_max < 1e-9

    eval_only_log = ProjectionLog()
    eonly = evaluate_model(d -> hard_projected_field(pv, d, x, w,
                                                     log = eval_only_log),
                           data, w)
    @test eonly.boundary_max < 1e-9
    @test eonly.mass_max < 1e-9
    @test get(eval_only_log.counts, :success, 0) == length(data)
end

@testset "DeepONet helper stress checks" begin
    K = 16
    rng = MersenneTwister(26)
    data, x, w = sample_heat_operator(2, K, rng = rng)
    p = init_deeponet(seed = 27)
    pred = deeponet_model(p, data[1].theta, x)

    impossible = full_heat_constraint(w, data[1].left, data[1].right,
                                      10.0)
    res = project(impossible, pred)
    @test res.status === :infeasible_constraint

    bad_bounds = boundary_box_constraint(K, data[1].left, data[1].right,
                                         lower = 2.0, upper = 1.0)
    bad = project(bad_bounds, pred)
    @test bad.status === :infeasible_constraint

    log = ProjectionLog()
    corrected = hard_projected_field(p, data[1], x, w, log = log)
    @test isapprox(field_rmse(corrected, pred),
                   log.correction_norms[end] / sqrt(length(pred)),
                   rtol = 1e-12, atol = 1e-12)
    @test log.correction_norms[end] > 0
end

@testset "DeepONet hard loss finite differences" begin
    K = 14
    rng = MersenneTwister(24)
    data, x, w = sample_heat_operator(3, K, rng = rng)
    p = init_deeponet(width = 10, rank = 6, seed = 25)
    L(pp) = hard_loss(pp, data, x, w)
    contexts = heat_correction_contexts(data, w, cached = true)
    Lctx(pp) = context_hard_loss(pp, contexts, x)
    g = Zygote.gradient(L, p)[1]
    gctx = Zygote.gradient(Lctx, p)[1]

    function perturb(q, field, ci, h)
        r = deepcopy(q)
        getfield(r, field)[ci] += h
        return r
    end

    h = 1e-6
    checks = [(:Wb1, CartesianIndex(2, 1)),
              (:Wb2, CartesianIndex(3, 4)),
              (:Wt1, CartesianIndex(5, 1)),
              (:c, CartesianIndex(1))]
    for (field, ci) in checks
        fd = (L(perturb(p, field, ci, h)) -
              L(perturb(p, field, ci, -h))) / (2h)
        @test isapprox(getfield(g, field)[ci], fd, rtol = 2e-4,
                       atol = 1e-8)
        fdctx = (Lctx(perturb(p, field, ci, h)) -
                 Lctx(perturb(p, field, ci, -h))) / (2h)
        @test isapprox(getfield(gctx, field)[ci], fdctx,
                       rtol = 2e-4, atol = 1e-8)
    end
end
