include("Pendulum.jl")
using .Pendulum
using Test, Random, Statistics, Zygote

@testset "M2 pendulum baseline harness" begin
    dt = 0.1
    nobs = 20

    # Data: energies in the band, reference integrator conserves energy, no wrap.
    rng = MersenneTwister(1)
    data = sample_band(5, dt, nobs; rng = rng)
    for d in data
        @test 0.3 <= d.H0 <= 1.2
        @test energy_violation(d.traj, d.H0).max < 1e-3   # reference conserves H
        @test maximum(abs, d.traj[1, :]) < pi             # angle never wraps
    end

    # Model rollout is finite and the right shape.
    p = init_mlp(seed = 2)
    pred = rollout(p, [0.1, 0.2], dt, nobs)
    @test size(pred) == (2, nobs + 1)
    @test all(isfinite, pred)

    # Zygote gradient flows through the rollout and is nonzero.
    g = Zygote.gradient(pp -> vanilla_loss(pp, data, dt), p)[1]
    @test all(isfinite, g.W1)
    @test sum(abs, g.W1) > 0

    # Training reduces the loss (Adam over the NamedTuple params).
    _, hist = train!(pp -> vanilla_loss(pp, data, dt), p; steps = 60, lr = 1e-2)
    @test hist[end] < hist[1]

    # The soft-penalty loss is also differentiable and trainable.
    gs = Zygote.gradient(pp -> soft_loss(pp, data, dt; beta = 2.0), p)[1]
    @test all(isfinite, gs.W3)
    _, hist_s = train!(pp -> soft_loss(pp, data, dt; beta = 2.0), p; steps = 60, lr = 1e-2)
    @test hist_s[end] < hist_s[1]

    # Metrics are finite.
    @test isfinite(traj_rmse(pred, data[1].traj))
    @test isfinite(energy_drift(p, data[1], dt, nobs * 5))

    # Failure counter (I10) tallies statuses.
    fc = FailureCounter()
    record!(fc, :success); record!(fc, :success); record!(fc, :nonunique_input)
    @test fc.counts[:success] == 2
    @test fc.counts[:nonunique_input] == 1
end

@testset "M3 projected (hard-constraint) rollout" begin
    dt = 0.1
    nobs = 20
    rng = MersenneTwister(7)
    data = sample_band(4, dt, nobs; rng = rng)
    p = init_mlp(seed = 3)
    d = data[1]

    # The projected rollout holds energy on the manifold to projection tolerance,
    # and every step projects cleanly (:success), recorded in the counter (I10).
    fc = FailureCounter()
    traj = projected_rollout_status!(p, d.z0, dt, nobs, d.H0, fc)
    @test size(traj) == (2, nobs + 1)
    @test energy_violation(traj, d.H0).max < 1e-6
    @test get(fc.counts, :success, 0) == nobs

    # The differentiable projected rollout backprops through the hard-constraint
    # layer: gradient flows and is finite and nonzero.
    g = Zygote.gradient(pp -> projected_loss(pp, data, dt), p)[1]
    @test all(isfinite, g.W1)
    @test sum(abs, g.W1) > 0

    # And it trains.
    _, hist = train!(pp -> projected_loss(pp, data, dt), p; steps = 40, lr = 1e-2)
    @test hist[end] < hist[1]
end

@testset "M3 projected gradient vs finite differences" begin
    # End-to-end correctness: the Zygote gradient of the projected loss (which
    # backprops through the hard-constraint rrule inside the RK4 rollout) must
    # match central finite differences. Spot-check a few parameter entries.
    dt = 0.1
    nobs = 10
    rng = MersenneTwister(11)
    data = sample_band(3, dt, nobs; rng = rng)
    p = init_mlp(seed = 5)
    L(pp) = projected_loss(pp, data, dt)
    g = Zygote.gradient(L, p)[1]

    perturb(q, f, ci, h) = (r = deepcopy(q); getfield(r, f)[ci] += h; r)
    h = 1e-6
    for (f, ci) in [(:W1, CartesianIndex(2, 1)), (:W3, CartesianIndex(1, 3)), (:b2, CartesianIndex(4))]
        fd = (L(perturb(p, f, ci, h)) - L(perturb(p, f, ci, -h))) / (2h)
        @test isapprox(getfield(g, f)[ci], fd; rtol = 1e-4, atol = 1e-8)
    end
end

@testset "M4 evaluation suite" begin
    dt = 0.1
    nobs = 10
    nlong = 40
    rng = MersenneTwister(21)
    train = sample_band(4, dt, nobs; rng = rng)
    test = sample_band(3, dt, nobs; rng = rng)
    p0 = init_mlp(seed = 9)

    pv, _ = train!(p -> vanilla_loss(p, train, dt), p0; steps = 30, lr = 1e-2)
    ev = evaluate_model((d, n) -> rollout(pv, d.z0, dt, n), test, dt, nobs, nlong)
    @test all(isfinite, (ev.rmse, ev.emax, ev.edrift, ev.rmse_long))

    # Projected model: long-horizon energy drift ~0, every projection :success.
    pp, _ = train!(p -> projected_loss(p, train, dt), p0; steps = 30, lr = 1e-2)
    fc = FailureCounter()
    evP = evaluate_model((d, n) -> projected_rollout_status!(pp, d.z0, dt, n, d.H0, fc),
                         test, dt, nobs, nlong)
    @test evP.edrift < 1e-6
    @test isfinite(evP.rmse_long)
    @test collect(keys(fc.counts)) == [:success]
end
