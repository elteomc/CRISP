"""
    Pendulum

M2 benchmark harness (PLAN.md Section 8.1): a Hamiltonian neural ODE on the
pendulum, trained on a band of libration energy levels. Provides the physics and
reference data, a small neural-ODE vector field with a fixed-step RK4 rollout,
the vanilla and soft-penalty losses, an Adam optimiser, and the Section 8 metrics
(energy violation, trajectory RMSE, long-horizon energy drift) with a per-status
failure counter (I10). The projected baseline (M3/M4) plugs into the same harness.
"""
module Pendulum

using Random, Statistics, Zygote, StructPINN

export H, sample_band, init_mlp, field, rollout,
       vanilla_loss, soft_loss, train!,
       energy_constraint, projected_rollout, projected_loss, projected_rollout_status!,
       energy_violation, traj_rmse, energy_drift,
       FailureCounter, record!

# ---------------- physics ----------------
# state z = (q, p); H(q,p) = 0.5 p^2 + (1 - cos q)
H(z) = 0.5 * z[2]^2 + (1 - cos(z[1]))
f_true(z) = [z[2], -sin(z[1])]                    # dq/dt = p, dp/dt = -sin q

function rk4_step(f, z, dt)
    k1 = f(z)
    k2 = f(z .+ (dt / 2) .* k1)
    k3 = f(z .+ (dt / 2) .* k2)
    k4 = f(z .+ dt .* k3)
    return z .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

# Reference trajectory at observation times 0, dt, ..., nobs*dt, using `substeps`
# fine RK4 steps between observations for accuracy. Not differentiated.
function reference_traj(z0, dt, nobs; substeps = 10)
    h = dt / substeps
    Z = zeros(2, nobs + 1)
    Z[:, 1] = z0
    z = copy(z0)
    for k in 1:nobs
        for _ in 1:substeps
            z = rk4_step(f_true, z, h)
        end
        Z[:, k + 1] = z
    end
    return Z
end

# ---------------- data: a band of libration energy levels ----------------
# Each trajectory carries its own energy H0_i (PLAN.md 8.1 per-trajectory).
function sample_band(N, dt, nobs; emin = 0.3, emax = 1.2, rng = Random.default_rng())
    data = NamedTuple[]
    for _ in 1:N
        H0 = emin + (emax - emin) * rand(rng)
        qmax = acos(1 - H0)                          # libration amplitude (H0 < 2)
        q0 = (2 * rand(rng) - 1) * 0.9 * qmax        # stay inside the turning points
        p2 = 2 * (H0 - (1 - cos(q0)))
        p0 = sqrt(max(p2, 0.0)) * (rand(rng) < 0.5 ? -1 : 1)
        z0 = [q0, p0]
        push!(data, (z0 = z0, traj = reference_traj(z0, dt, nobs), H0 = H0))
    end
    return data
end

# ---------------- model: small MLP vector field + fixed-step rollout ----------
function init_mlp(; nin = 2, nh = 16, nout = 2, seed = 0, scale = 0.1)
    rng = MersenneTwister(seed)
    r(d...) = scale .* randn(rng, d...)
    return (W1 = r(nh, nin), b1 = zeros(nh),
            W2 = r(nh, nh),  b2 = zeros(nh),
            W3 = r(nout, nh), b3 = zeros(nout))
end

field(p, z) = p.W3 * tanh.(p.W2 * tanh.(p.W1 * z .+ p.b1) .+ p.b2) .+ p.b3

# Fixed-step RK4 rollout of the learned field. Differentiable via a Zygote.Buffer.
function rollout(p, z0, dt, nsteps)
    buf = Zygote.Buffer(z0, length(z0), nsteps + 1)
    buf[:, 1] = z0
    z = z0
    for k in 1:nsteps
        z = rk4_step(zz -> field(p, zz), z, dt)
        buf[:, k + 1] = z
    end
    return copy(buf)
end

# ---------------- losses (shared data + shared init give fairness, I12) -------
function vanilla_loss(p, data, dt)
    s = 0.0
    for d in data
        nsteps = size(d.traj, 2) - 1
        pred = rollout(p, d.z0, dt, nsteps)
        s += sum(abs2, pred .- d.traj)
    end
    return s / length(data)
end

function soft_loss(p, data, dt; beta = 1.0)
    s = 0.0
    for d in data
        nsteps = size(d.traj, 2) - 1
        pred = rollout(p, d.z0, dt, nsteps)
        s += sum(abs2, pred .- d.traj)
        for k in 1:(nsteps + 1)              # penalty toward each trajectory's H0
            s += beta * (H(pred[:, k]) - d.H0)^2
        end
    end
    return s / length(data)
end

# ---------------- per-status failure counter (I10) ----------------
# Baselines record none; the projected model records each ProjectionResult status.
struct FailureCounter
    counts::Dict{Symbol,Int}
end
FailureCounter() = FailureCounter(Dict{Symbol,Int}())
record!(fc::FailureCounter, s::Symbol) = (fc.counts[s] = get(fc.counts, s, 0) + 1; fc)

# ---------------- hard-constraint (projected) model, M3 ----------------
# The energy manifold H(z) = H0 as a StructPINN nonlinear constraint, with its
# analytic Jacobian and Lagrangian Hessian.
function energy_constraint(H0)
    NonlinearConstraint(
        z -> [0.5 * z[2]^2 + (1 - cos(z[1])) - H0],
        z -> reshape([sin(z[1]), z[2]], 1, 2),
        (z, l) -> l[1] .* [cos(z[1]) 0.0; 0.0 1.0],
    )
end

# Projected rollout: after each RK4 step, project the state back onto H(z)=H0 via
# the differentiable hard-constraint layer (correct, with its rrule). This is the
# "projected neural ODE" baseline (PLAN.md 8.1 config 3). Differentiable for training.
function projected_rollout(p, z0, dt, nsteps, H0)
    ec = energy_constraint(H0)
    buf = Zygote.Buffer(z0, length(z0), nsteps + 1)
    buf[:, 1] = z0
    z = z0
    for k in 1:nsteps
        zr = rk4_step(zz -> field(p, zz), z, dt)
        z = correct(ec, zr)
        buf[:, k + 1] = z
    end
    return copy(buf)
end

function projected_loss(p, data, dt)
    s = 0.0
    for d in data
        nsteps = size(d.traj, 2) - 1
        pred = projected_rollout(p, d.z0, dt, nsteps, d.H0)
        s += sum(abs2, pred .- d.traj)
    end
    return s / length(data)
end

# Non-differentiable projected rollout that records each projection status into a
# FailureCounter (I10). Used for evaluation, not training.
function projected_rollout_status!(p, z0, dt, nsteps, H0, fc::FailureCounter)
    ec = energy_constraint(H0)
    z = copy(z0)
    states = [copy(z)]
    for _ in 1:nsteps
        zr = rk4_step(zz -> field(p, zz), z, dt)
        res = project(ec, zr)
        record!(fc, res.status)
        z = res.zstar
        push!(states, copy(z))
    end
    return reduce(hcat, states)
end

# ---------------- Adam over the NamedTuple of parameters ----------------
_ntmap(f, nts...) = NamedTuple{keys(nts[1])}(map(f, map(values, nts)...))

function train!(lossfn, p; steps = 300, lr = 1e-2, beta1 = 0.9, beta2 = 0.999, eps = 1e-8)
    m = _ntmap(zero, p)
    v = _ntmap(zero, p)
    history = Float64[]
    for t in 1:steps
        L, back = Zygote.pullback(lossfn, p)
        g = back(1.0)[1]
        push!(history, L)
        m = _ntmap((mk, gk) -> beta1 .* mk .+ (1 - beta1) .* gk, m, g)
        v = _ntmap((vk, gk) -> beta2 .* vk .+ (1 - beta2) .* (gk .^ 2), v, g)
        mh = _ntmap(mk -> mk ./ (1 - beta1^t), m)
        vh = _ntmap(vk -> vk ./ (1 - beta2^t), v)
        p = _ntmap((pk, a, b) -> pk .- lr .* a ./ (sqrt.(b) .+ eps), p, mh, vh)
    end
    return p, history
end

# ---------------- metrics (PLAN.md Section 8.1) ----------------
function energy_violation(traj, H0)
    e = [abs(H(traj[:, k]) - H0) for k in 1:size(traj, 2)]
    return (max = maximum(e), mean = mean(e))
end

traj_rmse(pred, truth) = sqrt(mean(abs2, pred .- truth))

# Long-horizon energy drift: roll the model out to nlong steps, report |H - H0|.
function energy_drift(p, d, dt, nlong)
    pred = rollout(p, d.z0, dt, nlong)
    return abs(H(pred[:, end]) - d.H0)
end

end # module
