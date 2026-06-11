# Shared scenario settings for the pendulum result suite. Defaults are paper
# oriented and every knob has a STRUCTPINN_PENDULUM_* environment override, so
# result scripts stay auditable without editing code.
module PendulumScenarios

export pendulum_results_dir, study_scenario, frequency_ablation_scenario

const DEFAULT_SOFT_CONFIGS = [
    (name = "soft_weak", beta = 1.0),
    (name = "soft_default", beta = 5.0),
    (name = "soft_strong", beta = 20.0),
]

pendulum_results_dir() =
    get(ENV, "STRUCTPINN_PENDULUM_RESULTS_DIR",
        joinpath(@__DIR__, "results"))

function parse_int_values(value)
    text = strip(value)
    isempty(text) && throw(ArgumentError("empty integer list"))
    if occursin(":", text)
        parts = split(text, ":")
        length(parts) == 2 ||
            throw(ArgumentError("ranges must use start:stop"))
        lo = parse(Int, strip(parts[1]))
        hi = parse(Int, strip(parts[2]))
        lo <= hi || throw(ArgumentError("range start must not exceed stop"))
        return collect(lo:hi)
    end
    parts = occursin(",", text) ? split(text, ",") : split(text)
    return [parse(Int, strip(part)) for part in parts if !isempty(strip(part))]
end

function env_int(name, default)
    haskey(ENV, name) || return default
    return parse(Int, strip(ENV[name]))
end

function env_int_vector(name, default)
    haskey(ENV, name) || return collect(default)
    return parse_int_values(ENV[name])
end

function study_scenario()
    return (name = :study,
            out = pendulum_results_dir(),
            dt = 0.1,
            nobs = 20,
            nlong = 200,
            seeds = env_int_vector("STRUCTPINN_PENDULUM_STUDY_SEEDS", 1:10),
            steps = env_int("STRUCTPINN_PENDULUM_STUDY_STEPS", 300),
            train_samples =
                env_int("STRUCTPINN_PENDULUM_STUDY_TRAIN_SAMPLES", 12),
            test_samples =
                env_int("STRUCTPINN_PENDULUM_STUDY_TEST_SAMPLES", 8),
            soft_configs = DEFAULT_SOFT_CONFIGS)
end

function frequency_ablation_scenario()
    return (name = :frequency_ablation,
            out = pendulum_results_dir(),
            dt = 0.1,
            nobs = 20,
            nlong = 200,
            seeds =
                env_int_vector("STRUCTPINN_PENDULUM_FREQUENCY_SEEDS", 1:5),
            steps = env_int("STRUCTPINN_PENDULUM_FREQUENCY_STEPS", 300),
            train_samples =
                env_int("STRUCTPINN_PENDULUM_FREQUENCY_TRAIN_SAMPLES", 12),
            test_samples =
                env_int("STRUCTPINN_PENDULUM_FREQUENCY_TEST_SAMPLES", 8),
            periods = (every_step = 1, every_2_steps = 2,
                       every_5_steps = 5))
end

end # module
