# Shared scenario settings for the field result suite. Defaults are paper
# oriented and every knob has a STRUCTPINN_FIELD_* environment override, so
# result scripts stay auditable without editing code.
module FieldScenarios

export field_results_dir, study_scenario, pde_study_scenario,
       stress_study_scenario

const DEFAULT_SOFT_CONFIGS = [
    (name = "soft_weak", beta = 5.0),
    (name = "soft_default", beta = 20.0),
    (name = "soft_strong", beta = 50.0),
]

field_results_dir() =
    get(ENV, "STRUCTPINN_FIELD_RESULTS_DIR", joinpath(@__DIR__, "results"))

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
            out = field_results_dir(),
            grid = env_int("STRUCTPINN_FIELD_STUDY_GRID", 32),
            seeds = env_int_vector("STRUCTPINN_FIELD_STUDY_SEEDS", 1:10),
            steps = env_int("STRUCTPINN_FIELD_STUDY_STEPS", 250),
            train_samples =
                env_int("STRUCTPINN_FIELD_STUDY_TRAIN_SAMPLES", 32),
            test_samples =
                env_int("STRUCTPINN_FIELD_STUDY_TEST_SAMPLES", 16),
            soft_configs = DEFAULT_SOFT_CONFIGS)
end

function stress_study_scenario()
    return (name = :stress_study,
            out = field_results_dir(),
            grid = env_int("STRUCTPINN_FIELD_STRESS_GRID", 32),
            seeds = env_int_vector("STRUCTPINN_FIELD_STRESS_SEEDS", 1:5),
            steps = env_int("STRUCTPINN_FIELD_STRESS_STEPS", 250),
            train_samples =
                env_int("STRUCTPINN_FIELD_STRESS_TRAIN_SAMPLES", 32),
            test_samples =
                env_int("STRUCTPINN_FIELD_STRESS_TEST_SAMPLES", 16),
            soft_beta = 20.0)
end

function pde_study_scenario()
    return (name = :pde_study,
            out = field_results_dir(),
            seeds = env_int_vector("STRUCTPINN_FIELD_PDE_SEEDS", 1:5),
            steps = env_int("STRUCTPINN_FIELD_PDE_STEPS", 100),
            train_samples =
                env_int("STRUCTPINN_FIELD_PDE_TRAIN_SAMPLES", 16),
            test_samples =
                env_int("STRUCTPINN_FIELD_PDE_TEST_SAMPLES", 8))
end

end # module
