module DeepONetScenarios

export deeponet_results_dir, study_scenario, large_study_scenario,
       profile_projection_scenario, larger_profile_scenario,
       expansion_gate_scenario, frequency_ablation_scenario,
       constraint_ablation_scenario, eval_only_scenario,
       scenario_rows, final_run_plan_rows, scenario_field

const DEFAULT_SOFT_CONFIGS = [
    (name = "soft_weak", beta_boundary = 1.0, beta_mass = 1.0,
     beta_box = 0.2),
    (name = "soft_default", beta_boundary = 10.0, beta_mass = 10.0,
     beta_box = 2.0),
    (name = "soft_strong", beta_boundary = 50.0, beta_mass = 50.0,
     beta_box = 10.0),
    (name = "soft_boundary_heavy", beta_boundary = 80.0,
     beta_mass = 10.0, beta_box = 2.0),
]

const METRICS = (:rmse, :boundary_max, :boundary_mean, :mass_max,
                 :mass_mean, :lower_max, :lower_mean, :upper_max,
                 :upper_mean)

deeponet_results_dir() =
    get(ENV, "STRUCTPINN_DEEPONET_RESULTS_DIR",
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

function env_int_tuple(name, default)
    haskey(ENV, name) || return default
    return Tuple(parse_int_values(ENV[name]))
end

function env_int_vector(name, default)
    haskey(ENV, name) || return collect(default)
    return parse_int_values(ENV[name])
end

scenario_field(row, key) = haskey(row, key) ? row[key] : ""

format_ints(values) = join(string.(collect(values)), " ")

function study_scenario()
    return (name = :study,
            out = deeponet_results_dir(),
            grid = env_int("STRUCTPINN_DEEPONET_STUDY_GRID", 32),
            seeds = env_int_vector("STRUCTPINN_DEEPONET_STUDY_SEEDS",
                                   1:10),
            steps = env_int("STRUCTPINN_DEEPONET_STUDY_STEPS", 90),
            train_samples =
                env_int("STRUCTPINN_DEEPONET_STUDY_TRAIN_SAMPLES", 18),
            test_samples =
                env_int("STRUCTPINN_DEEPONET_STUDY_TEST_SAMPLES", 8),
            soft_configs = DEFAULT_SOFT_CONFIGS)
end

function large_study_scenario()
    return (name = :large_study,
            out = deeponet_results_dir(),
            grids =
                env_int_tuple("STRUCTPINN_DEEPONET_LARGE_GRIDS", (64, 96)),
            seeds =
                env_int_vector("STRUCTPINN_DEEPONET_LARGE_SEEDS", 1:5),
            steps = env_int("STRUCTPINN_DEEPONET_LARGE_STEPS", 70),
            train_samples =
                env_int("STRUCTPINN_DEEPONET_LARGE_TRAIN_SAMPLES", 18),
            test_samples =
                env_int("STRUCTPINN_DEEPONET_LARGE_TEST_SAMPLES", 8))
end

function profile_projection_scenario()
    return (name = :profile_projection,
            out = deeponet_results_dir(),
            grids =
                env_int_tuple("STRUCTPINN_DEEPONET_PROFILE_GRIDS",
                              (32, 64, 96, 128, 192, 256)),
            samples =
                env_int("STRUCTPINN_DEEPONET_PROFILE_SAMPLES", 20),
            repeats =
                env_int("STRUCTPINN_DEEPONET_PROFILE_REPEATS", 8),
            trials =
                env_int("STRUCTPINN_DEEPONET_PROFILE_TRIALS", 4))
end

function larger_profile_scenario()
    return (name = :larger_profile,
            out = deeponet_results_dir(),
            grids =
                env_int_tuple("STRUCTPINN_DEEPONET_LARGER_PROFILE_GRIDS",
                              (384, 512)),
            samples =
                env_int("STRUCTPINN_DEEPONET_LARGER_PROFILE_SAMPLES", 8),
            repeats =
                env_int("STRUCTPINN_DEEPONET_LARGER_PROFILE_REPEATS", 4),
            trials =
                env_int("STRUCTPINN_DEEPONET_LARGER_PROFILE_TRIALS", 3))
end

function env_bool(name, default)
    haskey(ENV, name) || return default
    value = lowercase(strip(ENV[name]))
    return value in ("1", "true", "yes", "y")
end

function expansion_gate_scenario()
    return (name = :expanded_seed_gate,
            out = deeponet_results_dir(),
            allow_synthetic =
                env_bool("STRUCTPINN_DEEPONET_ALLOW_SYNTHETIC_EXPANSION",
                         true),
            require_summer =
                env_bool("STRUCTPINN_DEEPONET_EXPANSION_REQUIRES_SUMMER",
                         false),
            main_seed_target =
                env_int("STRUCTPINN_DEEPONET_EXPANDED_MAIN_SEEDS", 20),
            large_seed_target =
                env_int("STRUCTPINN_DEEPONET_EXPANDED_LARGE_SEEDS", 10))
end

function frequency_ablation_scenario()
    return (name = :frequency_ablation,
            out = deeponet_results_dir(),
            grid = env_int("STRUCTPINN_DEEPONET_FREQUENCY_GRID", 32),
            seeds =
                env_int_vector("STRUCTPINN_DEEPONET_FREQUENCY_SEEDS", 1:5),
            steps = env_int("STRUCTPINN_DEEPONET_FREQUENCY_STEPS", 80),
            train_samples =
                env_int("STRUCTPINN_DEEPONET_FREQUENCY_TRAIN_SAMPLES", 18),
            test_samples =
                env_int("STRUCTPINN_DEEPONET_FREQUENCY_TEST_SAMPLES", 8),
            periods = Dict("every_step" => 1, "every_2_steps" => 2,
                           "every_5_steps" => 5,
                           "every_10_steps" => 10))
end

function constraint_ablation_scenario()
    return (name = :constraint_ablation,
            out = deeponet_results_dir(),
            grid = env_int("STRUCTPINN_DEEPONET_CONSTRAINT_GRID", 32),
            seeds =
                env_int_vector("STRUCTPINN_DEEPONET_CONSTRAINT_SEEDS", 1:5),
            steps = env_int("STRUCTPINN_DEEPONET_CONSTRAINT_STEPS", 80),
            train_samples =
                env_int("STRUCTPINN_DEEPONET_CONSTRAINT_TRAIN_SAMPLES", 18),
            test_samples =
                env_int("STRUCTPINN_DEEPONET_CONSTRAINT_TEST_SAMPLES", 8),
            train_modes = (:boundary_only, :mass_only, :boundary_box,
                           :full),
            eval_modes = (:boundary_only, :box_only, :mass_only,
                          :boundary_box, :full))
end

function eval_only_scenario()
    return (name = :eval_only,
            out = deeponet_results_dir(),
            grid = env_int("STRUCTPINN_DEEPONET_EVAL_ONLY_GRID", 32),
            steps = env_int("STRUCTPINN_DEEPONET_EVAL_ONLY_STEPS", 90),
            train_samples =
                env_int("STRUCTPINN_DEEPONET_EVAL_ONLY_TRAIN_SAMPLES", 18),
            test_samples =
                env_int("STRUCTPINN_DEEPONET_EVAL_ONLY_TEST_SAMPLES", 8),
            seed = env_int("STRUCTPINN_DEEPONET_EVAL_ONLY_SEED", 31))
end

function scenario_rows()
    study = study_scenario()
    large = large_study_scenario()
    profile = profile_projection_scenario()
    larger_profile = larger_profile_scenario()
    expansion = expansion_gate_scenario()
    frequency = frequency_ablation_scenario()
    constraint = constraint_ablation_scenario()
    eval_only = eval_only_scenario()
    return [
        (script = "study.jl", scenario = string(study.name),
         grids = string(study.grid), seeds = format_ints(study.seeds),
         steps = study.steps, train_samples = study.train_samples,
         test_samples = study.test_samples),
        (script = "large_study.jl", scenario = string(large.name),
         grids = format_ints(large.grids), seeds = format_ints(large.seeds),
         steps = large.steps, train_samples = large.train_samples,
         test_samples = large.test_samples),
        (script = "profile_projection.jl", scenario = string(profile.name),
         grids = format_ints(profile.grids), seeds = "",
         steps = "", train_samples = profile.samples,
         test_samples = ""),
        (script = "profile_larger_outputs.jl",
         scenario = string(larger_profile.name),
         grids = format_ints(larger_profile.grids), seeds = "",
         steps = "", train_samples = larger_profile.samples,
         test_samples = ""),
        (script = "expanded_seed_gate.jl",
         scenario = string(expansion.name),
         grids = "32 64 96", seeds =
             "$(expansion.main_seed_target) main $(expansion.large_seed_target) large",
         steps = "", train_samples = "", test_samples = ""),
        (script = "frequency_ablation.jl",
         scenario = string(frequency.name),
         grids = string(frequency.grid),
         seeds = format_ints(frequency.seeds), steps = frequency.steps,
         train_samples = frequency.train_samples,
         test_samples = frequency.test_samples),
        (script = "constraint_ablation.jl",
         scenario = string(constraint.name),
         grids = string(constraint.grid),
         seeds = format_ints(constraint.seeds), steps = constraint.steps,
         train_samples = constraint.train_samples,
         test_samples = constraint.test_samples),
        (script = "eval_only_example.jl",
         scenario = string(eval_only.name),
         grids = string(eval_only.grid),
         seeds = string(eval_only.seed), steps = eval_only.steps,
         train_samples = eval_only.train_samples,
         test_samples = eval_only.test_samples),
    ]
end

function final_run_plan_rows()
    study = study_scenario()
    large = large_study_scenario()
    profile = profile_projection_scenario()
    larger_profile = larger_profile_scenario()
    expansion = expansion_gate_scenario()
    return [
        (priority = 1, batch = "core_helper_10_seed",
         command = "julia --project=benchmarks/deeponet benchmarks/deeponet/study.jl",
         env = "", grids = string(study.grid),
         seeds = format_ints(study.seeds), steps = study.steps,
         train_samples = study.train_samples,
         test_samples = study.test_samples,
         gate = "run after scenario manifest and DeepONet tests pass",
         purpose = "paper-facing main DeepONet helper table"),
        (priority = 2, batch = "large_helper_5_seed",
         command = "julia --project=benchmarks/deeponet benchmarks/deeponet/large_study.jl",
         env = "", grids = format_ints(large.grids),
         seeds = format_ints(large.seeds), steps = large.steps,
         train_samples = large.train_samples,
         test_samples = large.test_samples,
         gate = "run after profiling says cached contexts remain acceptable",
         purpose = "larger-grid DeepONet helper scaling table"),
        (priority = 3, batch = "larger_output_probe",
         command = "julia --project=benchmarks/deeponet benchmarks/deeponet/large_study.jl",
         env = "STRUCTPINN_DEEPONET_LARGE_GRIDS=128 192 STRUCTPINN_DEEPONET_LARGE_SEEDS=1:3",
         grids = "128 192", seeds = "1 2 3", steps = large.steps,
         train_samples = large.train_samples,
         test_samples = large.test_samples,
         gate = "run only if K=256 profiling still leaves sparse internals undecided",
         purpose = "train-time check before solver-internal sparse work"),
        (priority = 4, batch = "expanded_main_20_seed",
         command = "julia --project=benchmarks/deeponet benchmarks/deeponet/study.jl",
         env = "STRUCTPINN_DEEPONET_STUDY_SEEDS=1:20",
         grids = string(study.grid), seeds = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20",
         steps = study.steps, train_samples = study.train_samples,
         test_samples = study.test_samples,
         gate = "run only after the core 10 seed table is stable",
         purpose = "variance reduction for submission-ready claims"),
        (priority = 5, batch = "expanded_large_10_seed",
         command = "julia --project=benchmarks/deeponet benchmarks/deeponet/large_study.jl",
         env = "STRUCTPINN_DEEPONET_LARGE_SEEDS=1:10",
         grids = format_ints(large.grids),
         seeds = "1 2 3 4 5 6 7 8 9 10", steps = large.steps,
         train_samples = large.train_samples,
         test_samples = large.test_samples,
         gate = "run only if larger-grid rows become central claims",
         purpose = "submission-ready larger-grid variance check"),
        (priority = 6, batch = "profile_before_sparse_work",
         command = "julia --project=benchmarks/deeponet benchmarks/deeponet/profile_projection.jl",
         env = "", grids = format_ints(profile.grids), seeds = "",
         steps = "", train_samples = profile.samples,
         test_samples = "",
         gate = "run before any solver-internal sparse optimization",
         purpose = "projection overhead decision evidence"),
        (priority = 7, batch = "larger_output_sparse_profile",
         command = "julia --project=benchmarks/deeponet benchmarks/deeponet/profile_larger_outputs.jl",
         env = "", grids = format_ints(larger_profile.grids), seeds = "",
         steps = "", train_samples = larger_profile.samples,
         test_samples = "",
         gate = "run before solver-internal sparse work when K=256 remains inconclusive",
         purpose = "larger-output projection overhead decision evidence"),
        (priority = 8, batch = "expanded_seed_gate",
         command = "julia --project=benchmarks/deeponet benchmarks/deeponet/expanded_seed_gate.jl",
         env = "", grids = "32 64 96",
         seeds = "$(expansion.main_seed_target) main $(expansion.large_seed_target) large",
         steps = "", train_samples = "", test_samples = "",
         gate = "run before expanded seed-count result jobs",
         purpose = "guard expanded main and larger-grid seed runs"),
    ]
end

end
