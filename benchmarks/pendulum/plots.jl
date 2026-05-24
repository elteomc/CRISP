# M4 figures from the study CSVs: training curves and energy-over-time.
# Run after study.jl: julia --project=benchmarks/pendulum benchmarks/pendulum/plots.jl
using DelimitedFiles, Plots
gr()

const OUT = joinpath(@__DIR__, "results")

tc, _ = readdlm(joinpath(OUT, "training_curves.csv"), ',', header = true)
plot(tc[:, 1], tc[:, 2:4]; label = ["vanilla" "soft" "projected"],
     xlabel = "Adam step", ylabel = "training loss", yscale = :log10, lw = 2,
     title = "Training curves (seed 1)")
savefig(joinpath(OUT, "training_curves.png"))

et, _ = readdlm(joinpath(OUT, "energy_over_time.csv"), ',', header = true)
plot(et[:, 1], et[:, 3:5]; label = ["vanilla" "soft" "projected"],
     xlabel = "rollout step", ylabel = "H(z_t)", lw = 2,
     title = "Energy over a long-horizon rollout (seed 1)")
hline!([et[1, 2]]; label = "H0 (target)", ls = :dash, color = :black)
savefig(joinpath(OUT, "energy_over_time.png"))

println("wrote training_curves.png and energy_over_time.png to ", OUT)
