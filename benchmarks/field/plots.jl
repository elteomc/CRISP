# M5 and M6 field figures from study CSVs.
# Run after study.jl and pde_study.jl:
# julia --project=benchmarks/field benchmarks/field/plots.jl
using DelimitedFiles, Plots
gr()

const OUT = joinpath(@__DIR__, "results")

function _table(path)
    data, header = readdlm(path, ',', header = true)
    names = vec(String.(header))
    return data, names
end

function _col(names, name)
    idx = findfirst(==(name), names)
    idx === nothing && error("missing column $(name)")
    return idx
end

function _labels(data, names)
    return String.(data[:, _col(names, "model")])
end

function plot_fixed_grid()
    data, names = _table(joinpath(OUT, "results.csv"))
    models = _labels(data, names)
    rmse = Float64.(data[:, _col(names, "rmse_mean")])
    rmse_err = Float64.(data[:, _col(names, "rmse_std")])
    mass = Float64.(data[:, _col(names, "massmax_mean")])
    mass_err = Float64.(data[:, _col(names, "massmax_std")])
    neg = Float64.(data[:, _col(names, "negmax_mean")])
    upper = Float64.(data[:, _col(names, "uppermax_mean")])

    p1 = bar(models, rmse, yerror = rmse_err, legend = false,
             ylabel = "RMSE", title = "Fixed-grid field accuracy",
             xrotation = 35, margin = 8Plots.mm)
    savefig(p1, joinpath(OUT, "field_rmse.png"))

    p2 = plot(models, mass, yerror = mass_err, marker = :circle,
              yscale = :log10, legend = false, ylabel = "max |mass error|",
              title = "Mass conservation across field baselines",
              xrotation = 35, margin = 8Plots.mm)
    savefig(p2, joinpath(OUT, "field_mass_violation.png"))

    p3 = plot(models, max.(neg, eps()), marker = :circle, yscale = :log10,
              label = "lower bound", ylabel = "max violation",
              title = "Bound violations across field baselines",
              xrotation = 35, margin = 8Plots.mm)
    plot!(p3, models, max.(upper, eps()), marker = :square,
          label = "upper bound")
    savefig(p3, joinpath(OUT, "field_bound_violation.png"))
end

function plot_pde()
    path = joinpath(OUT, "pde_results.csv")
    isfile(path) || return nothing
    data, names = _table(path)
    problem = String.(data[:, _col(names, "problem")])
    rmse = Float64.(data[:, _col(names, "rmse_mean")])
    rmse_err = Float64.(data[:, _col(names, "rmse_std")])
    lower = Float64.(data[:, _col(names, "lowermax_mean")])
    upper = Float64.(data[:, _col(names, "uppermax_mean")])
    mass = Float64.(data[:, _col(names, "massmax_mean")])
    mass_enforced = lowercase.(string.(data[:, _col(names, "mass_enforced")])) .== "true"

    p1 = bar(problem, rmse, yerror = rmse_err, legend = false,
             ylabel = "RMSE", title = "Larger PDE integration accuracy",
             xrotation = 25, margin = 8Plots.mm)
    savefig(p1, joinpath(OUT, "pde_rmse.png"))

    mass_plot = ifelse.(mass_enforced, max.(mass, eps()), NaN)
    p2 = plot(problem, mass_plot, marker = :circle, yscale = :log10,
              label = "mass constrained", ylabel = "max violation",
              title = "PDE projection constraints",
              xrotation = 25, margin = 8Plots.mm)
    plot!(p2, problem, max.(lower, eps()), marker = :diamond,
          label = "lower")
    plot!(p2, problem, max.(upper, eps()), marker = :square,
          label = "upper")
    savefig(p2, joinpath(OUT, "pde_feasibility.png"))
end

plot_fixed_grid()
plot_pde()
println("wrote field and PDE plots to ", OUT)
