# DeepONet helper figures from study CSVs.
# Run after study.jl:
# julia --project=benchmarks/deeponet benchmarks/deeponet/plots.jl
using DelimitedFiles, Plots, Printf
gr()

const OUT = joinpath(@__DIR__, "results")
const FIG_KW = (size = (1300, 700), bottom_margin = 18Plots.mm,
                left_margin = 10Plots.mm, right_margin = 4Plots.mm,
                top_margin = 5Plots.mm)

function table(path)
    data, header = readdlm(path, ',', header = true)
    names = vec(String.(header))
    return data, names
end

function col(names, name)
    idx = findfirst(==(name), names)
    idx === nothing && error("missing column $(name)")
    return idx
end

function labels(data, names)
    return String.(data[:, col(names, "model")])
end

function plot_labels(models)
    replacements = Dict(
        "soft_boundary_heavy" => "soft_bheavy",
        "eval_only_full" => "eval_full",
        "hard_boundary_box" => "hard_bbox",
        "hard_full_cached" => "hard_full_c",
        "soft_plus_hard_boundary_box" => "sph_bbox",
        "soft_plus_hard_full" => "sph_full",
        "soft_plus_hard_full_cached" => "sph_full_c",
    )
    return [get(replacements, m, m) for m in models]
end

function write_runtime_svg(path, labels_plot, runtime)
    width = 1100
    height = 620
    left = 80
    right = 30
    top = 60
    bottom = 170
    plot_width = width - left - right
    plot_height = height - top - bottom
    ymax = max(maximum(runtime), 1e-9)
    bar_gap = 8
    bar_width = plot_width / length(runtime) - bar_gap
    open(path, "w") do io
        println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$width\" height=\"$height\" viewBox=\"0 0 $width $height\">")
        println(io, "<rect width=\"$width\" height=\"$height\" fill=\"white\"/>")
        println(io, "<text x=\"$(width / 2)\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"22\">DeepONet helper training time</text>")
        println(io, "<text x=\"25\" y=\"$(top + plot_height / 2)\" transform=\"rotate(-90 25 $(top + plot_height / 2))\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"16\">seconds</text>")
        println(io, "<line x1=\"$left\" y1=\"$(top + plot_height)\" x2=\"$(left + plot_width)\" y2=\"$(top + plot_height)\" stroke=\"black\"/>")
        println(io, "<line x1=\"$left\" y1=\"$top\" x2=\"$left\" y2=\"$(top + plot_height)\" stroke=\"black\"/>")
        for (i, value) in enumerate(runtime)
            x = left + (i - 1) * (bar_width + bar_gap) + bar_gap / 2
            h = plot_height * value / ymax
            y = top + plot_height - h
            println(io, "<rect x=\"$x\" y=\"$y\" width=\"$bar_width\" height=\"$h\" fill=\"#4477aa\"/>")
            lx = x + bar_width / 2
            println(io, "<text x=\"$lx\" y=\"$(top + plot_height + 18)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\" transform=\"rotate(45 $lx $(top + plot_height + 18))\">$(labels_plot[i])</text>")
            println(io, "<text x=\"$lx\" y=\"$(y - 5)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"10\">$(round(value, digits = 2))</text>")
        end
        println(io, "<text x=\"$left\" y=\"$(top - 10)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\">$(round(ymax, digits = 2))</text>")
        println(io, "</svg>")
    end
    return path
end

function write_correction_svg(path, labels_plot, mean_values, max_values)
    width = 1100
    height = 640
    left = 90
    right = 30
    top = 60
    bottom = 180
    plot_width = width - left - right
    plot_height = height - top - bottom
    ymax = max(maximum(max_values), maximum(mean_values), 1e-9)
    xstep = plot_width / max(length(labels_plot) - 1, 1)
    y(value) = top + plot_height - plot_height * value / ymax
    open(path, "w") do io
        println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$width\" height=\"$height\" viewBox=\"0 0 $width $height\">")
        println(io, "<rect width=\"$width\" height=\"$height\" fill=\"white\"/>")
        println(io, "<text x=\"$(width / 2)\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"22\">DeepONet helper evaluation corrections</text>")
        println(io, "<line x1=\"$left\" y1=\"$(top + plot_height)\" x2=\"$(left + plot_width)\" y2=\"$(top + plot_height)\" stroke=\"black\"/>")
        println(io, "<line x1=\"$left\" y1=\"$top\" x2=\"$left\" y2=\"$(top + plot_height)\" stroke=\"black\"/>")
        println(io, "<text x=\"$(left + plot_width - 120)\" y=\"$(top + 20)\" font-family=\"sans-serif\" font-size=\"14\" fill=\"#4477aa\">mean</text>")
        println(io, "<text x=\"$(left + plot_width - 120)\" y=\"$(top + 42)\" font-family=\"sans-serif\" font-size=\"14\" fill=\"#cc6677\">max</text>")
        for i in eachindex(labels_plot)
            x = left + (i - 1) * xstep
            println(io, "<circle cx=\"$x\" cy=\"$(y(mean_values[i]))\" r=\"4\" fill=\"#4477aa\"/>")
            println(io, "<circle cx=\"$x\" cy=\"$(y(max_values[i]))\" r=\"4\" fill=\"#cc6677\"/>")
            println(io, "<text x=\"$x\" y=\"$(top + plot_height + 18)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\" transform=\"rotate(45 $x $(top + plot_height + 18))\">$(labels_plot[i])</text>")
            if i > firstindex(labels_plot)
                px = left + (i - 2) * xstep
                println(io, "<line x1=\"$px\" y1=\"$(y(mean_values[i - 1]))\" x2=\"$x\" y2=\"$(y(mean_values[i]))\" stroke=\"#4477aa\"/>")
                println(io, "<line x1=\"$px\" y1=\"$(y(max_values[i - 1]))\" x2=\"$x\" y2=\"$(y(max_values[i]))\" stroke=\"#cc6677\"/>")
            end
        end
        println(io, "<text x=\"$left\" y=\"$(top - 10)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\">$(round(ymax, digits = 2))</text>")
        println(io, "</svg>")
    end
    return path
end

function plot_results()
    data, names = table(joinpath(OUT, "results.csv"))
    models = labels(data, names)
    xmodels = plot_labels(models)
    xs = collect(eachindex(xmodels))
    xtick_spec = (xs, xmodels)
    rmse = Float64.(data[:, col(names, "rmse_mean")])
    rmse_err = Float64.(data[:, col(names, "rmse_std")])
    boundary = Float64.(data[:, col(names, "boundarymax_mean")])
    boundary_err = Float64.(data[:, col(names, "boundarymax_std")])
    mass = Float64.(data[:, col(names, "massmax_mean")])
    mass_err = Float64.(data[:, col(names, "massmax_std")])
    lower = Float64.(data[:, col(names, "lowermax_mean")])
    upper = Float64.(data[:, col(names, "uppermax_mean")])
    runtime = Float64.(data[:, col(names, "train_seconds_mean")])

    p1 = bar(xs, rmse; yerror = rmse_err, legend = false,
             ylabel = "RMSE", title = "DeepONet helper accuracy",
             xticks = xtick_spec, xrotation = 25, FIG_KW...)
    savefig(p1, joinpath(OUT, "deeponet_rmse.png"))

    p2 = plot(xs, max.(boundary, eps()); yerror = boundary_err,
              marker = :circle, yscale = :log10, label = "boundary",
              ylabel = "max violation", title = "DeepONet helper feasibility",
              xticks = xtick_spec, xrotation = 25, FIG_KW...)
    plot!(p2, xs, max.(mass, eps()); yerror = mass_err,
          marker = :square, label = "mass")
    plot!(p2, xs, max.(lower, eps()); marker = :diamond,
          label = "lower")
    plot!(p2, xs, max.(upper, eps()); marker = :utriangle,
          label = "upper")
    savefig(p2, joinpath(OUT, "deeponet_feasibility.png"))

    write_runtime_svg(joinpath(OUT, "deeponet_runtime.svg"), xmodels,
                      runtime)
end

function plot_statuses()
    path = joinpath(OUT, "statuses.csv")
    isfile(path) || return nothing
    data, names = table(path)
    phase = String.(data[:, col(names, "phase")])
    model = String.(data[:, col(names, "model")])
    cmean = Float64.(data[:, col(names, "correction_mean")])
    cmax = Float64.(data[:, col(names, "correction_max")])
    keep = phase .== "eval"
    labels_eval = model[keep]
    xlabels_eval = plot_labels(labels_eval)
    xs = collect(eachindex(xlabels_eval))
    xtick_spec = (xs, xlabels_eval)
    mean_eval = cmean[keep]
    max_eval = cmax[keep]

    write_correction_svg(joinpath(OUT, "deeponet_correction_norm.svg"),
                         xlabels_eval, mean_eval, max_eval)
end

function metric_from(path, model, metric)
    data, names = table(path)
    models = String.(data[:, col(names, "model")])
    idx = findfirst(==(model), models)
    idx === nothing && error("missing row $(model)")
    return Float64(data[idx, col(names, metric)])
end

function panel_values(values, logscale)
    vals = Float64.(values)
    if !logscale
        ymax = max(maximum(vals), 1e-12)
        return vals ./ ymax, ymax, 0.0
    end
    clipped = max.(vals, eps())
    logs = log10.(clipped)
    lo = minimum(logs)
    hi = maximum(logs)
    hi <= lo && (hi = lo + 1)
    return (logs .- lo) ./ (hi - lo), hi, lo
end

function write_bar_panel(io, x0, y0, width, height, title, labels_plot,
                         values, logscale)
    top = y0 + 34
    bottom = y0 + height - 62
    left = x0 + 60
    right = x0 + width - 18
    plot_height = bottom - top
    plot_width = right - left
    scaled, ymax, ymin = panel_values(values, logscale)
    bar_gap = 8
    bar_width = plot_width / length(values) - bar_gap
    println(io, "<text x=\"$(x0 + width / 2)\" y=\"$(y0 + 22)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"18\">$(title)</text>")
    println(io, "<line x1=\"$left\" y1=\"$bottom\" x2=\"$right\" y2=\"$bottom\" stroke=\"#222\"/>")
    println(io, "<line x1=\"$left\" y1=\"$top\" x2=\"$left\" y2=\"$bottom\" stroke=\"#222\"/>")
    for (i, value) in enumerate(values)
        x = left + (i - 1) * (bar_width + bar_gap) + bar_gap / 2
        h = plot_height * scaled[i]
        y = bottom - h
        println(io, "<rect x=\"$x\" y=\"$y\" width=\"$bar_width\" height=\"$h\" fill=\"#4477aa\"/>")
        lx = x + bar_width / 2
        println(io, "<text x=\"$lx\" y=\"$(bottom + 16)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"10\" transform=\"rotate(35 $lx $(bottom + 16))\">$(labels_plot[i])</text>")
        println(io, "<text x=\"$lx\" y=\"$(y - 4)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"10\">$(@sprintf("%.2g", value))</text>")
    end
    if logscale
        println(io, "<text x=\"$(left - 8)\" y=\"$top\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"10\">1e$(@sprintf("%.1f", ymax))</text>")
        println(io, "<text x=\"$(left - 8)\" y=\"$bottom\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"10\">1e$(@sprintf("%.1f", ymin))</text>")
    else
        println(io, "<text x=\"$(left - 8)\" y=\"$top\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"10\">$(@sprintf("%.2g", ymax))</text>")
    end
    return nothing
end

function plot_report_bundle()
    result_path = joinpath(OUT, "results.csv")
    frequency_path = joinpath(OUT, "frequency_ablation.csv")
    constraint_path = joinpath(OUT, "constraint_ablation.csv")
    profile_path = joinpath(OUT, "projection_profile.csv")
    all(isfile, (result_path, frequency_path, constraint_path,
                 profile_path)) || return nothing

    main_models = ["vanilla", "soft_weak", "eval_only_full",
                   "hard_full_cached", "soft_plus_hard_full_cached"]
    main_labels = ["vanilla", "soft", "eval", "hard", "soft+hard"]
    rmse = [metric_from(result_path, m, "rmse_mean") for m in main_models]
    mass = [metric_from(result_path, m, "massmax_mean") for m in main_models]

    freq_models = ["no_correction", "eval_only_full", "every_step",
                   "every_2_steps", "every_5_steps", "every_10_steps"]
    freq_labels = ["none", "eval", "1", "2", "5", "10"]
    freq_rmse = [metric_from(frequency_path, m, "rmse_mean")
                 for m in freq_models]

    constraint_models = ["eval_boundary_only", "eval_box_only",
                         "eval_mass_only", "eval_boundary_box",
                         "eval_full"]
    constraint_labels = ["boundary", "box", "mass", "boundary+box",
                         "full"]
    constraint_rmse = [metric_from(constraint_path, m, "rmse_mean")
                       for m in constraint_models]

    path = joinpath(OUT, "deeponet_report_bundle.svg")
    open(path, "w") do io
        println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1500\" height=\"950\" viewBox=\"0 0 1500 950\">")
        println(io, "<rect width=\"1500\" height=\"950\" fill=\"white\"/>")
        write_bar_panel(io, 20, 20, 710, 430, "Main helper rows",
                        main_labels, rmse, false)
        write_bar_panel(io, 770, 20, 710, 430, "Main mass feasibility",
                        main_labels, mass, true)
        write_bar_panel(io, 20, 490, 710, 430, "Projection frequency",
                        freq_labels, freq_rmse, false)
        write_bar_panel(io, 770, 490, 710, 430, "Constraint families",
                        constraint_labels, constraint_rmse, false)
        println(io, "</svg>")
    end
    return nothing
end

plot_results()
plot_statuses()
plot_report_bundle()
println("wrote DeepONet helper plots to ", OUT)
