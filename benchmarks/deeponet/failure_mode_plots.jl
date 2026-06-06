# Failure-mode figure and compact CSV from DeepONet stress diagnostics.
# Run after stress_diagnostics.jl:
# julia --project=benchmarks/deeponet benchmarks/deeponet/failure_mode_plots.jl
using DelimitedFiles, Printf

const OUT = joinpath(@__DIR__, "results")

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

function short_label(name)
    replacements = Dict(
        "infeasible_mass_under_bounds" => "infeasible balance",
        "large_correction_norm" => "large correction",
        "box_only_kink" => "box kink",
        "malformed_adapter_rows" => "bad rows",
    )
    return get(replacements, name, name)
end

function fmt(x)
    return @sprintf("%.4g", Float64(x))
end

function status_color(status)
    status == "success" && return "#2f7d32"
    status == "infeasible_constraint" && return "#b42318"
    status == "nonunique_input" && return "#b54708"
    status == "argument_error" && return "#6f42c1"
    return "#57606a"
end

function write_failure_svg(path, labels, statuses, pass, norms)
    width = 1200
    height = 620
    left = 90
    right = 40
    top = 76
    bottom = 170
    plot_width = width - left - right
    plot_height = height - top - bottom
    ymax = max(maximum(norms), 1e-9)
    gap = 28
    bar_width = plot_width / length(labels) - gap
    open(path, "w") do io
        println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$width\" height=\"$height\" viewBox=\"0 0 $width $height\">")
        println(io, "<rect width=\"$width\" height=\"$height\" fill=\"white\"/>")
        println(io, "<text x=\"$(width / 2)\" y=\"32\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"24\">DeepONet failure-mode diagnostics</text>")
        println(io, "<text x=\"$(width / 2)\" y=\"56\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">Bars show correction norm. Color shows observed status.</text>")
        println(io, "<line x1=\"$left\" y1=\"$(top + plot_height)\" x2=\"$(left + plot_width)\" y2=\"$(top + plot_height)\" stroke=\"#222\"/>")
        println(io, "<line x1=\"$left\" y1=\"$top\" x2=\"$left\" y2=\"$(top + plot_height)\" stroke=\"#222\"/>")
        for i in eachindex(labels)
            x = left + (i - 1) * (bar_width + gap) + gap / 2
            h = plot_height * norms[i] / ymax
            y = top + plot_height - h
            color = status_color(statuses[i])
            println(io, "<rect x=\"$x\" y=\"$y\" width=\"$bar_width\" height=\"$h\" fill=\"$color\"/>")
            cx = x + bar_width / 2
            mark = pass[i] ? "pass" : "fail"
            println(io, "<text x=\"$cx\" y=\"$(y - 8)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">$(fmt(norms[i]))</text>")
            println(io, "<text x=\"$cx\" y=\"$(top + plot_height + 22)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\" transform=\"rotate(25 $cx $(top + plot_height + 22))\">$(labels[i])</text>")
            println(io, "<text x=\"$cx\" y=\"$(top + plot_height + 94)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">$(statuses[i])</text>")
            println(io, "<text x=\"$cx\" y=\"$(top + plot_height + 114)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">$(mark)</text>")
        end
        println(io, "<text x=\"$(left - 10)\" y=\"$top\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"12\">$(fmt(ymax))</text>")
        println(io, "</svg>")
    end
    return path
end

stress_path = joinpath(OUT, "stress_diagnostics.csv")
isfile(stress_path) || error("missing stress_diagnostics.csv")

data, names = table(stress_path)
cases = String.(data[:, col(names, "case")])
labels = short_label.(cases)
statuses = String.(data[:, col(names, "status")])
expected = String.(data[:, col(names, "expected")])
pass = string.(data[:, col(names, "pass")]) .== "true"
norms = Float64.(data[:, col(names, "correction_norm")])

open(joinpath(OUT, "failure_mode_summary.csv"), "w") do io
    println(io, "case,label,status,expected,pass,correction_norm")
    for i in eachindex(cases)
        println(io, @sprintf("%s,%s,%s,%s,%s,%.8e",
                             cases[i], labels[i], statuses[i],
                             expected[i], pass[i], norms[i]))
    end
end

write_failure_svg(joinpath(OUT, "failure_mode_diagnostics.svg"),
                  labels, statuses, pass, norms)

println("wrote failure_mode_summary.csv and failure_mode_diagnostics.svg to ",
        OUT)
