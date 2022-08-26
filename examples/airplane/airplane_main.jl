using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using BilinearControl
using ThreadsX
using ProgressMeter
using PGFPlotsX
using Statistics

include("airplane_utils.jl")
include("../plotting_constants.jl")

## Visualizer
model = BilinearControl.NominalAirplane()
vis = Visualizer()
delete!(vis)
set_airplane!(vis, model)
open(vis)

## Generate Training Data
gen_airplane_data(num_train=50, num_test=50, dt=0.04, dp_window=fill(0.5, 3))

## Test the model with an increasing number of training samples
num_train = [2; 5:5:100]
prog = Progress(length(num_train))
results = map(num_train) do N
    res = test_airplane(train_airplane(N)...)
    next!(prog)
    res
end
jldsave(AIRPLANE_RESULTS; results)

## Plot the Results
results = load(AIRPLANE_RESULTS)["results"]
airplane_data = load(AIRPLANE_DATAFILE)
num_test =  size(airplane_data["X_test"],2)
# num_train = load(AIRPLANE_RESULTS)["num_train"] 

did_track(x) = x<1e1

function get_average_error(results, method)
    map(x->mean(filter(did_track, x[method])), results)
end

function get_min_error(results, method)
    min_empty(x) = if isempty(x) 0 else minimum(x) end
    min_err = map(x->min_empty(filter(did_track, x[method])), results)
    return min_err
end

function get_max_error(results, method)
    max_empty(x) = if isempty(x) 0 else maximum(x) end
    max_err = map(x->max_empty(filter(did_track, x[method])), results)
    return max_err
end

function get_error_ci(results, method)
    tracked_res = map(x->filter(did_track, x[method]), results)
    err_std = map(x -> stdm(x, mean(x)), tracked_res)
    err_ci = 1.959964 .* err_std ./ sqrt(length(tracked_res[1]))
    return err_ci
end

function get_error_quantile(results, method; p=0.05)
    quantile_empty(x, i) = if isempty(x) 0 else quantile(x, i) end

    min_quant = map(x->quantile_empty(filter(did_track, x[method]), p), results)
    max_quant = map(x->quantile_empty(filter(did_track, x[method]), 1-p), results)
    return min_quant, max_quant
end

function get_success_rate(results, method)
    map(x->count(did_track, x[method]) / num_test, results)
end

function invalidate_by_success_rate!(err,sr)
    for i = 1:length(err)
        if sr[i] < 0.95
            err[i] = NaN
        end
    end
end
results[1]

err_nom  = get_average_error(results, :nominal) 
err_eDMD = get_average_error(results, :eDMD) 
err_jDMD = get_average_error(results, :jDMD) 

# min_err_nom  = get_min_error(results, :nominal) 
# min_err_eDMD = get_min_error(results, :eDMD) 
# min_err_jDMD = get_min_error(results, :jDMD) 

# max_err_nom  = get_max_error(results, :nominal) 
# max_err_eDMD = get_max_error(results, :eDMD) 
# max_err_jDMD = get_max_error(results, :jDMD)

# ci_err_nom  = get_error_ci(results, :nominal) 
# ci_err_eDMD = get_error_ci(results, :eDMD) 
# ci_err_jDMD = get_error_ci(results, :jDMD) 

quant_min_nom, quant_max_nom  = get_error_quantile(results, :nominal) 
quant_min_eDMD, quant_max_eDMD = get_error_quantile(results, :eDMD) 
quant_min_jDMD, quant_max_jDMD = get_error_quantile(results, :jDMD) 
jerr_nom  = get_average_error(results, :jerr_nominal) 
jerr_eDMD = get_average_error(results, :jerr_eDMD) 
jerr_jDMD = get_average_error(results, :jerr_jDMD) 

sr_nom  = get_success_rate(results, :nominal) 
sr_eDMD = get_success_rate(results, :eDMD) 
sr_jDMD = get_success_rate(results, :jDMD) 

invalidate_by_success_rate!(err_nom, sr_nom)
invalidate_by_success_rate!(err_eDMD, sr_eDMD)
invalidate_by_success_rate!(err_jDMD, sr_jDMD)

invalidate_by_success_rate!(quant_min_nom, sr_nom)
invalidate_by_success_rate!(quant_min_eDMD, sr_eDMD)
invalidate_by_success_rate!(quant_min_jDMD, sr_jDMD)

invalidate_by_success_rate!(quant_max_nom, sr_nom)
invalidate_by_success_rate!(quant_max_eDMD, sr_eDMD)
invalidate_by_success_rate!(quant_max_jDMD, sr_jDMD)

p_err = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        # ymode="log",
        xlabel = "Number of Training Trajectories",
        ylabel = "Tracking Error",
        legend_style = "{at={(0.97,0.7)},anchor=east}",
        # ymax = 0.11,
    },

    PlotInc({lineopts..., color=color_nominal, solid, thick}, Coordinates(num_train, err_nom)),

    # PlotInc({lineopts..., "name path=A", "black!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, quant_min_nom)),
    # PlotInc({lineopts..., "name_path=B", "black!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, quant_max_nom)),
    # PlotInc({lineopts..., "black!10", "forget plot"}, "fill between [of=A and B]"),

    PlotInc({lineopts..., color=color_eDMD, solid, thick}, Coordinates(num_train, err_eDMD)),
    
    # PlotInc({lineopts..., "name_path=C", "orange!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, quant_min_eDMD)),
    # PlotInc({lineopts..., "name_path=D", "orange!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, quant_max_eDMD)),
    # PlotInc({lineopts..., "orange!10", "forget plot"}, "fill between [of=C and D]"),

    PlotInc({lineopts..., color=color_jDMD, solid, thick}, Coordinates(num_train, err_jDMD)),
    # PlotInc({lineopts..., "name_path=E", "cyan!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, quant_min_jDMD)),
    # PlotInc({lineopts..., "name_path=F","cyan!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, quant_max_jDMD)),
    # PlotInc({lineopts..., "cyan!10", "forget plot"}, "fill between [of=E and F]"),

    Legend(["Nominal MPC", "EDMD", "JDMD"])

)
pgfsave(joinpath(BilinearControl.FIGDIR, "airplane_error_by_num_train.tikz"), p_err, 
    include_preamble=false)

p_jerr = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        # ymode="log",
        xlabel = "Number of training trajectories",
        ylabel = "Jacobian Error",
        legend_style="{at={(0.97,0.65)},anchor=east}"
    },
    PlotInc({lineopts..., color=color_nominal}, Coordinates(num_train, jerr_nom)),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(num_train, jerr_eDMD)),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(num_train, jerr_jDMD)),
    Legend(["Nominal MPC", "EDMD", "JDMD"])
)
pgfsave(joinpath(BilinearControl.FIGDIR, "airplane_jacobian_error.tikz"), p_err, 
    include_preamble=false)
## TODO: Add waypoints plot
