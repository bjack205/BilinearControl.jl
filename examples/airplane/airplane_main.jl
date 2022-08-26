using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using BilinearControl
using ThreadsX
using ProgressMeter
using PGFPlotsX
using Statistics
using LaTeXStrings

include("airplane_utils.jl")
include("../plotting_constants.jl")

#############################################
## Functions
#############################################

function get_average_error(results, method)
    map(x->mean(filter(did_track, x[method])), results)
end

function get_median_error(results, method)
    median_empty(x) = if isempty(x) NaN else median(sort(x)) end
    map(x->median_empty(filter(did_track, x[method])), results)
end

function get_min_error(results, method)
    min_empty(x) = if isempty(x) NaN else minimum(x) end
    min_err = map(x->min_empty(filter(did_track, x[method])), results)
    return min_err
end

function get_max_error(results, method)
    max_empty(x) = if isempty(x) NaN else maximum(x) end
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

function invalidate_by_success_rate!(err,sr; thresh=0.95)
    for i = 1:length(err)
        if sr[i] < thresh 
            err[i] = NaN
        end
    end
    err
end

#############################################
## Visualizer
#############################################
model = BilinearControl.NominalAirplane()
vis = Visualizer()
delete!(vis)
set_airplane!(vis, model)
open(vis)

#############################################
## Generate training data
#############################################

gen_airplane_data(num_train=50, num_test=50, dt=0.04, dp_window=fill(0.5, 3))

#############################################
## Sample study
#############################################

num_train = [2; 5:5:100]
prog = Progress(length(num_train))
results = map(num_train) do N
    test_airplane(train_airplane(N)...)
    next!(prog)
end
jldsave(AIRPLANE_RESULTS; results)

## Plot the Results
results = load(AIRPLANE_RESULTS)["results"]
airplane_data = load(AIRPLANE_DATAFILE)
num_test =  size(airplane_data["X_test"],2)
# num_train = load(AIRPLANE_RESULTS)["num_train"] 

did_track(x) = x<1e1

err_nom  = get_median_error(results, :nominal) 
err_eDMD = get_median_error(results, :eDMD) 
err_jDMD = get_median_error(results, :jDMD) 

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
jerr_nom_lo, jerr_nom_up = get_error_quantile(results, :jerr_nominal) 
jerr_eDMD_lo, jerr_eDMD_up = get_error_quantile(results, :jerr_eDMD) 
jerr_jDMD_lo, jerr_jDMD_up = get_error_quantile(results, :jerr_jDMD) 

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

invalidate_by_success_rate!(jerr_nom, sr_nom)
invalidate_by_success_rate!(jerr_nom_up, sr_nom)
invalidate_by_success_rate!(jerr_nom_lo, sr_nom)
invalidate_by_success_rate!(jerr_eDMD, sr_eDMD)
invalidate_by_success_rate!(jerr_eDMD_up, sr_eDMD)
invalidate_by_success_rate!(jerr_eDMD_lo, sr_eDMD)
invalidate_by_success_rate!(jerr_jDMD, sr_jDMD)
invalidate_by_success_rate!(jerr_jDMD_up, sr_jDMD)
invalidate_by_success_rate!(jerr_jDMD_lo, sr_jDMD)

p_err = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        # ymode="log",
        xlabel = "Number of Training Trajectories",
        ylabel = "Tracking Error",
        legend_style = "{at={(0.97,0.7)},anchor=east}",
        ymax = 0.22,
    },

    PlotInc({lineopts..., color=color_nominal, solid, thick}, Coordinates(num_train, err_nom)),

    PlotInc({lineopts..., "name path=A", "black!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, quant_min_nom)),
    PlotInc({lineopts..., "name_path=B", "black!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, quant_max_nom)),
    PlotInc({lineopts..., "black!10", "forget plot"}, "fill between [of=A and B]"),

    PlotInc({lineopts..., color=color_eDMD, solid, thick}, Coordinates(num_train, err_eDMD)),
    
    PlotInc({lineopts..., "name_path=C", "orange!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, quant_min_eDMD)),
    PlotInc({lineopts..., "name_path=D", "orange!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, quant_max_eDMD)),
    PlotInc({lineopts..., "orange!10", "forget plot"}, "fill between [of=C and D]"),

    PlotInc({lineopts..., color=color_jDMD, solid, thick}, Coordinates(num_train, err_jDMD)),
    PlotInc({lineopts..., "name_path=E", "cyan!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, quant_min_jDMD)),
    PlotInc({lineopts..., "name_path=F","cyan!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, quant_max_jDMD)),
    PlotInc({lineopts..., "cyan!10", "forget plot"}, "fill between [of=E and F]"),

    Legend(["Nominal MPC", "EDMD", "JDMD"])

);
pgfsave(joinpath(BilinearControl.FIGDIR, "airplane_error_by_num_train.tikz"), p_err, 
    include_preamble=false)

p_jerr = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        ymode="log",
        xlabel = "Number of Training Trajectories",
        ylabel = "Jacobian Error",
        # legend_style = "{at={(0.97,0.7)},anchor=east}",
        "legend_pos" = "north east",
        # ymax = 0.11,
    },

    PlotInc({lineopts..., color=color_nominal, solid, thick}, Coordinates(num_train, jerr_nom)),

    PlotInc({lineopts..., "name path=A", "black!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, jerr_nom_lo)),
    PlotInc({lineopts..., "name_path=B", "black!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, jerr_nom_up)),
    PlotInc({lineopts..., "black!10", "forget plot"}, "fill between [of=A and B]"),

    PlotInc({lineopts..., color=color_eDMD, solid, thick}, Coordinates(num_train, jerr_eDMD)),
    
    PlotInc({lineopts..., "name_path=C", "orange!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, jerr_eDMD_lo)),
    PlotInc({lineopts..., "name_path=D", "orange!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, jerr_eDMD_up)),
    PlotInc({lineopts..., "orange!10", "forget plot"}, "fill between [of=C and D]"),

    PlotInc({lineopts..., color=color_jDMD, solid, thick}, Coordinates(num_train, jerr_jDMD)),
    PlotInc({lineopts..., "name_path=E", "cyan!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, jerr_jDMD_lo)),
    PlotInc({lineopts..., "name_path=F", "cyan!10", "forget plot", solid, line_width=0.1}, Coordinates(num_train, jerr_jDMD_up)),
    PlotInc({lineopts..., "cyan!10", "forget plot"}, "fill between [of=E and F]"),

    Legend(["Nominal", "EDMD", "JDMD"])
);
pgfsave(joinpath(BilinearControl.FIGDIR, "airplane_jacobian_error.tikz"), p_jerr, 
    include_preamble=false)
#############################################
## Model Prediction Error 
#############################################

alpha = 0.0:0.05:0.9
prog = Progress(length(alpha))
results = map(alpha) do a
    closed_loop_res = test_airplane(train_airplane(20; α=a)...)
    open_loop_res = test_airplane_open_loop(train_airplane(20; α=a)...)
    next!(prog)
    (;closed_loop_res, open_loop_res)
end

fields = keys(results[1])
results_model_pred = Dict(Pair.(fields, map(x->getfield.(results, x), fields)))
jldsave(AIRPLANE_RESULTS_PREDICTION; results_model_pred)

##
airplane_data = load(AIRPLANE_DATAFILE)
num_test =  size(airplane_data["X_test"],2)
# num_train = load(AIRPLANE_RESULTS)["num_train"] 
alpha = 0.0:0.05:0.9
results_model_pred = load(AIRPLANE_RESULTS_PREDICTION)["results_model_pred"]

closed_loop_res = results_model_pred[:closed_loop_res]
open_loop_res = results_model_pred[:open_loop_res]

did_track(x) = x<1e1
edmd_err_ol = get_median_error(open_loop_res, :eDMD) 
jdmd_err_cl = get_median_error(closed_loop_res, :jDMD) 

edmd_err_cl = get_median_error(closed_loop_res, :eDMD) 

did_track(x) = x<1e1

jdmd_err_ol = get_median_error(open_loop_res, :jDMD) 

edmd_quant_min_cl, edmd_quant_max_cl = get_error_quantile(closed_loop_res, :eDMD) 
edmd_quant_min_ol, edmd_quant_max_ol = get_error_quantile(open_loop_res, :eDMD) 

jdmd_quant_min_cl, jdmd_quant_max_cl = get_error_quantile(closed_loop_res, :jDMD) 
jdmd_quant_min_ol, jdmd_quant_max_ol = get_error_quantile(open_loop_res, :jDMD) 

edmd_sr_cl  = get_success_rate(closed_loop_res, :eDMD) 
edmd_sr_ol = get_success_rate(open_loop_res, :eDMD) 

jdmd_sr_cl  = get_success_rate(closed_loop_res, :jDMD) 
jdmd_sr_ol = get_success_rate(open_loop_res, :jDMD) 


jdmd_err_ol_filtered = invalidate_by_success_rate!(copy(jdmd_err_ol), jdmd_sr_ol, thresh=0.9)
jdmd_err_cl_filtered = invalidate_by_success_rate!(copy(jdmd_err_cl), jdmd_sr_cl, thresh=0.9)

# prediction error
# p_err = @pgf Axis(
#     {
#         xmajorgrids,
#         ymajorgrids,
#         ymode="log",
#         xlabel = L"\alpha",
#         ylabel = "Error",
#         legend_pos = "north west",
#     },

#     PlotInc({lineopts..., "name_path=E", "orange!10", "forget plot", solid, line_width=0.1}, Coordinates(alpha, edmd_quant_min_ol)),
#     PlotInc({lineopts..., "name_path=F","orange!10", "forget plot", solid, line_width=0.1}, Coordinates(alpha, edmd_quant_max_ol)),
#     PlotInc({lineopts..., "orange!10", "forget plot"}, "fill between [of=E and F]"),

#     PlotInc({lineopts..., "name_path=G", "cyan!10", "forget plot", solid, line_width=0.1}, Coordinates(alpha, jdmd_quant_min_ol)),
#     PlotInc({lineopts..., "name_path=H","cyan!10", "forget plot", solid, line_width=0.1}, Coordinates(alpha, jdmd_quant_max_ol)),
#     PlotInc({lineopts..., "cyan!10", "forget plot"}, "fill between [of=G and H]"),

#     PlotInc({lineopts..., "name_path=A", "orange!20", "forget plot", solid, line_width=0.1}, Coordinates(alpha, edmd_quant_min_cl)),
#     PlotInc({lineopts..., "name_path=B","orange!20", "forget plot", solid, line_width=0.1}, Coordinates(alpha, edmd_quant_max_cl)),
#     PlotInc({lineopts..., "orange!20", "forget plot"}, "fill between [of=A and B]"),

#     PlotInc({lineopts..., "name_path=C", "cyan!20", "forget plot", solid, line_width=0.1}, Coordinates(alpha, jdmd_quant_min_cl)),
#     PlotInc({lineopts..., "name_path=D","cyan!20", "forget plot", solid, line_width=0.1}, Coordinates(alpha, jdmd_quant_max_cl)),
#     PlotInc({lineopts..., "cyan!20", "forget plot"}, "fill between [of=C and D]"),

#     PlotInc({lineopts..., color=color_eDMD, solid, thick}, Coordinates(alpha, edmd_err_cl)),
#     PlotInc({lineopts..., "orange!50", dashed, thick}, Coordinates(alpha, edmd_err_ol)),
#     PlotInc({lineopts..., color=color_jDMD, solid, thick}, Coordinates(alpha, jdmd_err_cl)),
#     PlotInc({lineopts..., "cyan!50", dashed, thick}, Coordinates(alpha, jdmd_err_ol)),

#     # Legend(["Closed-loop", "Open-loop"])

# )

# pgfsave(joinpath(BilinearControl.FIGDIR, "airplane_model_error_by_alpha.tikz"), p_err, 
#     include_preamble=false)

# prediction error just jdmd

p_err = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        ymode="log",
        xlabel = L"\alpha",
        ylabel = "Error",
        legend_pos = "north west",
    },

    PlotInc({lineopts..., color=color_jDMD, solid, thick}, Coordinates(alpha, jdmd_err_cl)),
    PlotInc({lineopts..., "name_path=E", "cyan!20", "forget plot", solid, line_width=0.1}, Coordinates(alpha, jdmd_quant_min_cl)),
    PlotInc({lineopts..., "name_path=F","cyan!20", "forget plot", solid, line_width=0.1}, Coordinates(alpha, jdmd_quant_max_cl)),
    PlotInc({lineopts..., "cyan!20", "forget plot"}, "fill between [of=E and F]"),

    PlotInc({lineopts..., "cyan!50", dashed, thick}, Coordinates(alpha, jdmd_err_ol)),
    PlotInc({lineopts..., "name_path=E", "cyan!10", "forget plot", solid, line_width=0.1}, Coordinates(alpha, jdmd_quant_min_ol)),
    PlotInc({lineopts..., "name_path=F","cyan!10", "forget plot", solid, line_width=0.1}, Coordinates(alpha, jdmd_quant_max_ol)),
    PlotInc({lineopts..., "cyan!10", "forget plot"}, "fill between [of=E and F]"),

    Legend(["Closed-loop", "Open-loop"])

);

pgfsave(joinpath(BilinearControl.FIGDIR, "airplane_model_error_by_alpha_just_jdmd.tikz"), p_err, 
    include_preamble=false)

# # success rate
# p_err = @pgf Axis(
#     {
#         xmajorgrids,
#         ymajorgrids,
#         xlabel = L"\alpha",
#         ylabel = "Success Rate",
#         legend_pos = "south west",
#     },

#     PlotInc({lineopts..., color=color_eDMD, solid, thick}, Coordinates(alpha, edmd_sr_cl)),
#     PlotInc({lineopts..., "orange!50", dashed, thick}, Coordinates(alpha, edmd_sr_ol)),

#     PlotInc({lineopts..., color=color_jDMD, solid, thick}, Coordinates(alpha, jdmd_sr_cl)),
#     PlotInc({lineopts..., "cyan!50", dashed, thick}, Coordinates(alpha, jdmd_sr_ol)),
#     Legend(["Closed-loop (EDMD)", "Open-loop (EDMD)", "Closed-loop (JDMD)", "Open-loop (JDMD)"])

# )
# pgfsave(joinpath(BilinearControl.FIGDIR, "airplane_success_rate_by_alpha.tikz"), p_err, 
#     include_preamble=false)

# # success rate just jdmd
# p_err = @pgf Axis(
#     {
#         xmajorgrids,
#         ymajorgrids,
#         xlabel = L"\alpha",
#         ylabel = "Success Rate",
#         legend_pos = "south west",
#     },

#     PlotInc({lineopts..., color=color_jDMD, solid, thick}, Coordinates(alpha, jdmd_sr_cl)),
#     PlotInc({lineopts..., "cyan!50", dashed, thick}, Coordinates(alpha, jdmd_sr_ol)),
#     Legend(["Closed-loop (JDMD)", "Open-loop (JDMD)"])

# )
# pgfsave(joinpath(BilinearControl.FIGDIR, "airplane_success_rate_by_alpha_just_jdmd.tikz"), p_err, 
#     include_preamble=false)

# p_jerr = @pgf Axis(
#     {
#         xmajorgrids,
#         ymajorgrids,
#         # ymode="log",
#         xlabel = "Number of training trajectories",
#         ylabel = "Jacobian Error",
#         legend_style="{at={(0.97,0.65)},anchor=east}"
#     },
#     PlotInc({lineopts..., color=color_nominal}, Coordinates(num_train, jerr_nom)),
#     PlotInc({lineopts..., color=color_eDMD}, Coordinates(num_train, jerr_eDMD)),
#     PlotInc({lineopts..., color=color_jDMD}, Coordinates(num_train, jerr_jDMD)),
#     Legend(["Nominal MPC", "EDMD", "JDMD"])
# )
# pgfsave(joinpath(BilinearControl.FIGDIR, "airplane_jacobian_error.tikz"), p_err, 
#     include_preamble=false)
## TODO: Add waypoints plot