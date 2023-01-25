using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using BilinearControl
using ThreadsX
using ProgressMeter
using PGFPlotsX
using Statistics
using LaTeXStrings
using MeshCat

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
## Generate training data
#############################################

gen_airplane_data(num_train=100, num_test=100, dt=0.04, dp_window=fill(0.5, 3))

#############################################
## Sample study
#############################################

num_train = [2; 5:5:100]
prog = Progress(length(num_train))
results = map(num_train) do N
    test_airplane(train_airplane(N)...)
    next!(prog)
end
jldsave(AIRPLANE_RESULTS; results, num_train)

## Plot the Results
results = load(AIRPLANE_RESULTS)["results"]
num_train = load(AIRPLANE_RESULTS)["num_train"]
airplane_data = load(AIRPLANE_DATAFILE)
num_test =  size(airplane_data["X_test"],2)
# num_train = load(AIRPLANE_RESULTS)["num_train"] 

did_track(x) = x<1e1

err_nom  = get_median_error(results, :nominal) 
err_eDMD = get_median_error(results, :eDMD) 
err_jDMD = get_median_error(results, :jDMD) 

quant_min_nom, quant_max_nom  = get_error_quantile(results, :nominal) 
quant_min_eDMD, quant_max_eDMD = get_error_quantile(results, :eDMD) 
quant_min_jDMD, quant_max_jDMD = get_error_quantile(results, :jDMD) 

jerr_nom  = get_median_error(results, :jerr_nominal) 
jerr_eDMD = get_median_error(results, :jerr_eDMD) 
jerr_jDMD = get_median_error(results, :jerr_jDMD) 
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
## Visualize trajectories 
#############################################
num_test = 100 
airplane_data = load(AIRPLANE_DATAFILE)
T_ref = airplane_data["T_ref"]
num_samples = size(airplane_data["X_ref"],2)
test_inds = num_samples - num_test + 1:num_samples
X_ref = airplane_data["X_ref"][:,test_inds]
U_ref = airplane_data["U_ref"][:,test_inds]

num_train_edmd = 24
num_train_jdmd = 12 
model_edmd, _ = train_airplane(num_train_edmd)
_, model_jdmd = train_airplane(num_train_jdmd, α=0.9)
model_edmd_projected = BilinearControl.ProjectedEDMDModel(model_edmd)
model_jdmd_projected = BilinearControl.ProjectedEDMDModel(model_jdmd)

## Closed-loop prediction errors
res_cl_dmd = test_airplane(model_edmd, model_jdmd; num_test)
res_cl_dmd[:eDMD]
res_cl_dmd[:jDMD]
mean(filter(isfinite,res_cl_dmd[:eDMD]))
mean(filter(isfinite,res_cl_dmd[:jDMD]))
count(isfinite,res_cl_dmd[:eDMD]) / num_test
count(isfinite,res_cl_dmd[:jDMD]) / num_test

## Open-loop prediction errors
res_ol_dmd = test_airplane_open_loop(model_edmd_projected, model_jdmd_projected; num_test)
mean(filter(x->abs(x) < 10,res_ol_dmd[:nominal]))
mean(filter(x->abs(x) < 10,res_ol_dmd[:eDMD]))
mean(filter(x->abs(x) < 10,res_ol_dmd[:jDMD]))
count(x->abs(x) < 10,res_ol_dmd[:eDMD]) / num_test
count(x->abs(x) < 10,res_ol_dmd[:jDMD]) / num_test

## Dynamics prediction
res_dp_dmd = airplane_dynamics_prediction_error(model_edmd_projected, model_jdmd_projected; num_test)
mean(filter(x->abs(x) < 10,res_dp_dmd[:eDMD]))
mean(filter(x->abs(x) < 10,res_dp_dmd[:jDMD]))
res_dp_dmd

## Jacobian errors
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model)
jac_err_nom = map(i->jacobian_error(dmodel_nom, X_ref[:,i], U_ref[:,i], T_ref), 1:num_test)
jac_err_eDMD = map(i->jacobian_error(model_edmd_projected, X_ref[:,i], U_ref[:,i], T_ref), 1:num_test)
jac_err_jDMD = map(i->jacobian_error(model_jdmd_projected, X_ref[:,i], U_ref[:,i], T_ref), 1:num_test)


## Visualize trajectories
using Plots
i = 8 
clamphi(x, hi) = isfinite(x) ? min(x, hi) : hi
bins = range(0,1.6,length=50)
p = histogram(res_ol_dmd[:nominal], label="nominal", c="red", lc="red", bins=bins, xlabel="open loop error", ylabel="number of samples", background_color=:transparent, legend=:topleft, alpha=0.5)
histogram!(clamphi.(res_ol_dmd[:eDMD], 1.5), label="model A", c="orange", lc="orange", bins=bins, alpha=0.5)
histogram!(clamphi.(res_ol_dmd[:jDMD], 1.5), label="model B", c="cyan", lc="cyan", bins=bins, alpha=0.5)
savefig(p, "open_loop_errors.png")

bins = range(0,1.6,length=50)
p = histogram(res_cl_dmd[:nominal], label="nominal", c="red", lc="red", bins=bins, xlabel="open loop error", ylabel="number of samples", background_color=:transparent, alpha=0.5)
histogram!(clamphi.(res_cl_dmd[:eDMD], 1.5), label="model A", c="orange", lc="orange", bins=bins, alpha=0.5)
histogram!(clamphi.(res_cl_dmd[:jDMD], 1.5), label="model B", c="cyan", lc="cyan", bins=bins, alpha=0.5)
savefig(p, "closed_loop_errors.png")

bins = range(0.4,1.6,length=50)
p = histogram(jac_err_nom, label="nominal", c="red", lc="red", bins=bins, xlabel="Jacobian error", ylabel="number of samples", background_color=:transparent, alpha=0.5)
histogram!(clamphi.(jac_err_eDMD, 1.5), label="model A", c="orange", lc="orange", bins=bins, alpha=0.5)
histogram!(clamphi.(jac_err_jDMD, 1.5), label="model B", c="cyan", lc="cyan", bins=bins, alpha=0.5)
savefig(p, "jacobian_errors.png")

X_nom_ol = res_ol_dmd[:X_nom][i]
X_eDMD_ol = res_ol_dmd[:X_eDMD][i]
X_jDMD_ol = res_ol_dmd[:X_jDMD][i]
X_nom = res_cl_dmd[:X_nom][i]
X_eDMD = res_cl_dmd[:X_eDMD][i]
X_jDMD = res_cl_dmd[:X_jDMD][i]

res_ol_dmd[:nominal][i]
res_ol_dmd[:eDMD][i]
res_ol_dmd[:jDMD][i]

res_cl_dmd[:nominal][i]
res_cl_dmd[:eDMD][i]
res_cl_dmd[:jDMD][i]

res_dp_dmd[:nominal][i]
res_dp_dmd[:eDMD][i]
res_dp_dmd[:jDMD][i]

X_jDMD_ol = map(X_jDMD_ol) do x
    norm(x) < 100 ? x : x * NaN
end

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