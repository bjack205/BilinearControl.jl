"""
    airplane_main.jl

Code to generate Figure 2b in the paper, showing the sample complexity of the fixed-wing 
airplane perching example.

Uses ALTRO to generate a set of reference perching trajectories using the true model, in 
order to get a dynamically feasible reference trajectory. These trajectories are then 
tracked using an MPC controller designed using the nominal model to generate the training
trajectories.

The main loop trains a new model using incrementally more and more training trajectories.
"""

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using BilinearControl
using ThreadsX
using ProgressMeter
using PGFPlotsX

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
# NOTE: This can blow your RAM
num_train = [2; 5:5:50]
prog = Progress(length(num_train))
results = map(num_train) do N
    test_airplane(train_airplane(N)...)
    next!(prog)
end

## Plot the Results
results = load(AIRPLANE_RESULTS)["results"]
airplane_data = load(AIRPLANE_DATAFILE)
num_test =  size(airplane_data["X_test"],2)
num_train = load(AIRPLANE_RESULTS)["num_train"] 

did_track(x) = x<1e1
function get_average_error(results, method)
    map(x->mean(filter(did_track, x[method])), results)
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

sr_nom  = get_success_rate(results, :nominal) 
sr_eDMD = get_success_rate(results, :eDMD) 
sr_jDMD = get_success_rate(results, :jDMD) 

invalidate_by_success_rate!(err_nom, sr_nom)
invalidate_by_success_rate!(err_eDMD, sr_eDMD)
invalidate_by_success_rate!(err_jDMD, sr_jDMD)

p_err = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        # ymode="log",
        xlabel = "Number of training trajectories",
        ylabel = "Tracking Error",
        legend_style="{at={(0.97,0.65)},anchor=east}"
    },
    PlotInc({lineopts..., color=color_nominal}, Coordinates(num_train, err_nom)),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(num_train, err_eDMD)),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(num_train, err_jDMD)),
    Legend(["Nominal MPC", "EDMD", "JDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "airplane_error_by_num_train.tikz"), p_err, 
    include_preamble=false)

#############################################
## Waypoints plot (Fig 1)
#############################################
# NOTE: This won't work unless you have the "piper" model in the data folder
#       Not shipped with supplementary materials because the size is too large
using Colors, MeshCat
airplane_data = load(AIRPLANE_DATAFILE)
num_train = size(airplane_data["X_train"],2)
X_ref0 = airplane_data["X_ref"][:,num_train+1:end]

i = 5
X_ref = X_ref0[:,i]
model_nom = BilinearControl.NominalAirplane()
delete!(vis["robot"])
waypoints!(vis, model_nom, X_ref, vis->set_airplane!(vis, model_nom), [1,5,10,15,20,25,30,35,40,51])
setprop!(vis["/Background"], "top_color", colorant"rgb(255,255,255)")
setprop!(vis["/Background"], "bottom_color", colorant"rgb(255,255,255)")