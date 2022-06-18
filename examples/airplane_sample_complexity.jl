using Pkg; Pkg.activate(joinpath(@__DIR__));
Pkg.instantiate();
using BilinearControl
using BilinearControl.Problems
using BilinearControl.EDMD
using Rotations
using StaticArrays
using Test
using LinearAlgebra 
using Altro
using RobotDynamics
using TrajectoryOptimization
const TO = TrajectoryOptimization
import RobotDynamics as RD
using BilinearControl: Problems
using JLD2
using Plots
using ProgressMeter
using Statistics
using PGFPlotsX

include("airplane_constants.jl")
include("constants.jl")

##
num_train = [2; 5:5:50]
results = @showprogress map(num_train) do N
    test_airplane(train_airplane(N)...)
end
jldsave(AIRPLANE_RESULTS; results, num_train)

##
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

p_sr = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Number of Training Trajectories",
        ylabel = "Success Rate",
    },
    PlotInc({lineopts..., color=color_nominal}, Coordinates(num_train, sr_nom)),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(num_train, sr_eDMD)),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(num_train, sr_jDMD)),
    Legend(["Nominal MPC", "eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "airplane_success_by_num_train.tikz"), p_sr, 
    include_preamble=false)
