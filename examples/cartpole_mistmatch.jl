using Pkg: Pkg;
Pkg.activate(joinpath(@__DIR__));
Pkg.instantiate();
using BilinearControl
using BilinearControl.Problems
using BilinearControl.EDMD
import RobotDynamics as RD
using LinearAlgebra
using RobotZoo
using JLD2
using SparseArrays
using Plots
using Distributions
using Distributions: Normal
using Random
using FiniteDiff, ForwardDiff
using StaticArrays
using Test
import TrajectoryOptimization as TO
using Altro
import BilinearControl.Problems
using Test
using ProgressMeter

include("constants.jl")
include("cartpole_utils.jl")
const CARTPOLE_MISMATCH_RESULTS = joinpath(
    Problems.DATADIR, "cartpole_mismatch_results.jld2"
)

#############################################
## Min Trajectories to Stabilize
#############################################

##
test_window_ratio = 0.5
reg = 1e-4
num_train = [2:25; 25:5:100]
mu_vals = 0:0.1:0.6
num_test = 50
repeats_required = 4

# Test jDMD with alpha = 0.01
α = 0.01
@time res_jDMD = find_min_sample_to_stabilize(
    mu_vals, num_train; num_test, alg=:jDMD, test_window_ratio, reg, α,
    repeats_required
)

# Test jDMD with alpha = 0.5
α = 0.5
@time res_jDMD_2 = find_min_sample_to_stabilize(
    mu_vals, num_train; num_test, alg=:jDMD, test_window_ratio, reg, α,
    repeats_required
)

# Test jDMD with alpha = 0.1
α = 0.1
@time res_jDMD_3 = find_min_sample_to_stabilize(
    mu_vals, num_train; num_test, alg=:jDMD, test_window_ratio, reg, α,
    repeats_required
)

# Test eDMD
@time res_eDMD = find_min_sample_to_stabilize(
    mu_vals, num_train; num_test, alg=:eDMD, test_window_ratio, reg=1e-6, α,
    repeats_required
)
res_eDMD

res_jDMD_all = [
    merge(res_jDMD, Dict(:α => 0.01)),
    merge(res_jDMD_3, Dict(:α => 0.1)),
    merge(res_jDMD_2, Dict(:α => 0.5)),
]
jldsave(CARTPOLE_MISMATCH_RESULTS; res_jDMD=res_jDMD_all, res_eDMD, mu_vals, num_train)

## Plot the results
using PGFPlotsX
using LaTeXStrings
using Colors
results_mismatch = load(CARTPOLE_MISMATCH_RESULTS)

mu_vals = results_mismatch["mu_vals"]
num_train = results_mismatch["num_train"]
setzerotonan(x) = x ≈ zero(x) ? 0 : x
getsamples(d) = map(k -> setzerotonan(getindex(d, k)), mu_vals)

res_jDMD_0p5 = getsamples(results_mismatch["res_jDMD"][3])
res_jDMD_0p1 = getsamples(results_mismatch["res_jDMD"][2])
res_jDMD_0p01 = getsamples(results_mismatch["res_jDMD"][1])
res_eDMD = getsamples(results_mismatch["res_eDMD"])

p_bar = @pgf Axis(
    {
        # height = "5cm",
        # width = "5in",
        bar_width = "5pt",
        # reverse_legend,
        ybar,
        ymax = 22,
        legend_pos = "outer north east",
        ylabel = "Training trajectories to Stabilize",
        xlabel = "Coloumb Friction Coefficient",
        nodes_near_coords,
        legend_cell_align = ["left"],
    },
    PlotInc({no_marks, color = colorant"forestgreen"}, Coordinates(mu_vals, res_jDMD_0p5)),
    PlotInc({no_marks, color = colorant"purple"}, Coordinates(mu_vals, res_jDMD_0p1)),
    PlotInc({no_marks, color = color_jDMD}, Coordinates(mu_vals, res_jDMD_0p01)),
    PlotInc({no_marks, color = color_eDMD}, Coordinates(mu_vals, res_eDMD)),
    # PlotInc({no_marks, color=color_eDMD}, Coordinates([eDMD_projected_samples,eDMD_samples], [0,1])),
    Legend([
        "jDMD " * L"(\alpha=0.5)",
        "jDMD " * L"(\alpha=0.1)",
        "jDMD " * L"(\alpha=0.01)",
        "eDMD",
    ]),
)
pgfsave(
    joinpath(Problems.FIGDIR, "cartpole_friction_mismatch.tikz"),
    p_bar;
    include_preamble=false,
)

#############################################
## Min Trajectories to beat MPC 
#############################################

dt = 0.05
num_train = 2:15 
repeats_required = 4

# stabilized with 2
res_jDMD = find_min_sample_to_beat_mpc(2:10, dt; alg=:jDMD, lifted=false, 
    repeats_required, α=0.1
)

# stabilized with 15 @ alpha = 0.1
res_jDMD_lifted = find_min_sample_to_beat_mpc(2:1:100, dt; alg=:jDMD, lifted=true, 
    repeats_required, α=0.1
)

# Stabilized with 18 samples
# - bumping up regularization to 1e-4 doesn't improve it
res_eDMD = find_min_sample_to_beat_mpc(2:40, dt; alg=:eDMD, lifted=false, repeats_required=4)

# Stabilized with 17 
res_eDMD_lifted = find_min_sample_to_beat_mpc(2:60, dt; alg=:eDMD, lifted=true, 
    repeats_required=4,
)

res_jDMD = 2
res_jDMD_lifted = 15
res_eDMD = 18
res_eDMD_lifted = 17

# Generate plot
p_bar = @pgf Axis(
    {
        reverse_legend,
        # width="4in",
        # height="4cm",
        xbar,
        ytick="data",
        yticklabels={"Projected", "Lifted"},
        xmax=27,
        enlarge_y_limits = 0.7,
        legend_pos = "south east",
        xlabel = "Training trajectories Rqd to Beat Nominal MPC",
        nodes_near_coords
    },
    PlotInc({no_marks, color=color_jDMD}, Coordinates([res_jDMD,res_jDMD_lifted], [0,1])),
    PlotInc({no_marks, color=color_eDMD}, Coordinates([res_eDMD,res_eDMD_lifted], [0,1])),
    Legend(["jDMD", "eDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "cartpole_lqr_samples.tikz"), p_bar, include_preamble=false)

    eigfuns = ["state", "sine", "cosine", "sine", "sine", "chebyshev"]
    eigorders = [[0], [1], [1], [2], [4], [2, 4]]
