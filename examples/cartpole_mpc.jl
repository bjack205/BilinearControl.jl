"""
    cartpole_mpc.jl

This script was used to generated Figure 2a in the paper
"""
import Pkg; Pkg.activate(joinpath(@__DIR__)); Pkg.instantiate();
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

# include("learned_models/edmd_utils.jl")
include("constants.jl")
include("cartpole_utils.jl")
const CARTPOLE_RESULTS_FILE = joinpath(Problems.DATADIR, "cartpole_results.jld2")

## Test MPC Controller with a given number of training samples
generate_cartpole_data()  # OPTIONAL (data file already exists)
train_cartpole_models(0,2, α=0.5, β=1.0, learnB=true, reg=1e-4)  # run once for JIT
num_swingup = 2:2:36
results = map(num_swingup) do N
    println("\nRunning with N = $N")
    res = train_cartpole_models(0,N, α=0.5, β=1.0, learnB=true, reg=1e-4)
    @show res.jDMD_err_avg
    @show res.eDMD_err_avg
    res
end
jldsave(CARTPOLE_RESULTS_FILE; results)

## Process results
using PGFPlotsX
results = load(CARTPOLE_RESULTS_FILE)["results"]
fields = keys(results[1])
res = Dict(Pair.(fields, map(x->getfield.(results, x), fields)))
res
p = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Number of training samples",
        ylabel = "Training time (sec)",
        legend_pos = "north west",
    },
    PlotInc({no_marks, "very thick", "orange"}, Coordinates(res[:nsamples], res[:t_train_eDMD])),
    PlotInc({no_marks, "very thick", "cyan"}, Coordinates(res[:nsamples], res[:t_train_jDMD])),
    Legend(["eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "cartpole_mpc_train_time.tikz"), p, include_preamble=false)

p = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Number of training trajectories",
        ylabel = "Tracking error",
        ymax=0.2,
    },
    PlotInc({lineopts..., color=color_nominal}, Coordinates(res[:num_swingup], res[:nom_err_avg])),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(res[:num_swingup], res[:eDMD_err_avg])),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(res[:num_swingup], res[:jDMD_err_avg])),
    # Legend(["Nominal", "eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "cartpole_mpc_test_error.tikz"), p, include_preamble=false)
