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
using Test
using PGFPlotsX
using Statistics
using LaTeXStrings

##
include("constants.jl")
const REX_PLANAR_QUADROTOR_RESULTS_FILE = joinpath(Problems.DATADIR, "rex_planar_quadrotor_mpc_results.jld2")
const CARTPOLE_RESULTS_FILE = joinpath(Problems.DATADIR, "cartpole_results.jld2")

## Cartpole
using PGFPlotsX
results_cartpole = load(CARTPOLE_RESULTS_FILE)["results"]
fields = keys(results_cartpole[1])
res_cartpole = Dict(Pair.(fields, map(x->getfield.(results_cartpole, x), fields)))

num_traj_cartpole = res_cartpole[:num_swingup]
good_inds = 1:length(num_traj_cartpole)
nom_err_avg_cartpole = res_cartpole[:nom_err_avg]
eDMD_err_avg_cartpole = res_cartpole[:eDMD_err_avg]
jDMD_err_avg_cartpole = res_cartpole[:jDMD_err_avg]

# Quadrotor
results_planar_quad = load(REX_PLANAR_QUADROTOR_RESULTS_FILE)["results"]
fields = keys(results_planar_quad[1])
res_planar_quad = Dict(Pair.(fields, map(x->getfield.(results_planar_quad, x), fields)))

num_traj_planar_quad = res_planar_quad[:num_mpc][good_inds]
nom_err_avg_planar_quad = res_planar_quad[:nom_err_avg][good_inds]
eDMD_err_avg_planar_quad = res_planar_quad[:eDMD_err_avg][good_inds]
jDMD_err_avg_planar_quad = res_planar_quad[:jDMD_err_avg][good_inds]

# renormalize based on nominal MPC error
eDMD_err_avg_cartpole ./= mean(nom_err_avg_cartpole) 
jDMD_err_avg_cartpole ./= mean(nom_err_avg_cartpole) 
nom_err_avg_cartpole ./= mean(nom_err_avg_cartpole) 

eDMD_err_avg_planar_quad ./= mean(nom_err_avg_planar_quad) 
jDMD_err_avg_planar_quad ./= mean(nom_err_avg_planar_quad) 
nom_err_avg_planar_quad ./= mean(nom_err_avg_planar_quad)

## 
p = @pgf Axis(
    {
        xmajorgrids, ymajorgrids,
        xlabel = "Number of trajectories",
        ylabel = "Tracking error",
        ymax=3,
        # legend_pos = "north east",
        legend_style="{at={(0.5,-0.1)},anchor=north}",
        legend_columns=2,
    },
    PlotInc({lineopts..., color=color_nominal}, Coordinates(num_traj_cartpole, nom_err_avg_cartpole)),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(num_traj_cartpole, eDMD_err_avg_cartpole)),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(num_traj_cartpole, jDMD_err_avg_cartpole)),

    # PlotInc({lineopts..., color=color_nominal, style ="{dashed}"}, Coordinates(num_traj_planar_quad, nom_err_avg_planar_quad))
    PlotInc({lineopts..., color=color_eDMD, style ="{dashed}"}, Coordinates(num_traj_planar_quad, eDMD_err_avg_planar_quad)),
    PlotInc({lineopts..., color=color_jDMD, style ="{dashed}"}, Coordinates(num_traj_planar_quad, jDMD_err_avg_planar_quad)),

    # Legend(["Nominal MPC", "eDMD (Cartpole)", "jDMD (Cartpole)", 
    #         "eDMD (Planar Quad)", "jDMD (Planar Quad)"])
)

pgfsave(joinpath(Problems.FIGDIR, "combined_mpc_test_error.tikz"), p, include_preamble=false)