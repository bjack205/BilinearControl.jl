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

##
using PGFPlotsX
results_cartpole = load(CARTPOLE_RESULTS_FILE)["results"]
fields = keys(results_cartpole[1])
res_cartpole = Dict(Pair.(fields, map(x->getfield.(results_cartpole, x), fields)))

good_inds = 1:18

samples_cartpole = res_cartpole[:nsamples][good_inds] / 100
nom_err_avg_cartpole = res_cartpole[:nom_err_avg][good_inds]
eDMD_err_avg_cartpole = res_cartpole[:eDMD_err_avg][good_inds]
jDMD_err_avg_cartpole = res_cartpole[:jDMD_err_avg][good_inds]

##
results_planar_quad = load(REX_PLANAR_QUADROTOR_RESULTS_FILE)["results"]
fields = keys(results_planar_quad[1])
res_planar_quad = Dict(Pair.(fields, map(x->getfield.(results_planar_quad, x), fields)))

good_inds = 1:length(res_planar_quad[:nsamples])

samples_planar_quad = res_planar_quad[:nsamples][good_inds] / 100
nom_err_avg_planar_quad = res_planar_quad[:nom_err_avg][good_inds]
eDMD_err_avg_planar_quad = res_planar_quad[:eDMD_err_avg][good_inds]
jDMD_err_avg_planar_quad = res_planar_quad[:jDMD_err_avg][good_inds]

## 
p = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Number of trajectories",
        ylabel = "Tracking error",
        ymax=2,
    },
    PlotInc({lineopts..., color=color_nominal}, Coordinates(samples_cartpole, nom_err_avg_cartpole ./ median(nom_err_avg_cartpole))),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(samples_cartpole, eDMD_err_avg_cartpole ./ median(nom_err_avg_cartpole))),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(samples_cartpole, jDMD_err_avg_cartpole ./ median(nom_err_avg_cartpole))),

    PlotInc({lineopts..., color=color_nominal}, Coordinates(samples_planar_quad, nom_err_avg_planar_quad ./ median(nom_err_avg_planar_quad))),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(samples_planar_quad, eDMD_err_avg_planar_quad ./ median(nom_err_avg_planar_quad))),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(samples_planar_quad, jDMD_err_avg_planar_quad ./ median(nom_err_avg_planar_quad))),

    Legend(["Nominal MPC (Cartpole)", "Nominal MPC (Planar Quad)", "eDMD (Cartpole)", "eDMD (Planar Quad)", 
        "jDMD (Cartpole)", "jDMD (Planar Quad)"])
)

pgfsave(joinpath(Problems.FIGDIR, "combined_mpc_test_error.tikz"), p, include_preamble=false)