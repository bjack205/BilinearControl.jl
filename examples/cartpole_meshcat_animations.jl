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
using TrajectoryOptimization
const TO = TrajectoryOptimization
using Altro
using BilinearControl: Problems
using Rotations
using RobotDynamics
using GeometryBasics, CoordinateTransformations
using Colors
using MeshCat
using ProgressMeter
using Statistics

include("constants.jl")

#############################################
## Load reference trajectory
#############################################

cartpole_traj = load(joinpath(Problems.DATADIR, "cartpole_swingup_data.jld2"))
X_ref = cartpole_traj["X_ref"]
t_ref = 5.0

#############################################
## Visualize trajectory
#############################################

model = RobotZoo.Cartpole()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
delete!(vis)
render(vis)

setprop!(vis["/Background"], "top_color", colorant"rgb(255,255,255)")
setprop!(vis["/Background"], "bottom_color", colorant"rgb(255,255,255)")

set_cartpole!(vis)

##
visualize!(vis, model, X_ref[1, 1])
visualize!(vis, model, 5.0, X_ref[:, 1])