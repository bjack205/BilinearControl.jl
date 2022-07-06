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

include("airplane_constants.jl")
include("constants.jl")

#############################################
## Functions
#############################################

function traj3!(vis, X::AbstractVector{<:AbstractVector}; inds=SA[1,2,3], kwargs...)
    pts = [Point{3,Float32}(x[inds]) for x in X] 
    setobject!(vis[:traj], MeshCat.Line(pts, LineBasicMaterial(; linewidth=3.0, kwargs...)))
end

#############################################
## Load data and models
#############################################

airplane_data = load(AIRPLANE_DATAFILE)
X_train = airplane_data["X_train"]
U_train = airplane_data["U_train"]
X_test = airplane_data["X_test"]
U_test = airplane_data["U_test"]
num_train = size(X_train,2)
num_test =  size(X_test,2)

X_ref0 = airplane_data["X_ref"][:,num_train+1:end]
U_ref0 = airplane_data["U_ref"][:,num_train+1:end]
T_ref = airplane_data["T_ref"]
dt = T_ref[2]
t_ref = T_ref[end]

airplane_models = load(AIRPLANE_MODELFILE)

eDMD_data = airplane_models["eDMD"]
jDMD_data = airplane_models["jDMD"]

model_eDMD = EDMDModel(airplane_models["eDMD"])
model_jDMD = EDMDModel(airplane_models["jDMD"])

# Analytical models
model_nom = Problems.NominalAirplane()
model_real = Problems.SimulatedAirplane()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

# Projected models
model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)

#############################################
## Create MPC controllers
#############################################

Nt = 21
Qk = Diagonal([fill(1e0, 3); fill(1e1, 3); fill(1e-1, 3); fill(2e-1, 3)])
Rk = Diagonal(fill(1e-3,4))
Qf = Diagonal([fill(1e-2, 3); fill(1e0, 3); fill(1e1, 3); fill(1e1, 3)]) * 10
u_trim = [41.66667789082778, 105.99999999471807, 74.65179381344494, 106.00000124622453]
xmax = [fill(0.5,3); fill(1.0, 3); fill(0.5, 3); fill(10.0, 3)]
xmin = -xmax
umin = fill(0.0, 4) - u_trim
umax = fill(255.0, 4) - u_trim

#############################################
## Track a trajectory
#############################################

i = 21
X_ref = X_ref0[:,i]
U_ref = U_ref0[:,i]

mpc_nom = EDMD.LinearMPC(dmodel_nom, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
    xmin,xmax,umin,umax
)
mpc_eDMD = EDMD.LinearMPC(model_eDMD_projected, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
    xmin,xmax,umin,umax
)
mpc_jDMD = EDMD.LinearMPC(model_jDMD_projected, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
    xmin,xmax,umin,umax
)

X_nom,  = simulatewithcontroller(dmodel_real, mpc_nom,  X_ref[1], t_ref, dt)
X_eDMD, = simulatewithcontroller(dmodel_real, mpc_eDMD, X_ref[1], t_ref, dt)
X_jDMD, = simulatewithcontroller(dmodel_real, mpc_jDMD, X_ref[1], t_ref, dt)

#############################################
## Visualize trajectories
#############################################

model = Problems.NominalAirplane()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
delete!(vis)
render(vis)

setprop!(vis["/Background"], "top_color", colorant"rgb(255,255,255)")
setprop!(vis["/Background"], "bottom_color", colorant"rgb(255,255,255)")

##
traj3!(vis["ref_traj"]["$i"], X_ref; color=colorant"rgb(204,0,43)")
set_airplane!(vis["ref_air"]["$i"], model, color=colorant"rgb(204,0,43)")
visualize!(vis["ref_air"]["$i"], model, X_ref[1])
visualize!(vis["ref_air"]["$i"], model, t_ref, X_ref)

##
traj3!(vis["nom_traj"]["$i"], X_nom; color=colorant"black")
set_airplane!(vis["nominal_air"]["$i"], model, color=colorant"black")
visualize!(vis["nominal_air"]["$i"], model, X_nom[1])
visualize!(vis["nominal_air"]["$i"], model, t_ref, X_nom)

##
traj3!(vis["eDMD_traj"]["$i"], X_eDMD; color=colorant"rgb(255,173,0)")
set_airplane!(vis["eDMD_air"]["$i"], model, color=colorant"rgb(255,173,0)")
visualize!(vis["eDMD_air"]["$i"], model, X_eDMD[1])
visualize!(vis["eDMD_air"]["$i"], model, t_ref, X_eDMD)

##
traj3!(vis["jDMD_traj"]["$i"], X_jDMD; color=colorant"rgb(0,193,208)")
set_airplane!(vis["jDMD_air"]["$i"], model, color=colorant"rgb(0,193,208)")
visualize!(vis["jDMD_air"]["$i"], model, X_jDMD[1])
visualize!(vis["jDMD_air"]["$i"], model, t_ref, X_jDMD)