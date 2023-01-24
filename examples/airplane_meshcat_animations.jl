import Pkg; Pkg.activate(joinpath(@__DIR__)); Pkg.instantiate();
using BilinearControl
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
using Rotations
using RobotDynamics
using GeometryBasics, CoordinateTransformations
using Colors
using MeshCat
using ProgressMeter
using Statistics

include("airplane/airplane_utils.jl")
include("plotting_constants.jl")

#############################################
## Functions
#############################################

function traj3!(vis, X::AbstractVector{<:AbstractVector}; inds=SA[1,2,3], kwargs...)
    pts = [Point{3,Float32}(x[inds]) for x in X] 
    setobject!(vis[:traj], MeshCat.Line(pts, LineBasicMaterial(; linewidth=50.0, kwargs...)))
end

function visualize_multiple(vis1, vis2, vis3, model, tf, X1, X2, X3)
    N = length(X1)
    fps = Int(floor((N-1)/tf))
    anim = MeshCat.Animation(fps)
    for k = 1:N
        atframe(anim, k) do 
            visualize!(vis1, model, X1[k])
            visualize!(vis2, model, X2[k])
            visualize!(vis3, model, X3[k])
        end
    end
    setanimation!(vis, anim)
end

#############################################
## Load data and models
#############################################

# gen_airplane_data(num_train=50, num_test=50, dt=0.04, dp_window=fill(0.5, 3))

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

model_eDMD, model_jDMD = train_airplane(5)

# Analytical models
model_nom = BilinearControl.NominalAirplane()
model_real = BilinearControl.SimulatedAirplane()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

# Projected models
model_eDMD_projected = BilinearControl.ProjectedEDMDModel(model_eDMD)
model_jDMD_projected = BilinearControl.ProjectedEDMDModel(model_jDMD)

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

i = 1
X_ref = X_ref0[:,i]
U_ref = U_ref0[:,i]

mpc_nom = BilinearControl.LinearMPC(dmodel_nom, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
    xmin,xmax,umin,umax
)
mpc_eDMD = BilinearControl.LinearMPC(model_eDMD_projected, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
    xmin,xmax,umin,umax
)
mpc_jDMD = BilinearControl.LinearMPC(model_jDMD_projected, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
    xmin,xmax,umin,umax
)

X_nom,  = simulatewithcontroller(dmodel_real, mpc_nom,  X_ref[1], t_ref, dt)
X_eDMD, = simulatewithcontroller(dmodel_real, mpc_eDMD, X_ref[1], t_ref, dt)
X_jDMD, = simulatewithcontroller(dmodel_real, mpc_jDMD, X_ref[1], t_ref, dt)

#############################################
## Visualize trajectories
#############################################

model = BilinearControl.NominalAirplane()
include(joinpath(BilinearControl.VISDIR, "visualization.jl"))
vis = Visualizer()
delete!(vis)
render(vis)

# setprop!(vis["/Background"], "top_color", colorant"rgb(255,255,255)")
# setprop!(vis["/Background"], "bottom_color", colorant"rgb(255,255,255)")

setprop!(vis["/Background"], "top_color", colorant"#262626")
setprop!(vis["/Background"], "bottom_color", colorant"#262626")

ref_color = colorant"rgb(204,0,43)";
nominal_color = colorant"rgb(255,255,255)";
edmd_color = colorant"rgb(255,173,0)";
jdmd_color = colorant"rgb(0,193,208)";

##
traj3!(vis["ref_traj"], X_ref; color=ref_color)
set_airplane!(vis["ref_air"], model, color=ref_color)
visualize!(vis["ref_air"], model, X_ref[end])

traj3!(vis["nom_traj"], X_nom; color=nominal_color)
set_airplane!(vis["nominal_air"], model, color=nominal_color)
visualize!(vis["nominal_air"], model, X_nom[1])
#visualize!(vis["nominal_air"], model, t_ref, X_nom)

traj3!(vis["eDMD_traj"], X_eDMD; color=edmd_color)
set_airplane!(vis["eDMD_air"], model, color=edmd_color)
visualize!(vis["eDMD_air"], model, X_eDMD[1])
# visualize!(vis["eDMD_air"], model, t_ref, X_eDMD)

traj3!(vis["jDMD_traj"], X_jDMD; color=jdmd_color)
set_airplane!(vis["jDMD_air"], model, color=jdmd_color)
visualize!(vis["jDMD_air"], model, X_jDMD[1])
# visualize!(vis["jDMD_air"], model, t_ref, X_jDMD)

#############################################
## Visualize trajectories at same time
#############################################

visualize_multiple(vis["eDMD_air"], vis["jDMD_air"], vis["nominal_air"],
            model, t_ref, X_eDMD, X_jDMD, X_nom)