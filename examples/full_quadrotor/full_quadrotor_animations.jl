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
import TrajectoryOptimization as TO
using Altro
import BilinearControl: orientation, translation
using Test
using Rotations
using RobotDynamics
using GeometryBasics, CoordinateTransformations
using Colors
using MeshCat

const QUADROTOR_RESULTS_FILE = joinpath(BilinearControl.DATADIR, "rex_full_quadrotor_mpc_results.jld2")
const QUADROTOR_MODELS_FILE = joinpath(BilinearControl.DATADIR, "rex_full_quadrotor_mpc_models.jld2")

#############################################
## Functions
#############################################

function set_quadrotor!(vis, model::L;
    scaling=1.0, color=colorant"black"
    ) where {L <: Union{RobotZoo.Quadrotor, RobotZoo.PlanarQuadrotor, BilinearControl.RexQuadrotor, BilinearControl.RexPlanarQuadrotor}}
     
    # urdf_folder = @__DIR__
    # if scaling != 1.0
    #     quad_scaling = 0.085 * scaling
    # obj = joinpath(urdf_folder, "quadrotor_scaled.obj")
    obj = joinpath("/Users/jeonghun/Scratch/Research/Robot_Meshes/quadrotor/", "drone.obj")
    if scaling != 1.0
        error("Scaling not implemented after switching to MeshCat 0.12")
    end
    robot_obj = MeshFileGeometry(obj)
    mat = MeshPhongMaterial(color=color)
    setobject!(vis["robot"]["geom"], robot_obj, mat)
    settransform!(vis["robot"]["geom"], LinearMap(RotXYZ(pi/2, 0, 0)))
end

function visualize!(vis, model::L, x::AbstractVector) where {L <: BilinearControl.RexQuadrotor}
    px, py,pz = x[1], x[2], x[3]
    rx, ry, rz = x[4], x[5], x[6]
    settransform!(vis, compose(Translation(px,py,pz), LinearMap(MRP(rx, ry, rz))))
end

function visualize!(vis, model, tf, X)
    N = length(X)
    fps = Int(floor((N-1)/tf))
    anim = MeshCat.Animation(fps)
    for k = 1:N
        atframe(anim, k) do 
            visualize!(vis, model, X[k])
        end
    end
    setanimation!(vis, anim)
end

function waypoints!(vis, model::L, Z::AbstractVector;
    interval=20, color=nothing
    ) where {L <: Union{RobotZoo.Quadrotor, RobotZoo.PlanarQuadrotor, BilinearControl.RexQuadrotor, BilinearControl.RexPlanarQuadrotor}}
    
    # N = size(Z,1)
    # if length > 0 && isempty(inds)
    #     inds = Int.(round.(range(1,N,length=length)))
    # elseif !isempty(inds) && length == 0
    #     length = size(inds,1)
    # else
    #     throw(ArgumentError("Have to pass either length or inds, but not both"))
    # end
    # if !isnothing(color)
    #     if isnothing(color_end)
    #         color_end = color
    #     end
    #     colors = range(color, color_end, length=size(inds,1))
    #     # set_quadrotor!(vis, model, color=colors[end])
    # end

    inds = Int.(round.(range(1,length(Z), interval)))

    delete!(vis["waypoints"])
    for i in inds
        if isnothing(color)
            set_quadrotor!(vis["waypoints"]["point$i"], model)
        else
            set_quadrotor!(vis["waypoints"]["point$i"], model, color=color)
        end
            
        visualize!(vis["waypoints"]["point$i"], model, Z[i])
    end
    visualize!(vis, model, Z[end])
end

function traj3!(vis, X::AbstractVector{<:AbstractVector}; inds=SA[1,2,3], kwargs...)
    pts = [Point{3,Float32}(x[inds]) for x in X] 
    setobject!(vis[:traj], MeshCat.Line(pts, LineBasicMaterial(; linewidth=3.0, kwargs...)))
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
## Load Results
#############################################

MPC_test_results = load(QUADROTOR_RESULTS_FILE)["MPC_test_results"]
mpc_lqr_traj = load(joinpath(BilinearControl.DATADIR, "rex_full_quadrotor_mpc_tracking_data.jld2"))

tf = mpc_lqr_traj["tf"]
dt = mpc_lqr_traj["dt"]

ref_trajectories = MPC_test_results[:X_ref]
nom_MPC_trajectories = MPC_test_results[:X_mpc_nom]
eDMD_MPC_trajectories = MPC_test_results[:X_mpc_eDMD]
jDMD_MPC_trajectories = MPC_test_results[:X_mpc_jDMD]

nom_errs  = MPC_test_results[:nom_errs]
eDMD_errs = MPC_test_results[:eDMD_errs]
jDMD_errs = MPC_test_results[:jDMD_errs]

T_mpc_vecs = MPC_test_results[:T_mpc]

#############################################
## Visualize a trajectory
#############################################

model = BilinearControl.RexQuadrotor()
vis = Visualizer()
delete!(vis)
render(vis)

## Some cool trajectories: 38, 45, 46, 47, 48
i = 46

ref = ref_trajectories[i]
nom_MPC = nom_MPC_trajectories[i]
eDMD_MPC = eDMD_MPC_trajectories[i]
jDMD_MPC = jDMD_MPC_trajectories[i]
T_mpc = T_mpc_vecs[i]
set_quadrotor!(vis, model, color=colorant"rgb(204,0,43)")

visualize!(vis, model, T_mpc[end], jDMD_MPC)

println("")
println("Traj #$i Summary:")
println("  Model  |  Tracking Error ")
println("---------|-------------------")
println(" nom MPC |  ", MPC_test_results[:nom_errs][i])
println("  eDMD   |  ", MPC_test_results[:eDMD_errs][i])
println("  jDMD   |  ", MPC_test_results[:jDMD_errs][i])
println("")

#############################################
## Plot layout of all initial conditions
#############################################

model = BilinearControl.RexQuadrotor()
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

set_quadrotor!(vis, model, color=ref_color)

i = 1
for ref in ref_trajectories
    set_quadrotor!(vis["ref_start"]["$i"], model, color=nominal_color)
    # set_quadrotor!(vis["ref_start"]["$i"], model)
    visualize!(vis["ref_start"]["$i"], model, ref[1])
    traj3!(vis["ref_traj"]["$i"], ref; color=ref_color)
    i+=1
end

#############################################
## Trajectory plotting with waypoints
#############################################

model = BilinearControl.RexQuadrotor()
vis = Visualizer()
delete!(vis)
render(vis)

# setprop!(vis["/Background"], "top_color", colorant"rgb(255,255,255)")
# setprop!(vis["/Background"], "bottom_color", colorant"rgb(255,255,255)")

setprop!(vis["/Background"], "top_color", colorant"#262626")
setprop!(vis["/Background"], "bottom_color", colorant"#262626")

set_quadrotor!(vis, model, color=colorant"rgb(204,0,43)")

ref_color = colorant"rgb(204,0,43)";
nominal_color = colorant"rgb(255,255,255)";
edmd_color = colorant"rgb(255,173,0)";
jdmd_color = colorant"rgb(0,193,208)";

i = 46
ref = ref_trajectories[i]
nom_MPC = nom_MPC_trajectories[i]
eDMD_MPC = eDMD_MPC_trajectories[i]
jDMD_MPC = jDMD_MPC_trajectories[i]
T_mpc = T_mpc_vecs[i]

traj3!(vis["ref_traj"]["$i"], ref; color=ref_color)
traj3!(vis["nom_traj"]["$i"], nom_MPC; color=nominal_color)
traj3!(vis["eDMD_traj"]["$i"], eDMD_MPC[1:24]; color=edmd_color)
traj3!(vis["jDMD_traj"]["$i"], jDMD_MPC; color=jdmd_color)

waypoints!(vis, model, jDMD_MPC; color=colorant"rgb(70,70,70)", interval=15)

#############################################
## Trajectory plotting with moving quad
#############################################

model = BilinearControl.RexQuadrotor()
vis = Visualizer()
delete!(vis)
render(vis)

setprop!(vis["/Background"], "top_color", colorant"#262626")
setprop!(vis["/Background"], "bottom_color", colorant"#262626")

# setprop!(vis["/Background"], "top_color", colorant"rgb(255,255,255)")
# setprop!(vis["/Background"], "bottom_color", colorant"rgb(255,255,255)")

##
set_quadrotor!(vis, model, color=colorant"rgb(204,0,43)")

i = 46
ref = ref_trajectories[i]
nom_MPC = nom_MPC_trajectories[i]
eDMD_MPC = eDMD_MPC_trajectories[i]
jDMD_MPC = jDMD_MPC_trajectories[i]
T_mpc = T_mpc_vecs[i]

ref_color = colorant"rgb(204,0,43)";
nominal_color = colorant"rgb(255,255,255)";
edmd_color = colorant"rgb(255,173,0)";
jdmd_color = colorant"rgb(0,193,208)";

traj3!(vis["ref_traj"]["$i"], ref; color=ref_color)
set_quadrotor!(vis["ref_quad"]["$i"], model, color=ref_color)
visualize!(vis["ref_quad"]["$i"], model, ref[end])
# visualize!(vis["ref_quad"]["$i"], model, 5.0, ref)

traj3!(vis["nom_traj"]["$i"], nom_MPC; color=nominal_color)
set_quadrotor!(vis["nominal_quad"]["$i"], model, color=nominal_color)
visualize!(vis["nominal_quad"]["$i"], model, nom_MPC[1])
# visualize!(vis["nominal_quad"]["$i"], model, T_mpc[end], nom_MPC)

traj3!(vis["eDMD_traj"]["$i"], eDMD_MPC[1:24]; color=edmd_color)
set_quadrotor!(vis["eDMD_quad"]["$i"], model, color=edmd_color)
visualize!(vis["eDMD_quad"]["$i"], model, eDMD_MPC[1])
# visualize!(vis["eDMD_quad"]["$i"], model, T_mpc[end], eDMD_MPC)

traj3!(vis["jDMD_traj"]["$i"], jDMD_MPC; color=jdmd_color)
set_quadrotor!(vis["jDMD_quad"]["$i"], model, color=jdmd_color)
visualize!(vis["jDMD_quad"]["$i"], model, jDMD_MPC[1])
# visualize!(vis["jDMD_quad"]["$i"], model, T_mpc[end], jDMD_MPC)

## multiple trajectories

visualize_multiple(vis["eDMD_quad"]["$i"], vis["jDMD_quad"]["$i"], vis["nominal_quad"]["$i"],
            model, T_mpc[end], eDMD_MPC, jDMD_MPC, nom_MPC)