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
import BilinearControl.Problems: orientation, translation
using Test
using Rotations
using RobotDynamics
using GeometryBasics, CoordinateTransformations
using Colors
using MeshCat

include("constants.jl")
const QUADROTOR_RESULTS_FILE = joinpath(Problems.DATADIR, "rex_full_quadrotor_mpc_results.jld2")
const QUADROTOR_MODELS_FILE = joinpath(Problems.DATADIR, "rex_full_quadrotor_mpc_models.jld2")

#############################################
## Functions
#############################################

function set_quadrotor!(vis, model::L;
    scaling=1.0, color=colorant"black"
    ) where {L <: Union{RobotZoo.Quadrotor, RobotZoo.PlanarQuadrotor, Problems.Quadrotor, Problems.RexQuadrotor, Problems.RexPlanarQuadrotor}}
     
    # urdf_folder = @__DIR__
    # if scaling != 1.0
    #     quad_scaling = 0.085 * scaling
    # obj = joinpath(urdf_folder, "quadrotor_scaled.obj")
    obj = joinpath("/home/jeonghun/Scratch/Research/Robot_Meshes/quadrotor/", "drone.obj")
    if scaling != 1.0
        error("Scaling not implemented after switching to MeshCat 0.12")
    end
    robot_obj = MeshFileGeometry(obj)
    mat = MeshPhongMaterial(color=color)
    setobject!(vis["robot"]["geom"], robot_obj, mat)
    settransform!(vis["robot"]["geom"], LinearMap(RotXYZ(pi/2, 0, 0)))
end

function visualize!(vis, model::L, x::AbstractVector) where {L <: Union{Problems.Quadrotor, Problems.RexQuadrotor}}
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
    ) where {L <: Union{RobotZoo.Quadrotor, RobotZoo.PlanarQuadrotor, Problems.Quadrotor, Problems.RexQuadrotor, Problems.RexPlanarQuadrotor}}
    
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
    setobject!(vis[:traj], MeshCat.Line(pts, LineBasicMaterial(;kwargs...)))
end

#############################################
## Load Results
#############################################

MPC_test_results = load(QUADROTOR_RESULTS_FILE)["MPC_test_results"]
mpc_lqr_traj = load(joinpath(Problems.DATADIR, "rex_full_quadrotor_mpc_tracking_data.jld2"))

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

model = Problems.RexQuadrotor()
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

model = Problems.RexQuadrotor()
vis = Visualizer()
delete!(vis)
render(vis)

setprop!(vis["/Background"], "top_color", colorant"rgb(255,255,255)")
setprop!(vis["/Background"], "bottom_color", colorant"rgb(255,255,255)")

set_quadrotor!(vis, model, color=colorant"rgb(204,0,43)")

i = 1
for ref in ref_trajectories
    set_quadrotor!(vis["ref_start"]["$i"], model, color=colorant"rgb(70,70,70)")
    # set_quadrotor!(vis["ref_start"]["$i"], model)
    visualize!(vis["ref_start"]["$i"], model, ref[1])
    traj3!(vis["ref_traj"]["$i"], ref; color=colorant"rgb(23,75,63)")
    i+=1
end

#############################################
## Trajectory plotting with waypoints
#############################################

model = Problems.RexQuadrotor()
vis = Visualizer()
delete!(vis)
render(vis)

setprop!(vis["/Background"], "top_color", colorant"rgb(255,255,255)")
setprop!(vis["/Background"], "bottom_color", colorant"rgb(255,255,255)")

set_quadrotor!(vis, model, color=colorant"rgb(204,0,43)")

i = 46
ref = ref_trajectories[i]
nom_MPC = nom_MPC_trajectories[i]
eDMD_MPC = eDMD_MPC_trajectories[i]
jDMD_MPC = jDMD_MPC_trajectories[i]
T_mpc = T_mpc_vecs[i]

traj3!(vis["ref_traj"]["$i"], ref; color=colorant"rgb(23,75,63)")
traj3!(vis["nom_traj"]["$i"], nom_MPC; color=colorant"rgb(0,193,208)")
traj3!(vis["eDMD_traj"]["$i"], eDMD_MPC[1:24]; color=colorant"rgb(255,75,63)")
traj3!(vis["jDMD_traj"]["$i"], jDMD_MPC; color=colorant"rgb(255,173,0)")

waypoints!(vis, model, jDMD_MPC; color=colorant"rgb(70,70,70)", interval=15)

#############################################
## Trajectory plotting with moving quad
#############################################

model = Problems.RexQuadrotor()
vis = Visualizer()
delete!(vis)
render(vis)

setprop!(vis["/Background"], "top_color", colorant"rgb(255,255,255)")
setprop!(vis["/Background"], "bottom_color", colorant"rgb(255,255,255)")

set_quadrotor!(vis, model, color=colorant"rgb(204,0,43)")

i = 3
ref = ref_trajectories[i]
nom_MPC = nom_MPC_trajectories[i]
eDMD_MPC = eDMD_MPC_trajectories[i]
jDMD_MPC = jDMD_MPC_trajectories[i]
T_mpc = T_mpc_vecs[i]

traj3!(vis["ref_traj"]["$i"], ref; color=colorant"rgb(23,75,63)")
traj3!(vis["nom_traj"]["$i"], nom_MPC; color=colorant"rgb(0,193,208)")
traj3!(vis["eDMD_traj"]["$i"], eDMD_MPC; color=colorant"rgb(255,75,63)")
traj3!(vis["jDMD_traj"]["$i"], jDMD_MPC; color=colorant"rgb(255,173,0)")

set_quadrotor!(vis["jDMD_quad"]["$i"], model, color=colorant"rgb(70,70,70)")
visualize!(vis["jDMD_quad"]["$i"], model, T_mpc[end], jDMD_MPC)

#############################################
## Plot tracking performance vs equilibrium change
#############################################

equilibrium_results = load(QUADROTOR_RESULTS_FILE)["equilibrium_results"]
distances = 0:0.1:2.0

fields = keys(equilibrium_results[1])
res_equilibrium = Dict(Pair.(fields, map(x->getfield.(equilibrium_results, x), fields)))

p_equilibrium_err = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Equilibirum offset",
        ylabel = "Tracking error",
        legend_pos = "north west",
        ymax = 3,
        
    },
    PlotInc({lineopts..., color="black"}, Coordinates(distances, res_equilibrium[:err_nom_MPC])),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(distances, res_equilibrium[:error_eDMD_projected])),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(distances, res_equilibrium[:error_jDMD_projected])),

    Legend(["nominal MPC", "eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "rex_full_quadrotor_mpc_error_by_equilibrium_change.tikz"), p_equilibrium_err, include_preamble=false)

p_equilibrium_err = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Equilibirum offset",
        ylabel = "Success rate",
        legend_pos = "north west",
        ymax = 1.2,
        
    },
    PlotInc({lineopts..., color="black"}, Coordinates(distances, res_equilibrium[:nom_success][2:end] ./ 10.0)),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(distances, res_equilibrium[:eDMD_success][2:end] ./ 10.0)),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(distances, res_equilibrium[:jDMD_success][2:end] ./ 10.0)),

    Legend(["nominal MPC", "eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "rex_full_quadrotor_mpc_success_rate_by_equilibrium_change.tikz"), p_equilibrium_err, include_preamble=false)

#############################################
## Plot tracking performance vs test window
#############################################

window_results = load(QUADROTOR_RESULTS_FILE)["window_results"]
percentages = 0.1:0.1:2

fields = keys(window_results[1])
window_results = Dict(Pair.(fields, map(x->getfield.(window_results, x), fields)))

p_window_error = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Training range",
        ylabel = "Tracking error",
        legend_pos = "north west",
        
    },
    PlotInc({lineopts..., color="black"}, Coordinates(percentages, window_results[:error_nom])),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(percentages, window_results[:error_eDMD])),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(percentages, window_results[:error_jDMD])),

    Legend(["nominal MPC", "eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "rex_full_quadrotor_mpc_error_by_training_range.tikz"), p_window_error, include_preamble=false)

p_window_success = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Training range",
        ylabel = "Success rate",
        legend_pos = "north east",
        ymax = 1.2,
        
    },
    PlotInc({lineopts..., color="black"}, Coordinates(percentages, window_results[:num_success_nom] ./ 50)),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(percentages, window_results[:num_success_eDMD]./ 50)),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(percentages, window_results[:num_success_jDMD]./ 50)),

    Legend(["nominal MPC", "eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "rex_full_quadrotor_success_rate_by_training_range.tikz"), p_window_success, include_preamble=false)