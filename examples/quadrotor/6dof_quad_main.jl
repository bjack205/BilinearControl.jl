using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using BilinearControl
using SparseArrays
using PGFPlotsX
using LaTeXTabulars
using RobotZoo

include("6dof_quad_utils.jl")
include("../plotting_constants.jl")


#############################################
## Generate quadrotor data
#############################################
generate_quadrotor_data()

#############################################
## Train bilinear models
#############################################

num_lqr = 10
num_mpc = 20

res = train_quadrotor_models(num_lqr, num_mpc, α=0.5, β=1.0, learnB=true, reg=1e-6)

# Save model information to file
eDMD_data = res.eDMD_data
jDMD_data = res.jDMD_data
kf = jDMD_data[:kf]
G = sparse(jDMD_data[:g])
dt = res.dt
model_info = (; eDMD_data, jDMD_data, G, kf, dt)
jldsave(FULL_QUAD_MODEL_DATA; model_info)

#############################################
## Test bilinear models
#############################################
## Run test trajectories
MPC_test_results = test_full_quadrotor()

## Print Summary
println("Test Summary:")
println("  Model  |  Success Rate ")
println("---------|-------------------")
println(" nom MPC |  ", MPC_test_results[:nom_success])
println("  eDMD   |  ", MPC_test_results[:eDMD_success])
println("  jDMD   |  ", MPC_test_results[:jDMD_success])
println("")
println("Test Summary:")
println("  Model  |  Avg Tracking Err ")
println("---------|-------------------")
println(" nom MPC |  ", MPC_test_results[:nom_err_avg])
println("  eDMD   |  ", MPC_test_results[:eDMD_err_avg])
println("  jDMD   |  ", MPC_test_results[:jDMD_err_avg])

## Save results to latex table 
MPC_test_results = load(FULL_QUAD_RESULTS)["MPC_test_results"]
latex_tabular(joinpath(BilinearControl.FIGDIR, "tables", "full_quad_mpc.tex"),
    Tabular("cccc"),
    [
        Rule(:top),
        ["", 
            "{\\color{black} \\textbf{Nominal}}",
            "{\\color{orange} \\textbf{EDMD}}",
            "{\\color{cyan} \\textbf{JDMD}}",
        ],
        Rule(:mid),
        ["Tracking Err.", 
            round(MPC_test_results[:nom_err_avg], digits=2), 
            round(MPC_test_results[:eDMD_err_avg], digits=2), 
            round(MPC_test_results[:jDMD_err_avg], digits=2), 
        ],
        ["Success Rate", 
            string(round(MPC_test_results[:nom_success] * 100, digits=2)) * "\\%", 
            string(round(MPC_test_results[:eDMD_success] * 100, digits=2)) * "\\%", 
            string(round(MPC_test_results[:jDMD_success] * 100, digits=2)) * "\\%", 
        ],
        Rule(:bottom)
    ]
)

#############################################
## Generate Images
#############################################
using GeometryBasics, CoordinateTransformations, Colors, MeshCat
const ROBOT_MESHES_DIR = joinpath(homedir(), "Code", "robot_meshes")

function BilinearControl.set_quadrotor!(vis;
    scaling=1.0, color=nothing
    ) where {L <: Union{RobotZoo.Quadrotor, RobotZoo.PlanarQuadrotor, BilinearControl.RexQuadrotor, BilinearControl.RexPlanarQuadrotor}}
     
    if isdir(ROBOT_MESHES_DIR)
        meshfile = joinpath(ROBOT_MESHES_DIR, "quadrotor", "drone.obj")
        if isnothing(color)
            obj = MeshFileObject(meshfile)
            setobject!(vis["drone"], obj)
        else
            geom = MeshFileGeometry(meshfile)
            mat = MeshPhongMaterial(color=color)
            mat = MeshLambertMaterial(color=color)
            setobject!(vis["drone"], geom, mat)
        end
        settransform!(vis["drone"], LinearMap(scaling * RotX(pi/2)))
    else
        meshfile = joinpath(BilinearControl.DATADIR, "quadrotor.obj")
        obj = MeshFileGeometry(meshfile)
        mat = MeshPhongMaterial(color=color)
        settransform!(vis, LinearMap(scaling * 0.25 * I(3)))
    end
end

## Visualizer
vis = Visualizer()
open(vis)
set_quadrotor!(vis["robot"], color=colorant"red")

## Load trajectories
MPC_test_results = load(FULL_QUAD_RESULTS)["MPC_test_results"]

ref_trajectories = MPC_test_results[:X_ref]
nom_MPC_trajectories = MPC_test_results[:X_mpc_nom]
eDMD_MPC_trajectories = MPC_test_results[:X_mpc_eDMD]
jDMD_MPC_trajectories = MPC_test_results[:X_mpc_jDMD]
T_mpc_vecs = MPC_test_results[:T_mpc]

#############################################
## Plot all Initial Conditions (Fig 4a)
#############################################
# WARNING: This is slow!
setprop!(vis["/Background"], "top_color", colorant"rgb(255,255,255)")
setprop!(vis["/Background"], "bottom_color", colorant"rgb(255,255,255)")

delete!(vis)
model = BilinearControl.NominalRexQuadrotor()
set_quadrotor!(vis["robot"], color=colorant"rgb(204,0,43)")

for i = 1:length(ref_trajectories)
    println("Visualizing Initial condition $i / $(length(ref_trajectories))")
    ref = ref_trajectories[i]
    set_quadrotor!(vis["ref_start"]["$i"], color=colorant"rgb(70,70,70)")
    # set_quadrotor!(vis["ref_start"]["$i"], model)
    visualize!(vis["ref_start"]["$i"], model, ref[1])
    traj3!(vis["ref_traj"]["$i"], ref; color=colorant"rgb(23,75,63)")
end
render(vis)

#############################################
## Trajectory plotting with waypoints
#############################################

model = Problems.RexQuadrotor()
vis = Visualizer()
delete!(vis)
open(vis)

setprop!(vis["/Background"], "top_color", colorant"rgb(255,255,255)")
setprop!(vis["/Background"], "bottom_color", colorant"rgb(255,255,255)")

set_quadrotor!(vis, color=colorant"rgb(204,0,43)")

i = 46
ref = ref_trajectories[i]
nom_MPC = nom_MPC_trajectories[i]
eDMD_MPC = eDMD_MPC_trajectories[i]
jDMD_MPC = jDMD_MPC_trajectories[i]
T_mpc = T_mpc_vecs[i]

traj3!(vis["ref_traj"]["$i"], ref; color=colorant"rgb(204,0,43)")
traj3!(vis["nom_traj"]["$i"], nom_MPC; color=colorant"black")
traj3!(vis["eDMD_traj"]["$i"], eDMD_MPC[1:24]; color=colorant"rgb(255,173,0)")
traj3!(vis["jDMD_traj"]["$i"], jDMD_MPC; color=colorant"rgb(0,193,208)")

waypoints!(vis, model, jDMD_MPC; color=colorant"rgb(70,70,70)", interval=15)
