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
import BilinearControl.Problems

include("constants.jl")
include("cartpole_utils.jl")

const CARTPOLE_RESULTS_FILE = joinpath(Problems.DATADIR, "cartpole_meshcat_results.jld2")

#############################################
## Functions
#############################################

function generate_trajectories(;
    μ=0.1,
    μ_nom=μ,
    t_sim=4.0,
    num_train=20,
    α=1e-1,
    reg=1e-6,
    x_window=[1, deg2rad(40), 0.5, 0.5],
    ρ=1e-6,
    Nt=21,
)

    ## Define the models
    model_nom = Problems.NominalCartpole()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" Cartpole Model
    model_real = Problems.SimulatedCartpole(; μ=μ) # this model has damping
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    # Generate data with the new damping term
    X_train, U_train, _, _, _, _, metadata = generate_cartpole_data(;
        save_to_file=false,
        num_swingup=0,
        num_lqr=num_train,
        μ=μ,
        μ_nom=μ_nom,
        max_lqr_samples=600,
        x_window,
    )
    dt = metadata.dt

    # Train new model
    eigfuns = ["state", "sine", "cosine", "sine", "sine", "chebyshev"]
    eigorders = [[0], [1], [1], [2], [4], [2, 4]]

    model_edmd = run_eDMD(
        X_train,
        U_train,
        dt,
        eigfuns,
        eigorders;
        reg=reg,
        name="cartpole_eDMD",
        alg=:qr_rls,
    )

    model_jdmd = run_jDMD(
        X_train,
        U_train,
        dt,
        eigfuns,
        eigorders,
        dmodel_nom;
        reg=reg,
        name="cartpole_jDMD",
        learnB=true,
        α=α,
    )

    x0 = [-0.7,pi-deg2rad(40),-1,0.5]

    # Generate an MPC controller
    model_projected = EDMD.ProjectedEDMDModel(model_edmd)
    mpc = generate_stabilizing_mpc_controller(model_projected, t_sim, dt; Nt, ρ)
    X_edmd, = simulatewithcontroller(dmodel_real, mpc, x0, t_sim, dt)

    model_projected = EDMD.ProjectedEDMDModel(model_jdmd)
    mpc = generate_stabilizing_mpc_controller(model_projected, t_sim, dt; Nt, ρ)
    X_jdmd, = simulatewithcontroller(dmodel_real, mpc, x0, t_sim, dt)

    mpc = generate_stabilizing_mpc_controller(dmodel_nom, t_sim, dt; Nt, ρ)
    X_nom, = simulatewithcontroller(dmodel_real, mpc, x0, t_sim, dt)

    return X_nom, X_edmd, X_jdmd
end

function visualize_multiple(vis1, vis2, model, tf, X1, X2, )
    N = length(X1)
    fps = Int(floor((N-1)/tf))
    anim = MeshCat.Animation(fps)
    for k = 1:N
        atframe(anim, k) do 
            visualize!(vis1, model, X1[k])
            visualize!(vis2, model, X2[k])
        end
    end
    setanimation!(vis, anim)
end

#############################################
## Load reference trajectory
#############################################

X_nom_01, X_edmd_01, X_jdmd_01 = generate_trajectories(;μ=0.1)
X_nom_05, X_edmd_05, X_jdmd_05 = generate_trajectories(;μ=0.5)

#############################################
## Visualize 0.1 coeff trajectories
#############################################

model = RobotZoo.Cartpole()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
delete!(vis)
render(vis)

setprop!(vis["/Background"], "top_color", colorant"rgb(255,255,255)")
setprop!(vis["/Background"], "bottom_color", colorant"rgb(255,255,255)")


ref_color = colorant"rgb(204,0,43)";
edmd_color = colorant"rgb(255,173,0)";
jdmd_color = colorant"rgb(0,193,208)";

##
set_cartpole!(vis["ref_cart"]; color=ref_color, color2=ref_color)
visualize!(vis["ref_cart"], model, [0,pi,0,0])

set_cartpole!(vis["edmd_cart"]; color=edmd_color, color2=edmd_color)
visualize!(vis["edmd_cart"], model, X_edmd_01[1])

set_cartpole!(vis["jdmd_cart"]; color=jdmd_color, color2=jdmd_color)
visualize!(vis["jdmd_cart"], model, X_jdmd_01[1])

##
visualize_multiple(vis["edmd_cart"], vis["jdmd_cart"],
            model, 7.0, X_edmd_01, X_jdmd_01)

# visualize!(vis["ref_cart"], model, 7.0, X_nom_01)
# visualize!(vis["edmd_cart"], model, 7.0, X_edmd_01)
# visualize!(vis["jdmd_cart"], model, 7.0, X_jdmd_01)

#############################################
## Visualize 0.5 coeff trajectories
#############################################

model = RobotZoo.Cartpole()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
delete!(vis)
render(vis)

setprop!(vis["/Background"], "top_color", colorant"rgb(255,255,255)")
setprop!(vis["/Background"], "bottom_color", colorant"rgb(255,255,255)")


##
set_cartpole!(vis["ref_cart"]; color=ref_color, color2=ref_color)
visualize!(vis["ref_cart"], model, [0,pi,0,0])

set_cartpole!(vis["edmd_cart"]; color=edmd_color, color2=edmd_color)
visualize!(vis["edmd_cart"], model, X_edmd_05[1])

set_cartpole!(vis["jdmd_cart"]; color=jdmd_color, color2=jdmd_color)
visualize!(vis["jdmd_cart"], model, X_jdmd_05[1])

##
visualize_multiple(vis["edmd_cart"], vis["jdmd_cart"],
            model, 7.0, X_edmd_05, X_jdmd_05)

# visualize!(vis["ref_cart"], model, 7.0, X_nom_05)
# visualize!(vis["edmd_cart"], model, 7.0, X_edmd_05)
# visualize!(vis["jdmd_cart"], model, 7.0, X_jdmd_05)