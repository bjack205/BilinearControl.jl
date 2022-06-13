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

# Params
μ = 0.1
t_sim = 4.0
num_train = 20
num_test = 10
err_thresh = 0.1
alg = :eDMD
α = 1e-1
reg = 1e-6

## Define the models
model_nom = Problems.NominalCartpole()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

# Define Mismatched "Real" Cartpole Model
model_real = Problems.SimulatedCartpole(;μ=μ) # this model has damping
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

# Generate data with the new damping term
X_train, U_train,_,_,_,_,metadata = generate_cartpole_data(
    save_to_file=false, num_swingup=0, num_lqr=num_train,
    μ=μ, μ_nom=μ, max_lqr_samples=600
)
dt = metadata.dt

# Train new model
eigfuns = ["state", "sine", "cosine", "sine"]
eigorders = [[0],[1],[1],[2],[4],[2, 4]]

model = let alg=alg, α=α, reg=reg
    if alg == :eDMD
        model_eDMD = run_eDMD(X_train, U_train, dt, eigfuns, eigorders, 
            reg=reg, name="cartpole_eDMD", alg=:qr_rls)
    elseif alg == :jDMD
        model_jDMD = run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom;
            reg=reg, name="cartpole_jDMD", learnB=true, α=α)
    end
end

# Generate an MPC controller
# Reference Trajectory
mpc = let model=model
    Nt = 21
    xe = [0,pi,0,0]
    ue = [0.0]
    T_ref = range(0,t_sim,step=dt)
    X_ref = [copy(xe) for t in T_sim]
    U_ref = [copy(ue) for t in T_sim]
    Qmpc = Diagonal(fill(1e-0,4))
    Rmpc = Diagonal(fill(1e-3,1))
    Qfmpc = Diagonal([1e2,1e2,1e1,1e1])
    model_projected = EDMD.ProjectedEDMDModel(model)
    TrackingMPC(model_projected, 
        X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
    )
end

# Test mpc controller
sr, ae = let mpc=mpc, num_test=num_test, model_real=dmodel_real, t_sim=t_sim, 
             err_thresh=err_thresh, xg=xe, dt=dt
    # Set seed so that all are tested on the same conditions
    Random.seed!(100) 

    # Generate initial conditions to test
    x0_sampler = Product([
        Uniform(-1.0,1.0),
        Uniform(pi-deg2rad(30),pi+deg2rad(30)),
        Uniform(-.5,.5),
        Uniform(-.5,.5),
    ])
    x0_test = [rand(x0_sampler) for i = 1:num_test]

    # Run controller for each initial condition
    errors = map(x0_test) do x0
        X_sim, = simulatewithcontroller(model_real, mpc, x0, t_sim, dt)
        norm(X_sim[end] - xg)
    end
    average_error = mean(filter(x->x<err_thresh, errors))
    success_rate = count(x->x<err_thresh, errors) / num_test
    return success_rate, average_error 
end