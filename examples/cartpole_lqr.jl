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

## Visualizer
model = RobotZoo.Cartpole()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
delete!(vis)
set_cartpole!(vis)
render(vis)

#############################################
## Define the models 
#############################################

# Nominal Simulated Cartpole Model
model_nom = RobotZoo.Cartpole(mc=1.0, mp=0.2, l=0.5)
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

# Mismatched "Real" Cartpole Model
model_real = Cartpole2(mc=1.05, mp=0.19, l=0.52, b=0.02)  # this model has damping
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)


#############################################
## Generate Training and Test Data 
#############################################
tf = 2.0
dt = 0.02

## Generate Data From Mismatched Model
Random.seed!(1)

# Number of trajectories
num_test = 50
num_train = 50

# Generate a stabilizing LQR controller about the top
Qlqr = Diagonal([0.2,10,1e-2,1e-2])
Rlqr = Diagonal([1e-3])
xe = [0,pi,0,0]
ue = [0.0]
ctrl_lqr = LQRController(dmodel_real, Qlqr, Rlqr, xe, ue, dt)

# Sample a bunch of initial conditions for the LQR controller
x0_sampler = Product([
    Uniform(-0.7,0.7),
    Uniform(pi-pi/4,pi+pi/4),
    Uniform(-.2,.2),
    Uniform(-.2,.2),
])

initial_conditions_lqr = [rand(x0_sampler) for _ in 1:num_test]
initial_conditions_test = [rand(x0_sampler) for _ in 1:num_train]

# Create data set
X_train, U_train = create_data(dmodel_real, ctrl_lqr, initial_conditions_lqr, tf, dt)
X_test, U_test = create_data(dmodel_real, ctrl_lqr, initial_conditions_test, tf, dt)

#############################################
## fit the training data
#############################################

## Define basis functions
eigfuns = ["state", "sine", "cosine", "sine", "sine", "chebyshev"]
eigorders = [[0],[1],[1],[2],[4],[2, 4]]

eigfuns = ["state", "sine", "cosine", "sine"]
eigorders = [[0],[1],[1],[2],[4],[2, 4]]

model_eDMD = run_eDMD(X_train, U_train, dt, eigfuns, eigorders, reg=1e0, name="cartpole_eDMD")
model_jDMD = run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom, reg=1e0, name="cartpole_jDMD")

EDMD.open_loop_error(model_eDMD, X_test, U_test)
EDMD.open_loop_error(model_jDMD, X_test, U_test)
BilinearControl.EDMD.fiterror(model_eDMD, X_test, U_test)
BilinearControl.EDMD.fiterror(model_jDMD, X_test, U_test)