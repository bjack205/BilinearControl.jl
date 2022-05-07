import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using BilinearControl
using BilinearControl.Problems
using BilinearControl.EDMD
import RobotDynamics as RD
import TrajectoryOptimization as TO
using RobotZoo
using LinearAlgebra
using StaticArrays
using SparseArrays
# using MeshCat, GeometryBasics, Colors, CoordinateTransformations, Rotations
using Plots
using Distributions
using Distributions: Normal
using Random
using JLD2
using Altro
using BilinearControl: Problems
using QDLDL
using Test

include("edmd_utils.jl")
include("cartpole_model.jl")

## Visualization 
visdir = Problems.VISDIR
include(joinpath(visdir, "visualization.jl"))
vis = Visualizer()
open(vis)
delete!(vis)
set_cartpole!(vis)

## Simulate the two models
model0 = RobotZoo.Cartpole()
b = 0.02  # damping 
model_nominal = RD.DiscretizedDynamics{RD.RK4}(model0)
model_true = RD.DiscretizedDynamics{RD.RK4}(Cartpole2(model0.mc, model0.mp, model0.l, model0.g, b))
n,m = RD.dims(model_nominal)
dt = 0.01

x0 = [0, pi-0.1, 0.1, 0.0]
t_sim = 4.0
times_sim = range(0,t_sim,step=dt)
U = [zeros(n) for t in times_sim]

Xsim_nom = simulate(model_nominal, U, x0, t_sim, dt)
Xsim_tru = simulate(model_true, U, x0, t_sim, dt)

plot(times_sim, reduce(hcat, Xsim_nom)[1:2,:]', c=[1 2], label="nominal")
plot!(times_sim, reduce(hcat, Xsim_tru)[1:2,:]', c=[1 2], s=:dash, label="true")

visualize!(vis, RobotZoo.Cartpole(), t_sim, Xsim_nom)
visualize!(vis, RobotZoo.Cartpole(), t_sim, Xsim_tru)

## Generate some trajectories from the "true" model
Random.seed!(1)
num_lqr = 400
Qlqr = Diagonal([0.2,10,1e-2,1e-2])
Rlqr = Diagonal([1e-3])
xe = [0,pi,0,0]
ue = [0.0]
ctrl_lqr = LQRController(model_nominal, Qlqr, Rlqr, xe, ue, dt)  # generate controller using nominal model
x0_sampler = Product([
    Uniform(-1.0,1.0),
    Uniform(pi-pi/3,pi+pi/3),
    Uniform(-.5,.5),
    Uniform(-.5,.5),
])
initial_conditions_lqr = [rand(x0_sampler) for _ in 1:num_lqr]

# Generate data using "true" model
t_train = 2.0
X_train_lqr, U_train_lqr = create_data(model_true, ctrl_lqr, initial_conditions_lqr, t_train, dt)
visualize!(vis, model0, t_train, X_train_lqr[:,4])

# Calculate "error" output
X_err = calc_error(model_nominal, X_train_lqr, U_train_lqr, dt)

