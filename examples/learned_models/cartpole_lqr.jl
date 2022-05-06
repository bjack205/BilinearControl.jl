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
using RecipesBase

include("edmd_utils.jl")
include("cartpole_model.jl")

@userplot PlotStates 

@recipe function f(ps::PlotStates; inds=1:length(ps.args[end][1]))
    Xvec = ps.args[end]
    if length(ps.args) == 1
        times = 1:length(Xvecs)
    else
        times = ps.args[1]
    end
    Xmat = reduce(hcat,Xvec)[inds,:]'
    (times,Xmat)
end

## Visualizer
model = Cartpole2()
visdir = Problems.VISDIR
include(joinpath(visdir, "visualization.jl"))
vis = Visualizer()
open(vis)
delete!(vis)
set_cartpole!(vis)

## Generate Data 
Random.seed!(1)
model = Cartpole2()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
num_traj = 100
tf = 3.0
dt = 0.01

# LQR Training data
Random.seed!(1)
num_lqr = 50
Qlqr = Diagonal([0.2,10,1e-2,1e-2])
Rlqr = Diagonal([1e-3])
xe = [0,pi,0,0]
ue = [0.0]
ctrl_lqr = LQRController(dmodel, Qlqr, Rlqr, xe, ue, dt)
x0_sampler = Product([
    Uniform(-1.0,1.0),
    Uniform(pi-pi/3,pi+pi/3),
    Uniform(-.5,.5),
    Uniform(-.5,.5),
])
initial_conditions_lqr = [rand(x0_sampler) for _ in 1:num_lqr]
X_train, U_train = create_data(dmodel, ctrl_lqr, initial_conditions_lqr, tf, dt)
initial_conditions_test = [rand(x0_sampler) for _ in 1:num_lqr]
X_test, U_test = create_data(dmodel, ctrl_lqr, initial_conditions_test, tf, dt)
@test mapreduce(x->norm(x[2]-xe[2],Inf), max, X_train[end,:]) < deg2rad(5)
# visualize!(vis, RobotZoo.Cartpole(), tf, X_train_lqr[:,2])

## Fit the Data
eigfuns = ["state", "sine", "cosine", "sine", "sine", "chebyshev"]
eigorders = [0,0,0,2,4,4]
Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders)
Z_test, Zu_test, kf = build_eigenfunctions(X_test, U_test, eigfuns, eigorders)

F, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
    ["ridge", "lasso"]; 
    edmd_weights=[10.1], 
    mapping_weights=[0.1], 
    algorithm=:qr
);
BilinearControl.EDMD.fiterror(F,C,g,kf, X_train, U_train)
BilinearControl.EDMD.fiterror(F,C,g,kf, X_test, U_test)

model_bilinear = EDMDModel(F,C,g,kf,dt,"cartpole")
n,m = RD.dims(model_bilinear)
n0 = originalstatedim(model_bilinear)

## Compare linearization about equilibrium
xe = zeros(n0)
ue = zeros(m)
ye = expandstate(model_bilinear, xe)

ctrl_nom = ZeroController(model)
ctrl_bil = ZeroController(model_bilinear)

t_sim = 3.0
times_sim = range(0,t_sim,step=dt)
x0 = [0,deg2rad(20),0,0]
Xsim_nom = simulatewithcontroller(dmodel, ctrl_nom, x0, t_sim, dt)

plotstates(times_sim, Xsim_nom)