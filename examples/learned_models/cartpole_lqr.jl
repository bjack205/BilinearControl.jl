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

matdensity(A) = nnz(sparse(A)) / length(A)

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
tf = 2.0
dt = 0.02
times = range(0,tf,step=dt)

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
@test mapreduce(x->norm(x[2]-xe[2],Inf), max, X_train[end,:]) < deg2rad(10)
# visualize!(vis, RobotZoo.Cartpole(), tf, X_train_lqr[:,2])

## Fit the Data
eigfuns = ["state", "sine", "cosine", "sine", "sine", "chebyshev"]
eigorders = [0,0,0,2,4,4]
Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders)
Z_test, Zu_test, kf = build_eigenfunctions(X_test, U_test, eigfuns, eigorders)

F, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
    ["ridge", "lasso"]; 
    edmd_weights=[10.1], 
    mapping_weights=[0.0], 
    algorithm=:qr
);
BilinearControl.EDMD.fiterror(F,C,g,kf, X_train, U_train)
BilinearControl.EDMD.fiterror(F,C,g,kf, X_test, U_test)

model_bilinear = EDMDModel(F,C,g,kf,dt,"cartpole")
n,m = RD.dims(model_bilinear)
n0 = originalstatedim(model_bilinear)

## Compare linearizations about equilibrium 
xe = [0,pi,0,0.] 
ue = zeros(m)
ze = RD.KnotPoint{n0,m}(xe,ue,0.0,dt)
ye = expandstate(model_bilinear, xe)

J = zeros(n0,n0+m)
xn = zeros(n0)
RD.jacobian!(RD.InPlace(), RD.ForwardAD(), dmodel, J, xn, ze)
A_nom = J[:,1:n0]
B_nom = J[:,n0+1:end]

function dynamics_bilinear(x,u,t,dt)
    y = expandstate(model_bilinear, x)
    yn = zero(y)
    RD.discrete_dynamics!(model_bilinear, yn, y, u, t, dt)
    originalstate(model_bilinear, yn)
end

A_bil = FiniteDiff.finite_difference_jacobian(x->dynamics_bilinear(x,ue,0.0,dt), xe)
B_bil = FiniteDiff.finite_difference_jacobian(u->dynamics_bilinear(xe,u,0.0,dt), ue)
[A_nom zeros(n0) A_bil]
[B_nom zeros(n0) B_bil]
sign.(A_nom) ≈ sign.(A_bil)
sign.(B_nom) ≈ sign.(B_bil)

# Design a stabilizing LQR controller for both
Qlqr = Diagonal([1.0,10.0,1e-2,1e-2])
Rlqr = Diagonal([1e-4])
K_nom = dlqr(A_nom, B_nom, Qlqr, Rlqr)
K_bil = dlqr(A_bil, B_bil, Qlqr, Rlqr)
maximum(abs.(eigvals(A_nom - B_nom*K_nom))) < 1.0
maximum(abs.(eigvals(A_bil - B_bil*K_bil))) < 1.0
maximum(abs.(eigvals(A_nom - B_nom*K_bil))) < 1.0  # unstable!

# Simulate nominal model with LQR gain from bilinear model
ctrl_lqr = LQRController(K_bil, xe, ue)

t_sim = 10.0
times_sim = range(0,t_sim,step=dt)
x0 = [0,pi-deg2rad(1),0,0]
Xsim_lqr, = simulatewithcontroller(dmodel, ctrl_lqr, x0, t_sim, dt)
plotstates(times_sim, Xsim_lqr, inds=1:2)

#############################################
## Train with derivative info
#############################################

# Generate extra data from training data
xn = zeros(n0)
jacobians = map(CartesianIndices(U_train)) do cind
    k = cind[1]
    x = X_train[cind]
    u = U_train[cind]
    z = RD.KnotPoint{n0,m}(x,u,times[k],dt)
    J = zeros(n0,n0+m)
    RD.jacobian!(
        RD.InPlace(), RD.ForwardAD(), dmodel, J, xn, z 
    )
    J
end
A_train = map(J->J[:,1:n0], jacobians)
B_train = map(J->J[:,n0+1:end], jacobians)
Z_train = map(kf, X_train)
F_train = map(@view X_train[1:end-1,:]) do x
    sparse(ForwardDiff.jacobian(x->expandstate(model_bilinear,x), x))
end
model_bilinear.g
# F = map(Finit)
@test size(A_train[1]) == (n0,n0)
@test size(B_train[1]) == (n0,m)
@test size(Z_train[1]) == (n,)
@test size(F_train[1]) == (n,n0)

G = spdiagm(n0,n,1=>ones(n0)) 
@test norm(G - model_bilinear.g) < 1e-8
W,s = BilinearControl.EDMD.build_edmd_data(
    Z_train, U_train, A_train, B_train, F_train, model_bilinear.g
)
Wsparse = sparse(W)