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
using Test

include("edmd_utils.jl")

## Generate data stabilizing about the top with an LQR controller 
Random.seed!(1)
model = RobotZoo.Pendulum()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
n0,m = RD.dims(model)
num_traj = 400
tf = 4.0
dt = 0.05
xe = [pi,0]
ue = [0.0]

# Training data
Random.seed!(1)
Q = Diagonal([1.0, 1e-1])
R = Diagonal(fill(1e-3, m))
ctrl_lqr = LQRController(dmodel, Q, R, xe, ue, dt)

x_ub = [pi+pi/4,+2]
x_lb = [pi-pi/4,-2]
x0_sampler = Product([Uniform(x_lb[1], x_ub[1]), Normal(x_lb[2], x_ub[2])])
initial_conditions = tovecs(rand(x0_sampler, num_traj), length(x0_sampler))
X_train, U_train = create_data(dmodel, ctrl_lqr, initial_conditions, tf, dt)
t_train = range(0,tf,step=dt)
u_bnd = norm(U_train,Inf)

@test all(1:num_traj) do i
    simulate(dmodel, U_train[:,i], initial_conditions[i], tf, dt) ≈ X_train[:,i]
end

# Test data
num_test = 10
initial_conditions_test = tovecs(rand(x0_sampler, num_test), length(x0_sampler))
X_test, U_test = create_data(dmodel, ctrl_lqr, initial_conditions_test, tf, dt)

# Linear a model about the top equilibrium
eigfuns = ["state", "sine", "cosine", "chebyshev"]
eigorders = [0,0,0,4]
Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders)

F, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
    ["ridge", "lasso"]; 
    edmd_weights=[0.1], 
    mapping_weights=[10.0], 
    algorithm=:cholesky
);

# Build Model
model_lqr = EDMDModel(F,C,g,kf,dt,"pendulum")

# Check performance on test data
norm(F)
maximum(
    norm(bilinearerror(model_lqr, dmodel, X_test[:,i], U_test[:,i]), Inf) for i = 1:num_test
)
maximum(
    norm(bilinearerror(model_lqr, dmodel, X_train[:,i], U_train[:,i]), Inf) for i = 1:num_traj
)
n = RD.state_dim(model_lqr)

## Generate MPC controller
N = 1001
Xref = [copy(xe) for k = 1:N]
Uref = [copy(ue) for k = 1:N]
tref = range(0,length=N,step=dt)
Nmpc = 21
Qmpc = Diagonal([1.0,0.1])
Rmpc = Diagonal([1e-3])
ctrl_mpc = BilinearMPC(
    model_lqr, Nmpc, initial_conditions[1], Qmpc, Rmpc, Xref, Uref, tref
)

# Test on test data
tsim = 1.0
times_sim = range(0,tsim,step=dt)
p = plot(times_sim, reduce(hcat, Xref[1:length(times_sim)])', 
    label=["θ" "ω"], lw=2
)
for i = 1:num_test
    x0 = initial_conditions_test[i]
    Xmpc, Umpc = simulatewithcontroller(
        dmodel, ctrl_mpc, x0, tsim, dt
    )
    plot!(p,times_sim, reduce(hcat,Xmpc)', c=[1 2], s=:dash, label="", lw=1, legend=:bottom, ylim=(-4,4))
end
display(p)