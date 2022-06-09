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
import TrajectoryOptimization as TO
using Altro

include("learned_models/edmd_utils.jl")

function gencartpoleproblem(x0=zeros(4), Qv=1e-2, Rv=1e-1, Qfv=1e2, u_bnd=3.0, tf=5.0; 
    dt=0.05, constrained=true)

    model = Cartpole2()
    dmodel = RD.DiscretizedDynamics{RD.RK4}(model) 
    n,m = RD.dims(model)
    N = round(Int, tf/dt) + 1

    Q = Qv*Diagonal(@SVector ones(n)) * dt
    Qf = Qfv*Diagonal(@SVector ones(n))
    R = Rv*Diagonal(@SVector ones(m)) * dt
    xf = @SVector [0, pi, 0, 0]
    obj = TO.LQRObjective(Q,R,Qf,xf,N)

    conSet = TO.ConstraintList(n,m,N)
    bnd = TO.BoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd)
    goal = TO.GoalConstraint(xf)
    if constrained
    TO.add_constraint!(conSet, bnd, 1:N-1)
    TO.add_constraint!(conSet, goal, N:N)
    end

    X0 = [@SVector fill(NaN,n) for k = 1:N]
    u0 = @SVector fill(0.01,m)
    U0 = [u0 for k = 1:N-1]
    Z = TO.SampledTrajectory(X0,U0,dt=dt*ones(N-1))
    prob = TO.Problem(dmodel, obj, x0, tf, constraints=conSet, xf=xf) 
    TO.initial_trajectory!(prob, Z)
    TO.rollout!(prob)
    prob
end;

## Visualizer
model = RobotZoo.Cartpole()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
set_cartpole!(vis)
open(vis)

## Generate test swing-up trajectories
test_params = [
    (zeros(4), 1e-3, 1e-2, 1e2, 10.0, tf),
    (zeros(4), 1e-3, 1e-2, 1e2, 5.0, tf),
    (zeros(4), 1e-3, 1e-2, 1e2, 4.5, tf),
    (zeros(4), 1e-3, 1e0, 1e2, 10.0, tf),
    (zeros(4), 1e-3, 1e-0, 1e2, 5.0, tf),
    (zeros(4), 1e-3, 1e-0, 1e2, 4.5, tf),
    (zeros(4), 1e-0, 1e-2, 1e2, 10.0, tf),
    (zeros(4), 1e-0, 1e-2, 1e2, 5.0, tf),
    (zeros(4), 1e-0, 1e-1, 1e2, 4.5, tf),
    (zeros(4), 1e-0, 1e-2, 1e-2, 10.0, tf),
]
prob = gencartpoleproblem(test_params[1]..., dt=dt)
solver = ALTROSolver(prob)
Altro.solve!(solver)
visualize!(vis, model, tf, TO.states(solver))

Qtvlqr = [Diagonal(fill(1e-3,4)) for k in 1:length(X_ref)]
Qtvlqr[end] = Diagonal(fill(1e3,4))
Rtvlqr = [Diagonal([1e-2]) for k in 1:length(X_ref)]
X_ref = Vector.(TO.states(solver))
U_ref = Vector.(TO.controls(solver))
push!(U_ref, zeros(1))
T_ref = TO.gettimes(solver)
tvlqr_true = TVLQRController(dmodel, Qtvlqr, Rtvlqr, X_ref, U_ref, T_ref)

dx = [0.,0,0,0]
X_tvlqr_true,_,times = simulatewithcontroller(dmodel, tvlqr_true, X_ref[1] + dx, 1.5*tf, dt)
plotstates(T_ref, X_ref, inds=1:2, label="reference", xlabel="time (s)", ylabel="states", c=:black, ylim=(-4,4))
plotstates!(times, X_tvlqr_true, inds=1:2, label="true", xlabel="time (s)", ylabel="states", c=[1 2])


test_trajectories = map(test_params) do params
    solver = Altro.solve!(ALTROSolver(gencartpoleproblem(params...; dt), show_summary=false))
    if Altro.status(solver) != Altro.SOLVE_SUCCEEDED
        @show params
        @warn "ALTRO Solve failed"
    end
    X = TO.states(solver)
    U = TO.controls(solver)
    Vector.(X), Vector.(U)
end
X_test_altro = mapreduce(x->getindex(x,1), hcat, test_trajectories)
U_test_altro = mapreduce(x->getindex(x,2), hcat, test_trajectories)
time = range(0,tf,step=dt)

# Plot test trajectories
p = plot(ylabel="states", xlabel="time (s)")
for i = 1:size(X_test_altro,2)
    plot!(p, time, reduce(hcat, X_test_altro[:,i])[1:2,:]', label="", c=[1 2])
end

## Define the models
model_nom = RobotZoo.Cartpole(mc=1.0, mp=0.2, l=0.5)
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom);

model_real = Cartpole2(mc=1.05, mp=0.19, l=0.52, b=0.02)  # this model has damping
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

model = Cartpole2()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)

## Import Testing and Training data
altro_lqr_traj = load(joinpath(Problems.DATADIR, "cartpole_lqr_and_altro_trajectories.jld2"))

X_train = altro_lqr_traj["X_train"]
U_train = altro_lqr_traj["U_train"]
X_test_swing_up = altro_lqr_traj["X_test_swing_up"]
U_test_swing_up = altro_lqr_traj["U_test_swing_up"]
X_test_stabilize = altro_lqr_traj["X_test_stabilize"]
U_test_stabilize = altro_lqr_traj["U_test_stabilize"]
tf = altro_lqr_traj["tf"]
dt = altro_lqr_traj["dt"]

X_test_swing_up[end,:]

T_ref = range(0,tf,step=dt);

## Import nominal eDMD model 
cartpole_data = load(joinpath(Problems.DATADIR, "cartpole_tvlqr_nominal_eDMD_data.jld2"))
A_nom = cartpole_data["A"]
B_nom = cartpole_data["B"]
C_nom = cartpole_data["C"]
g = cartpole_data["g"]
kf = cartpole_data["kf"]
tf = cartpole_data["tf"]
dt = cartpole_data["dt"]
model_bilinear_nom = EDMDModel(A_nom, B_nom, C_nom, g, kf, dt, "cartpole_nom")
model_bilinear_nom_projected = BilinearControl.Problems.ProjectedEDMDModel(model_bilinear_nom)

## Import Jacobian eDMD model
cartpole_data = load(joinpath(Problems.DATADIR, "cartpole_tvlqr_jacobian_penalized_eDMD_data.jld2"))
A_jac = cartpole_data["A"]
B_jac = cartpole_data["B"]
C_jac = cartpole_data["C"]
model_bilinear_jac = EDMDModel(A_jac, B_jac, C_jac, g, kf, dt, "cartpole_jac")
model_bilinear_jac_projected = BilinearControl.Problems.ProjectedEDMDModel(model_bilinear_jac)

## Stabilize about the top
xeq = [0,pi,0,0]
ueq = [0.]
Qlqr = Diagonal([1.0,5.0,1e-5,1e-5])
Rlqr = Diagonal([1e-1])
lqr_true = LQRController(dmodel, Qlqr, Rlqr, xeq, ueq, dt)
lqr_nom_projected = LQRController(model_bilinear_nom_projected, Qlqr, Rlqr, xeq, ueq, dt)
lqr_jac_projected = LQRController(model_bilinear_jac_projected, Qlqr, Rlqr, xeq, ueq, dt)

i = 1  # index of test initial condition
X_lqr_true,_,times = simulatewithcontroller(dmodel, lqr_true, X_test_stabilize[1,i], 3.0, dt)
X_lqr_nom_projected,_,times = simulatewithcontroller(dmodel, lqr_nom_projected, X_test_stabilize[1,i], 3.0, dt)
X_lqr_jac_projected,_,times = simulatewithcontroller(dmodel, lqr_jac_projected, X_test_stabilize[1,i], 3.0, dt)
X_lqr_nom_projected
plotstates(times, X_lqr_true, inds=1:2, c=[1 2], label="true")
plotstates!(times, X_lqr_nom_projected, inds=1:2, c=[1 2], s=:dot, label="eDMD")
plotstates!(times, X_lqr_jac_projected, inds=1:2, c=[1 2], s=:dash, label="JDMD")

## Run TVLQR controller on test swing-up trajectories
i = 1  # index of test trajectory
X_ref = deepcopy(X_test_swing_up[:,i])
U_ref = deepcopy(U_test_swing_up[:,i])

# Make the last state of the reference trajectory the desired final state
X_ref[end] .= xeq
push!(U_ref, ueq)

# Design the TVLQR controller
Qtvlqr = [Diagonal([1.0,1.0,1e-3,1e-3]) for k in 1:length(X_ref)]
Rtvlqr = [Diagonal([1e-2]) for k in 1:length(X_ref)]

tvlqr_true = TVLQRController(dmodel, Qtvlqr, Rtvlqr, X_ref, U_ref, T_ref)
tvlqr_nom_projected = TVLQRController(model_bilinear_nom_projected, Qtvlqr, Rtvlqr, X_ref, U_ref, T_ref)
tvlqr_jac_projected = TVLQRController(model_bilinear_jac_projected, Qtvlqr, Rtvlqr, X_ref, U_ref, T_ref)

dx = [0.00,0.00,0.01,0]
X_tvlqr_true,_,times = simulatewithcontroller(dmodel, tvlqr_true, X_ref[1] + dx, 1.5*tf, dt)
X_tvlqr_nom_projected,_,times = simulatewithcontroller(dmodel, tvlqr_nom_projected, X_ref[1] + dx, 1.5*tf, dt)
X_tvlqr_jac_projected,_,times = simulatewithcontroller(dmodel, tvlqr_jac_projected, X_ref[1] + dx, 1.5*tf, dt)
plotstates(T_ref, X_ref, inds=1:2, label="reference", xlabel="time (s)", ylabel="states", c=:black, ylim=(-4,4))
plotstates!(times, X_tvlqr_true, inds=1:2, label="true", xlabel="time (s)", ylabel="states", c=[1 2])
plotstates!(times, X_tvlqr_nom_projected, inds=1:2, label="eDMD", c=[1 2], s=:dot)
plotstates!(times, X_tvlqr_jac_projected, inds=1:2, label="JDMD", c=[1 2], s=:dash)