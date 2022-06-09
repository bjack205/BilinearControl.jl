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

include("learned_models/edmd_utils.jl")

function gencartpoleproblem(x0=zeros(4), Qv=1e-2, Rv=1e-1, Qfv=1e2, u_bnd=3.0, tf=5.0; 
    dt=0.05, constrained=true)

    model = Problems.NominalCartpole()  # NOTE: this should exactly match RobotZoo.Cartpole()
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
end

## Visualizer
model = RobotZoo.Cartpole()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
set_cartpole!(vis)
open(vis)

## Setup training data
num_train = 15
Random.seed!(1)
train_params = map(1:num_train) do i
    Qv = 1e-2
    Rv = Qv * 10^rand(Uniform(-1,3.0))
    Qfv = Qv * 10^rand(Uniform(1,5.0)) 
    u_bnd = rand(Uniform(4.5, 8.0))
    (zeros(4), Qv, Rv, Qfv, u_bnd, tf)
end

train_trajectories = map(train_params) do params
    solver = Altro.solve!(ALTROSolver(gencartpoleproblem(params..., dt=dt), 
        show_summary=false, projected_newton=true))
    if Altro.status(solver) != Altro.SOLVE_SUCCEEDED
        @warn "ALTRO Solve failed"
    end
    X = TO.states(solver)
    U = TO.controls(solver)
    Vector.(X), Vector.(U)
end
X_train_altro = mapreduce(x->getindex(x,1), hcat, train_trajectories)
U_train_altro = mapreduce(x->getindex(x,2), hcat, train_trajectories)
T_test_altro = range(0,tf,step=dt)

## Setup test data 
dt = 0.05
tf = 5.0
test_params = [
    (zeros(4), 1e-2, 1e-1, 1e2,  3.0, tf)
    (zeros(4), 1e-0, 1e-1, 1e2,  5.0, tf)
    (zeros(4), 1e1,  1e-2, 1e2, 10.0, tf)
    (zeros(4), 1e-1, 1e-0, 1e2, 10.0, tf)
    (zeros(4), 1e-2, 1e-0, 1e1, 10.0, tf)
    (zeros(4), 1e-2, 1e-0, 1e1,  3.0, tf)
    (zeros(4), 1e1,  1e-3, 1e2, 10.0, tf)
    (zeros(4), 1e1,  1e-3, 1e2,  5.0, tf)
    (zeros(4), 1e3,  1e-3, 1e3, 10.0, tf)
    (zeros(4), 1e0,  1e-2, 1e2,  4.0, tf)
]
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
T_test_altro = range(0,tf,step=dt)

# Make sure all trajectories are complete swing-ups
xg = [0,pi,0,0]
@test all(x->norm(x-xg) < 1e-6, X_train_altro[end,:])
@test all(x->norm(x-xg) < 1e-6, X_test_altro[end,:])

## Setup MPC Controller
model_nom = Problems.NominalCartpole()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)
model_true = Problems.SimulatedCartpole()
dmodel_true = RD.DiscretizedDynamics{RD.RK4}(model_true)

i = 10 
X_ref = deepcopy(X_test_altro[:,i])
U_ref = deepcopy(U_test_altro[:,i])
push!(U_ref, zeros(RD.control_dim(solver)))
T_ref = TO.gettimes(solver)
Qmpc = Diagonal(fill(1e-0,4))
Rmpc = Diagonal(fill(1e-3,1))
Qfmpc = Diagonal(fill(1e2,4))
Nt = 41 
mpc = TrackingMPC(dmodel_nom, X_ref, U_ref, T_ref, Qmpc, Rmpc, Qfmpc; Nt=Nt)

# Run sim w/ MPC controller w/ large initial offset
dx = [0.9,deg2rad(-30),0,0.]  # large initial offset
X_sim,U_sim,T_sim = simulatewithcontroller(dmodel, mpc, X_ref[1] + dx, T_ref[end]*1.5, T_ref[2])
plotstates(T_ref, X_ref, inds=1:2, c=:black, legend=:topleft)
plotstates!(T_sim, X_sim, inds=1:2, c=[1 2])

# Compare open loop trajectories for true and nominal models
x,u = rand(model)
X_nom,_,T_nom = simulate(dmodel_nom, U_ref, X_ref[1], T_ref[end], T_ref[2])
X_true,_,T_true = simulate(dmodel_true, U_ref, X_ref[1], T_ref[end], T_ref[2])
plotstates(T_ref, X_ref, inds=1:2, c=:black, legend=:topleft, label="reference")
plotstates!(T_nom, X_nom, inds=1:2, c=[1 2], label="nominal model")
plotstates!(T_true, X_true, inds=1:2, c=[1 2], s=:dash, label="true model")

# Try using MPC with model mismatch
#  NOTE: The performance of the MPC should be pretty bad, but should still stabilize
#        We want the MPC with the learn model to show a dramatic improvement in tracking performance
dx = [0.5,deg2rad(20),1,-1.] * 0
X_mismatch,_,T_mismatch = simulatewithcontroller(dmodel_true, mpc, X_ref[1] + dx, T_ref[end]*1.5, T_ref[2])
plotstates(T_ref, X_ref, inds=1:2, c=:black, legend=:topleft, ylim=(-4,4), label=["reference" ""],
    xlabel="time (s)", ylabel="states"
)
plotstates!(T_mismatch, X_mismatch, inds=1:2, c=[1 2], label=["MPC" ""])

## Define TVLQR Controller
dx = [0.01,0,0,0]
Qtvlqr = [copy(Qmpc) for k in 1:length(X_ref)]
Qtvlqr[end] = copy(Qfmpc) 
Rtvlqr = [Diagonal(fill(1e0,1)) for k in 1:length(X_ref)]
tvlqr_nom = TVLQRController(dmodel, Qtvlqr, Rtvlqr, X_ref, U_ref, T_ref)
X_tvlqr,U_tvlqr,T_tvlqr= simulatewithcontroller(dmodel, tvlqr_nom, X_ref[1] + dx, T_ref[end]*1.5, T_ref[2])

plotstates(T_ref, X_ref, inds=1:2, c=:black, legend=:topleft, ylim=(-4,4), label=["reference" ""],
    xlabel="time (s)", ylabel="states"
)
plotstates!(T_sim, X_sim, inds=1:2, c=[1 2], label=["MPC" ""])
plotstates!(T_tvlqr, X_tvlqr, inds=1:2, c=[1 2], label=["TVLQR" ""], s=:dash)