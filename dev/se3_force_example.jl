import Pkg; Pkg.activate(@__DIR__)

using BilinearControl
using BilinearControl.RD
using BilinearControl.TO
import BilinearControl.RD
import BilinearControl.TO
using Test
using FiniteDiff
using LinearAlgebra
using Statistics
using SparseArrays
using StaticArrays
import BilinearControl.RD

include("se3_force_model.jl")
using BilinearControl: getA, getB, getC, getD

function buildse3forceproblem()
    # Model
    mass = 2.9
    model = Se3ForceDynamics(mass)
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

    # Discretization
    tf = 3.0
    N = 101

    # Dimensions
    nx = RD.state_dim(model)
    nu = RD.control_dim(model)
    nb = base_state_dim(model)
    ns = nx - nb 

    # Initial and final conditions
    x0_ = [0;0;0; vec(RotZ(deg2rad(0))); zeros(3)]
    xf_ = [3;0;1; vec(RotZ(deg2rad(90)) * RotX(deg2rad(150))); zeros(3)]
    x0 = expandstate(model, x0_)
    xf = expandstate(model, xf_)

    # Objective
    Q = Diagonal([fill(1e-1, 3); fill(1e-2, 9); fill(1e-2, 3); fill(0.0, ns)])
    R = Diagonal([fill(1e-2, 3); fill(1e-2, 3)])
    Qf = 100 * Q 
    obj = LQRObjective(Q,R,Qf,xf,N)

    # Goal state
    cons = ConstraintList(nx, nu, N)
    goalcon = GoalConstraint(xf, 1:nb)  # only constraint the original states
    add_constraint!(cons, goalcon, N)

    # Initial Guess
    U0 = [fill(0.1,nu) for k = 1:N-1] 

    # Build the problem
    prob = Problem(dmodel, obj, x0, tf, xf=xf, constraints=cons, U0=U0)
    rollout!(prob)
    prob
end

## Visualization
visdir = joinpath(@__DIR__, "..", "visualization")
include(joinpath(visdir, "visualization.jl"))
using MeshCat
vis = Visualizer()
open(vis)
setbackgroundimage(vis, joinpath(visdir, "stars.jpg"), 30)
setendurance!(vis, scale=1/4)

## Try solving with ALTRO
using Altro
prob = buildse3forceproblem()
altro = ALTROSolver(prob)
altro.opts.dynamics_diffmethod = RD.ImplicitFunctionTheorem(RD.UserDefined())
altro.opts.projected_newton = false
solve!(altro)

model = Se3ForceDynamics(2.0)
visualize!(vis, model, prob.tf, states(altro))

## Try solving with ADMM
prob = buildse3forceproblem()
admm = BilinearADMM(prob)
X = extractstatevec(prob)
U = extractcontrolvec(prob)

BilinearControl.setpenalty!(admm, 1e4)
admm.opts.penalty_threshold = 1e2
Xsol, Usol = BilinearControl.solve(admm, X, U, max_iters=800)
n,m = RD.dims(prob.model[1])
Xs = collect(eachcol(reshape(Xsol, n, :)))
Us = collect(eachcol(reshape(Usol, m, :)))
visualize!(vis, model, prob.tf, Xs)
norm([det(reshape(x[4:12], 3,3)) - 1 for x in Xs], Inf)


# Check it reaches the goal
@test norm(Xs[end] - prob.xf) < 1e-3

# Check the rotation matrices
@test norm([det(reshape(x[4:12],3,3)) - 1 for x in Xs], Inf) < 1e-2

# Test that the controls are smooth
@test norm(mean(diff(Us)), Inf) < 0.1