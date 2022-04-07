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
using Rotations

include("se3_integrator_model.jl")
using BilinearControl: getA, getB, getC, getD

function buildse3integratorproblem()
    # Model
    mass = 2.9
    J = I(3) 
    model = Se3IntegratorDynamics(mass, diag(J)...)
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
    x0_ = [0;0;0; vec(RotZ(deg2rad(0))); zeros(6)]
    xf_ = [0;0;0; vec(RotZ(deg2rad(90)) * RotX(deg2rad(0))); zeros(6)]
    x0 = expandstate(model, x0_)
    xf = expandstate(model, xf_)

    # Objective
    Q = Diagonal([fill(1e-1, 3); fill(1e-2, 9); fill(1e-2, 6); fill(0.0, ns)])
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

## Test dynamics
mass = 2.9
J = I(3) 
model = Se3IntegratorDynamics(mass, diag(J)...)
n,m = RD.dims(model)
@test n == 45 
@test m == 6
base_state_dim(model) == 18
x,u = rand(model)

# Extract states
r = x[1:3]
R = SMatrix{3,3}(x[4:12]...) 
v = x[13:15]
ω = x[16:18]
F = u[1:3]
τ = u[4:6]

# Test rotation matrix
@test det(R) ≈ 1
@test norm(norm.(eachcol(R)) .- 1, Inf) < 1e-12

# Test dynamics
xdot = zeros(n)
RD.dynamics!(model, xdot, x, u)
@test xdot[1:18] ≈ [v; vec(R*skew(ω)); R*F/mass; J\τ]

# Test bilinear dynamics
A,B,C,D = getA(model), getB(model), getC(model), getD(model)
A*x + B*u + sum(u[i]*C[i]*x for i = 1:m) + D ≈ xdot

## Try solving with Altro
using Altro
prob = buildse3integratorproblem()
altro = ALTROSolver(prob)
altro.opts.dynamics_diffmethod = RD.ImplicitFunctionTheorem(RD.UserDefined())
altro.opts.iterations = 100
altro.opts.verbose = 4
solve!(altro)
visualize!(vis, model, prob.tf, states(altro))

## Try solving with ADMM
prob = buildse3integratorproblem()
admm = BilinearADMM(prob)
X = extractstatevec(prob)
U = extractcontrolvec(prob)

BilinearControl.setpenalty!(admm, 1e4)
admm.opts.penalty_threshold = 1e4
Xsol, Usol = BilinearControl.solve(admm, X, U, max_iters=100)
n,m = RD.dims(prob.model[1])
Xs = collect(eachcol(reshape(Xsol, n, :)))
Us = collect(eachcol(reshape(Usol, m, :)))
visualize!(vis, model, prob.tf, Xs)
norm([det(reshape(x[4:12], 3,3)) - 1 for x in Xs], Inf)
