import Pkg; Pkg.activate(@__DIR__)
using Altro
using BilinearControl
import BilinearControl.TO
import BilinearControl.RD
using BilinearControl.TO
using ForwardDiff
using FiniteDiff
using LinearAlgebra
using Random
using StaticArrays
using Test
using Statistics
using Rotations

include(joinpath(@__DIR__, "../test/models/se3_models.jl"))
using BilinearControl: getA, getB, getC, getD

function buildse3problem()
    # Model
    model = SE3Kinematics()
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

    # Discretization
    tf = 3.0
    N = 301

    # Dimensions
    nx = RD.state_dim(model)
    nu = RD.control_dim(model)

    # Initial and final conditions
    x0 = [zeros(3); vec(I(3))]
    xf = [5; 0; 1; vec(RotZ(deg2rad(90)) * RotX(deg2rad(90)))]

    # Objective
    Q = Diagonal([fill(1e-1, 3); fill(1e-2, 9)])
    R = Diagonal([fill(1e-2, 3); fill(1e-2, 3)])
    Qf = Q*10
    # costs = map(1:N) do k
    #     q = -xf  # tr(Rf'R)
    #     r = zeros(nu)
    #     TO.DiagonalCost(Q,R,q,r,0.0)
    # end
    # obj = TO.Objective(costs)
    obj = LQRObjective(Q,R,Qf,xf,N)

    # Goal state
    cons = ConstraintList(nx, nu, N)
    goalcon = GoalConstraint(xf)
    add_constraint!(cons, goalcon, N)

    # Initial Guess
    U0 = [fill(0.1,nu) for k = 1:N-1] 

    # Build the problem
    Problem(dmodel, obj, x0, tf, xf=xf, constraints=cons, U0=U0)
end

## Visualization
visdir = joinpath(@__DIR__, "..", "visualization")
include(joinpath(visdir, "visualization.jl"))
using MeshCat
vis = Visualizer()
open(vis)
setbackgroundimage(vis, joinpath(visdir, "stars.jpg"), 30)
setendurance!(vis, scale=1/4)

## Test kinematics
model = SE3Kinematics()
n,m = RD.dims(model)
x,u = rand(model)
@test length(x) == 12
@test length(u) == 6

v = u[1:3]
ω = u[4:end]
R = RotMatrix{3}(x[4:end])
@test RD.dynamics(model, x, u) ≈ [R*v; vec(R*skew(ω))]

# Test custom Jacobian
J = zeros(n, n+m)
xdot = zeros(n)
RD.jacobian!(model, J, xdot, x, u)
Jfd = zero(J)
FiniteDiff.finite_difference_jacobian!(
    Jfd, (y,z)->RD.dynamics!(model, y, z[1:12], z[13:end]), Vector([x;u])
)
@test Jfd ≈  J

# Test dynamics match bilinear dynamics
A,B,C,D = getA(model), getB(model), getC(model), getD(model)
RD.dynamics!(model, xdot, x, u)
@test xdot ≈ A*x + B*u + sum(u[i]*C[i]*x for i = 1:m) + D

## Try solving with ALTRO
prob = buildse3problem()
altro = ALTROSolver(prob)
altro.opts.dynamics_diffmethod = RD.ImplicitFunctionTheorem(RD.UserDefined())
altro.opts.projected_newton = false
solve!(altro)

visualize!(vis, model, prob.tf, states(altro))

## Try solving with ADMM
prob = buildse3problem()
admm = BilinearADMM(prob)
X = extractstatevec(prob)
U = extractcontrolvec(prob)
Xsol, Usol = BilinearControl.solve(admm, X, U)
Xs = collect(eachcol(reshape(Xsol, n, :)))
visualize!(vis, model, prob.tf, Xs)

## 
