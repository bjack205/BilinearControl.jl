import Pkg; Pkg.activate(@__DIR__)
using Altro
using BilinearControl
import BilinearControl.TO
import BilinearControl.RD
using BilinearControl.TO
using BilinearControl.RD
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

using BilinearControl: getnzindsA, getnzindsB

Ahat = BilinearControl.getAhat(admm, U)
nnz(Ahat) == sum(nnz(C) for C in admm.C) + nnz(admm.A)
nzindsA = map(admm.C) do C
    getnzindsA(Ahat, C)
end
pushfirst!(nzindsA, getnzindsA(Ahat, admm.A))

Ahat2 = similar(Ahat)
@test !(Ahat2 ≈ Ahat)
BilinearControl.updateAhat!(admm, Ahat2, U, nzindsA)
@test Ahat2 ≈ Ahat

Bhat = BilinearControl.getBhat(admm, X)
nzindsB = map(eachindex(admm.C)) do i
    getnzindsB(Bhat, admm.C[i], i)
end
pushfirst!(nzindsB, getnzindsA(Bhat, admm.B))
nzindsB[1]

Bhat2 = similar(Bhat)
@test !(Bhat2 ≈ Bhat)
BilinearControl.updateBhat!(admm, Bhat2, X, nzindsB)
@test Bhat2 ≈ Bhat

Xsol, Usol = BilinearControl.solve(admm, X, U, max_iters=200)
Xsol2, Usol2 = BilinearControl.solve(admm, Xsol, Usol, max_iters=200)
Xs = collect(eachcol(reshape(Xsol, n, :)))
Us = collect(eachcol(reshape(Usol, m, :)))

# Check it reaches the goal
@test norm(Xs[end] - prob.xf) < 1e-3

# Check the rotation matrices
@test norm([det(reshape(x[4:end],3,3)) - 1 for x in Xs], Inf) < 1e-3

# Test that the controls are smooth
@test norm(mean(diff(Us)), Inf) < 0.1

@profview BilinearControl.solve(admm, X, U, max_iters=20)


using TimerOutputs
using StatProfilerHTML
@profilehtml BilinearControl.solve(admm, X, U, max_iters=20)
let x = X, z = U, solver = admm
    w = solver.w 
    to = TimerOutput()
    @timeit to "r" r = BilinearControl.primal_residual(solver, x, z)
    @timeit to "s" s = BilinearControl.dual_residual(solver, x, z)
    @timeit to "solvex" BilinearControl.solvex(solver, z, w)
    @timeit to "solvez" BilinearControl.solvez(solver, x, w)
    @timeit to "updatew" BilinearControl.updatew(solver, x, z, w)
    println(to)
end

admm.C[1] * X
admm.C[1]
B
Bhat = BilinearControl.getBhat(admm, X)
Bhat'Bhat
