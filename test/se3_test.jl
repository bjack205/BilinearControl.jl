using BilinearControl
import BilinearControl.TO
import BilinearControl.RD
using BilinearControl.TO
using FiniteDiff
using LinearAlgebra
using Random
using StaticArrays
using Rotations
using Statistics
using Test

using BilinearControl: getA, getB, getC, getD

include("models/se3_models.jl")
enablevis = false 
enablevis = true 

function buildse3problem(model, x̄0::RBState, x̄f::RBState;
        Qd = fill(1e-1, RD.state_dim(model)),
        Rd = fill(1e-2, RD.control_dim(model)),
        Qfd = fill(1.0, RD.state_dim(model)),
    )
    # Discretization
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)
    tf = 5.0
    N = 101

    # Dimensions
    nx = RD.state_dim(model)
    nu = RD.control_dim(model)
    nb = base_state_dim(model)

    # Initial and final conditions
    r0,q0 = x̄0.r, x̄0.q 
    rf,qf = x̄f.r, x̄f.q 

    x0 = buildstate(model, x̄0) 
    xf = buildstate(model, x̄f) 

    # Initial Guess
    aa = AngleAxis(q0\qf)
    X0 = map(range(0,1,length=N)) do t
        R = q0 * AngleAxis(aa.theta * t, aa.axis_x, aa.axis_y, aa.axis_z)
        x̄ = x̄0 + RBState((x̄f - x̄0) * t)
        x̄ = RBState(x̄.r, R, x̄.v, x̄.ω)
        buildstate(model, x̄)
    end
    # X0 = [x0 + (xf - x0)*t for t in range(0,1, length=N)]
    U0 = [randn(nu) for k = 1:N-1]
    Zref = SampledTrajectory{nx,nu}(X0, U0, tf=tf)

    # Objective
    Q = Diagonal(Qd)
    R = Diagonal(Rd)
    Qf = Diagonal(Qfd)
    obj = TO.LQRObjective(Q,R,Qf,xf,N)
    costs = map(1:N) do k
        Q_ = k < N ? Q : Qf
        x = X0[k]
        q = -Q_*x
        q[4:12] = -x[4:12]  # equal to tr(Rf'R)
        r = zeros(nu)
        c = 0.5*x'Qf*x
        TO.DiagonalCost(Q_,R,q,r,c)
    end
    obj = TO.Objective(costs)
    # obj = TO.TrackingObjective(Q,R,Zref, Qf=Qf) 

    # Goal state
    cons = ConstraintList(nx, nu, N)
    goalcon = GoalConstraint(xf, 1:nb)  # constrain only the base states
    add_constraint!(cons, goalcon, N)

    # Build the problem
    Problem(dmodel, obj, x0, tf, xf=xf, constraints=cons, X0=X0, U0=U0)
end

## Problem 
r0 = @SVector zeros(3)
q0 = SA[1,0,0,0.]
v0 = @SVector zeros(3)
ω0 = @SVector zeros(3)
x̄0 = RBState(r0, q0, v0, ω0)

rf = SA[10,15,4.0]
qf = Rotations.params(UnitQuaternion(RotX(deg2rad(90)) * RotZ(deg2rad(180))))
vf = @SVector zeros(3)
ωf = @SVector zeros(3)
x̄f = RBState(rf, qf, vf, ωf)

## Visualize it
if enablevis
    visdir = joinpath(@__DIR__, "..", "visualization")
    include(joinpath(visdir, "visualization.jl"))
    using MeshCat
    vis = Visualizer()
    open(vis)
    setbackgroundimage(vis, joinpath(visdir, "stars.jpg"), 30)
    setendurance!(vis, scale=1/4)
end

if enablevis
    visualize!(vis, r0, q0)
end

if enablevis
    visualize!(vis, rf, qf)
end

# @testset "SE(3) Kinematics" begin
## Kinematics
model = SE3Kinematics()
@test RD.dims(model) == (12,6,12)

# Test the dynamics
let
    x̄ = rand(RBState)
    r,q = x̄.r, Rotations.params(x̄.q)
    v,ω = x̄.v, x̄.ω
    u = [v; ω]
    x = [r; vec(qrot(q))]
    xdot = zeros(RD.state_dim(model))
    RD.dynamics!(model, xdot, x, u)
    @test xdot[1:3] ≈ UnitQuaternion(q)*v
    @test xdot[4:12] ≈ vec(qrot(q)*skew(ω))

    # Test it matches the bilinear dynamics
    A,B,C,D = getA(model), getB(model), getC(model), getD(model)
    m = size(B,2)
    @test xdot ≈ A*x + B*u + sum(u[i]*C[i]*x for i = 1:m) + D
end

Q = [fill(1e-1, 3); fill(0.0, 9)] 
R = [fill(1e-3, 3); fill(1e-3, 3)]
Qf = fill(10.0, 12)
prob = buildse3problem(model, x̄0, x̄f)
admm = BilinearADMM(prob)
admm.opts.penalty_threshold = 1e5
BilinearControl.setpenalty!(admm, 1e3)
X = extractstatevec(prob)
U = extractcontrolvec(prob)
Xsol, Usol, Ysol = BilinearControl.solve(admm, X, U, max_iters=300)

BilinearControl.setpenalty!(admm, 1e2)
Xsol2, Usol2 = BilinearControl.solve(admm, Xsol, Usol, Ysol, max_iters=100)

# Test that it got to the goal
n,m = RD.dims(model)
Xs = collect(eachcol(reshape(Xsol, n, :)))
x̄N = RBState(Xs[end][1:3], RotMatrix{3}(Xs[end][4:12]), zeros(3), zeros(3))
@test norm(x̄N ⊖ x̄f) < 2e-4


# Test that the rotation matrices are still valid 
R = [reshape(x[4:end], 3, 3) for x in Xs]
maxerr = maximum(abs(det(r) - 1) for r in R)
@test maxerr < 1e-1

if enablevis
    n,m = RD.dims(prob.model[1])
    Xs = collect(eachcol(reshape(Xsol, n, :)))
    visualize!(vis, model, prob.tf, Xs)
end
# end

@variables R[1:3,1:3] Rg[1:3,1:3]
R_ = SMatrix{3,3}(R...)
Rg_ = SMatrix{3,3}(Rg...)

using Plots
Us = reshape(Usol, m, :)
plot(Us')

tr(Rg_'R_)
# ## SE(3) with angular velocity
# model = SE3AngVelBilinearDynamics(2.0)
# Qd = [fill(1e-1, 3); fill(1e-3, 4); fill(1e-1, 3); fill(0e-8, RD.state_dim(model) - base_state_dim(model))]
# Rd = fill(1e-3, 6)
# Qfd = Qd*1
# prob = buildse3problem(model, x̄0, x̄f, Qd=Qd, Rd=Rd, Qfd=Qfd)
# visualize!(vis, model, prob.tf, states(prob))
# admm = BilinearADMM(prob)
# BilinearControl.setpenalty!(admm, 1e4)
# X = extractstatevec(prob)
# U = extractcontrolvec(prob)
# Xsol, Usol = BilinearControl.solve(admm, X, U, max_iters=100)

# if enablevis
#     n,m = RD.dims(prob.model[1])
#     Xs = collect(eachcol(reshape(Xsol, n, :)))
#     visualize!(vis, model, prob.tf, Xs)
# end

# ## SE(3) Dynamics
# model = SE3BilinearDynamics(2.0, I(3))
# Qd = [
#     fill(1e-1, 3); 
#     fill(1e-3, 4); 
#     fill(1e-1, 3); 
#     fill(1e-2, 3); 
#     fill(0e-8, RD.state_dim(model) - base_state_dim(model))
# ]
# Rd = fill(1e-2, 6)
# Qfd = Qd*10
# prob = buildse3problem(model, x̄0, x̄f, Qd=Qd, Rd=Rd, Qfd=Qfd)
# visualize!(vis, model, prob.tf, states(prob))
# admm = BilinearADMM(prob)
# BilinearControl.getpenalty(admm)
# BilinearControl.setpenalty!(admm, 1e3)
# X = extractstatevec(prob)
# U = extractcontrolvec(prob)
# Xsol, Usol = BilinearControl.solve(admm, X, U, max_iters=30)