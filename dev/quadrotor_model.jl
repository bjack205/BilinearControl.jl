import Pkg; Pkg.activate(@__DIR__)

using BilinearControl
using BilinearControl.RD
using BilinearControl.TO
using Test
using FiniteDiff
using LinearAlgebra
using StaticArrays
using Statistics
using Rotations
import BilinearControl.RD
import BilinearControl.TO
include(joinpath(@__DIR__, "../examples/models/rotation_utils.jl"))

##
struct QuadrotorSE23 <: RD.ContinuousDynamics
    mass::Float64
    gravity::Float64
end
QuadrotorSE23(mass=2.0, gravity=9.81) = QuadrotorSE23(mass, gravity)

RD.state_dim(::QuadrotorSE23) = 15
RD.control_dim(::QuadrotorSE23) = 4

BilinearControl.Problems.translation(::QuadrotorSE23, x) = SVector{3}(x[1], x[2], x[3])
BilinearControl.Problems.orientation(::QuadrotorSE23, x) = RotMatrix{3}(x[4:12]...)


function Base.rand(::QuadrotorSE23)
    x = [
            @SVector randn(3);
            vec(qrot(normalize(@SVector randn(4))));
            @SVector randn(3)
    ]
    u = push((@SVector randn(3)), rand())
    x,u
end

function RD.dynamics(model::QuadrotorSE23, x, u)
    mass = model.mass
    g = model.gravity 
    R = SA[
        x[4] x[7] x[10]
        x[5] x[8] x[11]
        x[6] x[9] x[12]
    ]
    v = SA[x[13], x[14], x[15]]
    ω = SA[u[1], u[2], u[3]]
    Fbody = [0, 0, u[4]]

    rdot = v;
    Rdot = R * Rotations.skew(ω)
    vdot = R*Fbody ./ mass - [0,0,g]
    return [rdot; vec(Rdot); vdot]
end

function RD.jacobian!(model::QuadrotorSE23, J, xdot, x, u)
    R = SA[
        x[4] x[7] x[10]
        x[5] x[8] x[11]
        x[6] x[9] x[12]
    ]
    for i = 1:3
        J[i,12+i] = 1.0

        J[6+i,9+i] = +1.0 * u[1]
        J[9+i,6+i] = -1.0 * u[1]
        J[3+i,9+i] = -1.0 * u[2]
        J[9+i,3+i] = +1.0 * u[2]
        J[3+i,6+i] = +1.0 * u[3]
        J[6+i,3+i] = -1.0 * u[3]
        J[12+i,9+i] = 1/model.mass * u[4]

        J[3+i,17] = -R[i,3]
        J[3+i,18] = +R[i,2]
        J[6+i,16] = +R[i,3]
        J[6+i,18] = -R[i,1]
        J[9+i,16] = -R[i,2]
        J[9+i,17] = +R[i,1]
        
        J[12+i,19] = R[i,3] / model.mass 
    end
end

function BilinearControl.getA(::QuadrotorSE23)
    A = zeros(15,15)
    for i = 1:3
        A[i,12+i] = 1.0
    end
    A
end

BilinearControl.getB(::QuadrotorSE23) = zeros(15,4)

function BilinearControl.getC(model::QuadrotorSE23)
    m = model.mass
    C = [zeros(15,15) for i = 1:4]
    for i = 1:3
        C[1][6+i,9+i] = +1.0
        C[1][9+i,6+i] = -1.0
        C[2][3+i,9+i] = -1.0
        C[2][9+i,3+i] = +1.0
        C[3][3+i,6+i] = +1.0
        C[3][6+i,3+i] = -1.0
        C[4][12+i,9+i] = 1/m
    end
    C
end

function BilinearControl.getD(model::QuadrotorSE23)
    g = model.gravity 
    d = zeros(15)
    d[end] = -g
    d
end

using BilinearControl: getA, getB, getC, getD

model = QuadrotorSE23(2.0, 0.0)
r = randn(3)
R = qrot(normalize(randn(4)))
v = randn(3)
ω = randn(3)
F = rand()*10

x = [r; vec(R); v]
u = [ω; F]
xdot = [
    v;
    vec(R*skew(ω));
    (R*[0,0,F] + [0,0,-model.mass*model.gravity]) / model.mass
]
@test xdot ≈ RD.dynamics(model, x, u)

# Test custom Jacobian
n,m = RD.dims(model)
J = zeros(n, n+m)
RD.jacobian!(model, J, xdot, x, u)
Jfd = zero(J)
z = [x;u]
FiniteDiff.finite_difference_jacobian!(
    Jfd, (y,z)->RD.dynamics!(model, y, z[1:15], z[16:end]), Vector([x;u])
)
@test Jfd ≈ J

# Test dynamics match bilinear dynamics
A,B,C,D = getA(model), getB(model), getC(model), getD(model)
@test xdot ≈ A*x + B*u + sum(u[i]*C[i]*x for i = 1:length(u)) + D


## Visualization 
using MeshCat, Colors, CoordinateTransformations
vis = Visualizer()
open(vis)
visdir = joinpath(@__DIR__, "../examples/visualization")
include(joinpath(visdir, "visualization.jl"))
setquadrotor!(vis)

## Build Problem
model = QuadrotorSE23(2.0)
dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

# Discretization
tf = 3.0
N = 101

# Dimensions
nx = RD.state_dim(model)
nu = RD.control_dim(model)

# Initial and final state
x0 = [0; 0; 1.0; vec(I(3)); zeros(3)]
xf = [5; 0; 1.0; vec(RotZ(deg2rad(90))); zeros(3)]

# Objective
Q = Diagonal([fill(1e-2, 3); fill(1e-2, 9); fill(1e-2, 3)])
R = Diagonal([fill(1e-2,3); 1e-2])
Qf = (N-1)*Q
uhover = SA[0,0,0,model.mass*model.gravity]
obj = LQRObjective(Q,R,Qf,xf,N, uf=uhover)

# Goal state
cons = ConstraintList(nx, nu, N)
goalcon = GoalConstraint(xf)

# Initial Guess
U0 = [copy(uhover) for i = 1:N-1]

prob = Problem(dmodel, obj, x0, tf, constraints=cons, U0=U0)
rollout!(prob)

## Solve with ADMM
admm = BilinearADMM(prob)
X = extractstatevec(prob)
U = extractcontrolvec(prob)
admm.opts.penalty_threshold = 1e2
BilinearControl.setpenalty!(admm, 1e4)
Xsol, Usol = BilinearControl.solve(admm, X, U, verbose=true, max_iters=400)

Xs = collect(eachcol(reshape(Xsol, n, :)))
visualize!(vis, model, TO.get_final_time(prob), Xs)
Us = collect(eachrow(reshape(Usol, m, :)))
times = TO.gettimes(prob)

using Plots
plot(times[1:end-1], Us[4])

import BilinearControl.Problems: translation, orientation
translation(model, Xs[1])
orientation(model, Xs[2])