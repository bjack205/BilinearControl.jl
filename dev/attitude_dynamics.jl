import Pkg; Pkg.activate(@__DIR__)

using BilinearControl
using Test
using FiniteDiff
using LinearAlgebra
using Statistics
using BilinearControl.Problems: qrot, skew
using StaticArrays
using Rotations
import RobotDynamics as RD

using BilinearControl: getA, getB, getC, getD
import BilinearControl.Problems: translation, orientation

struct FullAttitudeDynamics <: RD.ContinuousDynamics 
    J::Diagonal{Float64, SVector{3,Float64}}
end

RD.state_dim(::FullAttitudeDynamics) = 12
RD.control_dim(::FullAttitudeDynamics) = 6
RD.default_diffmethod(::FullAttitudeDynamics) = RD.UserDefined()

translation(::FullAttitudeDynamics, x) = @SVector zeros(3)
orientation(::FullAttitudeDynamics, x) = RotMatrix{3}(x[1:9]...)

function Base.rand(::FullAttitudeDynamics)
    x = [vec(qrot(normalize(@SVector randn(4)))); (@SVector randn(3))] 
    u = @SVector randn(6)
    x,u
end

function RD.dynamics(model::FullAttitudeDynamics, x, u)
    J = model.J
    R = SA[
        x[1] x[4] x[7]
        x[2] x[5] x[8]
        x[3] x[6] x[9]
    ]
    ω = SA[x[10], x[11], x[12]]
    τ = SA[u[1], u[2], u[3]]
    w = SA[u[4], u[5], u[6]]
    Rdot = R*skew(w)
    ωdot = J\(τ - w × (J*ω))
    [vec(Rdot); ωdot]
end

BilinearControl.getA(::FullAttitudeDynamics) = @SMatrix zeros(12,12)

function BilinearControl.getB(model::FullAttitudeDynamics)
    J1 = 1 / model.J[1,1]
    J2 = 1 / model.J[2,2]
    J3 = 1 / model.J[3,3]
    [
        @SMatrix zeros(9,6);
        SA[
            J1 0 0 0 0 0
            0 J2 0 0 0 0
            0 0 J3 0 0 0
        ]
    ]
end

function BilinearControl.getC(model::FullAttitudeDynamics)
    J1 = model.J[1,1]
    J2 = model.J[2,2]
    J3 = model.J[3,3]
    C = [zeros(12,12) for i = 1:6]
    for i = 1:3
        C[4][3+i, 6+i] = 1
        C[4][6+i, 3+i] = -1
        C[5][0+i, 6+i] = -1
        C[5][6+i, 0+i] = 1
        C[6][0+i, 3+i] = 1
        C[6][3+i, 0+i] = -1

        C[4][11,12] = J3/J2
        C[4][12,11] = -J2/J3
        C[5][10,12] = -J3/J1
        C[5][12,10] = J1/J3
        C[6][10,11] = J2/J1
        C[6][11,10] = -J1/J2
    end
    C
end

BilinearControl.getD(::FullAttitudeDynamics) = @SVector zeros(12)

struct ConsensusDynamics <: RD.DiscreteDynamics
    J::Diagonal{Float64, SVector{3,Float64}}
end

RD.state_dim(::ConsensusDynamics) = 12
RD.control_dim(::ConsensusDynamics) = 6
RD.default_diffmethod(::ConsensusDynamics) = RD.UserDefined()
RD.output_dim(::ConsensusDynamics) = 15

function Base.rand(::ConsensusDynamics)
    x = [vec(qrot(normalize(@SVector randn(4)))); (@SVector randn(3))] 
    u = @SVector randn(6)
    x,u
end

function RD.dynamics_error(model::ConsensusDynamics, z2::RD.AbstractKnotPoint, z1::RD.AbstractKnotPoint)
    x1,u1 = RD.state(z1), RD.control(z1)
    h = RD.timestep(z1)
    x2 = RD.state(z2)

    J = model.J
    R1 = SA[
        x1[1] x1[4] x1[7]
        x1[2] x1[5] x1[8]
        x1[3] x1[6] x1[9]
    ]
    ω1 = SA[x1[10], x1[11], x1[12]]
    R2 = SA[
        x2[1] x2[4] x2[7]
        x2[2] x2[5] x2[8]
        x2[3] x2[6] x2[9]
    ]
    ω2 = SA[x2[10], x2[11], x2[12]]

    τ = SA[u1[1], u1[2], u1[3]]
    w = SA[u1[4], u1[5], u1[6]]

    R = (R1 + R2)/2
    ω = (ω1 + ω2)/2
    Rdot = R*skew(w)
    ωdot = J\(τ - w × (J*ω))
    [h*[vec(Rdot); ωdot] + x1 - x2; ω1 - w]
end

function BilinearControl.getA(::ConsensusDynamics, h)
    n = 12
    A = zeros(15,2n)
    for i = 1:3
        A[12+i,9+i] = 1
    end
    for i = 1:n
        A[i,i] = 1
        A[i,i+n] = -1
    end
    A
end

function BilinearControl.getB(model::ConsensusDynamics, h)
    J1 = h / model.J[1,1]
    J2 = h / model.J[2,2]
    J3 = h / model.J[3,3]
    [
        zeros(9,12);
        [
            J1 0 0 0 0 0
            0 J2 0 0 0 0
            0 0 J3 0 0 0
            0 0 0 -1 0 0
            0 0 0 0 -1 0
            0 0 0 0 0 -1
        ] zeros(6,6)
    ]
end

function BilinearControl.getC(model::ConsensusDynamics, h)
    J1 = model.J[1,1]
    J2 = model.J[2,2]
    J3 = model.J[3,3]
    C = [zeros(15,24) for i = 1:12]
    n = 12
    for i = 1:3
        for j in (0,n)
            C[4][3+i, 6+i+j] = 1 
            C[4][6+i, 3+i+j] = -1
            C[5][0+i, 6+i+j] = -1
            C[5][6+i, 0+i+j] = 1
            C[6][0+i, 3+i+j] = 1
            C[6][3+i, 0+i+j] = -1

            C[4][11,12+j] = J3/J2
            C[4][12,11+j] = -J2/J3
            C[5][10,12+j] = -J3/J1
            C[5][12,10+j] = J1/J3
            C[6][10,11+j] = J2/J1
            C[6][11,10+j] = -J1/J2
        end
    end
    C * h / 2
end

BilinearControl.getD(::ConsensusDynamics, h) = @SVector zeros(15)

##
J = Diagonal(SA[1,2,1.])
model = FullAttitudeDynamics(J)
n,m = RD.dims(model)
R = qrot(randn(4))
ω = randn(3)
T = randn(3)
w = copy(ω)

x = [vec(R); ω]
u = [T; w]

xdot = RD.dynamics(model, x, u)
@test xdot ≈ [vec(R*skew(ω)); J\(T - cross(ω, J*ω))]
B,C = getB(model), getC(model)
@test xdot ≈ B*u + sum(u[i] * C[i] * x for i = 1:length(u))

## Discrete model
dmodel = ConsensusDynamics(J)
R1 = qrot(randn(4))
ω1 = randn(3)
T1 = randn(3)
w1 = copy(ω1)

R2 = qrot(randn(4))
ω2 = randn(3)
T2 = randn(3)
w2 = copy(ω2)

x1 = [vec(R1); ω1]
x2 = [vec(R2); ω2]
u1 = [T1; w1]
u2 = [T2; w2]
h = 0.1

z1 = RD.KnotPoint{n,m}(n,m, [x1; u1], 0, h)
z2 = RD.KnotPoint{n,m}(n,m, [x2; u2], h, h)

err = RD.dynamics_error(dmodel, z2, z1)
xdot_mid = RD.dynamics(model, (x1+x2)/2, u1)
@test err[1:n] ≈ h*xdot_mid + x1 - x2
@test err[n+1:end] ≈ ω1 - w1

Ad,Bd,Cd,Dd = getA(dmodel,h), getB(dmodel,h), getC(dmodel,h), getD(dmodel,h)
x12 = [x1;x2]
u12 = [u1;u2]

@test Ad*x12 ≈ [x1 - x2;  ω1]
@test Bd*u12 ≈ [zeros(9); h*(J\T1); -w1]
tmp = sum(u1[i]*Cd[i]*x12 for i = 1:m)
@test tmp[1:9] ≈ vec(h*(R1 + R2)/2 * skew(w1))
@test tmp[10:12] ≈ -h*inv(J)*(w1 × (J*(ω1+ω2)/2))
@test tmp[13:15] ≈ zeros(3)
@test Ad*x12 + Bd*u12 + tmp ≈ err

x0 = [vec(I(3)); zeros(3)]
xf = [vec(RotX(deg2rad(45)) * RotZ(deg2rad(180))); zeros(3)]
u0 = zeros(m)
N = 11
tf = 2.0
h = tf / (N-1)

Ad,Bd,Cd,Dd = getA(dmodel,h), getB(dmodel,h), getC(dmodel,h), getD(dmodel,h)
A,B,C,D = BilinearControl.buildbilinearconstraintmatrices(dmodel, x0, xf, h, N)
Xs = [Vector(rand(model)[1]) for k = 1:N]
Us = [Vector(rand(model)[2]) for k = 1:N]
X = vcat(Xs...)
U = vcat(Us...)
c = A*X + B*U + sum(U[i] * C[i] * X for i = 1:length(U)) + D
p = size(Ad,1)
@test c[1:n] ≈ x0 - Xs[1]
@test all(1:N-1) do k
    c[n+1+p*(k-1):n+p*k] ≈ Ad * [Xs[k]; Xs[k+1]] + Bd * [Us[k]; Us[k+1]] + 
        sum(Us[k][i] * Cd[i] * [Xs[k]; Xs[k+1]] for i = 1:m) + Dd
end
@test c[end-n+1:end] ≈ xf - Xs[end]

## 
Q = Diagonal([fill(1e-2, 9); fill(1e-2, 3)])
Qf = Q*(N-1)
R = Diagonal([fill(1e-3,3); fill(1e-3,3)])
Qbar = Diagonal(vcat([diag(Q) for i = 1:N-1]...))
Qbar = Diagonal([diag(Qbar); diag(Qf)])
Rbar = Diagonal(vcat([diag(R) for i = 1:N]...))
q = repeat(-Q*xf, N)
r = repeat(-R*u0, N)
c = 0.5*sum(dot(xf,Q,xf) for k = 1:N-1) + 0.5*dot(xf,Qf,xf) + 
    0.5*sum(dot(u0,R,u0) for k = 1:N)

X = repeat(x0, N)
U = repeat(u0, N)
admm = BilinearADMM(A,B,C,D, Qbar,q,Rbar,r,c)
admm.x .= X
admm.z .= U
admm.opts.penalty_threshold = 1e2
BilinearControl.setpenalty!(admm, 1e4)

Xsol,Usol = BilinearControl.solve(admm, X, U, verbose=true, max_iters=1000)

## Visualization
using MeshCat, Plots
visdir = joinpath(@__DIR__, "../examples/visualization")
include(joinpath(visdir, "visualization.jl"))
vis = Visualizer()
open(vis)
setendurance!(vis)

Xs = collect(eachcol(reshape(Xsol, n, :)))
Us = collect(eachrow(reshape(Usol, m, :)))
visualize!(vis, model, tf, Xs)

plot(Us[1])