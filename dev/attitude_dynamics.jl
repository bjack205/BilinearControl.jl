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

##
function test_so3_dynamics()
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
end

function test_se3_torque_bilinear_constraint()
    J = Diagonal(SA[1,2,1.])
    dmodel = ConsensusDynamics(J)
    x0 = [vec(I(3)); zeros(3)]
    xf = [vec(RotX(deg2rad(45)) * RotZ(deg2rad(180))); zeros(3)]
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
end

## Visualization
using MeshCat, Plots
visdir = joinpath(@__DIR__, "../examples/visualization")
include(joinpath(visdir, "visualization.jl"))
vis = Visualizer()
open(vis)
setendurance!(vis)


admm = SE3TorqueProblem(N=51, tf=3.0)
X = copy(admm.x)
U = copy(admm.z)
# admm.opts.ϵ_rel_dual = 1.0
# admm.opts.ϵ_abs_dual = 1.0
Xsol,Usol = BilinearControl.solve(admm, X, U, verbose=true, max_iters=1000)

Xs = collect(eachcol(reshape(Xsol, n, :)))
Us = reshape(Usol, m, :)
visualize!(vis, model, tf, Xs)

plot(Us[1:3,1:end-1]')