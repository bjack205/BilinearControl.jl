import Pkg; Pkg.activate(joinpath(@__DIR__)); Pkg.instantiate();
using Altro
using BilinearControl
using BilinearControl.Problems
using BilinearControl.EDMD
import BilinearControl: Problems
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
using StaticArrays

include("learned_models/edmd_utils.jl")

##
function gencartpoleproblem(x0=zeros(4), Qv=1e-2, Rv=1e-1, Qfv=1e2, u_bnd=3.0, tf=5.0; 
    dt=0.05, constrained=true)

    # model = Cartpole2()
    model = RobotZoo.Cartpole()
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

struct TrackingMPC{T,L} <: AbstractController
    # Reference trajectory
    Xref::Vector{Vector{T}}
    Uref::Vector{Vector{T}}
    Tref::Vector{T}

    # Dynamics 
    model::L
    A::Vector{Matrix{T}}
    B::Vector{Matrix{T}}
    f::Vector{Vector{T}}

    # Cost
    Q::Vector{Diagonal{T,Vector{T}}}
    R::Vector{Diagonal{T,Vector{T}}}
    q::Vector{Vector{T}}
    r::Vector{Vector{T}}

    # Storage
    K::Vector{Matrix{T}}
    d::Vector{Vector{T}}
    P::Vector{Matrix{T}}
    p::Vector{Vector{T}}
    X::Vector{Vector{T}}
    U::Vector{Vector{T}}
    λ::Vector{Vector{T}}
    Nt::Vector{Int}  # horizon length
end

mpchorizon(mpc::TrackingMPC) = mpc.Nt[1]

function TrackingMPC(model::L, Xref, Uref, Tref, Qk, Rk, Qf; Nt=length(Xref)) where {L<:RD.DiscreteDynamics}
    N = length(Xref)
    n = length(Xref[1])
    m = length(Uref[1])
    A,B = linearize(model, Xref, Uref, Tref)
    f = map(1:N) do k
        dt = k < N ? Tref[k+1] - Tref[k] : Tref[k] - Tref[k-1] 
        xn = k < N ? Xref[k+1] : Xref[k]
        Vector(RD.discrete_dynamics(model, Xref[k], Uref[k], Tref[k], dt) - xn)
    end

    Q = [copy(Qk) for k in 1:Nt-1]
    push!(Q, Qf)
    R = [copy(Rk) for k = 1:Nt] 
    q = [zeros(n) for k in 1:Nt]
    r = [zeros(m) for k in 1:Nt]

    K = [zeros(m, n) for k in 1:(Nt - 1)]
    d = [zeros(m) for k in 1:(Nt - 1)]
    P = [zeros(n, n) for k in 1:Nt]
    p = [zeros(n) for k in 1:Nt]
    X = [zeros(n) for k in 1:Nt]
    U = [zeros(m) for k in 1:Nt]
    λ = [zeros(n) for k in 1:Nt]
    TrackingMPC(Xref, Uref, Tref, model, A, B, f, Q, R, q, r, K, d, P, p, X, U, λ, [Nt])
end

function backwardpass!(mpc::TrackingMPC, i)
    A, B = mpc.A, mpc.B
    Q, q = mpc.Q, mpc.q
    R, r = mpc.R, mpc.r
    P, p = mpc.P, mpc.p
    f = mpc.f
    N = length(mpc.Xref)
    Nt = mpchorizon(mpc)

    P[Nt] .= Q[Nt]
    p[Nt] .= q[Nt]

    for j in reverse(1:(Nt - 1))
        k = min(N, j + i - 1)
        P′ = P[j + 1]
        Ak = A[k]
        Bk = B[k]
        fk = f[k]

        Qx = q[j] + Ak' * (P′*fk + p[j + 1])
        Qu = r[j] + Bk' * (P′*fk + p[j + 1])
        Qxx = Q[j] + Ak'P′ * Ak
        Quu = R[j] + Bk'P′ * Bk
        Qux = Bk'P′ * Ak

        cholQ = cholesky(Symmetric(Quu))
        K = -(cholQ \ Qux)
        d = -(cholQ \ Qu)

        P[j] .= Qxx .+ K'Quu * K .+ K'Qux .+ Qux'K
        p[j] = Qx .+ K'Quu * d .+ K'Qu .+ Qux'd
        mpc.K[j] .= K
        mpc.d[j] .= d
    end
end

function forwardpass!(mpc::TrackingMPC, x0, i)
    A,B,f = mpc.A, mpc.B, mpc.f
    X,U,λ = mpc.X, mpc.U, mpc.λ
    K,d = mpc.K, mpc.d
    X[1] = x0  - mpc.Xref[i]
    Nt = mpchorizon(mpc)
    for j = 1:Nt-1
        λ[j] = mpc.P[j]*X[j] .+ mpc.p[j]
        U[j] = K[j]*X[j] + d[j]
        X[j+1] = A[j]*X[j] .+ B[j]*U[j] .+ f[j]
    end
    λ[Nt] = mpc.P[Nt]*X[Nt] .+ mpc.p[Nt]
end

function solve!(mpc::TrackingMPC, x0, i=1)
    backwardpass!(mpc, i)
    forwardpass!(mpc, x0, i)
    return nothing
end

function cost(mpc::TrackingMPC)
    Nt = mpchorizon(mpc)
    mapreduce(+, 1:Nt) do k
        Jx = 0.5 * mpc.X[k]'mpc.Q[k]*mpc.X[k]
        Ju = 0.5 * mpc.U[k]'mpc.R[k]*mpc.U[k]
        Jx + Ju
    end
end

gettime(mpc::TrackingMPC) = mpc.Tref

function getcontrol(mpc::TrackingMPC, x, t)
    k = get_k(mpc, t) 
    solve!(mpc, x, k)
    return mpc.U[1] + mpc.Uref[k]
end

## Visualizer
model = RobotZoo.Cartpole()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
set_cartpole!(vis)
open(vis)

## Solve ALTRO problem
prob = gencartpoleproblem()
solver = ALTROSolver(prob)
Altro.solve!(solver)
visualize!(vis, model, TO.gettimes(solver)[end], TO.states(solver))

## MPC
dmodel = TO.get_model(solver)[1]
X_ref = Vector.(TO.states(solver))
U_ref = Vector.(TO.controls(solver))
push!(U_ref, zeros(RD.control_dim(solver)))
T_ref = TO.gettimes(solver)
Qmpc = Diagonal(fill(1e-0,4))
Rmpc = Diagonal(fill(1e-3,1))
Qfmpc = Diagonal(fill(1e2,4))
Nt = 21 
mpc = TrackingMPC(dmodel, X_ref, U_ref, T_ref, Qmpc, Rmpc, Qfmpc; Nt=Nt)

dx = [0.9,deg2rad(-30),0,0.]
X_sim,U_sim,T_sim = simulatewithcontroller(dmodel, mpc, X_ref[1] + dx, T_ref[end]*1.5, T_ref[2])
plotstates(T_ref, X_ref, inds=1:2, c=:black, legend=:topleft)
plotstates!(T_sim, X_sim, inds=1:2, c=[1 2])
# X_sim[101]
# plotstates(T_sim[1:end-1], U_sim)

## TVLQR
X_ref = Vector.(TO.states(solver))
U_ref = Vector.(TO.controls(solver))
push!(U_ref, zeros(RD.control_dim(solver)))
T_ref = TO.gettimes(solver)
dt = T_ref[2]
dmodel = TO.get_model(solver)[1]

Qtvlqr = [Diagonal(fill(1e-2,4)) for k in 1:length(X_ref)]
Qtvlqr[end] = Diagonal(fill(1e3,4))
Rtvlqr = [Diagonal([1e1]) for k in 1:length(X_ref)]
tvlqr_nom = TVLQRController(dmodel, Qtvlqr, Rtvlqr, X_ref, U_ref, T_ref)
norm(tvlqr_nom.xref - X_ref)
norm(tvlqr_nom.uref - U_ref)

##
dx = [0.0,deg2rad(5),0,0.]
X_sim,U_sim,times = simulatewithcontroller(dmodel, tvlqr_nom, X_ref[1] + dx, T_ref[end], dt)
# X_sim,_,times = simulate(dmodel, U_ref, X_ref[1], T_ref[end], dt)
plotstates(T_ref, X_ref, inds=1:2, c=:black, legend=:topleft)
plotstates!(times, X_sim, inds=1:2, c=[1 2])
# visualize!(vis, model, times[end], X_sim)
