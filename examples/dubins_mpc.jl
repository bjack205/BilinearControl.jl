import Pkg; Pkg.activate(@__DIR__)

using BilinearControl
using COSMOAccelerators
import RobotDynamics as RD
import TrajectoryOptimization as TO
using BilinearControl.Problems
using StaticArrays
using LinearAlgebra

# include("visualization/visualization.jl")
function DubinsMPCProblem(Zref; N=51, tf=2.0, kstart=1)
    model = BilinearDubins()
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)
    nx,nu = RD.dims(model)

    x0 = RD.state(Zref[1])
    dt = tf / (N-1)

    # cost
    Q = Diagonal(SA[1.0, 1.0, 1e-2, 1e-2])
    R = Diagonal(@SVector fill(1e-4, nu))
    Z = RD.SampledTrajectory(Zref[kstart - 1 .+ (1:N)])
    Z[end].dt = 0.0
    obj = TO.TrackingObjective(Q, R, Z)

    prob = TO.Problem(dmodel, obj, x0, tf)
    TO.initial_trajectory!(prob, Z)
    prob
end

include("scotty.jl")
scotty_interp = generate_scotty_trajectory(scale=0.01)
Nref = 501
tref = 50  # time of entire trajectory
dt = tref / (Nref - 1)
s = range(0, knots(scotty_interp).knots[end], length=Nref)
scotty = scotty_interp.(s)

using Plots
p = plot(Tuple.(scotty), aspect_ratio=:equal)

Zref = RD.SampledTrajectory(map(1:Nref) do k
    p = scotty[k]
    if k < Nref
        pn = scotty[k+1]
        i = normalize(SA[pn[1] - p[1], pn[2] - p[2]])
        z = SA[p[1], p[2], i[1], i[2], 0.0, 0.0]
    else
        pp = scotty[k-1]
        i = normalize(SA[p[1] - pp[1], p[2] - pp[2]])
        z = SA[p[1], p[2], i[1], i[2], 0.0, 0.0]
    end
    RD.KnotPoint{4,2}(4, 2, z, (k-1)*dt, dt)
end)

## Generate MPC problem
prob = DubinsMPCProblem(Zref)
n,m = RD.dims(prob,1)

# Plot first horizon
p = plot(Tuple.(scotty), aspect_ratio=:equal)
RD.traj2!(SVector{4}.(TO.states(prob)), xlim=xlims(p), ylim=ylims(p))

## Generate Solver
admm = BilinearADMM(prob)
admm.q[1:4] ≈ -Q*RD.state(Zref[1])
X = extractstatevec(prob)
U = extractcontrolvec(prob)
admm.opts.penalty_threshold = 1e2
Xsol, Usol = BilinearControl.solve(admm, X, U, verbose=true)
Xs = collect(eachrow(reshape(Xsol, 4, :))) 
J0 = BilinearControl.cost(admm, Xsol, Usol)


p = plot(Tuple.(scotty), aspect_ratio=:equal)
RD.traj2!(Xs[1], Xs[2], xlim=xlims(p), ylim=ylims(p))

## Advance one step
# Update cost
Q = prob.obj[1].Q
q0 = copy(admm.q)
BilinearControl.updatetrajectory!(admm, Zref, Q, 2)
admm.q[1:4] ≈ -Q*RD.state(Zref[2])
admm.q[5:8] ≈ -Q*RD.state(Zref[3])
admm.q ≉ q0
J1 = BilinearControl.cost(admm, Xsol, Usol)
J1 > J0

# Update initial state
BilinearControl.setinitialstate!(admm, RD.state(Zref[2]))
admm.d[1:4] ≈ RD.state(Zref[2])

# Shift the trajectories
x0 = copy(admm.x)
z0 = copy(admm.z)
BilinearControl.shiftfill!(admm, 4, 2)
x0 ≉ admm.x
z0 ≉ admm.z
admm.x[1:end-n] ≈ x0[n+1:end]
admm.z[1:end-m] ≈ z0[m+1:end]
admm.x[end-n+1:end] ≈ x0[end-n+1:end]
admm.z[end-m+1:end] ≈ z0[end-m+1:end]
J2 = BilinearControl.cost(admm, Xsol, Usol)
J0 < J2 < J1

## Solve updated problem
X1,U1 = copy(Xsol), copy(Usol)
Xsol2, Usol = BilinearControl.solve(admm, verbose=true)

Xs = collect(eachrow(reshape(Xsol2, 4, :))) 
RD.traj2!(Xs[1], Xs[2], xlim=xlims(p), ylim=ylims(p))

## MPC step 
function mpc_step!(admm::BilinearADMM, Zref, k; doplot=true)
    Xsol, Usol = BilinearControl.solve(admm, verbose=true)
    BilinearControl.updatetrajectory!(admm, Zref, Q, k+1)
    BilinearControl.shiftfill!(admm, 4, 2)
    if doplot
        Xs = collect(eachrow(reshape(Xsol, 4, :))) 
        p = plot(Tuple.(scotty), aspect_ratio=:equal)
        RD.traj2!(Xs[1], Xs[2], xlim=xlims(p), ylim=ylims(p))
        display(p)
    end
    Xsol, Usol
end

prob = DubinsMPCProblem(Zref)
admm = BilinearADMM(prob)
admm.opts.penalty_threshold = 1e2
X = [zeros(n) for _ = 1:Nref]
X[1] .= prob.x0
iters = Int[]
model = prob.model[1] 
for k = 1:length(Zref) - prob.N - 1
    Xsol, Usol = mpc_step!(admm, Zref, k, doplot=false)
    push!(iters, admm.stats.iterations)
    u = Usol[1:m]
    RD.discrete_dynamics!(model, X[k+1], X[k], u, 0.0, dt)
    BilinearControl.setinitialstate!(admm, X[k+1])
end
px = [x[1] for x in X[1:Nref-prob.N-1]]
py = [x[2] for x in X[1:Nref-prob.N-1]]
p = plot(Tuple.(scotty), aspect_ratio=:equal, label="reference")
RD.traj2!(px, py, label="mpc", legend=:topleft)

using Statistics
mean(iters)
median(iters)
std(iters)
prob.N
Nref