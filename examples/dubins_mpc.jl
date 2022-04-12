import Pkg; Pkg.activate(@__DIR__)

using BilinearControl
using COSMOAccelerators
import RobotDynamics as RD
import TrajectoryOptimization as TO
using BilinearControl.Problems
using StaticArrays
using LinearAlgebra
using Plots

## Generate reference trajectory
include("scotty.jl")
scotty_interp = generate_scotty_trajectory(scale=0.01)
Nref = 501
tref = 50  # time of entire trajectory
dt = tref / (Nref - 1)
s = range(0, knots(scotty_interp).knots[end], length=Nref)
scotty = scotty_interp.(s)

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

p = plot(Tuple.(scotty), aspect_ratio=:equal)

## MPC step 
function mpc_step!(admm::BilinearADMM, Zref, k; doplot=true, verbose=false)
    Xsol, Usol = BilinearControl.solve(admm, verbose=verbose)
    if doplot
        Xs = collect(eachrow(reshape(Xsol, 4, :))) 
        p = plot(Tuple.(scotty), aspect_ratio=:equal)
        RD.traj2!(Xs[1], Xs[2], xlim=xlims(p), ylim=ylims(p))
        display(p)
    end
    Xsol, Usol
end

prob = Problems.DubinsMPCProblem(Zref)
admm = BilinearADMM(prob)
admm.opts.penalty_threshold = 1e2
X = [zeros(n) for _ = 1:Nref]
X[1] .= prob.x0
iters = Int[]
model = prob.model[1] 
begin
    tstart = time()
    for k = 1:length(Zref)-1
        Xsol, Usol = mpc_step!(admm, Zref, k, doplot=true)
        push!(iters, admm.stats.iterations)
        println("time step $k took $(iters[end]) iterations")
        u = Usol[1:m]
        RD.discrete_dynamics!(model, X[k+1], X[k], u, 0.0, dt)
        BilinearControl.updatetrajectory!(admm, Zref, Q, k+1)
        BilinearControl.shiftfill!(admm, 4, 2)
        BilinearControl.setinitialstate!(admm, X[k+1])
    end
    tmpc  = time() - tstart
end
tmpc
500 / tmpc
px = [x[1] for x in X]
py = [x[2] for x in X]
p = plot(Tuple.(scotty), aspect_ratio=:equal, label="reference")
RD.traj2!(px, py, label="mpc", legend=:topleft)

using Statistics
mean(iters)
median(iters)
std(iters)
prob.N
Nref