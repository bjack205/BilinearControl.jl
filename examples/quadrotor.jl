import Pkg; Pkg.activate(@__DIR__)

using BilinearControl
import RobotDynamics as RD
import TrajectoryOptimization as TO
using LinearAlgebra
using StaticArrays
using Statistics
using Rotations

## Visualization 
using MeshCat
vis = Visualizer()
open(vis)
visdir = joinpath(@__DIR__, "../examples/visualization")
include(joinpath(visdir, "visualization.jl"))
setquadrotor!(vis)

## Solve with ADMM 
prob = Problems.QuadrotorProblem()
model = prob.model[1].continuous_dynamics
admm = BilinearADMM(prob)
X = extractstatevec(prob)
U = extractcontrolvec(prob)
admm.opts.penalty_threshold = 1e2
BilinearControl.setpenalty!(admm, 1e4)
Xsol, Usol = BilinearControl.solve(admm, X, U, verbose=true)

Xs = collect(eachcol(reshape(Xsol, RD.state_dim(model), :)))
visualize!(vis, model, TO.get_final_time(prob), Xs)

using Plots
Us = collect(eachrow(reshape(Usol, RD.control_dim(model), :)))
times = TO.gettimes(prob)
p1 = plot(times[1:end-1], Us[1], label="ω₁", ylabel="angular rates (rad/s)", xlabel="time (s)")
plot!(times[1:end-1], Us[2], label="ω₂")
plot!(times[1:end-1], Us[3], label="ω₃")
savefig(p1, "quadrotor_angular_rates.png")

p2 = plot(times[1:end-1], Us[4], label="", ylabel="Thrust", xlabel="times (s)")
savefig(p2, "quadrotor_force.png")



