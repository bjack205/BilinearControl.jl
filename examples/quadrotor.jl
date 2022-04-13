import Pkg; Pkg.activate(@__DIR__)

using BilinearControl
using BilinearControl.Problems
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


## With Rate Limiting (by penalty, not constraint)
model2 = QuadrotorRateLimited()
admm = Problems.QuadrotorRateLimitedSolver()
BilinearControl.setpenalty!(admm, 1e4)
Xsol2, Usol2 = BilinearControl.solve(admm, verbose=true, max_iters=200)

X2s = collect(eachcol(reshape(Xsol2, RD.state_dim(model2), :)))
visualize!(vis, model2, TO.get_final_time(prob), X2s)

Us = collect(eachrow(reshape(Usol, RD.control_dim(model), :)))
U2s = collect(eachrow(reshape(Usol2, RD.control_dim(model2), :)))
times = TO.gettimes(prob)
p1 = plot(times[1:end-1], Us[1], label="ω₁", ylabel="angular rates (rad/s)", xlabel="time (s)")
plot!(times[1:end-1], Us[2], label="ω₂")
plot!(times[1:end-1], Us[3], label="ω₃")
plot!(times, U2s[1], c=1, s=:dash, label="")
plot!(times, U2s[2], c=2, s=:dash, label="")
plot!(times, U2s[3], c=3, s=:dash, label="")
savefig(p1, joinpath(@__DIR__, "..", "images", "quadrotor_angular_rates_comparison.png"))

## Visualize both
using Colors
function comparison(vis, model, tf, X1, X2)
    delete!(vis)
    setquadrotor!(vis["quad1"])
    setquadrotor!(vis["quad2"], color=RGBA(0,0,1,0.5))
    N = length(X1)
    fps = Int(floor((N-1)/tf))
    anim = MeshCat.Animation(fps)
    for k = 1:N
        atframe(anim, k) do 
            visualize!(vis["quad1"], model, X1[k])
            visualize!(vis["quad2"], model, X2[k])
        end
    end
    setanimation!(vis, anim)
end

tf = TO.get_final_time(prob)
comparison(vis, model, tf, X2s, Xs)