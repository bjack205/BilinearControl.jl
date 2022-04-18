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


## Alternative Rate limiter
O = 2
model = QuadrotorRateLimited{O}()
n,m = RD.dims(model)
tf = 3.0
N = 101
h = tf / (N-1)
admm = Problems.QuadrotorRateLimitedSolver(O, tf=tf, N=N)
x0 = Vector(admm.d[1:n])
xf = Vector(admm.d[end-n+1:end])
X = copy(admm.x)
U = copy(admm.z)
admm.opts.penalty_threshold = 1e2
admm.opts.ϵ_abs_dual = 1.0
admm.opts.ϵ_rel_dual = 1.0
BilinearControl.setpenalty!(admm, 1e6)
Xsol, Usol = BilinearControl.solve(admm, X, U, verbose=true, max_iters=400)
# BilinearControl.setpenalty!(admm, 1e3)
# admm.opts.penalty_threshold = 1e8
# Xsol2, Usol2 = BilinearControl.solve(admm, Xsol, Usol, verbose=true, max_iters=1000)

Xs = collect(eachcol(reshape(Xsol, RD.state_dim(model), :)))
Us = collect(eachcol(reshape(Usol, RD.control_dim(model), :)))
visualize!(vis, model, tf, Xs)

##
c = BilinearControl.eval_c(admm, Xsol, Usol)
c[1:n] ≈ (x0 - Xs[1])
c[n+1:2n][16:18] ≈ Xs[2][16:18]*h - (Us[1][1:3] - Xs[1][19:21])
c[n+1:2n][19:21] ≈ Xs[2][19:21] - Us[1][1:3]
c[n+1:2n][1:3] ≈ h*(Xs[1][13:15] + Xs[2][13:15]) / 2 + Xs[1][1:3] - Xs[2][1:3]

c = abs.(c)
findmax(c)
x0[10] - Xs[1][10]

## Test dynamics
using BilinearControl.Problems: qrot, skew
using BilinearControl: getA, getB, getC, getD
using Test
model = QuadrotorRateLimited{2}()
n,m = RD.dims(model)
@test n == 21
@test m == 4

r1,r2 = randn(3), randn(3) 
R1,R2 = qrot(normalize(randn(4))), qrot(normalize(randn(4)))
v1,v2 = randn(3), randn(3) 
α1,α2 = randn(3), randn(3) 
ω1,ω2 = randn(3), randn(3) 
ωp1, ωp2 = randn(3), randn(3)
F1,F2 = rand(), rand()

x1 = [r1; vec(R1); v1; α1; ωp1]
x2 = [r2; vec(R2); v2; α2; ωp2]
u1 = [ω1; F1]
u2 = [ω2; F2]

h = 0.1
z1 = RD.KnotPoint{n,m}(n,m,[x1;u1],0.0,h)
z2 = RD.KnotPoint{n,m}(n,m,[x2;u2],h,h)
err = RD.dynamics_error(model, z2, z1)

@test err[1:3] ≈ h * (v1 + v2) / 2 + r1 - r2
@test err[4:12] ≈ vec(h * (R1 + R2) /2 * skew(ω1) + R1 - R2)
@test err[13:15] ≈ h*( (R1 + R2) /2 * [0,0,F1]) / model.mass - 
    h*[0,0,model.gravity] + v1 - v2
@test err[16:18] ≈ h*α2 - (ωp2 - ωp1)
@test err[19:21] ≈ ωp2 - ω1

# Test dynamics match bilinear dynamics
A,B,C,D = getA(model,h), getB(model,h), getC(model,h), getD(model,h)
x12 = [x1;x2]
u12 = [u1;u2]
err2 = A*x12 + B*u12 + sum(u12[i]*C[i]*x12 for i = 1:length(u12)) + D
@test err ≈ err2