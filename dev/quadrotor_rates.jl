import Pkg; Pkg.activate(@__DIR__)

using BilinearControl
using BilinearControl.Problems
import RobotDynamics as RD
import TrajectoryOptimization as TO
using LinearAlgebra
using StaticArrays
using Statistics
using Rotations
using BilinearControl.Problems: qrot, skew
using SparseArrays
using Test

using BilinearControl: getA, getB, getC, getD


## Test dynamics



# Build cost
u0 = [0,0,0,model.mass*model.gravity]
Q = Diagonal([fill(1e-2, 3); fill(1e-2, 9); fill(1e-2, 3); fill(1e-1, 3)])
Qf = Q*(N-1)
R = Diagonal([fill(1e-2,3); 1e-2])
Qbar = Diagonal(vcat([diag(Q) for i = 1:N-1]...))
Qbar = Diagonal([diag(Qbar); diag(Qf)])
Rbar = Diagonal(vcat([diag(R) for i = 1:N]...))
q = repeat(-Q*xf, N)
r = repeat(-R*u0, N)
c = 0.5*sum(dot(xf,Q,xf) for k = 1:N-1) + 0.5*dot(xf,Qf,xf) + 0.5*sum(dot(u0,R,u0) for k = 1:N)

# Build Solver
admm = BilinearADMM(Abar,Bbar,Cbar,Dbar, Qbar,q,Rbar,r,c)
X = repeat(x0, N)
U = repeat(u0, N)
admm.opts.penalty_threshold = 1e2
BilinearControl.setpenalty!(admm, 1e4)
Xsol, Usol = BilinearControl.solve(admm, X, U, verbose=true, max_iters=200)

Xs = collect(eachcol(reshape(Xsol, RD.state_dim(model), :)))
visualize!(vis, model, TO.get_final_time(prob), Xs)

# Plot controls
using Plots
Us = collect(eachrow(reshape(Usol, RD.control_dim(model), :)))
times = range(0,tf, length=N)
p1 = plot(times, Us[1], label="ω₁", ylabel="angular rates (rad/s)", xlabel="time (s)")
plot!(times, Us[2], label="ω₂")
plot!(times, Us[3], label="ω₃")
# savefig(p1, "quadrotor_angular_rates.png")

p2 = plot(times, Us[4], label="", ylabel="Thrust", xlabel="times (s)")
# savefig(p2, "quadrotor_force.png")