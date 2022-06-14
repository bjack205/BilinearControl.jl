using Pkg; Pkg.activate(joinpath(@__DIR__, ".."));
Pkg.instantiate();
using BilinearControl
using BilinearControl.Problems
using BilinearControl.EDMD
using Rotations
using StaticArrays
using Test
using LinearAlgebra 
using Altro
using RobotDynamics
using TrajectoryOptimization
const TO = TrajectoryOptimization
import RobotDynamics as RD
using BilinearControl: Problems
using JLD2
using Plots

include("airplane_problem.jl")

## Visualizer
model = Problems.NominalAirplane()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
delete!(vis)
set_airplane!(vis, model)
open(vis)

##
model_nom = Problems.NominalAirplane()
model_real = Problems.SimulatedAirplane()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

##
tf = 2.0
dt = 0.05
prob = AirplaneProblem(;tf, dt, Qv=10, Qw=5, dp=[-1.5,-2,2])
u_trim = copy(TO.controls(prob)[1])
X = states(prob)
Altro.cost(prob)
solver = ALTROSolver(prob, verbose=0)
solve!(solver)
visualize!(vis, model_real, TO.get_final_time(prob), states(solver))
X_ref = Vector.(states(solver))
U_ref = push!(Vector.(controls(solver)), u_trim)
T_ref = TO.gettimes(solver)
norm(X_ref[end][7:end])
# jldsave(joinpath(Problems.DATADIR, "plane_data.jld2"); X_ref, U_ref, T_ref, u_trim)


## Simulate with OSQP controller
# res = load(joinpath(Problems.DATADIR, "plane_data.jld2"))
# X_ref = res["X_ref"]
# U_ref = res["U_ref"]
# T_ref = res["T_ref"]
# u_trim = res["u_trim"] 
# dt = T_ref[2] 

n,m = 12,4
Nt = 21
Qk = Diagonal([fill(1e0, 3); fill(1e1, 3); fill(1e-1, 3); fill(2e-1, 3)])
Rk = Diagonal(fill(1e-3,4))
Qf = Diagonal([fill(1e-2, 3); fill(1e0, 3); fill(1e1, 3); fill(1e1, 3)]) * 10
t_mpc = (Nt-1) * dt
xmax = [fill(0.5,3); fill(1.0, 3); fill(0.5, 3); fill(10.0, 3)]
xmin = -xmax
umin = fill(0.0, 4) - u_trim
umax = fill(255.0, 4) - u_trim
mpc = EDMD.LinearMPC(dmodel_nom, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
    xmin,xmax,umin,umax
)

t_sim = T_ref[end]
X_mpc,_,T_mpc = simulatewithcontroller(dmodel_real, mpc, X_ref[1], t_sim, dt)

visualize!(vis, model_real, t_sim, X_mpc)
plotstates(T_ref, X_ref, inds=[1,3,6,7], label=["x" "z" "pitch" "vx"], legend=:outerright)
plotstates!(T_mpc, X_mpc, inds=[1,3,6,7], s=:dash, c=[1 2 3 4], lw=2)

## Simulate open-loop with real dynamics
X_real, = simulate(dmodel_real, U_ref, X_ref[1], tf, dt)
visualize!(vis, model_real, tf, X_real)
norm(X_real[end][7:end])

## Simulate with TVLQR
ilqr = Altro.get_ilqr(solver)
tvlqr = EDMD.TVLQRController(ilqr.K, states(ilqr), controls(ilqr), TO.gettimes(ilqr))
usat(u) = clamp.(u, 0, 255)
X_sim,U_sim,T_sim = simulatewithcontroller(dmodel_real, tvlqr, X_ref[1], tf, dt, umod=usat)
visualize!(vis, model_real, tf, X_sim)
norm(X_sim[end][7:end])

##
#
dx = zeros(n)
dx[1] = 0.0
x = X_ref[1] + dx
k = 1
t = 0.0
N_ref = length(X_ref)

k = 1 
dx = zeros(n)
dx[1] = 0.1
x = X_ref[k] + dx

## Solve for control


##
t = (k-1)*dt
Nh = min(Nt, N_ref - k)
mpc_inds = k-1 .+ (1:Nh)
A = Aref[mpc_inds[1:end-1]]
B = Bref[mpc_inds[1:end-1]]
f = [zeros(n) for k = 1:Nh-1]
Q = [copy(Qk) for k = 1:Nh]
Q[end] .= Qf
R = [copy(Rk) for k = 1:Nh-1]
q = [zeros(n) for k = 1:Nh]
r = [zeros(m) for k = 1:Nh-1]

dx = x - X_ref[k]
# dX,dU, = EDMD.solve_lqr_osqp(Q,R,q,r,A,B,f,dx; xmin, xmax, umin, umax)
# dX,dU, = EDMD.solve_lqr_osqp(Q,R,q,r,A,B,f,dx)
# u = dU[1] + U_ref[k]
u = EDMD.getcontrol(mpc, x, t)
dX = mpc.X[1:Nh]
u
x

# Simulate forward
x = RD.discrete_dynamics(dmodel_real, x, u, t, dt)

X_mpc = X_ref[mpc_inds] .+ dX
T_mpc = t .+ range(0,length=Nh,step=dt)
t += dt
k += 1
@show k

T_ref[k-1]
X_ref[k-1]
X_mpc[1]
plotstates(t_ref, x_ref, inds=[1,3,6,7], label=["x" "z" "pitch" "vx"], legend=:outerright)
plotstates!(t_mpc, x_mpc, inds=[1,3,6,7], s=:dash, c=[1 2 3 4], lw=2)

## Simulate with Linear MPC
Qk = Diagonal([fill(1e-0, 3); fill(1e-0, 3); fill(1e-1, 3); fill(1e-1, 3)])
Rk = Diagonal(fill(1e-3,4))
Qf = 10 * copy(Qk)
mpc = EDMD.TrackingMPC(dmodel_nom, X_ref, U_ref, T_ref, Qk, Rk, Qf, Nt=21)
EDMD.getcontrol(mpc, X_ref[1], 0.0)
X_sim, = simulatewithcontroller(dmodel_real, mpc, X_ref[1], tf, dt)
visualize!(vis, model_real, tf, X_sim)