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

tf = 2.0
dt = 0.05
prob = AirplaneProblem(;tf, dt)
u_trim = copy(TO.controls(prob)[1])
X = states(prob)
Altro.cost(prob)
solver = ALTROSolver(prob, verbose=4)
solve!(solver)
visualize!(vis, model_real, TO.get_final_time(prob), states(solver))
X_ref = Vector.(states(solver))
U_ref = push!(Vector.(controls(solver)), u_trim)
T_ref = TO.gettimes(solver)
norm(X_ref[end][7:9])

## Simulate open-loop with real dynamics
X_real, = simulate(dmodel_real, U_ref, X_ref[1], tf, dt)
visualize!(vis, model_real, tf, X_real)

## Simulate with Linear MPC
Qk = Diagonal([fill(1e-0, 3); fill(1e-0, 3); fill(1e-1, 3); fill(1e-1, 3)])
Rk = Diagonal(fill(1e-3,4))
Qf = 10 * copy(Qk)
mpc = EDMD.TrackingMPC(dmodel_nom, X_ref, U_ref, T_ref, Qk, Rk, Qf, Nt=21)
EDMD.getcontrol(mpc, X_ref[1], 0.0)
X_sim, = simulatewithcontroller(dmodel_real, mpc, X_ref[1], tf, dt)
visualize!(vis, model_real, tf, X_sim)