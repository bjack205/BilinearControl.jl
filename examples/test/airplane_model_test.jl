using Pkg; Pkg.activate(joinpath(@__DIR__, ".."));
Pkg.instantiate();
using BilinearControl
using BilinearControl.Problems
using BilinearControl.EDMD
using Rotations
using StaticArrays
using Test
using LinearAlgebra 
import RobotDynamics as RD

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
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

x,u = rand(model_real)
RD.dynamics(model_real, x, u)
RD.dynamics(model_nom, x, u)

# Get trim
p0 = MRP(0.997156, 0., 0.075366) # initial orientation
x_trim = Vector(RD.build_state(model_nom, [-3,0,1.5], p0, [5,0,0], [0,0,0]))
u_guess = fill(124.0,4)
u_trim = Vector(Problems.get_trim(model_real, x_trim, u_guess))
@test norm(RD.dynamics(model_nom, x_trim, u_trim)[2:end]) < 1e-4

# Simulate forward
dt = 0.05
t_sim = 2.0
T_sim = range(0,t_sim,step=dt)
U = [copy(u_trim) for t in T_sim]
X_sim, = EDMD.simulate(dmodel_real, U, x_trim, t_sim, dt)
visualize!(vis, model_real, t_sim, X_sim)