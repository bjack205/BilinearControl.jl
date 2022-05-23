import Pkg; Pkg.activate(@__DIR__)
using Rotations

using BilinearControl
using BilinearControl.Problems
import RobotDynamics as RD
import TrajectoryOptimization as TO
using LinearAlgebra
using StaticArrays
using Statistics
using Rotations
using SparseArrays

using BilinearControl.Problems: simulatewithcontroller

## Visualization 
using MeshCat
vis = Visualizer()
open(vis)
visdir = joinpath(@__DIR__, "../examples/visualization")
include(joinpath(visdir, "visualization.jl"))
model = RoverKinematics()
delete!(vis)
set_rover!(vis["robot"], model, tire_width=0.07)

## Try simulating a few trajectories
forward(w) = [w,w,w,w]
turn(w) = [-w,w,-w,w]
ctrl = Problems.ConstController(forward(2.0) + turn(-1.0))
model = RoverKinematics(vxl=-0.0, vxr=-0.0)
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
dt = 0.01
tf = 1.0
x0 = [zeros(3); 1; zeros(3)]
Xsim, Usim = simulatewithcontroller(dmodel, ctrl, x0, tf, dt)
model.Aν*(-model.B * forward(1.0))
model.Aω*(-model.B * forward(1.0))


visualize!(vis, model, tf, Xsim)
[x[1] for x in Xsim]