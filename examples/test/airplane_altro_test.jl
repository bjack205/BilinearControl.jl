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

## Visualizer
model = Problems.NominalAirplane()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
delete!(vis)
set_airplane!(vis, model)
open(vis)

##

##
tf = 3.0
prob = AirplaneProblem()
X = states(prob)
Altro.cost(prob)
solver = ALTROSolver(prob, verbose=4)
solve!(solver)
visualize!(vis, model_real, TO.get_final_time(prob), states(solver))
states(solver)[end][7]