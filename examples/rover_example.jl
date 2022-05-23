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

using BilinearControl.Problems: simulatewithcontroller, simulate

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

## Get collected data
using DelimitedFiles
using DataFrames, CSV
using Plots
datafile = joinpath(@__DIR__, "../data/slam_bag1_2022-04-06-19-50-04/rover_data_processed.csv")
data = DataFrame(CSV.File(datafile))
row2state(row) = [
    row.vicon_x, row.vicon_y, row.vicon_z, 
    row.vicon_qw, row.vicon_qx, row.vicon_qy, row.vicon_qz,
]
row2control(row) = [
    # row.cmd_fl,
    # row.cmd_fr,
    # row.cmd_rl,
    # row.cmd_rr,
    row.wheel_vel_fl,
    row.wheel_vel_fr,
    row.wheel_vel_rl,
    row.wheel_vel_rr,
]
norm(data.wheel_vel_fl)
X_train = map(row2state, eachrow(data))
U_train = map(row2control, eachrow(data))
dt_train = round(mean(diff(data.time)), digits=4)
tf_train = data.time[end] - data.time[1]
X_sim = simulate(dmodel, U_train, X_train[1], tf_train, dt_train)

## Compare actual data to kinematic model
RD.traj2(getindex.(X_train, 1), getindex.(X_train, 2))
model = RoverKinematics(radius=0.02, width=0.2, length=0.3, vxl=-0.0, vxr=-0.0)
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
X_sim = simulate(dmodel, U_train, X_train[1], tf_train, dt_train)
RD.traj2!(getindex.(X_sim, 1), getindex.(X_sim, 2), label="simulated")