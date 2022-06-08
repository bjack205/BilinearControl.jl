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
using ForwardDiff

using BilinearControl.EDMD: chebyshev
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

## Generate training data
using DelimitedFiles
using DataFrames, CSV
using Plots
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
X_train = reshape(map(row2state, eachrow(data)), :, 1)
U_train = reshape(map(row2control, eachrow(data)), :, 1)
times_train = copy(data.time)
dt_train = round(mean(diff(data.time)), digits=4)
tf_train = data.time[end] - data.time[1]
X_sim = simulate(dmodel, U_train, X_train[1], tf_train, dt_train)

## Compare actual data to kinematic model
RD.traj2(getindex.(X_train[:,1], 1), getindex.(X_train[:,1], 2))
model = RoverKinematics(radius=0.02, width=0.2, length=0.3, vxl=-0.0, vxr=-0.0)
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
X_sim = simulate(dmodel, U_train, X_train[1], tf_train, dt_train)
RD.traj2!(getindex.(X_sim, 1), getindex.(X_sim, 2), label="simulated")

## Learn bilinear model

# Koopman function
function kf(x)
    p = x[1:3]
    q = x[4:7]
    A = Matrix(UnitQuaternion(q..., false))
    L = Rotations.lmult(SVector{4}(q))
    Aν = model.Aν
    Aω = model.Aω
    v1 = Aν'A'p
    v2 = Aω'Rotations.vmat() * q
    [
        1; x;
        A'p; 
        L'q;
        v1;
        v2;
        chebyshev(p,order=2);
        chebyshev(q,order=2);
        chebyshev(v1,order=2);
        chebyshev(v2,order=2);
    ]
end
n = length(kf(X_train[1]))

# Generate Jacobians
n0 = RD.state_dim(model)
xn = zeros(n0)
jacobians = map(CartesianIndices(U_train)) do cind
    n0 = RD.state_dim(model)
    m = RD.control_dim(model)
    k = cind[1]
    x = X_train[cind]
    u = U_train[cind]
    z = RD.KnotPoint{n0,m}(x,u,times_train[k],dt_train)
    J = zeros(n0,n0+m)
    RD.jacobian!(
        RD.InPlace(), RD.ForwardAD(), dmodel, J, xn, z 
    )
    J
end
A_train = map(J->J[:,1:n0], jacobians)
B_train = map(J->J[:,n0+1:end], jacobians)

# Convert states to lifted Koopman states
Y_train = map(kf, X_train)

# Calculate Jacobian of Koopman transform
F_train = map(@view X_train[1:end-1,:]) do x
    sparse(ForwardDiff.jacobian(kf, x))
end;

# Create a sparse version of the G Jacobian
G = spdiagm(n0,n,1=>ones(n0)) 

# Build Least Squares Problem
W,s = BilinearControl.EDMD.build_edmd_data(
    Y_train, U_train, A_train, B_train, F_train, G 
);
W
log10.(size(W))

@time Wsparse = sparse(W)