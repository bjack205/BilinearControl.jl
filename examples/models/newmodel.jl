using RobotDynamics
using LinearAlgebra
using StaticArrays

const RD = RobotDynamics

struct MyDubinsCar <: RD.ContinuousDynamics
    mass::Float64
end

RD.state_dim(::MyDubinsCar) = 3
RD.control_dim(::MyDubinsCar) = 2

function RD.dynamics(model::MyDubinsCar, x, u)
    xdot = SA[
        u[1] * cos(x[3])
        u[2] * sin(x[3])
        u[2]
    ]
end

function RD.dynamics!(model::MyDubinsCar, xdot, x, u)
    xdot[1] = u[1] * cos(x[3])
    xdot[2] = u[2] * sin(x[3])
    xdot[3] = u[2]
end

function RD.jacobian!(model::MyDubinsCar, J, xdot, x, u)
    J .= 1
end

model = MyDubinsCar(1.0)
x,u = rand(model)
RD.dynamics(model, x, u)
n,m = RD.dims(model)
xdot = zeros(n)
RD.dynamics!(model, xdot, x, u)

J = zeros(n, n+m)
z = KnotPoint(x, u, 0.0, NaN)
RD.jacobian!(RD.StaticReturn(), RD.UserDefined(), model, J, xdot, z)
RD.jacobian!(model, J, xdot, x, u)

# discretization
t = 0.0
dt = 0.1
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
RD.discrete_dynamics(dmodel, x, u, t, dt)
RD.jacobian!(RD.StaticReturn(), RD.UserDefined(), dmodel, J, xdot, z)

using TrajectoryOptimization
const TO = TrajectoryOptimization

# discretization
tf = 1.0
N = 101
dt = tf / (N-1)

# Initial state
x0 = zeros(n)
xf = [1,1,deg2rad(90)]

# Objective
Q = Diagonal(fill(1.0, n))
R = Diagonal(fill(0.1, m))