using RobotZoo
using Test
using ForwardDiff
using FiniteDiff
import RobotDynamics as RD
include("QOC/QOC.jl")
using .QOC

model0 = RD.DiscretizedDynamics{RD.RK4}(RobotZoo.DubinsCar())
model = QOC.ControlDerivative(model0)
@test RD.dims(model) == (7,2,7)

x0,u0 = Vector.(rand(model0))
x = [x0; zeros(4)]
dt = 0.1
z0 = KnotPoint{3,2}(x0,u0,0.0,dt)
z = KnotPoint{7,2}(x,u0,0.0,dt)
xn0 = zero(x0)
xn = zero(x)
RD.discrete_dynamics!(model, xn, z)
@test xn[1:3] ≈ RD.discrete_dynamics(model0, z0)
@test xn[4:5] ≈ u0
@test xn[6:7] ≈ u0/dt

n0,m = RD.dims(model0)
n = RD.state_dim(model)
J0 = zeros(n0, n0+m)
J = zeros(n, n+m)
RD.jacobian!(RD.InPlace(), RD.ForwardAD(), model0, J0, xn0, z0)
RD.jacobian!(RD.InPlace(), RD.ForwardAD(), model, J, xn, z)
@test J[1:n0,1:n0] ≈ J0[:,1:n0]
@test J[1:n0, n+1:end] ≈ J0[:,n0+1:end]

f!(y,z) = RD.discrete_dynamics!(model, y, z[1:n], z[n+1:end], 0.0, dt)
Jfd = zero(J)
FiniteDiff.finite_difference_jacobian!(Jfd, f!, z.z)
@test Jfd ≈ J