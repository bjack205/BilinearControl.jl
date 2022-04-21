import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using BilinearControl
using COSMOAccelerators
import RobotDynamics as RD
import TrajectoryOptimization as TO
using FiniteDiff 
using BilinearControl.Problems
using BilinearControl: getA, getB, getC, getD
using Test

include("swarm_model.jl")

P = 4
model0 = BilinearDubins() 
model = SwarmSE2{P}()
x,u = rand(model)
X = reshape(x, 4, P)
U = reshape(u, 2, P)

vis = Visualizer()
setswarm!(vis, model)
visualize!(vis, model, x)

# Check the phasors are normalized
@test all(eachcol(X)) do x
    norm(x[3:4]) ≈ 1.0
end

# Check dynamics match dubins car
xdot = zeros(4P)
Xdot = reshape(xdot, :, P)
RD.dynamics!(model, xdot, x, u)
@test all(1:P) do i
    Xdot[:,i] ≈ RD.dynamics(model0, X[:,i], U[:,i])
end

# Check Jacobian
Jfd = zeros(4P,6P)
FiniteDiff.finite_difference_jacobian!(Jfd, (y,z)->RD.dynamics!(model, y, z[1:4P], z[4P+1:end]), [x; u])
J = zeros(4P,6P)
RD.jacobian!(model, J, xdot, x, u)
@test J ≈ Jfd

# Check Bilinear matrices
A,B,C,D = getA(model), getB(model), getC(model), getD(model)
@test xdot ≈ A*x + B*u + sum(u[i]*C[i] * x for i = 1:length(u)) + D

# Formation constraints
x1 = vec(Float64[
    0 0 1 1
    1 0 1 0
    0 0 0 0
    1 1 1 1
])
x2 = vec(Float64[
    1 0 1 0
    0 0 -1 -1 
    1 1 1 1
    0 0 0 0
])
visualize!(vis, model, x1)

relcons = [
    (i=1,j=2,x=-1,y=0),
    (i=1,j=3,x=0,y=-1),
    (i=2,j=4,x=0,y=-1),
]
F = buildformationconstraint(model, relcons)
norm(F*x) > 1e-3
norm(F*x1) < 1e-8
norm(F*x2) < 1e-8
