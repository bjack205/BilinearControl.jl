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
include(joinpath(@__DIR__, "..", "..", "examples", "visualization", "visualization.jl"))

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
x0 = vec(Float64[
    0 0 1 1
    1 0 1 0
    0 0 0 0
    1 1 1 1
])
x1 = vec(Float64[
    1 0 1 0
    0 0 -1 -1 
    1 1 1 1
    0 0 0 0
])
xf = vec(Float64[
    0 0 1 1
    2 1 2 1
    0 0 0 0
    1 1 1 1
])
visualize!(vis, model, xf)

relcons = [
    (i=1,j=2,x=-1,y=0, α=1, β=0),
    (i=1,j=3,x=0,y=-1, α=1, β=0),
    (i=2,j=4,x=0,y=-1, α=1, β=0),
    # (i=3,j=4,x=-1,y=0, α=1, β=0),
]
F = buildformationconstraint(model, relcons)
norm(F*x) > 1e-3
norm(F*x0) < 1e-8
norm(F*x2) < 1e-8

# Bilinear Constraint (includes formation constraints)
Af = [
    Matrix(I,4, 4P);
    F
]
bf = [-xf[1:4]; zeros(size(F,1))]

N = 51
tf = 3.0
h = tf / (N-1)
Abar,Bbar,Cbar,Dbar = BilinearControl.buildbilinearconstraintmatrices(model, x0, Af, bf, h, N)

Xs = [rand(model)[1] for k = 1:N]
Us = [rand(model)[2] for k = 1:N-1]
X = vcat(Xs...)
U = vcat(Us...)

p = size(F,1)
c = Abar*X + Bbar*U + sum(U[i] * Cbar[i] * X for i = 1:length(U)) + Dbar
@test c[1:4P] ≈ x0 - Xs[1]
@test c[4P+1:8P] ≈ h*(A*(Xs[1] + Xs[2])/2 + B*Us[1] + sum(Us[1][i] * C[i] * (Xs[1]+Xs[2])/2 for i = 1:2P) + D) + Xs[1] - Xs[2]
p = 0
@test all(1:N-2) do k
    c[8P + (k-1)*(4P + p) .+ (1:4P)] ≈ h*(A*(Xs[k+1] + Xs[k+2])/2 + B*Us[k+1] + sum(Us[k+1][i] * C[i] * (Xs[k+1]+Xs[k+2])/2 for i = 1:2P) + D) + Xs[k+1] - Xs[k+2]
end
# @test all(1:N-2) do k
#     c[12P + (k-1)*(4P + p) .+ (1:p)] ≈ F*Xs[k+1]
# end
@test c[end-size(Af,1)+1:end] ≈ Af*Xs[end] + bf

# Objective 
Q = Diagonal(repeat([1,1,0.1,0.1], P))
R = Diagonal(repeat([0.1,0.1], P))
Qf = Q * (N-1)
Qbar,Rbar,qbar,rbar,cbar = BilinearControl.buildcostmatrices(Q,R,Qf,xf,N)

Qbar = blockdiag([sparse(F'F) for k = 1:N]...)
qbar = zeros(4P*N)

# Initial trajectory
Xs = map(range(0,1,length=N)) do θ 
    x0 .+ (xf .- x0) .* θ
end
X = vcat(Xs...)
U = zeros((N-1)*2P)

# ADMM Solver
solver = BilinearADMM(Abar, Bbar, Cbar, Dbar, Qbar, qbar, Rbar, rbar, cbar)
BilinearControl.setpenalty!(solver, 1e4)
solver.opts.penalty_threshold = 1e2
Xsol, Usol = BilinearControl.solve(solver, X, U, verbose=true, max_iters=500)

Xs = collect(eachcol(reshape(Xsol, 4P, :)))
visualize!(vis, model, tf, Xs)