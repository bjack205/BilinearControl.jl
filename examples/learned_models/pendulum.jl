import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using BilinearControl
using BilinearControl.Problems
using BilinearControl.EDMD
import RobotDynamics as RD
import TrajectoryOptimization as TO
using RobotZoo
using LinearAlgebra
using StaticArrays
using SparseArrays
# using MeshCat, GeometryBasics, Colors, CoordinateTransformations, Rotations
using Plots
using Distributions
using Distributions: Normal
using Random
using JLD2
using Altro
using BilinearControl: Problems
using Test

include("edmd_utils.jl")

function pendulum_kf(x)
    p,v = x
    s,c = sincos(p)
    s2,c2 = sincos(2p)
    s3,c3 = sincos(3p)
    [1,s,c,s2,s3, p*s,p*c, v*s,v*c]
end

function genpendulumproblem(x0=[0.,0.], Qv=1.0, Rv=1.0, Qfv=1000.0, u_bnd=3.0, tf=3.0; dt=0.05)
    model = RobotZoo.Pendulum()
    dmodel = RD.DiscretizedDynamics{RD.RK4}(model) 
    n,m = RD.dims(model)
    N = floor(Int, tf / dt) + 1

    # cost
    Q = Diagonal(@SVector fill(Qv,n))
    R = 1e-3*Diagonal(@SVector fill(Rv,m))
    Qf = 1e-0*Diagonal(@SVector fill(Qfv,n))
    xf = @SVector [pi, 0.0]  # i.e. swing up
    obj = TO.LQRObjective(Q*dt,R*dt,Qf,xf,N)

    # constraints
    conSet = TO.ConstraintList(n,m,N)
    bnd = TO.BoundConstraint(n,m,u_min=-u_bnd,u_max=u_bnd)
    goal = TO.GoalConstraint(xf)
    TO.add_constraint!(conSet, bnd, 1:N-1)
    TO.add_constraint!(conSet, goal, N:N)

    # problem
    times = range(0,tf,length=N)
    U = [SA[cos(t/2)] for t in times]
    pendulum_static = TO.Problem(dmodel, obj, x0, tf, constraints=conSet, xf=xf)
    TO.initial_controls!(pendulum_static, U)
    TO.rollout!(pendulum_static)
    return pendulum_static
end

mutable struct PendulumParams 
    x0::Product{Continuous, Uniform{Float64}, Vector{Uniform{Float64}}}
    QRratio::Uniform{Float64}
    Qfratio::Uniform{Float64}  # log of ratio
    tf::Uniform{Float64}
    u_bnd::Uniform{Float64}
    dt::Float64
    function PendulumParams(;x0_bnd=[pi/2,10], QRratio=[0.5, 10], Qfratio=[1.0, 3], 
                             tf=[2.0, 6.0], u_bnd=[2.0, 6.0], dt=0.05)
        x0_sampler = Product([Uniform(-x0_bnd[i],x0_bnd[i]) for i = 1:2])
        QR_sampler = Uniform(QRratio[1], QRratio[2])
        Qf_sampler = Uniform(Qfratio[1], Qfratio[2])
        tf_sampler = Uniform(tf[1], tf[2])
        u_bnd_sampler = Uniform(u_bnd[1], u_bnd[2])
        new(x0_sampler, QR_sampler, Qf_sampler, tf_sampler, u_bnd_sampler, dt)
    end
end

function Base.rand(params::PendulumParams) 
    x0 = rand(params.x0) 
    R = 1.0 
    Q = rand(params.Qfratio)
    Qf = 10^(rand(params.Qfratio)) * Q
    u_bnd = rand(params.u_bnd)
    tf_raw = rand(params.tf)
    N = round(Int, tf_raw / params.dt) + 1
    tf = params.dt * (N - 1)
    (x0=x0, Qv=Q, Rv=R, Qfv=Qf, u_bnd=u_bnd, tf=tf)
end

## Visualization
using MeshCat
model = RobotZoo.Pendulum()
visdir = joinpath(@__DIR__, "../../examples/visualization/")
include(joinpath(visdir, "visualization.jl"))
vis = Visualizer()
open(vis)
delete!(vis)
set_pendulum!(vis)

## Generate ALTRO data
Random.seed!(1)
model = RobotZoo.Pendulum()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
num_traj = 100
tf = 4.0
dt = 0.05

# Training data
params_sampler = PendulumParams(tf=[tf-eps(tf),tf+eps(tf)])  # restrict it to a set horizon, for now
opts = Altro.SolverOptions(show_summary=false)
ctrl = ALTROController(genpendulumproblem, params_sampler, opts=opts)
resetcontroller!(ctrl, zeros(2))
X,U = simulatewithcontroller(dmodel, ctrl, zeros(2), tf, dt)

x0_sampler = Product([Uniform(-pi/4,pi/4), Normal(-2, 2)])
initial_conditions = tovecs(rand(x0_sampler, num_traj), length(x0_sampler))
X_train, U_train = create_data(dmodel, ctrl, initial_conditions, tf, dt)

all(1:num_traj) do i
    simulate(dmodel, U_train[:,i], initial_conditions[i], tf, dt) ≈ X_train[:,i]
end

# Test data
num_traj_test = 10
x0_sampler_test = Product([Uniform(-eps(),eps()), Normal(-eps(), eps())])
initial_conditions_test = tovecs(rand(x0_sampler_test, num_traj_test), length(x0_sampler_test))
params_sampler_test = PendulumParams(tf=[tf-eps(tf),tf+eps(tf)], QRratio=[1.0, 1.1], Qfratio=[2.0,3.0], u_bnd=[4.0,5.5])
ctrl_test = ALTROController(genpendulumproblem, params_sampler_test, opts=opts)
X_test, U_test = create_data(dmodel, ctrl_test, initial_conditions_test, tf, dt)
all(1:num_traj_test) do i
    simulate(dmodel, U_test[:,i], initial_conditions_test[i], tf, dt) ≈ X_test[:,i]
end
jldsave(joinpath(Problems.DATADIR, "pendulum_altro_trajectories.jld2"); X_train, U_train, X_test, U_test, tf, dt)

# ## Generate training data
# model = RobotZoo.Pendulum()
# dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
# num_traj = 1000
# tf = 3.0
# dt = 0.05
# ctrl_1 = RandomController(model, Uniform(-5.,5.))
# ctrl_2 = RandConstController(Product([Uniform(-7,7)]))
# Q = Diagonal([1.0, 0.1])
# R = Diagonal(fill(1e-4, 1))
# xeq = [pi,0]
# ueq = [0.]
# ctrl_3 = LQRController(dmodel, Q, R, xeq, ueq, dt)

# x0_sampler_1 = Product([Uniform(-eps(),0), Normal(0.0, 0.0)])
# initial_conditions_1 = tovecs(rand(x0_sampler_1, num_traj), length(x0_sampler_1))
# X_train_1, U_train_1 = create_data(dmodel, ctrl_1, initial_conditions_1, tf, dt)

# x0_sampler_2 = Product([Uniform(-pi/4,pi/4), Normal(0.0, 2.0)])
# initial_conditions_2 = tovecs(rand(x0_sampler_2, num_traj), length(x0_sampler_2))
# X_train_2, U_train_2 = create_data(dmodel, ctrl_2, initial_conditions_1, tf, dt)

# x0_sampler_3 = Product([Uniform(pi-pi, pi+pi), Normal(0.0, 4.0)])
# initial_conditions_3 = tovecs(rand(x0_sampler_3, num_traj), length(x0_sampler_3))
# X_train_3, U_train_3 = create_data(dmodel, ctrl_2, initial_conditions_3, tf, dt)

# X_train = hcat(X_train_1, X_train_2, X_train_3)
# U_train = hcat(U_train_1, U_train_2, U_train_3)

# ## Generate test data
# Random.seed!(1)
# num_traj_test = 8
# tf_test = tf
# initial_conditions = tovecs(rand(x0_sampler_1, num_traj_test), length(x0_sampler_1))
# # initial_conditions = [zeros(2) for i = 1:num_traj_test]
# X_test, U_test = create_data(dmodel, ctrl_1, initial_conditions, tf_test, dt)

## Learn Bilinear Model
model = RobotZoo.Pendulum()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)

altro_datafile = joinpath(Problems.DATADIR, "pendulum_altro_trajectories.jld2")
lqr_datafile = joinpath(Problems.DATADIR, "pendulum_lqr_trajectories.jld2")
X_train_altro = load(altro_datafile, "X_train")
U_train_altro = load(altro_datafile, "U_train")
X_train_lqr = load(lqr_datafile, "X_train")[:,1:100]
U_train_lqr = load(lqr_datafile, "U_train")[:,1:100]
X_test = load(altro_datafile, "X_test")
U_test = load(altro_datafile, "U_test")
tf = load(altro_datafile, "tf")
dt = load(altro_datafile, "dt")
@assert load(lqr_datafile, "tf") == tf
@assert load(lqr_datafile, "dt") == dt 
X_train = [X_train_altro X_train_lqr]
U_train = [U_train_altro U_train_lqr]

# Learn bilinear model
eigfuns = ["state", "sine", "cosine", "chebyshev"]
eigorders = [0,0,0,5]
Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders)

F, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
    ["ridge", "lasso"]; 
    edmd_weights=[10.1], 
    mapping_weights=[0.0], 
    algorithm=:cholesky
);

model_bilinear = EDMDModel(F,C,g,kf,dt,"pendulum")
RD.dims(model_bilinear)

norm(bilinearerror(model_bilinear, X_train, U_train)) / length(U_train)
norm(bilinearerror(model_bilinear, X_test, U_test)) / length(U_test)

## MPC Controller
n,m = RD.dims(model_bilinear)
n0 = originalstatedim(model_bilinear)
i = 1  # test trajectory index
x_max = [2pi, 1000]
u_max = [20]
x_min = -x_max 
u_min = -u_max 

# Reference trajectory
i = 1  # test trajectory index
x0 = copy(X_test[1,i])
xf = [pi,0]
N = length(X_test[:,i])
Xref = X_test[:,i]
Uref = U_test[:,i]
tref = range(0,length=N,step=dt)
Xref[end] = xf
push!(Uref, zeros(m))

# MPC controller
Nmpc = 41
Qmpc = Diagonal([10.0,0.1])
Rmpc = Diagonal([1e-2])
ctrl_mpc = BilinearMPC(
    model_bilinear, Nmpc, x0, Qmpc, Rmpc, Xref, Uref, tref;
    x_max, x_min, u_max, u_min
)

## Test MPC controller
Nx = Nmpc*n0
Ny = Nmpc*n
Nu = (Nmpc-1)*m
@test size(ctrl_mpc.A,2) ≈ Ny+Nu
@test size(ctrl_mpc.A,1) == Nmpc*n + (Nmpc-1)*(n0+m)

# Generate random input
X = [randn(n0) for k = 1:Nmpc]
U = [randn(m) for k = 1:Nmpc-1]
Y = map(x->expandstate(model_bilinear, x), X)

# Convert to vectors
Yref = map(x->expandstate(model_bilinear, x), Xref)
x̄ = reduce(vcat, Xref[j-1 .+ (1:Nmpc)])
ū = reduce(vcat, Uref[j-1 .+ (1:Nmpc-1)])
ȳ = reduce(vcat, Yref[j-1 .+ (1:Nmpc)])
z̄ = [ȳ;ū]
x = reduce(vcat, X)
u = reduce(vcat, U)
y = reduce(vcat, Y)
z = [y;u]

# Test cost
J = 0.5 * dot(z, ctrl_mpc.P, z) + dot(ctrl_mpc.q,z) + sum(ctrl_mpc.c)
@test J ≈ sum(1:Nmpc) do k
    J = 0.5 * (X[k]-Xref[k])'Qmpc*(X[k]-Xref[k])
    if k < Nmpc
        J += 0.5 * (U[k] - Uref[k])'Rmpc*(U[k] - Uref[k])
    end
    J
end

# Test dynamics constraint
c = mapreduce(vcat, 1:Nmpc-1) do k
    J = zeros(n,n+m)
    yn = zeros(n)
    z̄ = RD.KnotPoint(Yref[k], Uref[k], times[k], dt)
    RD.jacobian!(RD.InPlace(), RD.UserDefined(), model_bilinear, J, yn, z̄)
    A = J[:,1:n]
    B = J[:,n+1:end]
    dy = Y[k] - Yref[k]
    du = U[k] - Uref[k]
    dyn = Y[k+1] - Yref[k+1]
    RD.discrete_dynamics(model_bilinear, z̄) - Yref[k+1] + A*dy + B*du - dyn
end
c = [expandstate(model_bilinear, Xref[1]) - Y[1]; c]
ceq = Vector(ctrl_mpc.A*z - ctrl_mpc.l)[1:Nmpc*n]
@test c ≈ ceq

# Test bound constraints
clo = [
    mapreduce(vcat, 1:Nmpc-1) do k
        G*Y[k+1] - x_min
    end
    mapreduce(vcat, 1:Nmpc-1) do k
        U[k] - u_min
    end
]
chi = [
    mapreduce(vcat, 1:Nmpc-1) do k
        G*Y[k+1] - x_max
    end
    mapreduce(vcat, 1:Nmpc-1) do k
        U[k] - u_max
    end
]
@test clo ≈ (ctrl_mpc.A*z - ctrl_mpc.l)[Nmpc*n+1:end]
@test chi ≈ (ctrl_mpc.A*z - ctrl_mpc.u)[Nmpc*n+1:end]

## Try Solving single step
j = 1  # time index
zsol = solveqp!(ctrl_mpc, Xref[j], dt*(j-1))
Xsol = map(eachcol(reshape(view(zsol, 1:Nmpc*n), n, :))) do y
    originalstate(model_bilinear, y)
end
Usol = tovecs(view(zsol, Nmpc*n+1:length(zsol)), m)
@test getcontrol(ctrl_mpc, Xref[j], dt*(j-1)) ≈ Usol[1]
tmpc = dt*(j-1) .+ range(0,length=Nmpc,step=dt)
p = plot(tref, reduce(hcat, Xref)', 
    label=["θref" "ωref"], lw=1, c=[1 2]
)
plot!(p, tmpc, reduce(hcat, Xsol)',
    label=["θmpc" "ωmpc"], lw=2, c=[1 2], s=:dash
)
p = plot(tref, reduce(hcat, Uref)', 
    label=["θref" "ωref"], lw=1, c=[1 2]
)
plot!(p, tmpc[1:end-1], reduce(hcat, Usol)',
    label=["θmpc" "ωmpc"], lw=2, c=[1 2], s=:dash
)


## Try tracking with bilinear TVLQR 
i = 4
X = X_test[:,i]
U = U_test[:,i]
Y = map(x->expandstate(model_bilinear, x), X)
times = range(0,tf,step=dt)

# Open loop simulation
Y_ol = simulate(model_bilinear, U, Y[1], tf, dt)
X_ol = map(x->originalstate(model_bilinear, x), Y_ol)
plot(times, reduce(hcat, X)', c=[1 2], label="original")
plot!(times, reduce(hcat, X_ol)', legend=:bottom, c=[1 2], s=:dash, label="bilinear model (open loop)")

# Closed loop simulation
using BilinearControl.RandomLinearModels: iscontrollable
A,B = linearize(model_bilinear, Y, U, times)
map(zip(A,B)) do (Ak,Bk)
    iscontrollable(Ak,Bk)
end
n,m = RD.dims(model_bilinear)
Q = [Diagonal(fill(1.0, n)) for k = 1:length(X)]
R = [Diagonal(fill(1e2, m)) for k = 1:length(U)]
K,P = tvlqr(A,B,Q,R)
map(1:N-1) do k
    norm(eigvals(A[k] - B[k]*K[k]), Inf)
end

ctrl_bilinear = TVLQRController(K, Y, U, times)
Y_cl, = simulatewithcontroller(model_bilinear, ctrl_bilinear, Y[1], tf, dt)
X_cl = map(x->originalstate(model_bilinear, x), Y_cl)
plot(times, reduce(hcat, X)', c=[1 2], label="original")
plot!(times, reduce(hcat, X_cl)', legend=:bottom, c=[1 2], s=:dash, label="bilinear model")

# Create a controller for the nominal model that uses the bilinear one
ctrl_bl = BilinearController(model_bilinear, ctrl_bilinear)
X_bilinear, = simulatewithcontroller(dmodel, ctrl_bl, X[1] + randn(2)*0e-3, tf, dt)
p = plot(times, reduce(hcat, X)', c=[1 2], label="original")
plot!(times, reduce(hcat, X_bilinear)', legend=:bottom, c=[1 2], s=:dash, ylim=[-4,4], label="w/ bilinear TVLQR")
display(p)

visualize!(vis, model, tf, X)
visualize!(vis, model, tf, X_bilinear)

## MPC 
# Generate the QP data
n,m = RD.dims(model_bilinear)
n0 = originalstatedim(model_bilinear)
i = 1
Q = Diagonal(fill(1.0, n0))
R = Diagonal(fill(1e-3, m))
N = size(X_train,1)

Xref = X_train[:,i]
Uref = U_train[:,i] 
Xref[end] = [pi,0]
push!(Uref, zeros(m))
Yref = map(x->expandstate(model_bilinear, x), Xref)
x0 = copy(Xref[i])
Ny = sum(length, Yref)
Nu = sum(length, Uref)

# Create controller
ctrl_mpc = BilinearMPC(
    model_bilinear, N, x0, Q, R, Xref, Uref, times
)

# Generate random input
X = [randn(n0) for k = 1:Nmpc]
U = [randn(m) for k = 1:Nmpc-1]
Y = map(x->expandstate(model_bilinear, x), X)

# Convert to vectors
Yref = map(x->expandstate(model_bilinear, x), Xref)
x̄ = reduce(vcat, Xref[j-1 .+ (1:Nmpc)])
ū = reduce(vcat, Uref[j-1 .+ (1:Nmpc-1)])
ȳ = reduce(vcat, Yref[j-1 .+ (1:Nmpc)])
z̄ = [ȳ;ū]
x = reduce(vcat, X)
u = reduce(vcat, U)
y = reduce(vcat, Y)
z = [y;u]

# Test cost
z
J = 0.5 * dot(z, ctrl_mpc.P, z) + dot(ctrl_mpc.q,z) + sum(ctrl_mpc.c)
@test J ≈ sum(1:Nmpc) do k
    J = 0.5 * (X[k]-Xref[k])'Qmpc*(X[k]-Xref[k])
    if k < Nmpc
        J += 0.5 * (U[k] - Uref[k])'Rmpc*(U[k] - Uref[k])
    end
    J
end

# Test dynamics constraint
c = mapreduce(vcat, 1:Nmpc-1) do k
    J = zeros(n,n+m)
    yn = zeros(n)
    z̄ = RD.KnotPoint(Yref[k], Uref[k], times[k], dt)
    RD.jacobian!(RD.InPlace(), RD.UserDefined(), model_bilinear, J, yn, z̄)
    A = J[:,1:n]
    B = J[:,n+1:end]
    dy = Y[k] - Yref[k]
    du = U[k] - Uref[k]
    dyn = Y[k+1] - Yref[k+1]
    RD.discrete_dynamics(model_bilinear, z̄) - Yref[k+1] + A*dy + B*du - dyn
end
c = [expandstate(model_bilinear, Xref[1]) - Y[1]; c]
ceq = Vector(ctrl_mpc.A*z - ctrl_mpc.l)[1:Nmpc*n]
@test c ≈ ceq


zsol = solveqp!(ctrl_mpc, Xref[1], 0.0) 
ysol = zsol[1:Ny]
usol = zsol[Ny+1:end]
Xsol = map(eachcol(reshape(ysol, n, :))) do y
    originalstate(model_bilinear, y)
end
Usol = tovecs(usol, m)
norm(Xsol - Xref, Inf) < 0.2

plot(times, reduce(hcat,Xref)', c=[1 2], label="reference")
plot!(times, reduce(hcat,Xsol)', c=[1 2], s=:dash, label="MPC")
plot(times, reduce(hcat,Uref)', c=1, label="reference")
plot!(times[1:end-1], reduce(hcat,Usol)', c=1, s=:dash, label="MPC")

# Use shorter MPC horizon 
Nmpc = 21 
ctrl_mpc = BilinearMPC(
    model_bilinear, Nmpc, x0, Q, R, Xref, Uref, times
)
for i = 1:3
    zsol = solveqp!(ctrl_mpc, Xref[min(N,i)], (i-1)*dt) 
    Xsol = map(eachcol(reshape(view(zsol, 1:Nmpc*n), n, :))) do y
        originalstate(model_bilinear, y)
    end
    p = plot(times, reduce(hcat,Xref)', c=[1 2], label="reference")
    t_mpc = range(dt*(i-1), length=Nmpc, step=dt)
    plot!(p, 
        t_mpc, reduce(hcat,Xsol)', 
        c=[1 2], s=:dash, label="MPC", lw=2, legend=:bottom
    )
    @test norm(Xsol - Xref[i-1 .+ (1:Nmpc)], Inf) < 0.01
end

# Run MPC Controller
Xmpc, Umpc = simulatewithcontroller(dmodel, ctrl_mpc, Xref[1] + randn(n0) * 1e-1, tf, dt)
plot(times, reduce(hcat,Xref)', c=[1 2], label="reference")
plot!(times, reduce(hcat,Xmpc)', c=[1 2], s=:dash, label="MPC", lw=2, legend=:bottom, ylim=(-4,4))

