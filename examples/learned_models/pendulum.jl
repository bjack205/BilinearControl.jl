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
jldsave(joinpath(Problems.DATADIR, "pendulum_altro_trajectories.jld2"); X_train, U_train, X_test, U_test)

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
datafile = joinpath(Problems.DATADIR, "pendulum_altro_trajectories.jld2")
X_train = load(datafile, "X_train")
U_train = load(datafile, "U_train")
X_test = load(datafile, "X_test")
U_test = load(datafile, "U_test")

eigfuns = ["state", "sine", "cosine", "hermite"]
eigorders = [0, 0, 0, 2]
eigfuns = ["state", "chebyshev"]
eigorders = [0,5]
Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders)
# Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, pendulum_kf)

# learn bilinear model
F, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
    ["lasso", "lasso"]; 
    edmd_weights=[0.1], 
    mapping_weights=[0.0], 
    algorithm=:qr
);

model_bilinear = EDMDModel(F,C,g,kf,dt,"pendulum")
RD.dims(model_bilinear)

norm(bilinearerror(model_bilinear, dmodel, X_test[:,2], U_test[:,2]), Inf)

let i = 1
    compare_models(RD.InPlace(), model_bilinear, dmodel, initial_conditions[i], tf, 
        U_test[:,i], doplot=true)
end

const datadir = joinpath(dirname(pathof(BilinearControl)), "../data")
jldsave(joinpath(datadir, "pendulum_eDMD_data.jld2"); A=F, C, g, dt=dt, eigfuns, eigorders)

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
function build_qp!(model::EDMDModel, x0, Q, R, Xref, Uref, times)
    Yref = map(x->expandstate(model, x), Xref)

    n,m = RD.dims(model)
    N = length(Xref)
    Nx = sum(length, Xref)
    Nu = sum(length, Uref)
    Ny = sum(length, Yref)

    Ā,B̄ = linearize(model, Yref, Uref, times)
    d̄ = map(1:N-1) do k
        dt = times[k+1] - times[k]
        RD.discrete_dynamics(model, Yref[k], Uref[k], times[k], dt) - (Ā[k]*Yref[k] + B̄[k]*Uref[k])
    end
    G = model.g

    # Build Cost
    P = blockdiag(kron(sparse(I,N,N),G'Q*G), kron(sparse(I,N-1,N-1), R))
    q0 = mapreduce(vcat, 1:N) do k
        -G'Q*Xref[k]
    end
    r0 = mapreduce(vcat, 1:length(U)) do k
        -R*Uref[k]
    end
    q = [q0; r0]
    c = map(1:N) do k
        J = 0.5 * dot(Xref[k], Q, Xref[k])
        if k < N
            J += 0.5 * dot(Uref[k], R, Uref[k])
        end
        J
    end

    # Build dynamics
    D = spzeros(N*n, Ny + Nu) 
    d = zeros(N*n)
    ic = 1:n
    D[ic,1:n] .= -I(n)
    d[ic] .= -expandstate(model, x0)
    ic = ic .+ n
    for k = 1:N-1
        ix1 = (k-1)*n .+ (1:n)
        iu1 = (k-1)*m + Ny .+ (1:m)
        ix2 = ix1 .+ n
        D[ic,ix1] .= Ā[k]
        D[ic,iu1] .= B̄[k]
        D[ic,ix2] .= -I(n)
        d[ic] .= -d̄[k]

        ic = ic .+ n
    end
    A = D
    ub = d
    lb = d

    P, q, c, A, lb, ub 
end

# Generate the QP data
i = 1
Q = Diagonal(fill(1.0, n0))
R = Diagonal(fill(1e-3, m0))

Xref = X_train[:,i]
Uref = U_train[:,i] 
Yref = map(x->expandstate(model_bilinear, x), Xref)
x0 = copy(Xref[i])
Ny = sum(length, Yref)
Nu = sum(length, Uref)
P,q,c, A,lb,ub = build_qp!(model_bilinear, x0, Q, R, Xref, Uref, times)

# Generate random input
X = [randn(n0) for k = 1:N]
U = [randn(m0) for k = 1:N-1]
Y = map(x->expandstate(model_bilinear, x), X)

# Convert to vectors
x̄ = reduce(vcat, Xref)
ū = reduce(vcat, Uref)
ȳ = reduce(vcat, Yref)
z̄ = [ȳ;ū]
x = reduce(vcat, X)
u = reduce(vcat, U)
y = reduce(vcat, Y)
z = [y;u]

# Test cost
J = 0.5 * dot(z, P, z) + dot(q,z) + sum(c)
J ≈ sum(1:N) do k
    J = 0.5 * (X[k]-Xref[k])'Q*(X[k]-Xref[k])
    if k < N
        J += 0.5 * (U[k] - Uref[k])'R*(U[k] - Uref[k])
    end
    J
end

# Test dynamics constraint
c = mapreduce(vcat, 1:N-1) do k
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
c = [expandstate(model_bilinear, x0) - Y[1]; c]
c ≈ Vector(A*z - lb)

import OSQP
model = OSQP.Model()
OSQP.setup!(model, P=P, q=q, A=A, l=lb, u=ub)
res = OSQP.solve!(model)
zsol = res.x
ysol = zsol[1:Ny]
usol = zsol[Ny+1:end]
Xsol = map(eachcol(reshape(ysol, n, :))) do y
    originalstate(model_bilinear, y)
end
Usol = tovecs(usol, m0)
plot(times, reduce(hcat,Xref)', c=[1 2], label="reference")
plot!(times, reduce(hcat,Xsol)', c=[1 2], s=:dash, label="MPC")
plot(times[1:end-1], reduce(hcat,Uref)', c=1, label="reference")
plot!(times[1:end-1], reduce(hcat,Usol)', c=1, s=:dash, label="MPC")

