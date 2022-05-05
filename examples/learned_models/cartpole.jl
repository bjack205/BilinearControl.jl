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

function gencartpoleproblem(x0=zeros(4), Qv=1e-2, Rv=1e-1, Qfv=1e2, u_bnd=3.0, tf=5.0; 
                            dt=0.05, constrained=true)
    model = RobotZoo.Cartpole()
    dmodel = RD.DiscretizedDynamics{RD.RK4}(model) 
    n,m = RD.dims(model)
    N = round(Int, tf/dt) + 1

    Q = Qv*Diagonal(@SVector ones(n)) * dt
    Qf = Qfv*Diagonal(@SVector ones(n))
    R = Rv*Diagonal(@SVector ones(m)) * dt
    xf = @SVector [0, pi, 0, 0]
    obj = TO.LQRObjective(Q,R,Qf,xf,N)

    conSet = TO.ConstraintList(n,m,N)
    bnd = TO.BoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd)
    goal = TO.GoalConstraint(xf)
    if constrained
        TO.add_constraint!(conSet, bnd, 1:N-1)
        TO.add_constraint!(conSet, goal, N:N)
    end

    X0 = [@SVector fill(NaN,n) for k = 1:N]
    u0 = @SVector fill(0.01,m)
    U0 = [u0 for k = 1:N-1]
    Z = TO.SampledTrajectory(X0,U0,dt=dt*ones(N-1))
    prob = TO.Problem(dmodel, obj, x0, tf, constraints=conSet, xf=xf) 
    TO.initial_trajectory!(prob, Z)
    TO.rollout!(prob)
    prob
end

# mutable struct CartpoleParams 
#     x0::Product{Continuous, Uniform{Float64}, Vector{Uniform{Float64}}}
#     QRratio::Uniform{Float64}
#     Qfratio::Uniform{Float64}  # log of ratio
#     tf::Uniform{Float64}
#     u_bnd::Uniform{Float64}
#     dt::Float64
#     function CartpoleParams(;x0_bnd=[1.0,pi/2,10,10], QRratio=[0.1, 10], Qfratio=[1.0, 4.0], 
#                              tf=[4.0, 7.0], u_bnd=[2.0, 6.0], dt=0.05)
#         x0_sampler = Product([Uniform(-x0_bnd[i],x0_bnd[i]) for i = 1:4])
#         QR_sampler = Uniform(QRratio[1], QRratio[2])
#         Qf_sampler = Uniform(Qfratio[1], Qfratio[2])
#         tf_sampler = Uniform(tf[1], tf[2])
#         u_bnd_sampler = Uniform(u_bnd[1], u_bnd[2])
#         new(x0_sampler, QR_sampler, Qf_sampler, tf_sampler, u_bnd_sampler, dt)
#     end
# end

# function Base.rand(params::CartpoleParams) 
#     x0 = rand(params.x0) 
#     R = 1.0 
#     Q = rand(params.Qfratio)
#     Qf = 10^(rand(params.Qfratio)) * Q
#     u_bnd = rand(params.u_bnd)
#     tf_raw = rand(params.tf)
#     N = round(Int, tf_raw / params.dt) + 1
#     tf = params.dt * (N - 1)
#     (x0=x0, Qv=Q, Rv=R, Qfv=Qf, u_bnd=u_bnd, tf=tf)
# end

## Visualizer
model = RobotZoo.Cartpole()
visdir = Problems.VISDIR
include(joinpath(visdir, "visualization.jl"))
vis = Visualizer()
open(vis)
delete!(vis)
set_cartpole!(vis)

## Generate Data 
Random.seed!(1)
model = RobotZoo.Cartpole()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
num_traj = 100
tf = 5.0
dt = 0.05

# LQR Training data
Random.seed!(1)
num_lqr = 400
Qlqr = Diagonal([0.1,10,1e-2,1e-2])
Rlqr = Diagonal([1e-4])
xe = [0,pi,0,0]
ue = [0.0]
ctrl_lqr = LQRController(dmodel, Qlqr, Rlqr, xe, ue, dt)
x0_sampler = Product([
    Uniform(-0.5,0.5),
    Uniform(pi-pi/3,pi+pi/3),
    Uniform(-.5,.5),
    Uniform(-.5,.5),
])
initial_conditions_lqr = [rand(x0_sampler) for _ in 1:num_lqr]
X_train_lqr, U_train_lqr = create_data(dmodel, ctrl_lqr, initial_conditions_lqr, tf, dt)
@test mapreduce(x->norm(x-xe,Inf), max, X_train_lqr[end,:]) < 0.1
@test mapreduce(x->norm(x[2]-xe[2],Inf), max, X_train_lqr[end,:]) < deg2rad(1)
visualize!(vis, model, tf, X_train_lqr[:,8])

# ALTRO Training data
Random.seed!(1)
train_params = map(1:num_traj) do i
    Qv = 1e-2
    Rv = Qv * 10^rand(Uniform(-1,3.0))
    Qfv = Qv * 10^rand(Uniform(1,5.0)) 
    u_bnd = rand(Uniform(3.0, 8.0))
    (zeros(4), Qv, Rv, Qfv, u_bnd, tf)
end

train_trajectories = map(train_params) do params
    solver = Altro.solve!(ALTROSolver(gencartpoleproblem(params..., dt=dt), 
        show_summary=false, projected_newton=true))
    if Altro.status(solver) != Altro.SOLVE_SUCCEEDED
        @warn "ALTRO Solve failed"
    end
    X = TO.states(solver)
    U = TO.controls(solver)
    Vector.(X), Vector.(U)
end
X_train_altro = mapreduce(x->getindex(x,1), hcat, train_trajectories)
U_train_altro = mapreduce(x->getindex(x,2), hcat, train_trajectories)

# Test data
test_params = [
    (zeros(4), 1e-3, 1e-2, 1e2, 10.0, tf),
    (zeros(4), 1e-3, 1e-2, 1e2, 5.0, tf),
    (zeros(4), 1e-3, 1e-2, 1e2, 3.0, tf),
    (zeros(4), 1e-3, 1e0, 1e2, 10.0, tf),
    (zeros(4), 1e-3, 1e-0, 1e2, 5.0, tf),
    (zeros(4), 1e-3, 1e-0, 1e2, 3.0, tf),
    (zeros(4), 1e-0, 1e-2, 1e2, 10.0, tf),
    (zeros(4), 1e-0, 1e-2, 1e2, 5.0, tf),
    (zeros(4), 1e-0, 1e-1, 1e2, 3.0, tf),
    (zeros(4), 1e-0, 1e-2, 1e-2, 10.0, tf),
]
prob = gencartpoleproblem(test_params[end-1]...)
solver = ALTROSolver(prob)
solve!(solver)
visualize!(vis, model, tf, TO.states(solver))

test_trajectories = map(test_params) do params
    solver = Altro.solve!(ALTROSolver(gencartpoleproblem(params...), show_summary=false))
    if Altro.status(solver) != Altro.SOLVE_SUCCEEDED
        @show params
        @warn "ALTRO Solve failed"
    end
    X = TO.states(solver)
    U = TO.controls(solver)
    Vector.(X), Vector.(U)
end
X_test = mapreduce(x->getindex(x,1), hcat, test_trajectories)
U_test = mapreduce(x->getindex(x,2), hcat, test_trajectories)
time = range(0,tf,step=dt)

p = plot(ylabel="states", xlabel="time (s)")
for i = 1:size(X_test,2)
    plot!(p, time, reduce(hcat, X_test[:,i])[1:2,:]', label="", c=[1 2])
end
display(p)

jldsave(joinpath(Problems.DATADIR, "cartpole_altro_trajectories.jld2"); 
    X_train=X_train_altro, U_train=U_train_altro, X_test, U_test, tf, dt,
    X_lqr=X_train_lqr, U_lqr=U_train_lqr
)


## Learn Bilinear Model
model = RobotZoo.Cartpole()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)

altro_datafile = joinpath(Problems.DATADIR, "cartpole_altro_trajectories.jld2")
X_train_altro = load(altro_datafile, "X_train")
U_train_altro = load(altro_datafile, "U_train")
X_train_lqr = load(altro_datafile, "X_lqr")
U_train_lqr = load(altro_datafile, "U_lqr")
X_train = [X_train_altro[:,1:0] X_train_lqr[:,1:400]]
U_train = [U_train_altro[:,1:0] U_train_lqr[:,1:400]]
X_test = load(altro_datafile, "X_test")
U_test = load(altro_datafile, "U_test")
tf = load(altro_datafile, "tf")
dt = load(altro_datafile, "dt")

eigfuns = ["state", "sine", "cosine", "sine", "cosine", "chebyshev"]
eigorders = [0,0,0,2,2,2]
Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders)

F, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
    ["ridge", "lasso"]; 
    edmd_weights=[10.1], 
    mapping_weights=[0.0], 
    algorithm=:cholesky
);

norm(F)
model_bilinear = EDMDModel(F,C,g,kf,dt,"cartpole")
n,m = RD.dims(model_bilinear)
n0 = originalstatedim(model_bilinear)

norm(bilinearerror(model_bilinear, X_train, U_train)) / length(U_train)
norm(bilinearerror(model_bilinear, X_test, U_test)) / length(U_test)

## Stabilizing MPC Controller
xe = [0,pi,0,0]
ue = [0.0] 
N = 1001
Xref = [copy(xe) for k = 1:N]
Uref = [copy(ue) for k = 1:N]
tref = range(0,length=N,step=dt)
Nmpc = 61
Qmpc = Diagonal([20.1,10.,1e-2,1e-2])
Rmpc = Diagonal([1e-4])
ctrl_mpc = BilinearMPC(
    model_bilinear, Nmpc, Xref[1], Qmpc, Rmpc, Xref, Uref, tref
)

ctrl_mpc0 = BilinearMPC(
    dmodel, Nmpc, Xref[1], Qmpc, Rmpc, Xref, Uref, tref
)

t_mpc = (Nmpc-1)*dt 
t_sim = 3.0
times_mpc = range(0,t_mpc,step=dt)
times_sim = range(0,t_sim,step=dt)

x0 = [0.0,pi-deg2rad(10),0,0] 
zsol = solveqp!(ctrl_mpc, x0, 1)
X_osqp = map(eachcol(reshape(zsol[1:Nmpc*n], n, :))) do y
    originalstate(model_bilinear, y)
end
t_mpc = range(0,length=Nmpc,step=dt)
plot(t_mpc, reduce(hcat, X_osqp)[1:2,:]')
visualize!(vis, model, t_mpc[end], X_osqp)

zsol = solveqp!(ctrl_mpc0, x0, 1)
X_osqp = map(eachcol(reshape(zsol[1:Nmpc*n0], n0, :))) do y
    originalstate(dmodel, y)
end
t_mpc = range(0,length=Nmpc,step=dt)
plot!(t_mpc, reduce(hcat, X_osqp)[1:2,:]')

## Step through the MPC
k = 1
Xmpc = [copy(x0) for t in times_sim]
uind = Nmpc*n .+ (1:m)  # indices of first control

##
# zsol = solveqp!(ctrl_mpc, Xmpc[k], k)
# u = zsol[uind]
# X_osqp = map(eachcol(reshape(zsol[1:Nmpc*n], n, :))) do y
#     originalstate(model_bilinear, y)
# end
X_osqp, U_osqp = let n = n0, ctrl=ctrl_mpc0
    zsol = solveqp!(ctrl, Xmpc[k], 1)
    X = map(eachcol(reshape(zsol[1:Nmpc*n], n, :))) do y
        originalstate(dmodel, y)
    end
    U = tovecs(zsol[Nmpc*n + 1:end], m) 
    X,U
end
p = plot(times_sim, reduce(hcat, Xref[1:length(times_sim)])[1:2,:]', 
    label=["x" "θ"], lw=2)
plot!(p,times_mpc .+ (k-1)*dt, reduce(hcat, X_osqp)[1:2,:]', legend=:right, c=[1 2])
Xsim = simulate(dmodel, U_osqp, Xmpc[k], t_mpc, dt)

map(1:Nmpc-1) do k
    norm(RD.discrete_dynamics(dmodel, X_osqp[k], U_osqp[k], times_sim[k], dt) - X_osqp[k+1])
end
t_ref = 10
times_ref = range(0,t_ref, length=length(Xref))
A,B = linearize(dmodel, Xref, Uref, times_ref)
let x = X_osqp[1], u = U_osqp[1]
    dx = x - xe
    du = u - ue
    A[1]*dx + B[1]*du + RD.discrete_dynamics(dmodel, xe, ue, 0.0, dt) - X_osqp[2]
    RD.discrete_dynamics(dmodel, x, u, 0.0, dt) - X_osqp[2]
end

Xmpc[k+1] = RD.discrete_dynamics(dmodel, Xmpc[k], u, times_sim[k], dt)
plot!(p,times_mpc .+ (k-1)*dt, reduce(hcat, Xsim)[1:2,:]', legend=:right, c=[1 2])
k += 1
display(p)

## Test on test data
t_sim = 4.0
times_sim = range(0,t_sim,step=dt)
p = plot(times_sim, reduce(hcat, Xref[1:length(times_sim)])[1:2,:]', 
    label=["x" "θ"], lw=2
)
# for i = 1:size(X_test_lqr,2)
    x0 = [0.0,pi-deg2rad(20),0,0] 
    Xmpc, Umpc = simulatewithcontroller(
        dmodel, ctrl_mpc, x0, t_sim, dt
    )
    plot!(p,times_sim, reduce(hcat,Xmpc)[1:2,:]', 
        c=[1 2], s=:dash, label="", lw=1, legend=:bottom, ylim=(-1,4))
    Xlqr, Ulqr = simulatewithcontroller(
        dmodel, ctrl_lqr, x0, t_sim, dt
    )
    plot!(p,times_sim, reduce(hcat,Xlqr)[1:2,:]', 
        c=[1 2], s=:dash, label="", lw=1, legend=:bottom, ylim=(-1,4))
# end
visualize!(vis, model, t_sim, Xmpc)
visualize!(vis, model, t_sim, Xlqr)
display(p)