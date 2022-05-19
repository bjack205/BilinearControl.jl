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
using QDLDL
using Test

include("edmd_utils.jl")
include("cartpole_model.jl")

function gencartpoleproblem(x0=zeros(4), Qv=1e-2, Rv=1e-1, Qfv=1e2, u_bnd=3.0, tf=5.0; 
                            dt=0.05, constrained=true)
    model = Cartpole2()
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
model = Cartpole2()
visdir = Problems.VISDIR
include(joinpath(visdir, "visualization.jl"))
vis = Visualizer()
open(vis)
delete!(vis)
set_cartpole!(vis)

## Generate Data 
Random.seed!(1)
model = Cartpole2()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
num_traj = 100
tf = 3.0
dt = 0.01

# LQR Training data
Random.seed!(1)
num_lqr = 400
Qlqr = Diagonal([0.2,10,1e-2,1e-2])
Rlqr = Diagonal([1e-3])
xe = [0,pi,0,0]
ue = [0.0]
ctrl_lqr = LQRController(dmodel, Qlqr, Rlqr, xe, ue, dt)
x0_sampler = Product([
    Uniform(-1.0,1.0),
    Uniform(pi-pi/3,pi+pi/3),
    Uniform(-.5,.5),
    Uniform(-.5,.5),
])
initial_conditions_lqr = [rand(x0_sampler) for _ in 1:num_lqr]
X_train_lqr, U_train_lqr = create_data(dmodel, ctrl_lqr, initial_conditions_lqr, tf, dt)
# @test mapreduce(x->norm(x-xe,Inf), max, X_train_lqr[end,:]) < 0.1
@test mapreduce(x->norm(x[2]-xe[2],Inf), max, X_train_lqr[end,:]) < deg2rad(5)
visualize!(vis, RobotZoo.Cartpole(), tf, X_train_lqr[:,2])

# ALTRO Training data
Random.seed!(1)
train_params = map(1:num_traj) do i
    Qv = 1e-2
    Rv = Qv * 10^rand(Uniform(-1,3.0))
    Qfv = Qv * 10^rand(Uniform(1,5.0)) 
    u_bnd = rand(Uniform(4.5, 8.0))
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
    (zeros(4), 1e-3, 1e-2, 1e2, 4.5, tf),
    (zeros(4), 1e-3, 1e0, 1e2, 10.0, tf),
    (zeros(4), 1e-3, 1e-0, 1e2, 5.0, tf),
    (zeros(4), 1e-3, 1e-0, 1e2, 4.5, tf),
    (zeros(4), 1e-0, 1e-2, 1e2, 10.0, tf),
    (zeros(4), 1e-0, 1e-2, 1e2, 5.0, tf),
    (zeros(4), 1e-0, 1e-1, 1e2, 4.5, tf),
    (zeros(4), 1e-0, 1e-2, 1e-2, 10.0, tf),
]
prob = gencartpoleproblem(test_params[end]..., dt=dt)
solver = ALTROSolver(prob)
Altro.solve!(solver)
visualize!(vis, model, tf, TO.states(solver))

test_trajectories = map(test_params) do params
    solver = Altro.solve!(ALTROSolver(gencartpoleproblem(params...; dt), show_summary=false))
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
X_train = [X_train_altro[:,1:0] X_train_lqr[:,1:90]]
U_train = [U_train_altro[:,1:0] U_train_lqr[:,1:90]]
X_test = load(altro_datafile, "X_test")
U_test = load(altro_datafile, "U_test")
tf = load(altro_datafile, "tf")
dt = load(altro_datafile, "dt")
times = range(0,tf,step=dt)

eigfuns = ["monomial", "sine", "cosine", "sine", "cosine", "sine", "sine", "sine", "sine"]
eigorders = [4,0,0,0,2,2,3,4,6,8]
Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders)

F, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
    ["ridge", "lasso"]; 
    edmd_weights=[100.1], 
    mapping_weights=[0.1], 
    algorithm=:qr
);
length(U_train)

norm(F)
model_bilinear = EDMDModel(F,C,g,kf,dt,"cartpole")
n,m = RD.dims(model_bilinear)
n0 = originalstatedim(model_bilinear)

norm(bilinearerror(model_bilinear, X_train, U_train)) / length(U_train)
norm(bilinearerror(model_bilinear, X_test, U_test)) / length(U_test)

## Try simulating the training data
i = 1
Ysim = simulate(model_bilinear, U_train_lqr[:,i], expandstate(model_bilinear, X_train_lqr[1,i]), tf, dt)
Xsim0 = simulate(dmodel, U_train_lqr[:,i], X_train_lqr[1,i], tf, dt)
Xsim = map(x->originalstate(model_bilinear,x), Ysim)
plot(times, reduce(hcat, X_train_lqr[:,i])[1:2,:]', c=[1 2])
plot!(times, reduce(hcat, Xsim0)[1:2,:]', c=[1 2], s=:dot)
plot!(times, reduce(hcat, Xsim)[1:2,:]', c=[1 2], lw=2, s=:dash)

## Stabilizing MPC Controller
xe = [0,pi,0,0]
ue = [0.0] 
N = 1001
Xref = [copy(xe) for k = 1:N]
Uref = [copy(ue) for k = 1:N]
tref = range(0,length=N,step=dt)
Nmpc = 101
Qmpc = Diagonal([20.1,10.,1e-2,1e-2])
Rmpc = Diagonal([1e-4])
ctrl_mpc = BilinearMPC(
    model_bilinear, Nmpc, Xref[1], Qmpc, Rmpc, Xref, Uref, tref
)

## Build the QP
function solveqp!(y0; Nmpc=101)
    # Qmpc = Diagonal([0.3,5.,1e-1,1e-1])
    # Rmpc = Diagonal([1e-3])
    Qmpc = Diagonal([0.2,10,1e-2,1e-2])
    Rmpc = Diagonal([1e-3])

    # dx0 = x - xe
    if length(y0) == n0
        println("  Converting original state.")
        dy0 = expandstate(model_bilinear, y0) - ye
    else
        dy0 = y0 - ye 
    end
    Nx = Nmpc*n
    Np = Nmpc*n + (Nmpc-1)*m
    Nd = Nmpc*n
    G = spdiagm(n0,n,1=>ones(n0))
    Q = G'Qmpc*G + I*1e-8
    R = copy(Rmpc)
    P = blockdiag(kron(I(Nmpc), Q), kron(I(Nmpc-1), sparse(R)))
    q = zeros(Np)
    Ā = ctrl_mpc.Ā[1]
    B̄ = ctrl_mpc.B̄[1]
    A = sparse(-1.0I, Nmpc*n, Np)
    A += hcat(
        kron(spdiagm(Nmpc, Nmpc,-1=>ones(Nmpc-1)), Ā), 
        kron(spdiagm(Nmpc, Nmpc-1,-1=>ones(Nmpc-1)), B̄)
    )
    b = [-dy0; zeros((Nmpc-1)*n)]

    H = [P A'; A sparse(-1e-8I,Nd,Nd)]
    g = [q;b]
    F = QDLDL.qdldl(triu(H))
    z = QDLDL.solve(F, g)
    # z = H\g
    u = z[Nx .+ (1:m)]
    return u, z
end

Nmpc = 51
t_mpc = (Nmpc-1)*dt
t_sim = 1.0
times_sim = range(0,t_sim,step=dt)
times_mpc = range(0,t_mpc,step=dt)
ye = expandstate(model_bilinear, xe)

x0 = [0.2,pi-deg2rad(-20),0,0]
y0 = expandstate(model_bilinear, x0) 
Ysim = [copy(y0) for t in times_sim]
Xsim = [copy(x0) for t in times_sim]
Xsim0 = [copy(x0) for t in times_sim]
Usim = [zeros(m) for t in times_sim]

for k = 1:length(times_sim)-1
    @show k
    u,zsol = solveqp!(Ysim[k]; Nmpc)
    # u,zsol = solveqp!(Xsim0[k]; Nmpc)
    Usim[k] = u
    Xsim0[k+1] = RD.discrete_dynamics(dmodel, Xsim0[k], u, times_sim[k], dt)
    Ysim[k+1] = RD.discrete_dynamics(model_bilinear, Ysim[k], u, times_sim[k], dt)
    Xsim[k+1] = originalstate(model_bilinear, Ysim[k+1])
    X_qp = map(eachcol(reshape(zsol[1:Nmpc*n], n, :))) do dy
        originalstate(model_bilinear, ye + dy)
    end
    t0 = (k-1)*dt
    p = plot(times_mpc .+ t0, reduce(hcat, X_qp)[1:2,:]', xlim=(0,t0+t_mpc))
    display(p)

    # Ysim[k+1] = RD.discrete_dynamics(model_bilinear, Ysim[k], u, times_sim[k], dt)
    # Xsim[k+1] = originalstate(model_bilinear, Ysim[k+1])
end

visualize!(vis, model, t_sim[end], Xsim)
visualize!(vis, model, t_sim[end], Xsim0)
Xsim[1]
Xsim[2]
Usim[2]
plot(times_sim, reduce(hcat, Usim)')

##
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
X_osqp[end]

zsol = solveqp!(ctrl_mpc0, x0, 1)
X_osqp = map(eachcol(reshape(zsol[1:Nmpc*n0], n0, :))) do y
    originalstate(dmodel, y)
end
plot!(times_mpc, reduce(hcat, X_osqp)[1:2,:]')

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
X_osqp, U_osqp = let ctrl=ctrl_mpc, model=model_bilinear
    n = RD.state_dim(model)
    zsol = solveqp!(ctrl, Xmpc[k], 1)
    X = map(eachcol(reshape(zsol[1:Nmpc*n], n, :))) do y
        originalstate(model, y)
    end
    U = tovecs(zsol[Nmpc*n + 1:end], m) 
    X,U
end
p = plot(times_sim, reduce(hcat, Xref[1:length(times_sim)])[1:2,:]', 
    label=["x" "θ"], lw=2)
plot!(p,times_mpc .+ (k-1)*dt, reduce(hcat, X_osqp)[1:2,:]', legend=:right, c=[1 2])
# Xsim = simulate(dmodel, U_osqp, Xmpc[k], t_mpc, dt)
# plot!(p,times_mpc .+ (k-1)*dt, reduce(hcat, Xsim)[1:2,:]', legend=:right, c=[1 2])

u = U_osqp[1]
Xmpc[k+1] = RD.discrete_dynamics(dmodel, Xmpc[k], u, times_sim[k], dt)
# plot!(p,times_mpc .+ (k-1)*dt, reduce(hcat, Xsim)[1:2,:]', legend=:right, c=[1 2])
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