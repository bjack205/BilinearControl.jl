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
using Printf
using Test

include("edmd_utils.jl")

function pendulum_kf(x)
    p,v = x
    s,c = sincos(p)
    s2,c2 = sincos(2p)
    s3,c3 = sincos(3p)
    [1,s,c,s2,s3, p*s,p*c, v*s,v*c]
end

function genpendulumproblem(x0=[0.,0.], Qv=1e-3, Rv=1e-3, Qfv=1e-0, u_bnd=3.0, tf=3.0; dt=0.05)
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
    U = [SA[0*cos(t/2) + randn() * 1e-2] for t in times]
    pendulum_static = TO.Problem(dmodel, obj, x0, tf, constraints=conSet, xf=xf)
    TO.initial_controls!(pendulum_static, U)
    TO.rollout!(pendulum_static)
    return pendulum_static
end

# ## Visualization
# using MeshCat
# model = RobotZoo.Pendulum()
# visdir = joinpath(@__DIR__, "../../examples/visualization/")
# include(joinpath(visdir, "visualization.jl"))
# vis = Visualizer()
# open(vis)
# delete!(vis)
# set_pendulum!(vis)

## Generate ALTRO data
model = RobotZoo.Pendulum()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
num_traj = 400
tf = 4.0
dt = 0.05

# Training data
Random.seed!(1)
train_params = map(1:num_traj) do i
    Qv = 1e-3
    Rv = Qv * 10^rand(Uniform(-1,3.0))
    Qfv = Qv * 10^rand(Uniform(1,4.0)) 
    u_bnd = rand(Uniform(3.0, 8.0))
    (zeros(2), Qv, Rv, Qfv, u_bnd, tf)
end
train_trajectories = map(train_params) do params
    solver = Altro.solve!(ALTROSolver(genpendulumproblem(params...), 
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
    
all(1:num_traj) do i
    simulate(dmodel, U_train_altro[:,i], zeros(2), tf, dt) ≈ X_train_altro[:,i]
end

# Test data
test_params = [
    (zeros(2), 1e-3, 1e-3, 1e-0, 10.0, tf),
    (zeros(2), 1e-3, 1e-2, 1e-0, 3.0, tf),
    (zeros(2), 1e-3, 1e-3, 1e-0, 3.0, tf),
    (zeros(2), 1e-3, 1e-3, 1e-0, 4.0, tf),
    (zeros(2), 1e-3, 1e-3, 1e-0, 5.0, tf),
    (zeros(2), 1e-3, 1e-0, 1e-0, 5.0, tf),
    (zeros(2), 1e-1, 1e-3, 1e-0, 4.0, tf),
    (zeros(2), 1e-0, 1e-3, 1e-0, 7.0, tf),
    (zeros(2), 1e-0, 1e-3, 1e-0, 4.0, tf),
    (zeros(2), 1e-3, 1e-2, 1e-0, 4.0, tf),
]
test_trajectories = map(test_params) do params
    solver = Altro.solve!(ALTROSolver(genpendulumproblem(params...), show_summary=false))
    if Altro.status(solver) != Altro.SOLVE_SUCCEEDED
        @warn "ALTRO Solve failed"
    end
    X = TO.states(solver)
    U = TO.controls(solver)
    Vector.(X), Vector.(U)
end
X_test = mapreduce(x->getindex(x,1), hcat, test_trajectories)
U_test = mapreduce(x->getindex(x,2), hcat, test_trajectories)

jldsave(joinpath(Problems.DATADIR, "pendulum_altro_trajectories.jld2"); 
    X_train=X_train_altro, U_train=U_train_altro, X_test, U_test, tf, dt
)

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
X_test = load(altro_datafile, "X_test")
U_test = load(altro_datafile, "U_test")

X_train_lqr = load(lqr_datafile, "X_train")[:,1:00]
U_train_lqr = load(lqr_datafile, "U_train")[:,1:00]
X_test_lqr = load(lqr_datafile, "X_test")
U_test_lqr = load(lqr_datafile, "U_test")
tf = load(altro_datafile, "tf")
dt = load(altro_datafile, "dt")
@assert load(lqr_datafile, "tf") == tf
@assert load(lqr_datafile, "dt") == dt 
X_train = [X_train_altro X_train_lqr]
U_train = [U_train_altro U_train_lqr]

# Learn bilinear model
eigfuns = ["state", "sine", "cosine", "chebyshev"]
eigorders = [0,0,0,4]
Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders)

F, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
    ["ridge", "lasso"]; 
    edmd_weights=[1.1], 
    mapping_weights=[0.0], 
    algorithm=:cholesky
);

norm(F)
model_bilinear = EDMDModel(F,C,g,kf,dt,"pendulum")
RD.dims(model_bilinear)

norm(bilinearerror(model_bilinear, X_train, U_train)) / length(U_train)
norm(bilinearerror(model_bilinear, X_test, U_test)) / length(U_test)
norm(bilinearerror(model_bilinear, X_test_lqr, U_test_lqr)) / length(U_test_lqr)

## Stabilizing MPC Controller
xe = [pi,0]
ue = [0.0] 
N = 1001
Xref = [copy(xe) for k = 1:N]
Uref = [copy(ue) for k = 1:N]
tref = range(0,length=N,step=dt)
Nmpc = 21
Qmpc = Diagonal([10.0,0.1])
Rmpc = Diagonal([1e-4])
ctrl_mpc = BilinearMPC(
    model_bilinear, Nmpc, Xref[1], Qmpc, Rmpc, Xref, Uref, tref
)

# Test on test data
tsim = 1.0
times_sim = range(0,tsim,step=dt)
p = plot(times_sim, reduce(hcat, Xref[1:length(times_sim)])', 
    label=["θ" "ω"], lw=2
)
for i = 1:size(X_test_lqr,2)
    x0 = X_test_lqr[1,i] 
    Xmpc, Umpc = simulatewithcontroller(
        dmodel, ctrl_mpc, x0, tsim, dt
    )
    plot!(p,times_sim, reduce(hcat,Xmpc)', 
        c=[1 2], s=:dash, label="", lw=1, legend=:bottom, ylim=(-6,6))
end
display(p)

## MPC Tracking Controller
function run_tracking_mpc(Xref0, Uref0, tref;
        doplot=true,
        figname="",
        Nmpc = 41,
        Qmpc = Diagonal([10.0,0.1]),
        Rmpc = Diagonal([1e-4]),
        t_sim=5.0,
        x_max = [20pi, 1000],
        u_max = [200],
        x_min = -x_max,
        u_min = -u_max, 
    )
    n,m = RD.dims(model_bilinear)
    n0 = originalstatedim(model_bilinear)

    # Reference trajectory
    x0 = copy(Xref0[1])
    xf = [pi,0]
    Xref = deepcopy(Xref0) 
    Uref = deepcopy(Uref0) 
    tref = range(0,length=length(Xref),step=dt)

    # Make the end of the referernce match the final position and controls
    Xref[end] .= xf
    push!(Uref, zeros(m))

    # Generate controller
    ctrl_mpc = BilinearMPC(
        model_bilinear, Nmpc, x0, Qmpc, Rmpc, Xref, Uref, tref;
        x_max, x_min, u_max, u_min
    )

    # Run full controller
    time_sim = range(0,t_sim, step=dt)
    Xsim,Usim = simulatewithcontroller(dmodel, ctrl_mpc, Xref[1], t_sim, dt)
    if doplot
        p = plot(tref, reduce(hcat, ctrl_mpc.Xref)', 
            label=["θref" "ωref"], lw=1, c=[1 2],
            xlabel="time (s)", ylabel="states"
        )
        plot!(p, time_sim, reduce(hcat, Xsim)',
            label=["θmpc" "ωmpc"], lw=2, c=[1 2], s=:dash, legend=:outerright,
        )
        if !isempty(figname)
            savefig(p, joinpath(Problems.FIGDIR, figname * ".png"))
        end
        display(p)
    end
    Xsim,Usim, ctrl_mpc
end
tref = range(0,tf,step=dt)
for i = 1:size(X_test, 2)
    figname = @sprintf("pendulum_mpc/test_trajectory_%02d.png", i)
    run_tracking_mpc(X_test[:,i], U_test[:,i], tref; figname)
end

#############################################
## Test MPC controller
#############################################
n,m = RD.dims(model_bilinear)
n0 = originalstatedim(model_bilinear)
Nmpc = 41
Nx = Nmpc*n0
Ny = Nmpc*n
Nu = (Nmpc-1)*m
Nd = Nmpc*n
Qmpc = Diagonal([10.0,0.1])
Rmpc = Diagonal([1e-4])
x_max = [20pi, 1000]
u_max = [200.]

Xref = X_test[:,1]
Uref = U_test[:,1]
_,_,ctrl_mpc = run_tracking_mpc(Xref, Uref, tref; 
    doplot=false, Nmpc, Qmpc, Rmpc, x_max, u_max)
build_qp!(ctrl_mpc, Xref[1], 1)

@test size(ctrl_mpc.A,2) ≈ Ny+Nu
@test size(ctrl_mpc.A,1) == Nmpc*n + (Nmpc-1)*(n0+m)

# Generate random input
X = [randn(n0) for k = 1:Nmpc]
U = [randn(m) for k = 1:Nmpc-1]
Y = map(x->expandstate(model_bilinear, x), X)

# Convert to vectors
Yref = map(x->expandstate(model_bilinear, x), Xref)
x̄ = reduce(vcat, Xref[1:Nmpc])
ū = reduce(vcat, Uref[1:Nmpc-1])
ȳ = reduce(vcat, Yref[1:Nmpc])
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
    z̄ = RD.KnotPoint(Yref[k], Uref[k], tref[k], dt)
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
G = model_bilinear.g
clo = [
    mapreduce(vcat, 1:Nmpc-1) do k
        G*Y[k+1] - (-x_max)
    end
    mapreduce(vcat, 1:Nmpc-1) do k
        U[k] - (-u_max)
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
@test (ctrl_mpc.A*z)[Nd+1:end] ≈ [x[n0+1:end]; u]

