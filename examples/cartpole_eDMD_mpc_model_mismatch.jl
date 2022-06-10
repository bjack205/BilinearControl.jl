import Pkg; Pkg.activate(joinpath(@__DIR__)); Pkg.instantiate();
using BilinearControl
using BilinearControl.Problems
using BilinearControl.EDMD
import RobotDynamics as RD
using LinearAlgebra
using RobotZoo
using JLD2
using SparseArrays
using Plots
using Distributions
using Distributions: Normal
using Random
using FiniteDiff, ForwardDiff
using StaticArrays
using Test
import TrajectoryOptimization as TO
using Altro
import BilinearControl.Problems

@show Threads.nthreads()

include("learned_models/edmd_utils.jl")

## Create function for generating nominal cartpole problem for ALTRO
function gencartpoleproblem(x0=zeros(4), Qv=1e-2, Rv=1e-1, Qfv=1e2, u_bnd=3.0, tf=5.0; 
    dt=0.05, constrained=true)

    model = Problems.NominalCartpole()  # NOTE: this should exactly match RobotZoo.Cartpole()
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

## Visualizer
model = Problems.SimulatedCartpole()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
delete!(vis)
set_cartpole!(vis)
open(vis)

#############################################
## Define the Models
#############################################
# Define Nominal Simulated Cartpole Model
model_nom = Problems.NominalCartpole()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

# Define Mismatched "Real" Cartpole Model
model_real = Problems.SimulatedCartpole() # this model has damping
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

# Time parameters
tf = 5.0
dt = 0.05
Nt = 41  # MPC Horizon
t_sim = tf*1.2  # length of simulation (to capture steady-state behavior) 

#############################################
## Generate Training and Testing Data
#############################################

## Stabilization trajectories 
Random.seed!(1)
num_train_lqr = 3
num_test_lqr = 10

# Generate a stabilizing LQR controller about the top
Qlqr = Diagonal([1.0,10.0,1e-2,1e-2])
Rlqr = Diagonal([1e-3])
xe = [0,pi,0,0]
ue = [0.0]
ctrl_lqr = LQRController(dmodel_nom, Qlqr, Rlqr, xe, ue, dt)
T_lqr = range(0, t_sim, step=dt)

# Sample a bunch of initial conditions for the LQR controller
x0_sampler = Product([
    Uniform(-0.7,0.7),
    Uniform(pi-pi/4,pi+pi/4),
    Uniform(-.2,.2),
    Uniform(-.2,.2),
])
initial_conditions_train = [rand(x0_sampler) for _ in 1:num_train_lqr]
initial_conditions_test = [rand(x0_sampler) for _ in 1:num_test_lqr]

# Create data set
X_train_lqr, U_train_lqr = create_data(dmodel_real, ctrl_lqr, initial_conditions_train, t_sim, dt)
X_test_lqr, U_test_lqr = create_data(dmodel_real, ctrl_lqr, initial_conditions_test, t_sim, dt);

# Make sure they all stabilize
@test all(x->x<0.1, map(x->norm(x-xe), X_train_lqr[end,:]))
@test all(x->x<0.1, map(x->norm(x-xe), X_test_lqr[end,:]))

## Swing-up Trajectories 
Random.seed!(1)
num_train_swingup = 10
num_test_swingup = 10

train_params = map(1:num_train_swingup) do i
    Qv = 1e-2
    Rv = Qv * 10^rand(Uniform(-1,3.0))
    Qfv = Qv * 10^rand(Uniform(1,5.0)) 
    u_bnd = rand(Uniform(4.5, 8.0))
    (zeros(4), Qv, Rv, Qfv, u_bnd, tf)
end

Qmpc = Diagonal(fill(1e-0,4))
Rmpc = Diagonal(fill(1e-3,1))
Qfmpc = Diagonal(fill(1e2,4))

train_trajectories = map(train_params) do params
    solver = Altro.solve!(ALTROSolver(gencartpoleproblem(params..., dt=dt), 
        show_summary=false, projected_newton=true))
    if Altro.status(solver) != Altro.SOLVE_SUCCEEDED
        @warn "ALTRO Solve failed"
    end
    X = Vector.(TO.states(solver))
    U = Vector.(TO.controls(solver))
    T = Vector(range(0,tf,step=dt))

    push!(U, zeros(RD.control_dim(solver)))

    mpc = TrackingMPC(dmodel_nom, X, U, T, Qmpc, Rmpc, Qfmpc; Nt=Nt)
    X_sim,U_sim,T_sim = simulatewithcontroller(dmodel_real, mpc, X[1], t_sim, T[2])
    
    Vector.(X), Vector.(U[1:end-1]), Vector.(X_sim), Vector.(U_sim)
end

X_train_swingup_ref = mapreduce(x->getindex(x,1), hcat, train_trajectories)
U_train_swingup_ref = mapreduce(x->getindex(x,2), hcat, train_trajectories)
X_train_swingup = mapreduce(x->getindex(x,3), hcat, train_trajectories)
U_train_swingup = mapreduce(x->getindex(x,4), hcat, train_trajectories)

test_params = [
    (zeros(4), 1e-2, 1e-1, 1e2,  3.0, tf)
    (zeros(4), 1e-0, 1e-1, 1e2,  5.0, tf)
    (zeros(4), 1e1,  1e-2, 1e2, 10.0, tf)
    (zeros(4), 1e-1, 1e-0, 1e2, 10.0, tf)
    (zeros(4), 1e-2, 1e-0, 1e1, 10.0, tf)
    (zeros(4), 1e-2, 1e-0, 1e1,  3.0, tf)
    (zeros(4), 1e1,  1e-3, 1e2, 10.0, tf)
    (zeros(4), 1e1,  1e-3, 1e2,  5.0, tf)
    (zeros(4), 1e3,  1e-3, 1e3, 10.0, tf)
    (zeros(4), 1e0,  1e-2, 1e2,  4.0, tf)
]
test_trajectories = map(test_params) do params
    solver = Altro.solve!(ALTROSolver(gencartpoleproblem(params...; dt), show_summary=false))
    if Altro.status(solver) != Altro.SOLVE_SUCCEEDED
        @show params
        @warn "ALTRO Solve failed"
    end
    X = Vector.(TO.states(solver))
    U = Vector.(TO.controls(solver))
    T = Vector(range(0,tf,step=dt))

    push!(U, zeros(RD.control_dim(solver)))

    mpc = TrackingMPC(dmodel_nom, X, U, T, Qmpc, Rmpc, Qfmpc; Nt=Nt)
    X_sim,U_sim,T_sim = simulatewithcontroller(dmodel_real, mpc, X[1], t_sim, T[2])

    Vector.(X), Vector.(U[1:end-1]), Vector.(X_sim), Vector.(U_sim)
end

X_test_swingup_ref = mapreduce(x->getindex(x,1), hcat, test_trajectories)
U_test_swingup_ref = mapreduce(x->getindex(x,2), hcat, test_trajectories)
X_test_swingup = mapreduce(x->getindex(x,3), hcat, test_trajectories)
U_test_swingup = mapreduce(x->getindex(x,4), hcat, test_trajectories)
X_test_swingup[end,:]

## combine lqr and mpc training data
X_train = [X_train_lqr X_train_swingup]
U_train = [U_train_lqr U_train_swingup]

## Save generated training and test data
jldsave(joinpath(Problems.DATADIR, "mismatch_exp_cartpole_lqr_and_mpc_trajectories.jld2"); 
    X_train=X_train, U_train=U_train, 
    X_test_swingup, U_test_swingup, 
    X_test_swingup_ref, U_test_swingup_ref,
    X_test_lqr, U_test_lqr, 
    tf, t_sim, dt
)

## Import training and test data
altro_lqr_traj = load(joinpath(Problems.DATADIR, "mismatch_exp_cartpole_lqr_and_mpc_trajectories.jld2"))

X_train = altro_lqr_traj["X_train"]
U_train = altro_lqr_traj["U_train"]
X_test_swingup = altro_lqr_traj["X_test_swingup"]
U_test_swingup = altro_lqr_traj["U_test_swingup"]
X_test_swingup_ref = altro_lqr_traj["X_test_swingup_ref"]
U_test_swingup_ref = altro_lqr_traj["U_test_swingup_ref"]
X_test_lqr = altro_lqr_traj["X_test_lqr"]
U_test_lqr = altro_lqr_traj["U_test_lqr"]
tf = altro_lqr_traj["tf"]
t_sim = altro_lqr_traj["t_sim"]
dt = altro_lqr_traj["dt"]

T_altro = range(0,tf,step=dt)
T_sim = range(0,t_sim,step=dt)


#############################################
## Fit data using NOMINAL EDMD method
#############################################

# Define basis functions
eigfuns = ["state", "sine", "cosine", "sine", "sine", "chebyshev"]
eigorders = [[0],[1],[1],[2],[4],[2, 4]]

# Build the data 
Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders);

# Learn nominal model
t_train_eDMD = @elapsed A, B, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
    ["na", "na"]; 
    edmd_weights=[0.0], 
    mapping_weights=[0.0],
    algorithm=:qr
)

# Save nominal EDMD model
jldsave(joinpath(Problems.DATADIR,"mismatch_exp_cartpole_mpc_nominal_eDMD_data.jld2"); 
    A, B, C, g, kf, eigfuns, eigorders, tf, dt, t_train=t_train_eDMD
)

#############################################
## Fit data using Jacobian EDMD method
#############################################

# Generate Jacobians from nominal model
n0,m = RD.dims(model_nom)  # original dimensions
xn = zeros(n0)
n = length(kf(xn))         # lifted state dimension
jacobians = map(CartesianIndices(U_train)) do cind
    k = cind[1]
    x = X_train[cind]
    u = U_train[cind]
    z = RD.KnotPoint{n0,m}(x,u,T_sim[k],dt)
    J = zeros(n0,n0+m)
    RD.jacobian!(
        RD.InPlace(), RD.ForwardAD(), dmodel_nom, J, xn, z 
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
end

# Create a sparse version of the G Jacobian
G = spdiagm(n0,n,1=>ones(n0)) 
xn .= randn(n0)
@test G*kf(xn) ≈ xn

# Build Least Squares Problem
W,s = BilinearControl.EDMD.build_edmd_data(
    Z_train, U_train, A_train, B_train, F_train, G)

# Create sparse LLS matrix
@time Wsparse = sparse(W)
@show BilinearControl.matdensity(Wsparse)

# Solve with RLS
t_solve_jDMD = @elapsed x_rls = BilinearControl.EDMD.rls_qr(Vector(s), Wsparse; Q=1e-6)
E = reshape(x_rls,n,:)

# Extract out bilinear dynamics
A = E[:,1:n]
B = E[:,n .+ (1:m)]
C = E[:,n+m .+ (1:n*m)]

C_list = Matrix{Float64}[]
for i in 1:m
    C_i = C[:, (i-1)*n+1:i*n]
    push!(C_list, C_i)
end
C = C_list

# Save new jacobian-penalty EDMD model
jldsave(joinpath(Problems.DATADIR,"mismatch_exp_cartpole_mpc_jacobian_penalized_eDMD_data.jld2"); 
    A, B, C, g, kf, eigfuns, eigorders, tf, dt, t_solve=t_solve_jDMD
)

#############################################
## Model Comparisons 
#############################################

## Import eDMD model
cartpole_data = load(joinpath(Problems.DATADIR, "mismatch_exp_cartpole_mpc_nominal_eDMD_data.jld2"))
A_nom = cartpole_data["A"]
B_nom = cartpole_data["B"]
C_nom = cartpole_data["C"]
g = cartpole_data["g"]
kf = cartpole_data["kf"]
tf = cartpole_data["tf"]
dt = cartpole_data["dt"]
t_train_eDMD = cartpole_data["t_train"]
model_bilinear_eDMD = EDMDModel(A_nom,B_nom,C_nom,G,kf,dt,"cartpole_eDMD")
model_bilinear_eDMD_projected = Problems.ProjectedEDMDModel(model_bilinear_eDMD)

## Import jDMD model
cartpole_data = load(joinpath(Problems.DATADIR, "mismatch_exp_cartpole_mpc_jacobian_penalized_eDMD_data.jld2"))
A_jac = cartpole_data["A"]
B_jac = cartpole_data["B"]
C_jac = cartpole_data["C"]
g = cartpole_data["g"]
kf = cartpole_data["kf"]
tf = cartpole_data["tf"]
dt = cartpole_data["dt"]
model_bilinear_jDMD = EDMDModel(A_jac,B_jac,C_jac,G,kf,dt,"cartpole_eDMD")
model_bilinear_jDMD_projected = Problems.ProjectedEDMDModel(model_bilinear_jDMD)

## Training and test error
err_train_eDMD = BilinearControl.EDMD.fiterror(model_bilinear_eDMD, X_train, U_train)
err_test_swingup_ref_eDMD = BilinearControl.EDMD.fiterror(model_bilinear_eDMD, X_test_swingup_ref, U_test_swingup_ref)
err_test_swingup_eDMD = BilinearControl.EDMD.fiterror(model_bilinear_eDMD, X_test_swingup, U_test_swingup)
err_test_lqr_eDMD = BilinearControl.EDMD.fiterror(model_bilinear_eDMD, X_test_lqr, U_test_lqr)

err_train_jDMD = BilinearControl.EDMD.fiterror(model_bilinear_jDMD, X_train, U_train)
err_test_swingup_ref_jDMD = BilinearControl.EDMD.fiterror(model_bilinear_jDMD, X_test_swingup_ref, U_test_swingup_ref)
err_test_swingup_jDMD = BilinearControl.EDMD.fiterror(model_bilinear_jDMD, X_test_swingup, U_test_swingup)
err_test_lqr_jDMD = BilinearControl.EDMD.fiterror(model_bilinear_jDMD, X_test_lqr, U_test_lqr)

using LaTeXTabulars
error_table = [
    err_train_eDMD err_test_lqr_eDMD err_test_swingup_eDMD err_test_swingup_ref_eDMD
    err_train_jDMD err_test_lqr_jDMD err_test_swingup_jDMD err_test_swingup_ref_jDMD
]
latex_tabular(joinpath(Problems.FIGDIR, "tables", "cartpole_errors.tex"),
    Tabular("lll"),
    [
        Rule(:top),
        ["Method", "Training Error", "LQR Error", "Swingup Error", "Swingup Error (reference)"],
        Rule(:mid),
        [["eDMD"; "jDMD"] error_table],
        Rule(:bottom)
    ]
)
display(error_table)

## Compare Linearization
xe = [0.,pi,0,0]
ue = [0.0]
ze = RD.KnotPoint{n0,m}(xe,ue,0.0,dt)
ye = expandstate(model_bilinear_nom_EDMD, xe)
J = zeros(n0,n0+m)
xn = zeros(n0)
Qlqr = Diagonal([1.0,10,1e-2,1e-2])
Rlqr = Diagonal([1e-3])

models = [dmodel_real, dmodel_nom, model_bilinear_eDMD_projected, model_bilinear_jDMD_projected]
AB = map(models) do model
    RD.jacobian!(RD.InPlace(), RD.ForwardAD(), model, J, xn, ze)
    A = J[:,1:n0]
    B = J[:,n0+1:end]
    (A,B)
end
A_mats = getindex.(AB, 1)
B_mats = getindex.(AB, 2)

# Evaluate stability
K_mats = map(1:4) do i
    dlqr(A_mats[i], B_mats[i], Qlqr, Rlqr, max_iters=1000)[1]
end
A_cl = map(1:4) do i
    A_mats[i] - B_mats[i] * K_mats[i]
end
map(x->maximum(abs.(eigvals(x))), A_cl)

# Design LQR controllers
lqrs = map(K_mats) do K
    LQRController(K, xe, ue)
end

## Simulate nominal model with LQR gain from bilinear model
tf_sim = 6.0
Tsim_lqr_nominal = range(0,tf_sim,step=dt)

x0 = [+0.0, pi-deg2rad(00),0,0]
X_lqr,_,T_lqr = simulatewithcontroller(dmodel_true, lqrs[2], x0, tf_sim, dt)
plotstates(T_lqr, X_lqr, inds=1:2, legend=:none)
# X_lqr,_,T_lqr = simulatewithcontroller(dmodel_true, lqrs[4], x0, 10.0, dt)
# plotstates!(T_lqr, X_lqr, inds=1:2, legend=:none)
X_lqr,_,T_lqr = simulatewithcontroller(dmodel_true, lqrs[4], x0, tf_sim, dt)
plotstates!(T_lqr, X_lqr, inds=1:2, legend=:none)

X_ref = [copy(xe) for t in T_sim]
U_ref = [copy(ue) for t in T_sim]
mpc = TrackingMPC(model_bilinear_jDMD_projected, 
    X_ref, U_ref, Vector(T_sim), Qmpc, Rmpc, Qfmpc; Nt=Nt
)
RD.discrete_dynamics(dmodel_true, X_ref[1], U_ref[1], 0.0, dt)
X_mpc,U_mpc,T_mpc= simulatewithcontroller(dmodel_true, mpc, x0, tf_sim, dt)
plotstates!(T_mpc, X_mpc, inds=1:2, legend=:none)
cost(mpc)
map(x->x .= 0, mpc.X)
map(x->x .= 0, mpc.U)
map(x->x .= 0, mpc.f)

#############################################
## Compare MPC Performance
#############################################
i = 4 
tf_sim = T_altro[end]*1.5
X_ref = deepcopy(X_test_swingup_ref[:,i])
U_ref = deepcopy(U_test_swingup_ref[:,i])
T_ref = copy(T_altro)
push!(U_ref, zeros(RD.control_dim(model_bilinear_nom_EDMD)))

Qmpc = Diagonal(fill(1e-0,4))
Rmpc = Diagonal(fill(1e-3,1))
Qfmpc = Diagonal([1e4,1e2,1e1,1e1])

mpc_nom = TrackingMPC(dmodel_nom, 
    X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
)
mpc_eDMD = TrackingMPC(model_bilinear_eDMD_projected, 
    X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
)
mpc_jDMD = TrackingMPC(model_bilinear_jDMD_projected, 
    X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
)
X_mpc_nom, U_mpc_nom, T_mpc = simulatewithcontroller(dmodel_real, mpc_nom,  X_ref[1], tf_sim, T_altro[2])
X_mpc_eDMD,U_mpc_eDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_eDMD, X_ref[1], tf_sim, T_altro[2])
X_mpc_jDMD,U_mpc_jDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_jDMD, X_ref[1], tf_sim, T_altro[2])

plotstates(T_ref, X_ref, inds=1:2, xlabel="time (s)", ylabel="states",
            label=["x (ALTRO)" "θ (ALTRO)"], legend=:right, lw=1,
            linestyle=:solid, color=:black)
plotstates!(T_mpc, X_mpc_nom, inds=1:2, xlabel="time (s)", ylabel="states",
            label=["x (nominal MPC)" "θ (nominal MPC)"], legend=:right, lw=2,
            s=:dash,color=[1 2])
plotstates!(T_mpc, X_mpc_eDMD, inds=1:2, xlabel="time (s)", ylabel="states",
            label=["x (eDMD)" "θ (eDMD)"], legend=:right, lw=2,
            s=:dot, color=[1 2])
plotstates!(T_mpc, X_mpc_jDMD, inds=1:2, xlabel="time (s)", ylabel="states",
            label=["x (jDMD)" "θ (jDMD)"], legend=:outerright, lw=2,
            s=:solid, color=[1 2])
# ylims!((-3.,4))

## Learn new EDMD model with PENALTIES ON THE JACOBIANS

## Generate Jacobians

T_altro = range(0,tf,step=dt)

## Evaluate the new EDMD model fit

## Training and test error
err_train2 = BilinearControl.EDMD.fiterror(A_jacpen, B_jacpen, C_jacpen, g, kf, X_train, U_train)
err_test_mpc2 = BilinearControl.EDMD.fiterror(A_jacpen, B_jacpen, C_jacpen, g, kf, X_test_nom_mpc, U_test_nom_mpc)
err_test_lqr2 = BilinearControl.EDMD.fiterror(A_jacpen, B_jacpen, C_jacpen, g, kf, X_test_lqr, U_test_lqr)

println("Train Error: ", err_train)
println("Swing-up Test Error:  ", err_test_nom_mpc)
println("Stabilization Test Error:  ", err_test_lqr)
println("")
println("New Train Error: ", err_train2)
println("New Swing-up Test Error:  ", err_test_mpc2)
println("New Stabilization Test Error:  ", err_test_lqr2)

## Define bilinear model from new EDMD fit
model_bilinear_jacpen_EDMD = EDMDModel(A_jacpen, B_jacpen, C_jacpen, g, kf, dt, "cartpole")
model_bilinear_jacpen_EDMD_projected = Problems.ProjectedEDMDModel(model_bilinear_jacpen_EDMD)
n,m = RD.dims(model_bilinear_jacpen_EDMD)
n0 = originalstatedim(model_bilinear_jacpen_EDMD)

## Jacobians of new EDMD bilinear model 
J = zeros(n0,n0+m)
xn = zeros(n0)
RD.jacobian!(RD.InPlace(), RD.ForwardAD(), model_bilinear_jacpen_EDMD_projected, J, xn, ze)
A_bil_jacpen = J[:,1:n0]
B_bil_jacpen = J[:,n0+1:end]

## Display jacobians
@show A_og
@show A_bil_jacpen

@show B_og
@show B_bil_jacpen;

## Try stabilizing new EDMD model using LQR

## Determine LQR gains
K_bil_jacpen, = dlqr(A_bil_jacpen, B_bil_jacpen, Qlqr, Rlqr)

## Evaluate stability
isstable_bilinear2 = maximum(abs.(eigvals(A_bil_jacpen - B_bil_jacpen*K_bil_jacpen))) < 1.0
isstable_nominal_with_bilinear2 = maximum(abs.(eigvals(A_og - B_og*K_bil_jacpen))) < 1.0

println("Stability Summary:")
println("  Dynamics  |  Controller  |  is stable? ")
println("------------|--------------|--------------")
println("  Nominal   |  Nominal     |  ", isstable_nominal)
println("  Bilinear  |  Bilinear    |  ", isstable_bilinear2)
println("  Nominal   |  Bilinear    |  ", isstable_nominal_with_bilinear2)

## Simulate new model with LQR gain from bilinear model
tf_sim = 6.0
Tsim_lqr_jacpen = range(0,tf_sim,step=dt)

x0 = [-0.4,pi-deg2rad(40),0,0]

ctrl_lqr_jacpen = LQRController(K_bil_jacpen, xe, ue)
Xsim_lqr_jacpen, = simulatewithcontroller(dmodel_real, ctrl_lqr_jacpen, x0, tf_sim, dt)

## Plot the results
plotstates(Tsim_lqr_nominal, Xsim_lqr_og, inds=1:2, xlabel="time (s)", ylabel="states",
            label=["x (nominal MPC)" "θ (nominal MPC)"], legend=:right, lw=2,
            linestyle=:dash, color=[1 2])
plotstates!(Tsim_lqr_nominal, Xsim_lqr_nominal, inds=1:2, xlabel="time (s)", ylabel="states",
            label=["x (EDMD)" "θ (EDMD)"], legend=:right, lw=2,
            linestyle=:dashdot, color=[1 2])
plotstates!(Tsim_lqr_jacpen, Xsim_lqr_jacpen, inds=1:2, xlabel="time (s)", ylabel="states",
            label=["x (JDMD)" "θ (JDMD)"], legend=:right, lw=2,
            color=[1 2])
ylims!((-1.25,4))

## Render meshcat visualizer
render(vis)

## Visualize simulation in meshcat
# visualize!(vis, dmodel_real, tf_sim, Xsim_lqr_nominal)
visualize!(vis, dmodel_real, tf_sim, Xsim_lqr_jacpen)

## Try tracking swing-up trajectory using MPC

i = 1
tf_sim = T_altro[end]*2
X_ref = deepcopy(X_test_altro[:,i])
U_ref = deepcopy(U_test_altro[:,i])
X_nom_mpc = deepcopy(X_test_nom_mpc[:,i])
# push!(U_ref, zeros(RD.dims(model_bilinear_nom_EDMD)[2]))

Qmpc = Diagonal(fill(1e0,4))
Rmpc = Diagonal(fill(1e-3,1))
Qfmpc = Diagonal(fill(1e2,4))

mpc = TrackingMPC(model_bilinear_jacpen_EDMD_projected, X_ref, U_ref, Vector(T_altro), Qmpc, Rmpc, Qfmpc; Nt=Nt)
Xsim_mpc_jacpen_edmd,Usim_mpc_jacpen_edmd,Tsim_mpc_jacpen_edmd = simulatewithcontroller(dmodel_real, mpc, X_ref[1], tf_sim, T_altro[2])

plotstates(T_altro, X_ref, inds=1:2, xlabel="time (s)", ylabel="states",
            label=["x (ALTRO)" "θ (ALTRO)"], legend=:bottomright, lw=2,
            linestyle=:dot, color=[1 2])
plotstates!(T_nom_mpc, X_nom_mpc, inds=1:2, xlabel="time (s)", ylabel="states",
            label=["x (nominal MPC)" "θ (nominal MPC)"], legend=:top, lw=2,
            linestyle=:dash, color=[1 2])
plotstates!(T_mpc, X_mpc_eDMD, inds=1:2, xlabel="time (s)", ylabel="states",
            label=["x (EDMD)" "θ (EDMD)"], legend=:bottomright, lw=2,
            linestyle=:dashdotdot, color=[1 2])
plotstates!(Tsim_mpc_jacpen_edmd, Xsim_mpc_jacpen_edmd, inds=1:2, xlabel="time (s)", ylabel="states",
            label=["x (JDMD)" "θ (JDMD)"], legend=:bottomright, lw=2,
            color=[1 2])
ylims!((-1.5,4))

## Render meshcat visualizer
render(vis)

## Visualize simulation in meshcat
# visualize!(vis, dmodel_real, tf_sim, Xsim_mpc_nom_edmd)
visualize!(vis, dmodel_real, tf_sim, Xsim_mpc_jacpen_edmd)
