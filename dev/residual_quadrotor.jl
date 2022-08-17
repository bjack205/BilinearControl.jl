import Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra
using Random
using Statistics
using SparseArrays

using StaticArrays
using BilinearControl
using Distributions
import RobotDynamics as RD
using BilinearControl.Problems
using BilinearControl.EDMD
using Altro
using RobotZoo
using Plots
using BilinearControl: Problems, EDMD

include(joinpath(@__DIR__, "..", "examples", "quadrotor_utils.jl"))

#############################################
## Visualization
#############################################
if !isdefined(@__MODULE__, :vis)
    visdir = joinpath(@__DIR__, "../examples/visualization/")
    include(joinpath(visdir, "visualization.jl"))
    vis = Visualizer()
    open(vis)
    delete!(vis)
    RobotMeshes.setdrone!(vis["robot"]["geom"], scale=0.5)
end


#############################################
## Define the Models
#############################################

# Define Nominal Simulated Cartpole Model
model_nom = Problems.NominalQuadrotor()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

# Define Mismatched "Real" Cartpole Model
model_real = Problems.SimulatedQuadrotor()
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

# Time parameters
tf = 5.0
dt = 0.05
Nt = 41  # MPC Horizon
t_sim = tf*1.2  # length of simulation (to capture steady-state behavior) 
num_train = 50
num_lqr=10

#############################################
## Generate the training data
#############################################
opts = Altro.SolverOptions(show_summary=false, verbose=0,
    cost_tolerance=1e-3, cost_tolerance_intermediate=1e-2,
    expected_decrease_tolerance=1e-4,
    projected_newton=false,
)
X_train, U_train, X_test, U_test, X_ref = 
    generate_quadrotor_data(;tf, dt, Nt, num_train)
X_train
findall(x->!isfinite(norm(x)), X_train[end,:])
findall(x->!isfinite(norm(x)), X_test[end,:])
@assert mapreduce(norm, +, X_train) / length(X_train) < 20.0

#############################################
## Learn the residual
#############################################

# Calculate the error terms
E_train = map(CartesianIndices(U_train)) do idx
    x = X_train[idx]
    u = U_train[idx]
    xn = X_train[idx + CartesianIndex(1,0)]
    xhat = RD.discrete_dynamics(dmodel_nom,x, u, 0.0, dt)
    xn - xhat
end

# Define the Koopman function
eigfuns = ["state"]
eigorders = [[0], [2,4]]
kf(x) = EDMD.koopman_transform(x, eigfuns, eigorders)

n0 = length(X_train[1])
n = length(kf(X_train[1]))
m = length(U_train[1])

# Get lifted states
W_train = map(kf, E_train)  # lifted errors
Z_train = map(CartesianIndices(U_train)) do idx  # lifted states and controls
    x = X_train[idx]
    y = kf(x)
    u = U_train[idx]
    vcat(y, u, vec(y*u'))
end
@assert length(W_train[1]) == n
@assert length(Z_train[1]) == n + m + n*m

# Fit a bilinear model
W_mat = reduce(hcat, W_train)
Z_mat = reduce(hcat, Z_train)
model_data = EDMD.fitA(Z_mat, W_mat, rho=1e-6, algorithm=:qr_rls)

A = model_data[:, 1:n] 
B = model_data[:, n .+ (1:m)] 
C = [model_data[:, n + m + (i-1)*n .+ (1:n)] for i = 1:m]
G = spdiagm(n0,n,1=>ones(n0)) 
edmd_model = EDMDModel(A, B, C, G, kf, dt, "cartpole_error")

# Build model
model_res = EDMD.EDMDErrorModel(dmodel_nom, edmd_model)

#############################################
## Test MPC
#############################################

i = 4

# MPC Params
Q = Diagonal([1, 1, 10, 0.5, 0.5, 0.5, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])
R = Diagonal(fill(1e-0, 4))
Qf = copy(Q)
Nt = 21
xbnd = [fill(2.0,3); fill(0.2,3); fill(3.0,6)]
ubnd = fill(5.0, 4)

# Build MPC controllers
Xref = X_ref[:,i]
Uref = [Problems.trim_controls(model_nom) for x in Xref]
Tref = range(0,tf,step=dt)

mpc = EDMD.LinearMPC(model_res, Xref, Uref, collect(Tref), Q, R, Qf; Nt,
    # xmax=xbnd, xmin=-xbnd, 
    umax=ubnd, umin=-ubnd
)
x = copy(Xref[1])
i = 1

##
# EDMD.solve!(mpc, Xref[1], i)
t = (i-1)*dt
u = EDMD.getcontrol(mpc, x, t)
Xmpc = mpc.X .+ Xref[EDMD.get_ref_inds(mpc, i)]
Tmpc = range(0,step=dt, length=Nt) .+ t 
x = RD.discrete_dynamics(dmodel_real, x, u, t, dt)
# x = mpc.X[2] + mpc.Xref[i+1]
X = abs.(reduce(hcat, mpc.X))
maximum(X, dims=2)
norm(mpc.U, Inf)
mpc.U

p = plotstates(Tref, Xref, inds=1:3, c=[1 2 3], s=:dash, lw=2, label="ref")
plotstates!(Tmpc, Xmpc, inds=1:3, c=[1 2 3], s=:solid, label="nom")
display(p)
i += 1
##

mpc_nom = EDMD.LinearMPC(dmodel_nom, Xref, Uref, collect(Tref), Q, R, Qf; Nt,
    xmax=xbnd, xmin=-xbnd, 
    umax=ubnd, umin=-ubnd
)

mpc_res = EDMD.LinearMPC(model_res, Xref, Uref, collect(Tref), Q, R, Qf; Nt,
    xmax=xbnd, xmin=-xbnd, 
    umax=ubnd, umin=-ubnd
)

Xnom,_,Tsim = simulatewithcontroller(dmodel_real, mpc_nom, Xref[1], t_sim, dt)
Xres,_,Tsim = simulatewithcontroller(dmodel_real, mpc_res, Xref[1], t_sim, dt)
p = plotstates(Tref, Xref, inds=1:3, c=[1 2 3], s=:dash, lw=2, label="ref")
plotstates!(Tsim, Xnom, inds=1:3, c=[1 2 3], s=:solid, label="nom")

opts2 = copy(opts)
opts2.show_summary = true
opts2.verbose = 4
prob_res = build_mpc_problem(model_res, Xref[1], Xref, Uref, Tref; Nmpc=Nt)
mpc_res = AltroController(prob_res, Xref, Uref, Tref; opts=opts2)
solver = mpc_res.solver
TO.rollout!(solver)
TO.cost(solver)
TO.controls(solver)
TO.states(solver)
Altro.solve!(mpc_res.solver)

Xnom,_,Tsim = simulatewithcontroller(dmodel_real, mpc_nom, Xref[1], t_sim, dt)
plotstates(Tref, Xref, inds=1:3, c=[1 2 3], s=:dash, lw=2, label="ref")
plotstates!(Tsim, Xnom, inds=1:3, c=[1 2 3], s=:solid, label="nom")