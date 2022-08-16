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
using Altro
using RobotZoo
using Plots

include(joinpath(@__DIR__, "..", "examples", "cartpole_utils.jl"))

#############################################
## Visualization
#############################################
if !isdefined(@__MODULE__, :vis)
    model = RobotZoo.Cartpole()
    visdir = joinpath(@__DIR__, "../examples/visualization/")
    include(joinpath(visdir, "visualization.jl"))
    vis = Visualizer()
    open(vis)
    delete!(vis)
    set_cartpole!(vis)
end

#############################################
## Define the Models
#############################################
# Define Nominal Simulated Cartpole Model
μ_nom = 0.0
μ = 0.1
model_nom = Problems.NominalCartpole(;μ=μ_nom)
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

# Define Mismatched "Real" Cartpole Model
model_real = Problems.SimulatedCartpole(;μ=μ) # this model has damping
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

# Time parameters
tf = 5.0
dt = 0.05
Nt = 41  # MPC Horizon
t_sim = tf*1.2  # length of simulation (to capture steady-state behavior) 
num_train = 50
num_lqr=10

T_sim = range(0,t_sim,step=dt)

#############################################
## Generate the training data 
#############################################

X_train, U_train, X_test, U_test, X_ref, U_ref, metadata = generate_cartpole_data(;
    num_lqr, μ, μ_nom
)

visualize!(vis, RobotZoo.Cartpole(), t_sim, X_ref[:,1])

#############################################
## Calculate the dynamics error 
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
eigfuns = ["state", "chebyshev"]
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
model_data = EDMD.fitA(Z_mat, W_mat, rho=1e-6)

A = model_data[:, 1:n] 
B = model_data[:, n .+ (1:m)] 
C = [model_data[:, n + m + (i-1)*n .+ (1:n)] for i = 1:m]
G = spdiagm(n0,n,1=>ones(n0)) 
edmd_model = EDMDModel(A, B, C, G, kf, dt, "cartpole_error")

# Build model
res_model = EDMD.EDMDErrorModel(dmodel_nom, edmd_model)

#############################################
## Test MPC
#############################################

i = 1

# MPC Params
Nt = 41
Qmpc = Diagonal(fill(1e-0,4))
Rmpc = Diagonal(fill(1e-3,1))
Qfmpc = Diagonal([1e2,1e2,1e1,1e1])

# Build MPC controllers
Xref = X_ref[:,i+num_train]
Uref = U_ref[:,i+num_train]
Tref = range(0,tf,step=dt)
push!(Uref, zeros(m))

mpc_nom = TrackingMPC(dmodel_nom, Xref, Uref, Tref, Qmpc, Rmpc, Qfmpc; Nt) 
mpc_res = TrackingMPC(res_model,  Xref, Uref, Tref, Qmpc, Rmpc, Qfmpc; Nt) 

X_nom, = simulatewithcontroller(dmodel_real, mpc_nom, Xref[1], t_sim, dt)
X_res, = simulatewithcontroller(dmodel_real, mpc_res, Xref[1], t_sim, dt)
plotstates(Tref, Xref, inds=1:2, label="ref", lw=2, legend=:topleft)
plotstates!(T_sim, X_nom, c=[1 2], lw=2, s=:dash, inds=1:2, label="nominal")
plotstates!(T_sim, X_res, c=[1 2], lw=2, s=:dot,  inds=1:2, label="res")

visualize!(vis, RobotZoo.Cartpole(), t_sim, X_nom)
visualize!(vis, RobotZoo.Cartpole(), t_sim, X_res)