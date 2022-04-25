using BilinearControl

using RobotZoo
using RobotDynamics
using LinearAlgebra
using StaticArrays
using SparseArrays
using Convex
using SCS
using TrajectoryOptimization
using JLD2
using Test
const RD = RobotDynamics

function simulate_bilinear(F, C, g, x0, z0, U)
    
    x = x0
    z = z0
    Z = [z]
    X = [x]

    for k in 1:length(U)

        u = U[k]
        
        z = F * z + (C * z) .* u
        x = g * z

        push!(Z, z)
        push!(X, x)
        
    end

    return X, Z
end

## Load Reference Trajectory from file
const datadir = joinpath(@__DIR__, "..", "data")
ref_traj = load(joinpath(datadir, "cartpole_reference_trajectory.jld2"))
N = 601  # QUESTION: why not use entire trajectory?
X_sim = ref_traj["X_sim"][1:N]
U_sim = ref_traj["U_sim"][1:N-1]
T_sim = ref_traj["T_sim"][1:N]

## Define model
dt = T_sim[2] - T_sim[1]
tf = T_sim[end]

# Initial condition
x0 = copy(X_sim[1])

# Define the model
model = RobotZoo.Cartpole(1.0, 0.2, 0.5, 9.81)
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
n,m = RD.dims(model)

## Learn the bilinear dynamics
Z_sim, Zu_sim, z0 = BilinearControl.edmd.build_eigenfunctions(X_sim, U_sim, ["state", "sine", "cosine"], [0,0,0])
F, C, g = BilinearControl.edmd.learn_bilinear_model(
    X_sim, Z_sim, Zu_sim, ["lasso", "lasso"]; 
    edmd_weights=[0.0], mapping_weights=[0.0]
)

## Compare solutions
bi_X, bi_Z = simulate_bilinear(F, C, g, x0, z0, U_sim)

# Test that the discrete dynamics match
@test all(1:length(U_sim)) do k
    h = dt
    z = bi_Z[k]
    x = g * bi_Z[k]
    u = U_sim[k]
    xn0 = RD.discrete_dynamics(dmodel, x, U_sim[k], 0, h)
    zn = F*z + C*z * u[1]
    xn_bilinear = g*zn
    norm(xn0 - xn_bilinear) < 5e-2 
end

# Test that the trajectories are similar
@test norm(bi_X - X_sim, Inf) < 0.2