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
using Test

@show Threads.nthreads()

include("learned_models/edmd_utils.jl")

## Visualizer
model = Problems.RexPlanarQuadrotor()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
delete!(vis)
set_quadrotor!( vis, model)

## Render meshcat visualizer
render(vis)

## Define planar quadrotor model
model = Problems.RexPlanarQuadrotor()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)

## Time parameters
tf = 5.0
dt = 0.05

## Generate Data From Mismatched Model
Random.seed!(1)

# number of trajectories
num_train = 30
num_test = 20

# Generate a stabilizing LQR controller
Qlqr = Diagonal([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
Rlqr = Diagonal([1e-4, 1e-4])
xe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ue = Problems.trim_controls(model)
ctrl_lqr = LQRController(dmodel, Qlqr, Rlqr, xe, ue, dt)

# Sample a bunch of initial conditions for the LQR controller
x0_sampler = Product([
    Uniform(-1.0,1.0),
    Uniform(-1.0,1.0),
    Uniform(-deg2rad(45),deg2rad(45)),
    Uniform(-0.5,0.5),
    Uniform(-0.5,0.5),
    Uniform(-0.25,0.25)
])

initial_conditions_lqr = [rand(x0_sampler) for _ in 1:num_train]
initial_conditions_test = [rand(x0_sampler) for _ in 1:num_test]

# Create data set
X_train, U_train = create_data(dmodel, ctrl_lqr, initial_conditions_lqr, tf, dt)
X_test, U_test = create_data(dmodel, ctrl_lqr, initial_conditions_test, tf, dt)

## Save generated training and test data
jldsave(joinpath(Problems.DATADIR, "rex_planar_quadrotor_lqr_trajectories.jld2"); 
    X_train, U_train, X_test, U_test, tf, dt)

## Import training and test data
lqr_traj = load(joinpath(Problems.DATADIR, "rex_planar_quadrotor_lqr_trajectories.jld2"))

X_train = lqr_traj["X_train"]
U_train = lqr_traj["U_train"]
X_test = lqr_traj["X_test"]
U_test = lqr_traj["U_test"]
tf = lqr_traj["tf"]
dt = lqr_traj["dt"]

T_ref = range(0,tf,step=dt)

## Fit data using NOMINAL EDMD method

## Define basis functions
eigfuns = ["state", "sine", "cosine", "chebyshev"]
eigorders = [[0],[1],[1],[2,4]]

## Fit the data
Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders);

## Learn nominal model

#=
A, B, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
    ["ridge", "na"]; 
    edmd_weights=[1e-6], 
    mapping_weights=[0.0], 
    algorithm=:qr
);
=#

A, B, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
    ["na", "na"]; 
    edmd_weights=[0.0], 
    mapping_weights=[0.0],
    algorithm=:qr
)

## Save nominal EDMD model
jldsave(joinpath(Problems.DATADIR,"rex_planar_quadrotor_lqr_nominal_eDMD_data.jld2"); A, B, C, g, kf, eigfuns, eigorders, tf, dt)

## Import nominal EDMD model
cartpole_data = load(joinpath(Problems.DATADIR, "rex_planar_quadrotor_lqr_nominal_eDMD_data.jld2"))
A_nom = cartpole_data["A"]
B_nom = cartpole_data["B"]
C_nom = cartpole_data["C"]
g = cartpole_data["g"]
kf = cartpole_data["kf"]
tf = cartpole_data["tf"]
dt = cartpole_data["dt"]

## Evaluate the nominal EDMD model fit

## Training and test error
err_train = BilinearControl.EDMD.fiterror(A_nom, B_nom, C_nom, g, kf, X_train, U_train)
err_test = BilinearControl.EDMD.fiterror(A_nom, B_nom, C_nom, g, kf, X_test, U_test)
println("Train Error: ", err_train)
println("Test Error:  ", err_test)

## Define bilinear model from nominal EDMD fit
model_bilinear_nom_EDMD = EDMDModel(A_nom,B_nom,C_nom,g,kf,dt,"cartpole")
model_bilinear_nom_EDMD_projected = Problems.ProjectedEDMDModel(model_bilinear_nom_EDMD)
n,m = RD.dims(model_bilinear_nom_EDMD)
n0 = originalstatedim(model_bilinear_nom_EDMD)
println("New state dimension: ", n)

## Compare Linearization

# Define the equilibrium
xe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ue = Problems.trim_controls(model)
ze = RD.KnotPoint{n0,m}(xe,ue,0.0,dt)
ye = expandstate(model_bilinear_nom_EDMD, xe)

# True jacobians from mismatched "real" model
J = zeros(n0,n0+m)
xn = zeros(n0)
RD.jacobian!(RD.InPlace(), RD.ForwardAD(), dmodel, J, xn, ze)
A_og = J[:,1:n0]
B_og = J[:,n0+1:end]

## Jacobians of nominal EDMD bilinear model 
J = zeros(n0,n0+m)
xn = zeros(n0)
RD.jacobian!(RD.InPlace(), RD.ForwardAD(), model_bilinear_nom_EDMD_projected, J, xn, ze)
A_bil_nom = J[:,1:n0]
B_bil_nom = J[:,n0+1:end]

## Display jacobians
@show A_og
@show A_bil_nom

@show B_og
@show B_bil_nom;

## Try stabilizing nominal model using LQR

# Calculate LQR Gain 
Qlqr = Diagonal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
Rlqr = Diagonal([1e-4, 1e-4])

K_og, = dlqr(A_og, B_og, Qlqr, Rlqr)
K_bil_nom, = dlqr(A_bil_nom, B_bil_nom, Qlqr, Rlqr)

# Evaluate stability
isstable_nominal = maximum(abs.(eigvals(A_og - B_og*K_og))) < 1.0
isstable_bilinear = maximum(abs.(eigvals(A_bil_nom - B_bil_nom*K_bil_nom))) < 1.0
isstable_nominal_with_bilinear = maximum(abs.(eigvals(A_og - B_og*K_bil_nom))) < 1.0

println("Stability Summary:")
println("  Dynamics  |  Controller  |  is stable? ")
println("------------|--------------|--------------")
println("  Nominal   |  Nominal     |  ", isstable_nominal)
println("  Bilinear  |  Bilinear    |  ", isstable_bilinear)
println("  Nominal   |  Bilinear    |  ", isstable_nominal_with_bilinear)

## Simulate nominal model with LQR gain from bilinear model
tf_sim = 5.0
Tsim_lqr_nominal = range(0,tf_sim,step=dt)

x0 = [-0.5, 0.5, -deg2rad(20),-1.0,1.0,0.0]

ctrl_lqr_og = LQRController(K_og, xe, ue)
ctrl_lqr_nominal = LQRController(K_bil_nom, xe, ue)

Xsim_lqr_og, = simulatewithcontroller(dmodel, ctrl_lqr_og, x0, tf_sim, dt)
Xsim_lqr_nominal, = simulatewithcontroller(dmodel, ctrl_lqr_nominal, x0, tf_sim, dt)

plotstates(Tsim_lqr_nominal, Xsim_lqr_og, inds=1:3, xlabel="time (s)", ylabel="states",
            label=["x (og dynamics)" "y (og dynamics)" "θ (og dynamics)"], legend=:topright, lw=2,
            linestyle=:dash, color=[1 2 3])
plotstates!(Tsim_lqr_nominal, Xsim_lqr_nominal, inds=1:3, xlabel="time (s)", ylabel="states",
            label=["x (nominal eDMD)" "y (nominal eDMD)" "θ (nominal eDMD)"], legend=:topright, lw=2,
            color=[1 2 3])
ylims!((-1.25,1.25))

## Learn new EDMD model with PENALTIES ON THE JACOBIANS

## Generate Jacobians
xn = zeros(n0)
jacobians = map(CartesianIndices(U_train)) do cind
    k = cind[1]
    x = X_train[cind]
    u = U_train[cind]
    z = RD.KnotPoint{n0,m}(x,u,T_ref[k],dt)
    J = zeros(n0,n0+m)
    RD.jacobian!(
        RD.InPlace(), RD.ForwardAD(), dmodel, J, xn, z 
    )
    J
end
A_train = map(J->J[:,1:n0], jacobians)
B_train = map(J->J[:,n0+1:end], jacobians)

# Convert states to lifted Koopman states
Y_train = map(kf, X_train)

# Calculate Jacobian of Koopman transform
F_train = map(@view X_train[1:end-1,:]) do x
    sparse(ForwardDiff.jacobian(x->expandstate(model_bilinear_nom_EDMD,x), x))
end

## Build Least Squares Problem
W,s = BilinearControl.EDMD.build_edmd_data(
    Z_train, U_train, A_train, B_train, F_train, model_bilinear_nom_EDMD.g)

n = length(Z_train[1])

# Create sparse LLS matrix
@time Wsparse = sparse(W)
@show BilinearControl.matdensity(Wsparse)

## Solve with RLS
@time x_rls = BilinearControl.EDMD.rls_qr(Vector(s), Wsparse; Q=1e-4)
E = reshape(x_rls,n,:)

## Extract out bilinear dynamics
A = E[:,1:n]
B = E[:,n .+ (1:m)]
C = E[:,n+m .+ (1:n*m)]

C_list = Matrix{Float64}[]
    
for i in 1:m
    C_i = C[:, (i-1)*n+1:i*n]
    push!(C_list, C_i)
end

C = C_list

## Save new jacobian-penalty EDMD model
jldsave(joinpath(Problems.DATADIR,"rex_planar_quadrotor_lqr_jacobian_penalized_eDMD_data.jld2"); A, B, C, g, kf, eigfuns, eigorders, tf, dt)

## Import new jacobian-penalty EDMD model
cartpole_data = load(joinpath(Problems.DATADIR, "rex_planar_quadrotor_lqr_jacobian_penalized_eDMD_data.jld2"))
A_jacpen = cartpole_data["A"]
B_jacpen = cartpole_data["B"]
C_jacpen = cartpole_data["C"]
g = cartpole_data["g"]
kf = cartpole_data["kf"]
tf = cartpole_data["tf"]
dt = cartpole_data["dt"]

T_ref = range(0,tf,step=dt)

## Evaluate the new EDMD model fit

## Training and test error
err_train2 = BilinearControl.EDMD.fiterror(A_jacpen, B_jacpen, C_jacpen, g, kf, X_train, U_train)
err_test2 = BilinearControl.EDMD.fiterror(A_jacpen, B_jacpen, C_jacpen, g, kf, X_test, U_test)
println("Train Error: ", err_train)
println("Test Error:  ", err_test)
println("")
println("New Train Error: ", err_train2)
println("New Test Error:  ", err_test2)

## Define bilinear model from new EDMD fit
model_bilinear_jacpen_EDMD = EDMDModel(A_jacpen, B_jacpen, C_jacpen, g, kf, dt, "planar_quadrotor")
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

# Determine LQR gains
K_bil_jacpen, = dlqr(A_bil_jacpen, B_bil_jacpen, Qlqr, Rlqr)

# Evaluate stability
isstable_bilinear2 = maximum(abs.(eigvals(A_bil_jacpen - B_bil_jacpen*K_bil_jacpen))) < 1.0
isstable_nominal_with_bilinear2 = maximum(abs.(eigvals(A_og - B_og*K_bil_jacpen))) < 1.0

println("Stability Summary:")
println("  Dynamics  |  Controller  |  is stable? ")
println("------------|--------------|--------------")
println("  Nominal   |  Nominal     |  ", isstable_nominal)
println("  Bilinear  |  Bilinear    |  ", isstable_bilinear2)
println("  Nominal   |  Bilinear    |  ", isstable_nominal_with_bilinear2)

## Simulate new model with LQR gain from bilinear model
tf_sim = 5.0
Tsim_lqr_jacpen = range(0,tf_sim,step=dt)

x0 = [-0.5, 0.5, -deg2rad(20),-1.0,1.0,0.0]

ctrl_lqr_jacpen = LQRController(K_bil_jacpen, xe, ue)
Xsim_lqr_jacpen, = simulatewithcontroller(dmodel, ctrl_lqr_jacpen, x0, tf_sim, dt)

plotstates(Tsim_lqr_nominal, Xsim_lqr_og, inds=1:3, xlabel="time (s)", ylabel="states",
            label=["x (og dynamics)" "y (og dynamics)" "θ (og dynamics)"], legend=:bottomright, lw=3,
            linestyle=:dash, color=[1 2 3])
plotstates!(Tsim_lqr_nominal, Xsim_lqr_nominal, inds=1:3, xlabel="time (s)", ylabel="states",
            label=["x (nominal eDMD)" "y (nominal eDMD)" "θ (nominal eDMD)"], legend=:bottomright, lw=2,
            linestyle=:dot, color=[1 2 3])
plotstates!(Tsim_lqr_jacpen, Xsim_lqr_jacpen, inds=1:3, xlabel="time (s)", ylabel="states",
            label=["x (jacobian eDMD)" "y (jacobian eDMD)" "θ (jacobian eDMD)"], legend=:bottomright, lw=2,
            color=[1 2 3])
ylims!((-1.25,0.75))

## Render meshcat visualizer
render(vis)

## Visualize simulation in meshcat
# visualize!(vis, model, tf_sim, Xsim_lqr_nominal)
visualize!(vis, model, tf_sim, Xsim_lqr_jacpen)