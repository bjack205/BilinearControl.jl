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

include("learned_models/edmd_utils.jl")

model = Problems.RexQuadrotor()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
tf = 2.0
dt = 0.02

T_ref = range(0,tf,step=dt)

n, m = RD.dims(model)

## Generate training LQR trajectories

Random.seed!(1)

# number of trajectories
num_train = 30
num_test = 10

# Generate a stabilizing LQR controller
Qlqr = Diagonal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
Rlqr = Diagonal([1e-4, 1e-4, 1e-4, 1e-4])
xe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ue = [0.25*9.81, 0.25*9.81, 0.25*9.81, 0.25*9.81]
ctrl_lqr = LQRController(dmodel, Qlqr, Rlqr, xe, ue, dt)

# Sample a bunch of initial conditions for the LQR controller
x0_sampler = Product([
    Uniform(-1.0,1.0),
    Uniform(-1.0,1.0),
    Uniform(-1.0,1.0),
    Uniform(-deg2rad(45),deg2rad(45)),
    Uniform(-deg2rad(45),deg2rad(45)),
    Uniform(-deg2rad(45),deg2rad(45)),
    Uniform(-0.5,0.5),
    Uniform(-0.5,0.5),
    Uniform(-0.5,0.5),
    Uniform(-0.25,0.25),
    Uniform(-0.25,0.25),
    Uniform(-0.25,0.25)
])
initial_conditions_lqr = [rand(x0_sampler) for _ in 1:num_train]
initial_conditions_test = [rand(x0_sampler) for _ in 1:num_test]

# Create data set
X_train, U_train = create_data(dmodel, ctrl_lqr, initial_conditions_lqr, tf, dt)
X_test, U_test = create_data(dmodel, ctrl_lqr, initial_conditions_test, tf, dt)

#save data
jldsave(joinpath(Problems.DATADIR, "rex_quadrotor_lqr_trajectories.jld2"); 
    X_train, U_train, X_test, U_test, tf, dt)

# import data
lqr_traj = load(joinpath(Problems.DATADIR, "rex_quadrotor_lqr_trajectories.jld2"))

X_train = lqr_traj["X_train"]
U_train = lqr_traj["U_train"]
X_test = lqr_traj["X_test"]
U_test = lqr_traj["U_test"]
tf = lqr_traj["tf"]
dt = lqr_traj["dt"]

T_ref = range(0,tf,step=dt)

# Fit nominal EDMD model

# Define the basis functions
eigfuns = ["state", "sine", "cosine", "chebyshev"]
eigorders = [0,0,0,4]

# Fit the data
Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders)

# A, B, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
#     ["ridge", "lasso"]; 
#     edmd_weights=[10.1], 
#     mapping_weights=[0.0], 
#     algorithm=:qr
# );

A, B, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
    ["na", "na"]; 
    edmd_weights=[0.0], 
    mapping_weights=[0.0], 
    algorithm=:qr
)

# Save model
jldsave(joinpath(Problems.DATADIR,"rex_quadrotor_lqr_nominal_eDMD_data.jld2"); A, B, C, g, kf, eigfuns, eigorders, tf, dt)

# import nominal eDMD model
cartpole_data = load(joinpath(Problems.DATADIR, "rex_quadrotor_lqr_nominal_eDMD_data.jld2"))
A_nom = cartpole_data["A"]
B_nom = cartpole_data["B"]
C_nom = cartpole_data["C"]
g_nom = cartpole_data["g"]
kf = cartpole_data["kf"]
tf = cartpole_data["tf"]
dt = cartpole_data["dt"]

## Evaluate nominal eDMD fit
err_train = BilinearControl.EDMD.fiterror(A_nom, B_nom, C_nom, g_nom, kf, X_train, U_train)
err_test = BilinearControl.EDMD.fiterror(A_nom, B_nom, C_nom, g_nom, kf, X_test, U_test)
println("Train Error: ", err_train)
println("Test Error:  ", err_test)

model_bilinear = EDMDModel(A_nom,B_nom,C_nom,g_nom,kf,dt,"planar_quadrotor")
n,m = RD.dims(model_bilinear)
n0 = originalstatedim(model_bilinear)
println("New state dimension: ", n)

# Define the equilibrium
xe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ue = [0.25*9.81, 0.25*9.81, 0.25*9.81, 0.25*9.81]
ze = RD.KnotPoint{n0,m}(xe,ue,0.0,dt)
ye = expandstate(model_bilinear, xe)

# Nominal Jacobians
J = zeros(n0,n0+m)
xn = zeros(n0)
RD.jacobian!(RD.InPlace(), RD.ForwardAD(), dmodel, J, xn, ze)
A_og = J[:,1:n0]
B_og = J[:,n0+1:end]

# Bilinear Jacobians
function dynamics_bilinear(x,u,t,dt)
    y = expandstate(model_bilinear, x)
    yn = zero(y)
    RD.discrete_dynamics!(model_bilinear, yn, y, u, t, dt)
    originalstate(model_bilinear, yn)
end

A_bil_nom = FiniteDiff.finite_difference_jacobian(x->dynamics_bilinear(x,ue,0.0,dt), xe)
B_bil_nom = FiniteDiff.finite_difference_jacobian(u->dynamics_bilinear(xe,u,0.0,dt), ue)

# Calculate LQR Gain 
Qlqr = Diagonal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
Rlqr = Diagonal([1e-4, 1e-4, 1e-4, 1e-4])

K_og = dlqr(A_og, B_og, Qlqr, Rlqr)
K_bil_nom = dlqr(A_bil_nom, B_bil_nom, Qlqr, Rlqr)

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

# Simulate nominal model with LQR gain from bilinear model

tf_sim = 5.0
Tsim_lqr_nominal = range(0,tf_sim,step=dt)

# x0 = [0,pi-deg2rad(1),0,0]
x0 = [-0.5, -0.5, 0.5, -deg2rad(20), deg2rad(50), -deg2rad(20), -1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

ctrl_lqr_og = LQRController(K_og, xe, ue)
ctrl_lqr_nominal = LQRController(K_bil_nom, xe, ue)

Xsim_lqr_og, = simulatewithcontroller(dmodel, ctrl_lqr_og, x0, tf_sim, dt)
Xsim_lqr_nominal, = simulatewithcontroller(dmodel, ctrl_lqr_nominal, x0, tf_sim, dt)

plotstates(Tsim_lqr_nominal, Xsim_lqr_og, inds=1:3, xlabel="time (s)", ylabel="states",
            label=["x (og dynamics)" "y (og dynamics)" "z (og dynamics)"], legend=:right, lw=2,
            linestyle=:dash, color=[1 2 3])
plotstates!(Tsim_lqr_nominal, Xsim_lqr_nominal, inds=1:3, xlabel="time (s)", ylabel="states",
            label=["x (nominal eDMD)" "y (nominal eDMD)" "z (nominal eDMD)"], legend=:right, lw=2,
            color=[1 2 3])

plotstates(Tsim_lqr_nominal, Xsim_lqr_og, inds=4:6, xlabel="time (s)", ylabel="states",
            label=["MRP-x (og dynamics)" "MRP-y (og dynamics)" "MRP-z (og dynamics)"], legend=:topright, lw=2,
            linestyle=:dash, color=[1 2 3])
plotstates!(Tsim_lqr_nominal, Xsim_lqr_nominal, inds=4:6, xlabel="time (s)", ylabel="states",
            label=["MRP-x (nominal eDMD)" "MRP-y (nominal eDMD)" "MRP-z (nominal eDMD)"], legend=:topright, lw=2,
            color=[1 2 3])

## Evaluate new Jacobian-penalty eDMD model

# Generate Jacobians
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
    sparse(ForwardDiff.jacobian(x->expandstate(model_bilinear,x), x))
end

# Create a sparse version of the G Jacobian
G = spdiagm(n0,n,1=>ones(n0)) 
#@test norm(G - model_bilinear.g) < 1e-8

# Build Least Squares Problem
W,s = BilinearControl.EDMD.build_edmd_data(
    Z_train, U_train, A_train, B_train, F_train, model_bilinear.g
)

# solve LLS
@time Wsparse = sparse(W)
@time F = qr(Wsparse)
@time x = F \ s
norm(W*x - s)

BilinearControl.matdensity(Wsparse)

# Extract out bilinear dynamics
n = length(Z_train[1])
E = reshape(x,n,:)
A = E[:,1:n]
B = E[:,n .+ (1:m)]
C = E[:,n+m .+ (1:n*m)]

C_list = Matrix{Float64}[]
    
for i in 1:m
    C_i = C[:, (i-1)*n+1:i*n]
    push!(C_list, C_i)
end

return A, B, C_list, Matrix(g)

C = C_list

# Save data
jldsave(joinpath(Problems.DATADIR,"rex_quadrotor_lqr_jacobian_penalized_eDMD_data.jld2"); A, B, C, g, kf, eigfuns, eigorders, tf, dt)

# Import data
cartpole_data = load(joinpath(Problems.DATADIR, "rex_quadrotor_lqr_jacobian_penalized_eDMD_data.jld2"))
A_jacpen = cartpole_data["A"]
B_jacpen = cartpole_data["B"]
C_jacpen = cartpole_data["C"]
g_jacpen = cartpole_data["g"]
kf = cartpole_data["kf"]
tf = cartpole_data["tf"]
dt = cartpole_data["dt"]

T_ref = range(0,tf,step=dt)

## Evaluate the Fit
err_train2 = BilinearControl.EDMD.fiterror(A_jacpen, B_jacpen, C_jacpen, g_jacpen, kf, X_train, U_train)
err_test2 = BilinearControl.EDMD.fiterror(A_jacpen, B_jacpen, C_jacpen, g_jacpen, kf, X_test, U_test)
println("Train Error: ", err_train)
println("Test Error:  ", err_test)
println("")
println("New Train Error: ", err_train2)
println("New Test Error:  ", err_test2)

model_bilinear_jacpen = EDMDModel(A_jacpen, B_jacpen, C_jacpen, g_jacpen, kf, dt, "cartpole")
n,m = RD.dims(model_bilinear_jacpen)
n0 = originalstatedim(model_bilinear_jacpen)

# Get A,B for new system
function dynamics_bilinear_jacpen(x,u,t,dt)
    y = expandstate(model_bilinear_jacpen, x)
    yn = zero(y)
    RD.discrete_dynamics!(model_bilinear_jacpen, yn, y, u, t, dt)
    originalstate(model_bilinear_jacpen, yn)
end

A_bil_jacpen = FiniteDiff.finite_difference_jacobian(x->dynamics_bilinear_jacpen(x,ue,0.0,dt), xe)
B_bil_jacpen = FiniteDiff.finite_difference_jacobian(u->dynamics_bilinear_jacpen(xe,u,0.0,dt), ue)

K_bil_jacpen = dlqr(A_bil_jacpen, B_bil_jacpen, Qlqr, Rlqr)

# Evaluate stability
isstable_bilinear2 = maximum(abs.(eigvals(A_bil_jacpen - B_bil_jacpen*K_bil_jacpen))) < 1.0
isstable_nominal_with_bilinear2 = maximum(abs.(eigvals(A_og - B_og*K_bil_jacpen))) < 1.0

println("Stability Summary:")
println("  Dynamics  |  Controller  |  is stable? ")
println("------------|--------------|--------------")
println("  Nominal   |  Nominal     |  ", isstable_nominal)
println("  Bilinear  |  Bilinear    |  ", isstable_bilinear2)
println("  Nominal   |  Bilinear    |  ", isstable_nominal_with_bilinear2)

tf_sim = 5.0
Tsim_lqr_jacpen = range(0,tf_sim,step=dt)

x0 = [-0.5, -0.5, 0.5, -deg2rad(20), deg2rad(50), -deg2rad(20), -1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

ctrl_lqr_jacpen = LQRController(K_bil_jacpen, xe, ue)
Xsim_lqr_jacpen, = simulatewithcontroller(dmodel, ctrl_lqr_jacpen, x0, tf_sim, dt)

plotstates(Tsim_lqr_nominal, Xsim_lqr_og, inds=1:3, xlabel="time (s)", ylabel="states",
            label=["x (og dynamics)" "y (og dynamics)" "z (og dynamics)"], legend=:right, lw=2,
            linestyle=:dash, color=[1 2 3])
plotstates!(Tsim_lqr_nominal, Xsim_lqr_nominal, inds=1:3, xlabel="time (s)", ylabel="states",
            label=["x (nominal eDMD)" "y (nominal eDMD)" "z (nominal eDMD)"], legend=:right, lw=2,
            linestyle=:dot, color=[1 2 3])
plotstates!(Tsim_lqr_jacpen, Xsim_lqr_jacpen, inds=1:3, xlabel="time (s)", ylabel="states",
            label=["x (jacobian eDMD)" "y (jacobian eDMD)" "z (jacobian eDMD)"], legend=:right, lw=2,
            color=[1 2 3])
ylims!((-1,1))

plotstates(Tsim_lqr_nominal, Xsim_lqr_og, inds=4:6, xlabel="time (s)", ylabel="states",
            label=["MRP-x (og dynamics)" "MRP-y (og dynamics)" "MRP-z (og dynamics)"], legend=:right, lw=2,
            linestyle=:dash, color=[1 2 3])
plotstates!(Tsim_lqr_nominal, Xsim_lqr_nominal, inds=4:6, xlabel="time (s)", ylabel="states",
            label=["MRP-x (nominal eDMD)" "MRP-y (nominal eDMD)" "MRP-z (nominal eDMD)"], legend=:right, lw=2,
            linestyle=:dot, color=[1 2 3])
plotstates!(Tsim_lqr_jacpen, Xsim_lqr_jacpen, inds=4:6, xlabel="time (s)", ylabel="states",
            label=["MRP-x (jacobian eDMD)" "MRP-y (jacobian eDMD)" "MRP-z (jacobian eDMD)"], legend=:right, lw=2,
            color=[1 2 3])
ylims!((-1,1))