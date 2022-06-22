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
import BilinearControl.Problems: orientation, translation
using Test
using Rotations
using RobotDynamics
using GeometryBasics, CoordinateTransformations
using Colors
using MeshCat

include("constants.jl")
const QUADROTOR_NUMTRAJ_RESULTS_FILE = joinpath(Problems.DATADIR, "rex_full_quadrotor_num_traj_sweep_results.jld2")
const QUADROTOR_RESULTS_FILE = joinpath(Problems.DATADIR, "rex_full_quadrotor_mpc_results.jld2")
const QUADROTOR_MODELS_FILE = joinpath(Problems.DATADIR, "rex_full_quadrotor_mpc_models.jld2")

#############################################
## Functions
#############################################

#############################################
## Generate quadrotor data if need to
#############################################

generate_quadrotor_data()

#############################################
## Train models for MPC Tracking
#############################################

num_lqr = 10
num_mpc = 20

res = train_quadrotor_models(num_lqr, num_mpc, α=0.5, β=1.0, learnB=true, reg=1e-6)

eDMD_data = res.eDMD_data
jDMD_data = res.jDMD_data
kf = jDMD_data[:kf]
G = jDMD_data[:g]
dt = res.dt

model_info = (; eDMD_data, jDMD_data, G, kf, dt)
jldsave(QUADROTOR_MODELS_FILE; model_info)

#############################################
## Make eDMD models for MPC Tracking
#############################################

# Define Nominal Simulated Quadrotor Model
model_nom = Problems.NominalRexQuadrotor()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

# Define Mismatched "Real" Quadrotor Model
model_real = Problems.SimulatedRexQuadrotor()  # this model has aero drag
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

#############################################
## Load data and models
#############################################

# Load models
model_info = load(QUADROTOR_MODELS_FILE)["model_info"]

eDMD_data = model_info.eDMD_data
jDMD_data = model_info.jDMD_data
G = model_info.G
kf = model_info.kf
dt = model_info.dt

model_eDMD = EDMDModel(eDMD_data[:A],eDMD_data[:B],eDMD_data[:C],G,kf,dt,"quadrotor_eDMD")
model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
model_jDMD = EDMDModel(jDMD_data[:A],jDMD_data[:B],jDMD_data[:C],G,kf,dt,"quadrotor_jDMD")
model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)

# Load data
mpc_lqr_traj = load(joinpath(Problems.DATADIR, "rex_full_quadrotor_mpc_tracking_data.jld2"))

# Training data
X_train_lqr = mpc_lqr_traj["X_train_lqr"][:,1:num_lqr]
U_train_lqr = mpc_lqr_traj["U_train_lqr"][:,1:num_lqr]
X_train_mpc = mpc_lqr_traj["X_train_mpc"][:,1:num_mpc]
U_train_mpc = mpc_lqr_traj["U_train_mpc"][:,1:num_mpc]

# combine lqr and mpc training data
X_train = [X_train_lqr X_train_mpc]
U_train = [U_train_lqr U_train_mpc]

# Test data
X_nom_mpc = mpc_lqr_traj["X_nom_mpc"]
U_nom_mpc = mpc_lqr_traj["U_nom_mpc"]
X_test_infeasible = mpc_lqr_traj["X_test_infeasible"]
U_test_infeasible = mpc_lqr_traj["U_test_infeasible"]

# Metadata
tf = mpc_lqr_traj["tf"]
t_sim = 10.0
dt = mpc_lqr_traj["dt"]

T_ref = range(0,tf,step=dt)
T_sim = range(0,t_sim,step=dt)

#############################################
## MPC Tracking
#############################################

xe = zeros(12)
ue = Problems.trim_controls(model_real)
Nt = 20  # MPC horizon
N_sim = length(T_sim)
N_ref = length(T_ref)

Qmpc = Diagonal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
Rmpc = Diagonal(fill(1e-3, 4))
Qfmpc = Qmpc*100

N_test = size(X_nom_mpc,2)
test_results = map(1:N_test) do i
    X_ref = deepcopy(X_test_infeasible[:,i])
    U_ref = deepcopy(U_test_infeasible[:,i])
    X_ref[end] .= xe
    push!(U_ref, ue)

    X_ref_full = [X_ref; [copy(xe) for i = 1:N_sim - N_ref]]
    mpc_nom = TrackingMPC_no_OSQP(dmodel_nom, 
        X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
    )
    mpc_eDMD = TrackingMPC_no_OSQP(model_eDMD_projected, 
        X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
    )
    mpc_jDMD = TrackingMPC_no_OSQP(model_jDMD_projected, 
        X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
    )
    X_mpc_nom, U_mpc_nom, T_mpc = simulatewithcontroller(dmodel_real, mpc_nom,  X_ref[1], t_sim, dt)
    X_mpc_eDMD,U_mpc_eDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_eDMD, X_ref[1], t_sim, dt)
    X_mpc_jDMD,U_mpc_jDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_jDMD, X_ref[1], t_sim, dt)

    err_nom = norm(X_mpc_nom - X_ref_full) / N_sim
    err_eDMD = norm(X_mpc_eDMD - X_ref_full) / N_sim
    err_jDMD = norm(X_mpc_jDMD - X_ref_full) / N_sim

    (; err_nom, err_eDMD, err_jDMD, X_ref, X_mpc_nom, X_mpc_eDMD, X_mpc_jDMD, T_mpc)
end

nom_err_avg  = mean(filter(isfinite, map(x->x.err_nom, test_results)))
eDMD_err_avg = mean(filter(isfinite, map(x->x.err_eDMD, test_results)))
jDMD_err_avg = mean(filter(isfinite, map(x->x.err_jDMD, test_results)))
nom_success = count(x -> norm(x[end]-xe)<=10, map(x->x.X_mpc_nom, test_results))
eDMD_success = count(x -> norm(x[end]-xe)<=10, map(x->x.X_mpc_eDMD, test_results))
jDMD_success = count(x -> norm(x[end]-xe)<=10, map(x->x.X_mpc_jDMD, test_results))

nom_errs  = map(x->x.err_nom, test_results)
eDMD_errs = map(x->x.err_eDMD, test_results)
jDMD_errs = map(x->x.err_jDMD, test_results)

X_ref = map(x->x.X_ref, test_results)
X_mpc_nom = map(x->x.X_mpc_nom, test_results)
X_mpc_eDMD = map(x->x.X_mpc_eDMD, test_results)
X_mpc_jDMD = map(x->x.X_mpc_jDMD, test_results)
T_mpc = map(x->x.T_mpc, test_results)

G = model_jDMD.g
kf = model_jDMD.kf

MPC_test_results = (;X_ref, X_mpc_nom, X_mpc_eDMD, X_mpc_jDMD, T_mpc, nom_err_avg,
    nom_errs, eDMD_errs, jDMD_errs, nom_success, eDMD_err_avg,
    eDMD_success, jDMD_err_avg, jDMD_success)

@show MPC_test_results.nom_err_avg
@show MPC_test_results.eDMD_err_avg
@show MPC_test_results.jDMD_err_avg

#############################################
## Tracking performance vs equilibrium change
#############################################

distances = 0:0.1:2.0
equilibrium_results = map(distances) do dist

    println("equilibrium offset = $dist")

    if dist == 0
        xe_test = [zeros(12)]
    else
        xe_sampler = Product([
            Uniform(-dist, +dist),
            Uniform(-dist, +dist),
            Uniform(-dist, +dist),
        ])
        xe_test = [vcat(rand(xe_sampler), zeros(9)) for i = 1:10]
    end

    perc = 0.2
    x0_sampler = Product([
        Uniform(-5.0,5.0),
        Uniform(-5.0,5.0),
        Uniform(-5.0,5.0),
        Uniform(-deg2rad(20),deg2rad(20)),
        Uniform(-deg2rad(20),deg2rad(20)),
        Uniform(-deg2rad(20),deg2rad(20)),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.25*perc,0.25*perc),
        Uniform(-0.25*perc,0.25*perc),
        Uniform(-0.25*perc,0.25*perc)
    ])
    
    x0_test = [rand(x0_sampler) for i = 1:10]

    xe_results = map(xe_test) do xe

        nom_projected_x0s = mean(test_initial_conditions_offset(model_real, dmodel_nom, xe, x0_test, tf, t_sim, dt))
        error_eDMD_projected_x0s = mean(test_initial_conditions_offset(model_real, model_eDMD_projected, xe, x0_test, tf, t_sim, dt))
        error_jDMD_projected_x0s = mean(test_initial_conditions_offset(model_real, model_jDMD_projected, xe, x0_test, tf, t_sim, dt))

        if nom_projected_x0s > 1e3
            nom_projected_x0s = NaN
        end
        if error_eDMD_projected_x0s > 1e3
            error_eDMD_projected_x0s = NaN
        end
        if error_jDMD_projected_x0s > 1e3
            error_jDMD_projected_x0s = NaN
        end
        (; nom_projected_x0s, error_eDMD_projected_x0s, error_jDMD_projected_x0s)
    end

    err_nom_MPC = mean(filter(isfinite, map(x->x.nom_projected_x0s, xe_results)))
    error_eDMD_projected = mean(filter(isfinite, map(x->x.error_eDMD_projected_x0s, xe_results)))
    error_jDMD_projected = mean(filter(isfinite, map(x->x.error_jDMD_projected_x0s, xe_results)))

    nom_success = count(isfinite, map(x->x.nom_projected_x0s, xe_results))
    eDMD_success = count(isfinite, map(x->x.error_eDMD_projected_x0s, xe_results))
    jDMD_success = count(isfinite, map(x->x.error_jDMD_projected_x0s, xe_results))

    (;err_nom_MPC, error_eDMD_projected, error_jDMD_projected, nom_success, eDMD_success, jDMD_success)

end

#############################################
## Tracking performance vs test window
#############################################

percentages = 0.1:0.1:2
window_results = map(percentages) do perc

    println("percentage of training window = $perc")

    x0_sampler = Product([
        Uniform(-2.5*perc,2.5*perc),
        Uniform(-2.5*perc,2.5*perc),
        Uniform(-2.5*perc,2.5*perc),
        Uniform(-deg2rad(40*perc),deg2rad(40*perc)),
        Uniform(-deg2rad(40*perc),deg2rad(40*perc)),
        Uniform(-deg2rad(40*perc),deg2rad(40*perc)),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.25*perc,0.25*perc),
        Uniform(-0.25*perc,0.25*perc),
        Uniform(-0.25*perc,0.25*perc)
    ])

    x0_test = [rand(x0_sampler) for i = 1:50]

    error_nom, num_success_nom = test_initial_conditions(model_real, dmodel_nom, x0_test, tf, t_sim, dt)
    error_eDMD, num_success_eDMD = test_initial_conditions(model_real, model_eDMD_projected, x0_test, tf, t_sim, dt)
    error_jDMD, num_success_jDMD = test_initial_conditions(model_real, model_jDMD_projected, x0_test, tf, t_sim, dt)

    (; error_nom, error_eDMD, error_jDMD, num_success_nom, num_success_eDMD, num_success_jDMD)

end

#############################################
## Save results
#############################################

jldsave(QUADROTOR_RESULTS_FILE; MPC_test_results, window_results)

#############################################
## Load results and models
#############################################

MPC_test_results = load(QUADROTOR_RESULTS_FILE)["MPC_test_results"]
mpc_lqr_traj = load(joinpath(Problems.DATADIR, "rex_full_quadrotor_mpc_tracking_data.jld2"))

tf = mpc_lqr_traj["tf"]
t_sim = 10.0
dt = mpc_lqr_traj["dt"]

println("Time Summary:")
println("  Model  |  Training Time ")
println("---------|----------------")
println("  eDMD   |  ", eDMD_data[:t_train])
println("  jDMD   |  ", jDMD_data[:t_train])
println("")
println("Test Summary:")
println("  Model  |  Success Rate ")
println("---------|-------------------")
println(" nom MPC |  ", MPC_test_results[:nom_success])
println("  eDMD   |  ", MPC_test_results[:eDMD_success])
println("  jDMD   |  ", MPC_test_results[:jDMD_success])
println("")
println("Test Summary:")
println("  Model  |  Avg Tracking Err ")
println("---------|-------------------")
println(" nom MPC |  ", MPC_test_results[:nom_err_avg])
println("  eDMD   |  ", MPC_test_results[:eDMD_err_avg])
println("  jDMD   |  ", MPC_test_results[:jDMD_err_avg])