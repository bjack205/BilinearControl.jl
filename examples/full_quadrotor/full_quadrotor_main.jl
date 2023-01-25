using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using BilinearControl
import RobotDynamics as RD
using LinearAlgebra
using RobotZoo
using JLD2
using SparseArrays
using Distributions
using Distributions: Normal
using Random
using FiniteDiff, ForwardDiff
using StaticArrays
using Plots
using Test
import TrajectoryOptimization as TO
using Altro
import BilinearControl: orientation, translation
using Test
using Rotations
using RobotDynamics
using GeometryBasics, CoordinateTransformations
using Colors
using MeshCat
using Statistics

include("../plotting_constants.jl")

const QUADROTOR_DATA_FILE = joinpath(BilinearControl.DATADIR, "rex_full_quadrotor_mpc_trajectory_data.jld2")
const QUADROTOR_RESULTS_FILE = joinpath(BilinearControl.DATADIR, "rex_full_quadrotor_mpc_results.jld2")
const QUADROTOR_MODELS_FILE = joinpath(BilinearControl.DATADIR, "rex_full_quadrotor_mpc_models.jld2")

#############################################
## Functions
#############################################

function get_error_quantile(results; p=0.05)
    quantile_empty(x, i) = if isempty(x) 0 else quantile(x, i) end

    min_quant = quantile_empty(results, p)
    max_quant = quantile_empty(results, 1-p)
    return min_quant, max_quant
end

function full_quadrotor_kf(x)
    p = x[1:3]
    q = x[4:6]
    mrp = MRP(x[4], x[5], x[6])
    R = Matrix(mrp)
    v = x[7:9]
    w = x[10:12]
    vbody = R'v
    speed = vbody'vbody
    [1; x; BilinearControl.chebyshev(x, order=[2,2]); sin.(p); cos.(p); vbody; speed; p × v; p × w; w × w]
end

function test_initial_conditions(model, bilinear_model, ics, tf, t_sim, dt)

    dmodel = RD.DiscretizedDynamics{RD.RK4}(model)

    N_tf = round(Int, tf/dt) + 1    
    N_sim = round(Int, t_sim/dt) + 1 
    Nt = 20
    T_ref = range(0,tf,step=dt)

    Qmpc = Diagonal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    Rmpc = Diagonal(fill(1e-3, 4))
    Qfmpc = Qmpc*100

    results = map(ics) do x0

        X_ref = nominal_trajectory(x0,N_tf,dt)
        X_ref_full = [X_ref; [X_ref[end] for i = 1:N_sim - N_tf]]
        U_ref = [copy(BilinearControl.trim_controls(model)) for k = 1:N_tf]

        mpc = BilinearControl.TrackingMPC_no_OSQP(bilinear_model, 
            X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_sim,= simulatewithcontroller(dmodel, mpc, x0, t_sim, dt)

        track_err = norm(X_sim - X_ref_full) / N_sim
        stable_err = norm(X_sim[end] - X_ref_full[end]) 

        (; track_err, stable_err)
    end

    err_avg  = mean(filter(isfinite, map(x->x.track_err, results)))
    num_success = count(x -> x <=10, map(x->x.stable_err, results))

    return err_avg, num_success
end

function test_initial_conditions_offset(model, bilinear_model, xg, ics, tf, t_sim, dt)

    dmodel = RD.DiscretizedDynamics{RD.RK4}(model)

    N_tf = round(Int, tf/dt) + 1    
    N_sim = round(Int, t_sim/dt) + 1 
    Nt = 20
    T_ref = range(0,tf,step=dt)

    Qmpc = Diagonal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    Rmpc = Diagonal(fill(1e-3, 4))
    Qfmpc = Qmpc*100

    results = map(ics) do x0

        x0 += xg
        X_ref = nominal_trajectory(x0,N_tf,dt; xf=xg[1:3]) 
        X_ref_full = [X_ref; [X_ref[end] for i = 1:N_sim - N_tf]]
        U_ref = [copy(BilinearControl.trim_controls(model)) for k = 1:N_tf]

        mpc = BilinearControl.TrackingMPC_no_OSQP(bilinear_model, 
            X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_sim,= simulatewithcontroller(dmodel, mpc, x0, t_sim, dt)

        err = norm(X_sim - X_ref_full) / N_sim

        (; err)
    end

    err_avg  = mean(filter(isfinite, map(x->x.err, results)))

    return err_avg
end

function nominal_trajectory(x0,N,dt; xf=zeros(3))
    Xref = [fill(NaN, length(x0)) for k = 1:N]
    
    # TODO: Design a trajectory that linearly interpolates from x0 to the origin
    
    pos_0 = x0[1:3]
    mrp_0 = x0[4:6]
    
    pos_ref = reshape(LinRange(pos_0, xf, N), N, 1)
    mrp_ref = reshape(LinRange(mrp_0, zeros(3), N), N, 1)
    angle_ref = map((x) -> Vector(Rotations.params(RotXYZ(MRP(x[1], x[2], x[3])))), mrp_ref)

    vel_pos_ref = vcat([(pos_ref[2] - pos_ref[1]) ./ dt for k = 1:N-1], [zeros(3)])
    vel_attitude_ref = vcat([(angle_ref[2] - angle_ref[1]) ./ dt for k = 1:N-1], [zeros(3)])

    for i = 1:N
        
        Xref[i] = vcat(pos_ref[i], mrp_ref[i], vel_pos_ref[i], vel_attitude_ref[i])
    
    end
    
    return Xref
end

function generate_quadrotor_data()
    #############################################
    ## Define the Models
    #############################################

    # Define Nominal Simulated Quadrotor Model
    model_nom = BilinearControl.NominalRexQuadrotor()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" Quadrotor Model
    model_real = BilinearControl.SimulatedRexQuadrotor()  # this model has aero drag
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    # Time parameters
    tf = 5.0
    dt = 0.05
    Nt = 20  # MPC Horizon
    t_sim = tf*1.2 # length of simulation (to capture steady-state behavior) 

    #############################################
    ## LQR Training and Testing Data 
    #############################################

    # Stabilization trajectories 
    Random.seed!(1)
    num_train_lqr = 50
    num_test_lqr = 50

    # Generate a stabilizing LQR controller
    Qlqr = Diagonal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    Rlqr = Diagonal([1e-4, 1e-4, 1e-4, 1e-4])
    xe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ue = BilinearControl.trim_controls(model_real)
    ctrl_lqr_nom = LQRController(dmodel_nom, Qlqr, Rlqr, xe, ue, dt)

    # Sample a bunch of initial conditions for the LQR controller
    x0_train_sampler = Product([
        Uniform(-1.0,1.0),
        Uniform(-1.0,1.0),
        Uniform(-1.0,1.0),
        Uniform(-deg2rad(80),deg2rad(80)),
        Uniform(-deg2rad(80),deg2rad(80)),
        Uniform(-deg2rad(80),deg2rad(80)),
        Uniform(-0.5,0.5),
        Uniform(-0.5,0.5),
        Uniform(-0.5,0.5),
        Uniform(-0.25,0.25),
        Uniform(-0.25,0.25),
        Uniform(-0.25,0.25)
    ])

    x0_test_sampler = Product([
        Uniform(-2.0,2.0),
        Uniform(-2.0,2.0),
        Uniform(-2.0,2.0),
        Uniform(-deg2rad(80),deg2rad(80)),
        Uniform(-deg2rad(80),deg2rad(80)),
        Uniform(-deg2rad(80),deg2rad(80)),
        Uniform(-1,1),
        Uniform(-1,1),
        Uniform(-1,1),
        Uniform(-0.5,0.5),
        Uniform(-0.5,0.5),
        Uniform(-0.5,0.5)
    ])

    initial_conditions_train = [rand(x0_train_sampler) for _ in 1:num_train_lqr]
    initial_conditions_test = [rand(x0_test_sampler) for _ in 1:num_test_lqr]

    initial_conditions_train = map((x) -> vcat(x[1:3], Rotations.params(MRP(RotXYZ(x[4], x[5], x[6]))), 
        x[7:end]), initial_conditions_train)
    initial_conditions_test = map((x) -> vcat(x[1:3], Rotations.params(MRP(RotXYZ(x[4], x[5], x[6]))), 
        x[7:end]), initial_conditions_test)
        
    # Create data set
    X_train_lqr, U_train_lqr = create_data(dmodel_real, ctrl_lqr_nom, initial_conditions_train, t_sim, dt)
    X_test_lqr, U_test_lqr = create_data(dmodel_real, ctrl_lqr_nom, initial_conditions_test, t_sim, dt);

    # Make sure they all stabilize
    @test all(x->x<0.1, map(x->norm(x-xe), X_train_lqr[end,:]))
    @test all(x->x<0.1, map(x->norm(x-xe), X_test_lqr[end,:]))

    #############################################
    ## MPC Training and Testing Data 
    #############################################
    Random.seed!(1)
    num_train_mpc = 50
    num_test_mpc = 50

    # Sample a bunch of initial conditions for the LQR controller
    x0_train_sampler = Product([
        Uniform(-5.0,5.0),
        Uniform(-5.0,5.0),
        Uniform(-5.0,5.0),
        Uniform(-deg2rad(70),deg2rad(70)),
        Uniform(-deg2rad(70),deg2rad(70)),
        Uniform(-deg2rad(70),deg2rad(70)),
        Uniform(-0.5,0.5),
        Uniform(-0.5,0.5),
        Uniform(-0.5,0.5),
        Uniform(-0.25,0.25),
        Uniform(-0.25,0.25),
        Uniform(-0.25,0.25)
    ])

    x0_test_sampler = Product([
        Uniform(-5.0,5.0),
        Uniform(-5.0,5.0),
        Uniform(-5.0,5.0),
        Uniform(-deg2rad(80),deg2rad(80)),
        Uniform(-deg2rad(80),deg2rad(80)),
        Uniform(-deg2rad(80),deg2rad(80)),
        Uniform(-1.0,1.0),
        Uniform(-1.0,1.0),
        Uniform(-1.0,1.0),
        Uniform(-0.5,0.5),
        Uniform(-0.5,0.5),
        Uniform(-0.5,0.5)
    ])

    # Sample a bunch of initial conditions for the MPC controller
    initial_conditions_mpc_train = [rand(x0_train_sampler) for _ in 1:num_train_mpc]
    initial_conditions_mpc_test = [rand(x0_test_sampler) for _ in 1:num_test_mpc]

    initial_conditions_mpc_train = map((x) -> vcat(x[1:3], Rotations.params(MRP(RotXYZ(x[4], x[5], x[6]))), 
        x[7:end]), initial_conditions_mpc_train)
    initial_conditions_mpc_test = map((x) -> vcat(x[1:3], Rotations.params(MRP(RotXYZ(x[4], x[5], x[6]))), 
        x[7:end]), initial_conditions_mpc_test)

    Random.seed!(1)

    Qmpc = Diagonal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    Rmpc = Diagonal(fill(1e-4, 4))
    Qfmpc = Qmpc*10

    N_tf = round(Int, tf/dt) + 1
    N_tsim = round(Int, t_sim/dt) + 1

    X_train_mpc = Matrix{Vector{Float64}}(undef, N_tsim, num_train_mpc)
    U_train_mpc = Matrix{Vector{Float64}}(undef, N_tsim-1, num_train_mpc)

    for i = 1:num_train_mpc

        x0 = initial_conditions_mpc_train[i]
        X = nominal_trajectory(x0,N_tf,dt)
        U = [copy(BilinearControl.trim_controls(model_real)) for k = 1:N_tf]
        T = range(0,tf,step=dt)

        mpc = TrackingMPC_no_OSQP(dmodel_nom, X, U, Vector(T), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_sim,U_sim,T_sim = simulatewithcontroller(dmodel_real, mpc, X[1], t_sim, T[2])
        
        if maximum(X_sim[end]-X[end]) < maximum(100*x0)

            X_train_mpc[:,i] = X_sim
            U_train_mpc[:,i] = U_sim
        
        else

            X_train_mpc[:,i] = X_train_mpc[:, i-1]
            U_train_mpc[:,i] = U_train_mpc[:, i-1]

        end
    end

    ## Generate test data

    X_test_infeasible = Matrix{Vector{Float64}}(undef, N_tf, num_test_mpc)
    X_nom_mpc = Matrix{Vector{Float64}}(undef, N_tsim, num_test_mpc)

    U_test_infeasible = Matrix{Vector{Float64}}(undef, N_tf-1, num_test_mpc)
    U_nom_mpc = Matrix{Vector{Float64}}(undef, N_tsim-1, num_test_mpc)

    for i = 1:num_test_mpc

        x0 = initial_conditions_mpc_test[i]
        X = nominal_trajectory(x0,N_tf,dt)
        U = [copy(BilinearControl.trim_controls(model_real)) for k = 1:N_tf]
        T = range(0,tf,step=dt)
        
        mpc_nom = TrackingMPC_no_OSQP(dmodel_nom, X, U, Vector(T), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_nom,U_nom,T_nom = simulatewithcontroller(dmodel_real, mpc_nom, X[1], t_sim, T[2])

        X_test_infeasible[:,i] = X
        U_test_infeasible[:,i] = U[1:end-1]

        X_nom_mpc[:,i] = X_nom
        U_nom_mpc[:,i] = U_nom
    end

    ## Save generated training and test data
    jldsave(QUADROTOR_DATA_FILE; 
        X_train_lqr, U_train_lqr,
        X_train_mpc, U_train_mpc,
        X_nom_mpc, U_nom_mpc, 
        X_test_infeasible, U_test_infeasible,
        X_test_lqr, U_test_lqr, 
        tf, t_sim, dt
    )
end

function train_quadrotor_models(num_lqr::Int64, num_mpc::Int64;  α=0.5, learnB=true, β=1.0, reg=1e-6)

    #############################################
    ## Define the Models
    #############################################

    # Define Nominal Simulated Quadrotor Model
    model_nom = BilinearControl.NominalRexQuadrotor()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" Quadrotor Model
    model_real = BilinearControl.SimulatedRexQuadrotor()  # this model has aero drag
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    #############################################  
    ## Load Training and Test Data
    #############################################  
    mpc_lqr_traj = load(joinpath(BilinearControl.DATADIR, "rex_full_quadrotor_mpc_tracking_data.jld2"))

    # Training data
    X_train_lqr = mpc_lqr_traj["X_train_lqr"][:,1:num_lqr]
    U_train_lqr = mpc_lqr_traj["U_train_lqr"][:,1:num_lqr]
    X_train_mpc = mpc_lqr_traj["X_train_mpc"][:,1:num_mpc]
    U_train_mpc = mpc_lqr_traj["U_train_mpc"][:,1:num_mpc]

    ## combine lqr and mpc training data
    X_train = [X_train_lqr X_train_mpc]
    U_train = [U_train_lqr U_train_mpc]

    # Metadata
    dt = mpc_lqr_traj["dt"]

    #############################################
    ## Fit the training data
    #############################################

    # Define basis functions
    # eigfuns = ["state", "monomial"]
    # eigorders = [[0],[2, 2]];

    # println("Training eDMD model...")
    # t_train_eDMD = @elapsed model_eDMD = run_eDMD(X_train, U_train, dt, eigfuns, eigorders,
    #     reg=1e-1, name="planar_quadrotor_eDMD")

    # println("Training jDMD model...")
    # t_train_jDMD = @elapsed model_jDMD = run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom,
    #     reg=reg, name="planar_quadrotor_jDMD"; α, β, learnB, verbose=true)

    println("Training eDMD model...")
    t_train_eDMD = @elapsed model_eDMD = run_eDMD(X_train, U_train, dt, full_quadrotor_kf, nothing; 
        alg=:qr, showprog=true, reg=reg
    )

    println("Training jDMD model...")
    t_train_jDMD = @elapsed model_jDMD = run_jDMD(X_train, U_train, dt, full_quadrotor_kf, nothing,
        dmodel_nom; showprog=true, verbose=true, reg=reg, alg=:qr_rls, α=0.5
    )

    eDMD_data = Dict(
        :A=>model_eDMD.A, :B=>model_eDMD.B, :C=>model_eDMD.C, :g=>model_eDMD.g, :t_train=>t_train_eDMD, :kf=>full_quadrotor_kf
    )
    jDMD_data = Dict(
        :A=>model_jDMD.A, :B=>model_jDMD.B, :C=>model_jDMD.C, :g=>model_jDMD.g, :t_train=>t_train_jDMD, :kf=>full_quadrotor_kf
    )

    res = (; eDMD_data, jDMD_data, dt)
    return res
end

#############################################
## Generate quadrotor data if need to
#############################################

generate_quadrotor_data()

#############################################
## Train models for MPC Tracking
#############################################

num_lqr = 10
num_mpc = 20

##
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
model_nom = BilinearControl.NominalRexQuadrotor()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

# Define Mismatched "Real" Quadrotor Model
model_real = BilinearControl.SimulatedRexQuadrotor()  # this model has aero drag
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

model_eDMD = BilinearControl.EDMDModel(eDMD_data[:A],eDMD_data[:B],eDMD_data[:C],G,kf,dt,"quadrotor_eDMD")
model_eDMD_projected = BilinearControl.ProjectedEDMDModel(model_eDMD)
model_jDMD = BilinearControl.EDMDModel(jDMD_data[:A],jDMD_data[:B],jDMD_data[:C],G,kf,dt,"quadrotor_jDMD")
model_jDMD_projected = BilinearControl.ProjectedEDMDModel(model_jDMD)

# Load data
mpc_lqr_traj = load(QUADROTOR_DATA_FILE)

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
ue = BilinearControl.trim_controls(model_real)
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
    mpc_nom = BilinearControl.TrackingMPC_no_OSQP(dmodel_nom, 
        X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
    )
    mpc_eDMD = BilinearControl.TrackingMPC_no_OSQP(model_eDMD_projected, 
        X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
    )
    mpc_jDMD = BilinearControl.TrackingMPC_no_OSQP(model_jDMD_projected, 
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

nom_err_filtered = filter(isfinite, map(x->x.err_nom, test_results))
nom_err_avg  = mean(nom_err_filtered)
nom_err_std = stdm(nom_err_filtered, nom_err_avg)
nom_err_ci = 1.959964 .* (nom_err_std ./ sqrt(length(nom_err_filtered)))
nom_err_quanti_min, nom_err_quanti_max = get_error_quantile(nom_err_filtered; p=0.05)

eDMD_err_filtered = filter(isfinite, map(x->x.err_eDMD, test_results))
eDMD_err_avg  = mean(eDMD_err_filtered)
eDMD_err_std = stdm(eDMD_err_filtered, eDMD_err_avg)
eDMD_err_ci = 1.959964 .* (eDMD_err_std ./ sqrt(length(eDMD_err_filtered)))
eDMD_err_quanti_min, eDMD_err_quanti_max = get_error_quantile(eDMD_err_filtered; p=0.05)

jDMD_err_filtered = filter(isfinite, map(x->x.err_jDMD, test_results))
jDMD_err_avg  = mean(jDMD_err_filtered)
jDMD_err_std = stdm(jDMD_err_filtered, jDMD_err_avg)
jDMD_err_ci = 1.959964 .* (jDMD_err_std ./ sqrt(length(jDMD_err_filtered)))
jDMD_err_quanti_min, jDMD_err_quanti_max = get_error_quantile(jDMD_err_filtered; p=0.05)

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

MPC_test_results = (;X_ref, X_mpc_nom, X_mpc_eDMD, X_mpc_jDMD, T_mpc, 
    nom_errs, nom_success, nom_err_filtered, nom_err_avg, nom_err_std, nom_err_ci, nom_err_quanti_min, nom_err_quanti_max,
    eDMD_errs, eDMD_success, eDMD_err_filtered, eDMD_err_avg, eDMD_err_std, eDMD_err_ci, eDMD_err_quanti_min, eDMD_err_quanti_max,
    jDMD_errs, jDMD_success, jDMD_err_filtered, jDMD_err_avg, jDMD_err_std, jDMD_err_ci, jDMD_err_quanti_min, jDMD_err_quanti_max,
)

@show MPC_test_results.nom_err_avg
@show MPC_test_results.eDMD_err_avg
@show MPC_test_results.jDMD_err_avg

#############################################
## Save results
#############################################

jldsave(QUADROTOR_RESULTS_FILE; MPC_test_results)

#############################################
## Load results and models
#############################################

MPC_test_results = load(QUADROTOR_RESULTS_FILE)["MPC_test_results"]
mpc_lqr_traj = load(QUADROTOR_DATA_FILE)

tf = mpc_lqr_traj["tf"]
t_sim = 10.0
dt = mpc_lqr_traj["dt"]

println("Time Summary:")
println("  Model  |  Training Time ")
println("---------|----------------")
println("  eDMD   |  ", eDMD_data[:t_train])
println("  jDMD   |  ", jDMD_data[:t_train])
println("")
println("  Model  |  Success Rate ")
println("---------|-------------------")
println(" nom MPC |  ", MPC_test_results[:nom_success]/50)
println("  eDMD   |  ", MPC_test_results[:eDMD_success]/50)
println("  jDMD   |  ", MPC_test_results[:jDMD_success]/50)
println("")
println("  Model  |  Avg Tracking Err ")
println("---------|-------------------")
println(" nom MPC |  ", MPC_test_results[:nom_err_avg])
println("  eDMD   |  ", MPC_test_results[:eDMD_err_avg])
println("  jDMD   |  ", MPC_test_results[:jDMD_err_avg])
println("")
println("  Model  |  Tracking Err Std")
println("---------|-------------------")
println(" nom MPC |  ", MPC_test_results[:nom_err_std])
println("  eDMD   |  ", MPC_test_results[:eDMD_err_std])
println("  jDMD   |  ", MPC_test_results[:jDMD_err_std])
println("")
println("  Model  |  Tracking Err 95% CI")
println("---------|-------------------")
println(" nom MPC |  ", MPC_test_results[:nom_err_ci])
println("  eDMD   |  ", MPC_test_results[:eDMD_err_ci])
println("  jDMD   |  ", MPC_test_results[:jDMD_err_ci])
println("")
println("  Model  |  Tracking Err 5% Quantile")
println("---------|-------------------")
println(" nom MPC |  ", MPC_test_results[:nom_err_quanti_min])
println("  eDMD   |  ", MPC_test_results[:eDMD_err_quanti_min])
println("  jDMD   |  ", MPC_test_results[:jDMD_err_quanti_min])
println("")
println("  Model  |  Tracking Err 95% Quantile")
println("---------|-------------------")
println(" nom MPC |  ", MPC_test_results[:nom_err_quanti_max])
println("  eDMD   |  ", MPC_test_results[:eDMD_err_quanti_max])
println("  jDMD   |  ", MPC_test_results[:jDMD_err_quanti_max])