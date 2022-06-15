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
using Test
using Rotations
using RobotDynamics

include("constants.jl")
const QUADROTOR_RESULTS_FILE = joinpath(Problems.DATADIR, "quadrotor_mpc_results.jld2")
const QUADROTOR_MODELS_FILE = joinpath(Problems.DATADIR, "quadrotor_mpc_models.jld2")

#############################################
## Functions for generating data and training models
#############################################

function genQuadrotorProblem(;costfun=:Quadratic, normcon=false)

    model = Problems.NominalQuadrotor()
    n,m = RD.dims(model)

    opts = SolverOptions(
        penalty_scaling=100.,
        penalty_initial=0.1,
    )

    # discretization
    N = 101 # number of knot points
    tf = 5.0
    dt = tf/(N-1) # total time

    # Initial condition
    x0_pos = @SVector [0., -10., 1.]
    x0 = RobotDynamics.build_state(model, x0_pos, MRP(0.0, 0.0, 0.0), zeros(3), zeros(3))

    # cost
    costfun == :QuatLQR ? sq = 0 : sq = 1
    rm_quat = @SVector [1,2,3,4,5,6,8,9,10,11,12]
    Q_diag = RobotDynamics.fill_state(model, 1e-5, 1e-5*sq, 1e-3, 1e-3)
    Q = Diagonal(Q_diag)
    R = Diagonal(@SVector fill(1e-4,m))
    q_nom = MRP(0.0, 0.0, 0.0)
    v_nom, ω_nom = zeros(3), zeros(3)
    x_nom = RobotDynamics.build_state(model, zeros(3), q_nom, v_nom, ω_nom)

    if costfun == :QuatLQR
        cost_nom = TO.QuatLQRCost(Q*dt, R*dt, x_nom, w=0.0)
    elseif costfun == :ErrorQuad
        cost_nom = TO.ErrorQuadratic(model, Diagonal(Q_diag[rm_quat])*dt, R*dt, x_nom)
    else
        cost_nom = TO.LQRCost(Q*dt, R*dt, x_nom)
    end

    # waypoints
    wpts = [(@SVector [10.,0.,1.]),
            (@SVector [-10.,0.,1.]),
            (@SVector [0.,10.,1.])]
            
    times = [33., 66., 101.]
    Qw_diag = RobotDynamics.fill_state(model, 1e3,1*sq,1,1)
    Qf_diag = RobotDynamics.fill_state(model, 10., 100*sq, 10, 10)
    xf = RobotDynamics.build_state(model, wpts[end], MRP(0.0, 0.0, 0.0), zeros(3), zeros(3))

    costs = map(1:length(wpts)) do i
        r = wpts[i]
        xg = RobotDynamics.build_state(model, r, q_nom, v_nom, ω_nom)
        if times[i] == N
            Q = Diagonal(Qf_diag)
            w = 40.0
        else
            Q = Diagonal(1e-3*Qw_diag) * dt
            w = 0.1
        end
        if costfun == :QuatLQR
            TO.QuatLQRCost(Q, R*dt, xg, w=w)
        elseif costfun == :ErrorQuad
            Qd = diag(Q)
            TO.ErrorQuadratic(model, Diagonal(Qd[rm_quat]), R, xg)
        else
            TO.LQRCost(Q, R, xg)
        end
    end

    costs_all = map(1:N) do k
        i = findfirst(x->(x ≥ k), times)
        if k ∈ times
            costs[i]
        else
            cost_nom
        end
    end

    obj = TO.Objective(costs_all)

    # Initialization
    u0 = @SVector fill(0.5*9.81/4, m)
    U_hover = [copy(u0) for k = 1:N-1] # initial hovering control trajectory

    # Constraints
    conSet = TO.ConstraintList(n,m,N)
    if normcon
        if use_rot == :slack
            add_constraint!(conSet, QuatSlackConstraint(), 1:N-1)
        else
            add_constraint!(conSet, QuatNormConstraint(), 1:N-1)
            u0 = [u0; (@SVector [1.])]
        end
    end
    bnd = TO.BoundConstraint(n,m, u_min=0.0, u_max=12.0)
    TO.add_constraint!(conSet, bnd, 1:N-1)

    # Problem
    prob = TO.Problem(model, obj, x0, tf, xf=xf, constraints=conSet)
    TO.initial_controls!(prob, U_hover)
    TO.rollout!(prob)

    return prob, opts
end

function generate_zigzag_traj()

    # solver = ALTROSolver(genQuadrotorProblem()..., projected_newton=false)
    solver = ALTROSolver(Altro.Problems.Quadrotor(:zigzag, MRP{Float64})..., projected_newton=false)
    solver = Altro.solve!(solver)
    X = Vector.(TO.states(solver))
    U = Vector.(TO.controls(solver))

    return X, U
end

function test_initial_conditions(model, bilinear_model, ics, tf, t_sim, dt)

    dmodel = RD.DiscretizedDynamics{RD.RK4}(model)

    N_tf = round(Int, tf/dt) + 1    
    N_sim = round(Int, t_sim/dt) + 1 
    Nt = 41
    T_ref = range(0,tf,step=dt)

    Qmpc = Diagonal(fill(1.0, 12))
    Rmpc = Diagonal(fill(1e-4, 4))
    Qfmpc = 100*Qmpc

    results = map(ics) do x0

        X_ref = nominal_trajectory(x0,N_tf,dt)
        X_ref_full = [X_ref; [X_ref[end] for i = 1:N_sim - N_tf]]
        U_ref = [copy(Problems.trim_controls(model)) for k = 1:N_tf]

        mpc = TrackingMPC(bilinear_model, 
            X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_sim,= simulatewithcontroller(dmodel, mpc, x0, t_sim, dt)

        err = norm(X_sim - X_ref_full) / N_sim

        (; err)
    end

    err_avg  = mean(filter(isfinite, map(x->x.err, results)))

    return err_avg
end

function nominal_trajectory(x0,N,dt)
    Xref = [fill(NaN, length(x0)) for k = 1:N]
    
    # TODO: Design a trajectory that linearly interpolates from x0 to the origin
    
    pos_0 = x0[1:3]
    mrp_0 = x0[4:6]
    
    pos_ref = reshape(LinRange(pos_0, zeros(3), N), N, 1)
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
    model_nom = Problems.NominalQuadrotor()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" Quadrotor Model
    model_real = Problems.SimulatedQuadrotor()  # this model has aero drag
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
    num_test_lqr = 20

    # Generate a stabilizing LQR controller
    Qlqr = Diagonal(fill(1.0, 12))
    Rlqr = Diagonal([1e-4, 1e-4, 1e-4, 1e-4])
    xe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ue = Problems.trim_controls(model_real)
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
        Uniform(-0.25,0.25),
        Uniform(-0.25,0.25),
        Uniform(-0.25,0.25)
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
        Uniform(-deg2rad(60),deg2rad(60)),
        Uniform(-deg2rad(60),deg2rad(60)),
        Uniform(-deg2rad(60),deg2rad(60)),
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
        Uniform(-deg2rad(70),deg2rad(70)),
        Uniform(-deg2rad(70),deg2rad(70)),
        Uniform(-deg2rad(70),deg2rad(70)),
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
        U = [copy(Problems.trim_controls(model_real)) for k = 1:N_tf]
        T = range(0,tf,step=dt)

        mpc = TrackingMPC(dmodel_nom, X, U, Vector(T), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_sim,U_sim,T_sim = simulatewithcontroller(dmodel_real, mpc, X[1], t_sim, T[2])
        
        if maximum(X_sim[end]-X[end]) < maximum(x0)

            X_train_mpc[:,i] = X_sim
            U_train_mpc[:,i] = U_sim
        
        else

            X_train_mpc[:,i] = X_train_mpc[:, i-1]
            U_train_mpc[:,i] = U_train_mpc[:, i-1]

        end
    end

    ##
    # i = 10
    # T_ref = range(0,tf,step=dt)
    # T_sim = range(0,t_sim,step=dt)
    # X_ref = nominal_trajectory(initial_conditions_mpc_train[i],N_tf,dt)

    # plotstates(T_ref, X_ref, inds=1:3, xlabel="time (s)", ylabel="states",
    #             label=["x (ref)" "y (ref)" "z (ref)"], legend=:right, lw=2,
    #             linestyle=:dot, color=[1 2 3])
    # plotstates!(T_sim, X_train_mpc[:, i], inds=1:3, xlabel="time (s)", ylabel="states",
    #             label=["x (nom MPC)" "y (nom MPC)" "z (nom MPC)"], legend=:right, lw=2,
    #             linestyle=:solid, color=[1 2 3])

    # plotstates(T_ref, X_ref, inds=4:6, xlabel="time (s)", ylabel="states",
    #             label=["MRP-x (ref)" "MRP-y (ref)" "MRP-z (ref)"], legend=:right, lw=2,
    #             linestyle=:dot, color=[1 2 3])
    # plotstates!(T_sim, X_train_mpc[:, i], inds=4:6, xlabel="time (s)", ylabel="states",
    #             label=["MRP-x (nom MPC)" "MRP-y (nom MPC)" "MRP-z (nom MPC)"], legend=:right, lw=2,
    #             linestyle=:solid, color=[1 2 3])

    ## Generate test data

    X_test_infeasible = Matrix{Vector{Float64}}(undef, N_tf, num_test_mpc)
    X_nom_mpc = Matrix{Vector{Float64}}(undef, N_tsim, num_test_mpc)

    U_test_infeasible = Matrix{Vector{Float64}}(undef, N_tf-1, num_test_mpc)
    U_nom_mpc = Matrix{Vector{Float64}}(undef, N_tsim-1, num_test_mpc)

    for i = 1:num_test_mpc

        x0 = initial_conditions_mpc_test[i]
        X = nominal_trajectory(x0,N_tf,dt)
        U = [copy(Problems.trim_controls(model_real)) for k = 1:N_tf]
        T = range(0,tf,step=dt)
        
        mpc_nom = TrackingMPC(dmodel_nom, X, U, Vector(T), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_nom,U_nom,T_nom = simulatewithcontroller(dmodel_real, mpc_nom, X[1], t_sim, T[2])

        X_test_infeasible[:,i] = X
        U_test_infeasible[:,i] = U[1:end-1]

        if maximum(X_nom[end]-X[end]) < maximum(x0)

            X_nom_mpc[:,i] = X_nom
            U_nom_mpc[:,i] = U_nom
        
        else

            X_nom_mpc[:,i] = X_nom_mpc[:, i-1]
            U_nom_mpc[:,i] = U_nom_mpc[:, i-1]

        end

    end

    ##
    # i = 20
    # T_ref = range(0,tf,step=dt)
    # T_sim = range(0,t_sim,step=dt)
    # X_ref = X_test_infeasible[:, i]

    # plotstates(T_ref, X_ref, inds=1:3, xlabel="time (s)", ylabel="states",
    #             label=["x (ref)" "y (ref)" "z (ref)"], legend=:topright, lw=2,
    #             linestyle=:dot, color=[1 2 3])
    # plotstates!(T_sim, X_nom_mpc[:, i], inds=1:3, xlabel="time (s)", ylabel="states",
    #             label=["x (nom MPC)" "y (nom MPC)" "z (nom MPC)"], legend=:topright, lw=2,
    #             linestyle=:solid, color=[1 2 3])

    # plotstates(T_ref, X_ref, inds=4:6, xlabel="time (s)", ylabel="states",
    #             label=["MRP-x (ref)" "MRP-y (ref)" "MRP-z (ref)"], legend=:topright, lw=2,
    #             linestyle=:dot, color=[1 2 3])
    # plotstates!(T_sim, X_nom_mpc[:, i], inds=4:6, xlabel="time (s)", ylabel="states",
    #             label=["MRP-x (nom MPC)" "MRP-y (nom MPC)" "MRP-z (nom MPC)"], legend=:topright, lw=2,
    #             linestyle=:solid, color=[1 2 3])


    ## Save generated training and test data
    jldsave(joinpath(Problems.DATADIR, "quadrotor_mpc_tracking_data.jld2"); 
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
    model_nom = Problems.NominalQuadrotor()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" Quadrotor Model
    model_real = Problems.SimulatedQuadrotor()  # this model has aero drag
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    #############################################  
    ## Load Training and Test Data
    #############################################  
    mpc_lqr_traj = load(joinpath(Problems.DATADIR, "quadrotor_mpc_tracking_data.jld2"))

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
    eigfuns = ["state", "monomial"]
    eigorders = [[0],[2, 2]];

    println("Training eDMD model...")
    t_train_eDMD = @elapsed model_eDMD = run_eDMD(X_train, U_train, dt, eigfuns, eigorders,
        reg=1e-1, name="planar_quadrotor_eDMD")

    println("Training jDMD model...")
    t_train_jDMD = @elapsed model_jDMD = run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom,
        reg=reg, name="planar_quadrotor_jDMD"; α, β, learnB, verbose=true)

    eDMD_data = Dict(
        :A=>model_eDMD.A, :B=>model_eDMD.B, :C=>model_eDMD.C, :g=>model_eDMD.g, :t_train=>t_train_eDMD, :kf=>model_eDMD.kf
    )
    jDMD_data = Dict(
        :A=>model_jDMD.A, :B=>model_jDMD.B, :C=>model_jDMD.C, :g=>model_jDMD.g, :t_train=>t_train_jDMD, :kf=>model_jDMD.kf
    )

    res = (; eDMD_data, jDMD_data, dt)
    return res
end
            
#############################################
## Train models for MPC Tracking
#############################################

num_lqr = 10
num_mpc = 20

generate_quadrotor_data()
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
model_nom = Problems.NominalQuadrotor()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

# Define Mismatched "Real" Quadrotor Model
model_real = Problems.SimulatedQuadrotor()  # this model has aero drag
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
mpc_lqr_traj = load(joinpath(Problems.DATADIR, "quadrotor_mpc_tracking_data.jld2"))

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
t_sim = mpc_lqr_traj["t_sim"]
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

Qmpc = Diagonal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
Rmpc = Diagonal(fill(1e-4, 4))
Qfmpc = Qmpc*10

N_test = size(X_nom_mpc,2)
test_results = map(1:N_test) do i
    X_ref = deepcopy(X_test_infeasible[:,i])
    U_ref = deepcopy(U_test_infeasible[:,i])
    X_ref[end] .= xe
    push!(U_ref, ue)

    X_ref_full = [X_ref; [copy(xe) for i = 1:N_sim - N_ref]]
    mpc_nom = TrackingMPC(dmodel_nom, 
        X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
    )
    mpc_eDMD = TrackingMPC(model_eDMD_projected, 
        X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
    )
    mpc_jDMD = TrackingMPC(model_jDMD_projected, 
        X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
    )
    X_mpc_nom, U_mpc_nom, T_mpc = simulatewithcontroller(dmodel_real, mpc_nom,  X_ref[1], t_sim, dt)
    X_mpc_eDMD,U_mpc_eDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_eDMD, X_ref[1], t_sim, dt)
    X_mpc_jDMD,U_mpc_jDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_jDMD, X_ref[1], t_sim, dt)

    err_nom = norm(X_mpc_nom - X_ref_full) / N_sim
    err_eDMD = norm(X_mpc_eDMD - X_ref_full) / N_sim
    err_jDMD = norm(X_mpc_jDMD - X_ref_full) / N_sim

    (; err_nom, err_eDMD, err_jDMD)
end

nom_err_avg  = mean(filter(isfinite, map(x->x.err_nom, test_results)))
eDMD_err_avg = mean(filter(isfinite, map(x->x.err_eDMD, test_results)))
jDMD_err_avg = mean(filter(isfinite, map(x->x.err_jDMD, test_results)))
eDMD_success = count(isfinite, map(x->x.err_eDMD, test_results))
jDMD_success = count(isfinite, map(x->x.err_jDMD, test_results))

G = model_jDMD.g
kf = model_jDMD.kf

res = (;nom_err_avg, eDMD_err_avg, eDMD_success, jDMD_err_avg, jDMD_success)    

@show res.eDMD_err_avg
@show res.jDMD_err_avg
jldsave(QUADROTOR_RESULTS_FILE; res)

#############################################
## Load results and models
#############################################

res = load(QUADROTOR_RESULTS_FILE)["res"]
mpc_lqr_traj = load(joinpath(Problems.DATADIR, "quadrotor_mpc_tracking_data.jld2"))

tf = mpc_lqr_traj["tf"]
t_sim = mpc_lqr_traj["t_sim"]
dt = mpc_lqr_traj["dt"]

println("Time Summary:")
println("  Model  |  Training Time ")
println("---------|----------------")
println("  eDMD   |  ", eDMD_data[:t_train])
println("  jDMD   |  ", jDMD_data[:t_train])
println("")
println("Test Summary:")
println("  Model  |  Avg Tracking Err ")
println("---------|-------------------")
println(" nom MPC |  ", res[:nom_err_avg])
println("  eDMD   |  ", res[:eDMD_err_avg])
println("  jDMD   |  ", res[:jDMD_err_avg])

#############################################
## Plot a trajectory
#############################################

#############################################
## Try Tracking Zigzag pattern
#############################################

# dt = 0.05
# tf = 5.0
# t_sim = tf

# xe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# ue = Problems.trim_controls(model_real)
# X_ref, U_ref = generate_zigzag_traj()
# T_ref = range(0,tf,step=dt)
# T_sim = range(0,tf,step=dt)
# push!(U_ref, U_ref[end])
# x0 = X_ref[1]
# Nt = 20

# Qmpc = Diagonal([10, 10, 10, 10, 10, 10,
#     1, 1, 1, 1, 1, 1.0])
# Rmpc = Diagonal(fill(1e-3, 4))
# Qfmpc = Qmpc*100

# mpc_nom = TrackingMPC(dmodel_nom, 
#     X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
# )
# mpc_eDMD_projected = TrackingMPC(model_eDMD_projected, 
#     X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
# )
# mpc_jDMD_projected = TrackingMPC(model_jDMD_projected, 
#     X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
# )

# X_nom,= simulatewithcontroller(dmodel_real, mpc_nom, x0, t_sim, dt)
# X_eDMD,= simulatewithcontroller(dmodel_real, mpc_eDMD_projected, x0, t_sim, dt)
# X_jDMD,= simulatewithcontroller(dmodel_real, mpc_jDMD_projected, x0, t_sim, dt)

# jldsave(joinpath(Problems.DATADIR, "quadrotor_mpc_zigzag_tracking_results.jld2"); X_nom, X_eDMD, X_jDMD, X_ref, T_ref, T_sim)

# #############################################
# ## Plot results
# #############################################

# results = load(joinpath(Problems.DATADIR, "quadrotor_mpc_zigzag_tracking_results.jld2"))
# X_nom = results["X_nom"]
# X_eDMD = results["X_eDMD"]
# X_jDMD = results["X_jDMD"]
# X_ref = results["X_ref"]
# T_ref = results["T_ref"]
# T_sim = results["T_sim"]

# N_ref = length(T_ref)
# N_sim = length(T_sim)

# x_ref = map((x) -> x[1], X_ref)
# y_ref = map((x) -> x[2], X_ref)

# x_nom = map((x) -> x[1], X_nom)
# y_nom = map((x) -> x[2], X_nom)

# x_eDMD = map((x) -> x[1], X_eDMD)
# y_eDMD = map((x) -> x[2], X_eDMD)

# x_jDMD = map((x) -> x[1], X_jDMD)
# y_jDMD = map((x) -> x[2], X_jDMD)

# p_tracking = @pgf Axis(
#     {
#         xmajorgrids,
#         ymajorgrids,
#         xlabel = "X position (m)",
#         ylabel = "Y position (m)",
#         legend_pos = "north west",
#         # xmin = -10,
#         # xmax = 10,
#         # ymin = -10,
#         # ymax = 10,
#     },
#     PlotInc({lineopts..., color="teal"}, Coordinates(x_ref, y_ref)),
#     # PlotInc({lineopts..., color="black"}, Coordinates(x_nom, y_nom)),
#     # PlotInc({lineopts..., color=color_eDMD}, Coordinates(x_eDMD, y_eDMD)),
#     PlotInc({lineopts..., color=color_jDMD}, Coordinates(x_jDMD[1:12], y_jDMD[1:12])),
#     Legend(["Reference", "Nominal MPC", "eDMD", "jDMD"])
# )
# # pgfsave(joinpath(Problems.FIGDIR, "quadrotor_mpc_zigzag_tracking_trajectories.tikz"), p_tracking, include_preamble=false)