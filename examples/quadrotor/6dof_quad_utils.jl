using Test
using Distributions
using Random
using Rotations
using LinearAlgebra
using JLD2
using ThreadsX
import RobotDynamics as RD

# const QUADROTOR_NUMTRAJ_RESULTS_FILE = joinpath(BilinearControl.DATADIR, "rex_full_quadrotor_num_traj_sweep_results.jld2")
# const QUADROTOR_RESULTS_FILE = joinpath(BilinearControl.DATADIR, "rex_full_quadrotor_mpc_results.jld2")
# const QUADROTOR_MODELS_FILE = joinpath(BilinearControl.DATADIR, "rex_full_quadrotor_mpc_models.jld2")
const FULL_QUAD_DATA = joinpath(BilinearControl.DATADIR, "full_quad_data.jld2")
const FULL_QUAD_MODEL_DATA = joinpath(BilinearControl.DATADIR, "full_quad_model_data.jld2")
const FULL_QUAD_RESULTS = joinpath(BilinearControl.DATADIR, "full_quad_results.jld2")

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

"""
    generate_quadrotor_data()

Generates training and test data for the full quadrotor model. Generates trajectories using 
a stabilizing LQR controller about the origin, as well as a linear MPC policy that tracks 
straight line trajectories back to the origin.
"""
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
    X_train_lqr, U_train_lqr = BilinearControl.create_data(
        dmodel_real, ctrl_lqr_nom, initial_conditions_train, t_sim, dt)
    X_test_lqr, U_test_lqr = BilinearControl.create_data(
        dmodel_real, ctrl_lqr_nom, initial_conditions_test, t_sim, dt);

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

    # MPC Controller Params
    Qmpc = Diagonal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    Rmpc = Diagonal(fill(1e-4, 4))
    Qfmpc = Qmpc*10

    ## Generate training MPC trajectories
    N_tf = round(Int, tf/dt) + 1
    N_tsim = round(Int, t_sim/dt) + 1

    X_train_mpc = Matrix{Vector{Float64}}(undef, N_tsim, num_train_mpc)
    U_train_mpc = Matrix{Vector{Float64}}(undef, N_tsim-1, num_train_mpc)
    for i = 1:num_train_mpc

        x0 = initial_conditions_mpc_train[i]
        X = nominal_trajectory(x0,N_tf,dt)
        U = [copy(BilinearControl.trim_controls(model_real)) for k = 1:N_tf]
        T = range(0,tf,step=dt)

        mpc = BilinearControl.TrackingMPC_no_OSQP(dmodel_nom, X, U, Vector(T), Qmpc, Rmpc, Qfmpc; Nt=Nt)
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
        
        mpc_nom = BilinearControl.TrackingMPC_no_OSQP(dmodel_nom, X, U, Vector(T), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_nom,U_nom,T_nom = simulatewithcontroller(dmodel_real, mpc_nom, X[1], t_sim, T[2])

        X_test_infeasible[:,i] = X
        U_test_infeasible[:,i] = U[1:end-1]

        X_nom_mpc[:,i] = X_nom
        U_nom_mpc[:,i] = U_nom
    end

    ## Save generated training and test data
    jldsave(FULL_QUAD_DATA;
        X_train_lqr, U_train_lqr,
        X_train_mpc, U_train_mpc,
        X_nom_mpc, U_nom_mpc, 
        X_test_infeasible, U_test_infeasible,
        X_test_lqr, U_test_lqr, 
        tf, t_sim, dt
    )
end

# function test_initial_conditions(model, bilinear_model, ics, tf, t_sim, dt)

#     dmodel = RD.DiscretizedDynamics{RD.RK4}(model)

#     N_tf = round(Int, tf/dt) + 1    
#     N_sim = round(Int, t_sim/dt) + 1 
#     Nt = 20
#     T_ref = range(0,tf,step=dt)

#     Qmpc = Diagonal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
#     1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
#     Rmpc = Diagonal(fill(1e-3, 4))
#     Qfmpc = Qmpc*100

#     results = map(ics) do x0

#         X_ref = nominal_trajectory(x0,N_tf,dt)
#         X_ref_full = [X_ref; [X_ref[end] for i = 1:N_sim - N_tf]]
#         U_ref = [copy(BilinearControl.trim_controls(model)) for k = 1:N_tf]

#         mpc = BilinearControl.TrackingMPC_no_OSQP(bilinear_model, 
#             X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt)
#         X_sim,= simulatewithcontroller(dmodel, mpc, x0, t_sim, dt)

#         track_err = norm(X_sim - X_ref_full) / N_sim
#         stable_err = norm(X_sim[end] - X_ref_full[end]) 

#         (; track_err, stable_err)
#     end

#     err_avg  = mean(filter(isfinite, map(x->x.track_err, results)))
#     num_success = count(x -> x <=10, map(x->x.stable_err, results))

#     return err_avg, num_success
# end

# function test_initial_conditions_offset(model, bilinear_model, xg, ics, tf, t_sim, dt)

#     dmodel = RD.DiscretizedDynamics{RD.RK4}(model)

#     N_tf = round(Int, tf/dt) + 1    
#     N_sim = round(Int, t_sim/dt) + 1 
#     Nt = 20
#     T_ref = range(0,tf,step=dt)

#     Qmpc = Diagonal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
#     1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
#     Rmpc = Diagonal(fill(1e-3, 4))
#     Qfmpc = Qmpc*100

#     results = map(ics) do x0

#         x0 += xg
#         X_ref = nominal_trajectory(x0,N_tf,dt; xf=xg[1:3]) 
#         X_ref_full = [X_ref; [X_ref[end] for i = 1:N_sim - N_tf]]
#         U_ref = [copy(BilinearControl.trim_controls(model)) for k = 1:N_tf]

#         mpc = BilinearControl.TrackingMPC_no_OSQP(bilinear_model, 
#             X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt)
#         X_sim,= simulatewithcontroller(dmodel, mpc, x0, t_sim, dt)

#         err = norm(X_sim - X_ref_full) / N_sim

#         (; err)
#     end

#     err_avg  = mean(filter(isfinite, map(x->x.err, results)))

#     return err_avg
# end

"""
    nominal_trajectory(x0, N, dt)

Generate a straight line trajectory from `x0` to the origin with `N` steps and a time step 
of `dt` seconds.
"""
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


"""
    train_quadrotor_models(num_lqr, num_mpc)

Train eDMD and jDMD bilinear 6DOF quadrotor models using `num_lqr` LQR training 
trajectories and `num_mpc` MPC training trajectories.
"""
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

    println("Training eDMD model...")
    t_train_eDMD = @elapsed model_eDMD = run_eDMD(X_train, U_train, dt, full_quadrotor_kf, nothing; 
        alg=:qr, showprog=true, reg=reg
    )

    println("Training jDMD model...")
    t_train_jDMD = @elapsed model_jDMD = run_jDMD(X_train, U_train, dt, full_quadrotor_kf, nothing,
        dmodel_nom; showprog=true, verbose=false, reg=reg, alg=:qr_rls, α=0.5
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

"""
"""
function test_full_quadrotor(; save_to_file=true)
    # Load models
    model_info = load(FULL_QUAD_MODEL_DATA)["model_info"]

    eDMD_data = model_info.eDMD_data
    jDMD_data = model_info.jDMD_data
    G = model_info.G
    kf = model_info.kf
    dt = model_info.dt

    model_eDMD = EDMDModel(eDMD_data[:A],eDMD_data[:B],eDMD_data[:C],G,kf,dt,"quadrotor_eDMD")
    model_eDMD_projected = ProjectedEDMDModel(model_eDMD)
    model_jDMD = EDMDModel(jDMD_data[:A],jDMD_data[:B],jDMD_data[:C],G,kf,dt,"quadrotor_jDMD")
    model_jDMD_projected = ProjectedEDMDModel(model_jDMD)

    # Load data
    # mpc_lqr_traj = load(joinpath(BilinearControl.DATADIR, "rex_full_quadrotor_mpc_tracking_data.jld2"))
    mpc_lqr_traj = load(FULL_QUAD_DATA)

    # # Training data
    # X_train_lqr = mpc_lqr_traj["X_train_lqr"][:,1:num_lqr]
    # U_train_lqr = mpc_lqr_traj["U_train_lqr"][:,1:num_lqr]
    # X_train_mpc = mpc_lqr_traj["X_train_mpc"][:,1:num_mpc]
    # U_train_mpc = mpc_lqr_traj["U_train_mpc"][:,1:num_mpc]

    # # combine lqr and mpc training data
    # X_train = [X_train_lqr X_train_mpc]
    # U_train = [U_train_lqr U_train_mpc]

    # Test data
    # X_nom_mpc = mpc_lqr_traj["X_nom_mpc"]
    # U_nom_mpc = mpc_lqr_traj["U_nom_mpc"]
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

    # Define Nominal Simulated Quadrotor Model
    model_nom = BilinearControl.NominalRexQuadrotor()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" Quadrotor Model
    model_real = BilinearControl.SimulatedRexQuadrotor()  # this model has aero drag
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    xe = zeros(12)
    ue = BilinearControl.trim_controls(model_real)
    Nt = 20  # MPC horizon
    N_sim = length(T_sim)
    N_ref = length(T_ref)

    Qmpc = Diagonal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    Rmpc = Diagonal(fill(1e-3, 4))
    Qfmpc = Qmpc*100

    N_test = size(X_test_infeasible,2)
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
    nom_success = count(x -> norm(x[end]-xe)<=10, map(x->x.X_mpc_nom, test_results)) / N_test
    eDMD_success = count(x -> norm(x[end]-xe)<=10, map(x->x.X_mpc_eDMD, test_results)) / N_test
    jDMD_success = count(x -> norm(x[end]-xe)<=10, map(x->x.X_mpc_jDMD, test_results)) / N_test

    # nom_errs  = map(x->x.err_nom, test_results)
    # eDMD_errs = map(x->x.err_eDMD, test_results)
    # jDMD_errs = map(x->x.err_jDMD, test_results)

    X_ref = map(x->x.X_ref, test_results)
    X_mpc_nom = map(x->x.X_mpc_nom, test_results)
    X_mpc_eDMD = map(x->x.X_mpc_eDMD, test_results)
    X_mpc_jDMD = map(x->x.X_mpc_jDMD, test_results)
    T_mpc = map(x->x.T_mpc, test_results)

    MPC_test_results = (;
        X_ref, X_mpc_nom, X_mpc_eDMD, X_mpc_jDMD, T_mpc,
        nom_success, eDMD_success, jDMD_success, 
        nom_err_avg, eDMD_err_avg, jDMD_err_avg
    )
    if save_to_file
        jldsave(FULL_QUAD_RESULTS; MPC_test_results)
    end
    MPC_test_results
end