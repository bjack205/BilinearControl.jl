using LinearAlgebra
using Distributions
using Statistics
using Random
using ProgressMeter
using ThreadsX
using JLD2
using Test
import RobotDynamics as RD

const PLANAR_QUAD_DATA = joinpath(BilinearControl.DATADIR, "rex_planar_quadrotor_data.jld2"); 
const PLANAR_QUAD_LQR_RESULTS = joinpath(BilinearControl.DATADIR, "planar_quad_lqr_results.jld2")
const PLANAR_QUAD_MPC_RESULTS = joinpath(BilinearControl.DATADIR, "rex_planar_quadrotor_mpc_training_range_results.jld2")
"""
    generate_planar_quadrotor_data

Saves training and test trajectories to PLANAR_QUAD_DATA. Generates and saves both LQR 
stabilization and MPC tracking trajectories. All trajectories have the same length.
"""
function generate_planar_quadrotor_data()
    #############################################
    ## Define the Models
    #############################################

    ## Define Nominal Simulated REx Planar Quadrotor Model
    model_nom = BilinearControl.NominalPlanarQuadrotor()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" REx Planar Quadrotor Model
    model_real = BilinearControl.SimulatedPlanarQuadrotor()  # this model has aero drag
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    # Time parameters
    tf = 5.0
    dt = 0.05
    Nt = 41  # MPC Horizon
    t_sim = tf*1.2  # length of simulation (to capture steady-state behavior) 

    #############################################
    ## LQR Training and Testing Data 
    #############################################

    ## Stabilization trajectories 
    Random.seed!(1)
    num_train_lqr = 50
    num_test_lqr = 50

    # Generate a stabilizing LQR controller
    Qlqr = Diagonal([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    Rlqr = Diagonal([1e-4, 1e-4])
    xe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ue = BilinearControl.trim_controls(model_real)
    ctrl_lqr_nom = LQRController(dmodel_nom, Qlqr, Rlqr, xe, ue, dt)

    # Sample a bunch of initial conditions for the LQR controller
    x0_train_sampler = Product([
        Uniform(-1.0,1.0),
        Uniform(-1.0,1.0),
        Uniform(-deg2rad(40),deg2rad(40)),
        Uniform(-0.5,0.5),
        Uniform(-0.5,0.5),
        Uniform(-0.25,0.25)
    ])

    perc = 2.0
    x0_test_sampler = Product([
        Uniform(-1.0*perc,1.0*perc),
        Uniform(-1.0*perc,1.0*perc),
        Uniform(-deg2rad(40*perc),deg2rad(40*perc)),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.25*perc,0.25*perc)
    ])

    initial_conditions_train = [rand(x0_train_sampler) for _ in 1:num_train_lqr]
    initial_conditions_test = [rand(x0_test_sampler) for _ in 1:num_test_lqr]

    # Create data set
    X_train_lqr, U_train_lqr = BilinearControl.create_data(dmodel_real, ctrl_lqr_nom, initial_conditions_train, t_sim, dt)
    X_test_lqr, U_test_lqr = BilinearControl.create_data(dmodel_real, ctrl_lqr_nom, initial_conditions_test, t_sim, dt);

    # Make sure they all stabilize
    @test all(x->x<0.1, map(x->norm(x-xe), X_train_lqr[end,:]))
    @test all(x->x<0.1, map(x->norm(x-xe), X_test_lqr[end,:]))

    #############################################
    ## MPC Training and Testing Data 
    #############################################
    Random.seed!(1)
    num_train_mpc = 50
    num_test_mpc = 35

    x0_train_sampler = Product([
        Uniform(-2.0,2.0),
        Uniform(-2.0,2.0),
        Uniform(-deg2rad(20),deg2rad(20)),
        Uniform(-0.5,0.5),
        Uniform(-0.5,0.5),
        Uniform(-0.2,0.2)
    ])

    perc = 3.0
    x0_test_sampler = Product([
        Uniform(-2.0*perc,2.0*perc),
        Uniform(-2.0*perc,2.0*perc),
        Uniform(-deg2rad(20*perc),deg2rad(20*perc)),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.2*perc,0.2*perc)
    ])

    initial_conditions_mpc_train = [rand(x0_train_sampler) for _ in 1:num_train_mpc]
    initial_conditions_mpc_test = [rand(x0_test_sampler) for _ in 1:num_test_mpc]

    Random.seed!(1)

    Qmpc = Diagonal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    Rmpc = Diagonal([1e-3, 1e-3])
    Qfmpc = 100*Qmpc

    N_tf = round(Int, tf/dt) + 1
    N_tsim = round(Int, t_sim/dt) + 1

    X_train_mpc = Matrix{Vector{Float64}}(undef, N_tsim, num_train_mpc)
    U_train_mpc = Matrix{Vector{Float64}}(undef, N_tsim-1, num_train_mpc)

    for i = 1:num_train_mpc

        x0 = initial_conditions_mpc_train[i]
        X = nominal_trajectory(x0,N_tf,dt)
        U = [copy(BilinearControl.trim_controls(model_real)) for k = 1:N_tf]
        T = range(0,tf,step=dt)

        mpc = TrackingMPC(dmodel_nom, X, U, Vector(T), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_sim,U_sim,T_sim = simulatewithcontroller(dmodel_real, mpc, X[1], t_sim, T[2])
            
        X_train_mpc[:,i] = X_sim
        U_train_mpc[:,i] = U_sim
    end

    # Generate test data
    X_test_infeasible = Matrix{Vector{Float64}}(undef, N_tf, num_test_mpc)
    X_nom_mpc = Matrix{Vector{Float64}}(undef, N_tsim, num_test_mpc)

    U_test_infeasible = Matrix{Vector{Float64}}(undef, N_tf-1, num_test_mpc)
    U_nom_mpc = Matrix{Vector{Float64}}(undef, N_tsim-1, num_test_mpc)

    for i = 1:num_test_mpc

        x0 = initial_conditions_mpc_test[i]
        X = nominal_trajectory(x0,N_tf,dt)
        U = [copy(BilinearControl.trim_controls(model_real)) for k = 1:N_tf]
        T = range(0,tf,step=dt)
        
        mpc_nom = TrackingMPC(dmodel_nom, X, U, Vector(T), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_nom,U_nom,T_nom = simulatewithcontroller(dmodel_real, mpc_nom, X[1], t_sim, T[2])

        X_test_infeasible[:,i] = X
        U_test_infeasible[:,i] = U[1:end-1]

        X_nom_mpc[:,i] = X_nom
        U_nom_mpc[:,i] = U_nom
    end

    ## Save generated training and test data
    jldsave(joinpath(BilinearControl.DATADIR, "rex_planar_quadrotor_mpc_tracking_data.jld2"); 
        X_train_lqr, U_train_lqr,
        X_train_mpc, U_train_mpc,
        X_nom_mpc, U_nom_mpc, 
        X_test_infeasible, U_test_infeasible,
        X_test_lqr, U_test_lqr, 
        tf, t_sim, dt
    )
end


"""
    test_initial_conditions_offset(model, controller, xg, ics, tf, dt)

Test the performance of a stabilizing controller on the given discrete dynamics
`model`` for all of the initial conditions in `ics`, simulating for `tf` seconds
with a time step of `dt`.  Returns the distance of the final position from the
goal position `xg`. Offsets each initial condition by `xg`.
"""
function test_initial_conditions_offset(model, controller, xg, ics, tf, dt)
    map(ics) do x0
        X_sim, = simulatewithcontroller(model, controller, x0+xg, tf, dt)
        norm(X_sim[end] - xg)
    end
end

"""
    planar_quad_lqr_offset(;kwargs...)

Test the performance of an LQR stabilizing controller as the equilibrium posiiton shifts 
away from the point at which the model was trained. Returns a vector of average errors 
for projected LQR controllers for eDMD and jDMD, each with a high and low level of 
regularization.
"""
function planar_quad_lqr_offset(; num_train=30, verbose=true, save_to_file=true)
    Random.seed!(1)

    ############################################# 
    ## Define the Nominal and True Models
    ############################################# 
    model_nom = BilinearControl.NominalPlanarQuadrotor()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    model_real = BilinearControl.SimulatedPlanarQuadrotor()  # this model has aero drag
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    ############################################# 
    ## Read Training Data
    ############################################# 
    verbose && println("Reading Training Data...")
    num_lqr = num_train
    mpc_lqr_traj = load(joinpath(BilinearControl.DATADIR, "rex_planar_quadrotor_mpc_tracking_data.jld2"))
    X_train = mpc_lqr_traj["X_train_lqr"][:,1:num_lqr]
    U_train = mpc_lqr_traj["U_train_lqr"][:,1:num_lqr]
    dt = mpc_lqr_traj["dt"]

    ############################################# 
    ## Train the Bilinear Models 
    ############################################# 
    verbose && println("Training Bilinear Models...")
    eigfuns = ["state", "sine", "cosine", "chebyshev"]
    eigorders = [[0],[1],[1],[2,2]]

    model_eDMD = run_eDMD(
        X_train, U_train, dt, eigfuns, eigorders, reg=1e-1, name="planar_quadrotor_eDMD"
    )
    model_eDMD_unreg = run_eDMD(
        X_train, U_train, dt, eigfuns, eigorders, reg=0.0, name="planar_quadrotor_eDMD"
    )
    model_jDMD = run_jDMD(
        X_train, U_train, dt, eigfuns, eigorders, dmodel_nom, reg=1e-5, 
        name="planar_quadrotor_jDMD"
    )
    model_jDMD2 = run_jDMD(
        X_train, U_train, dt, eigfuns, eigorders, dmodel_nom, reg=1e-1, 
        name="planar_quadrotor_jDMD"
    )

    # Generate Projected Bilinear Models
    model_eDMD_projected = BilinearControl.ProjectedEDMDModel(model_eDMD)
    model_eDMD_projected_unreg = BilinearControl.ProjectedEDMDModel(model_eDMD_unreg)
    model_jDMD_projected = BilinearControl.ProjectedEDMDModel(model_jDMD)
    model_jDMD_projected2 = BilinearControl.ProjectedEDMDModel(model_jDMD2)

    ####################################################
    ## Test Performance as Equilibrium Position Changes
    ####################################################
    verbose && println("Testing Performance vs Equilibrium offset...")

    # MPC Parameters
    Qlqr = Diagonal([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    Rlqr = Diagonal([1e-4, 1e-4])
    xe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ue = BilinearControl.trim_controls(model_real)

    # Test increasing equilibrium offsets
    distances = 0:0.1:4
    prog = Progress(length(distances))
    errors = ThreadsX.map(distances) do dist

        # println("equilibrium offset = $dist")
        t_sim = 5.0

        if dist == 0
            xe_test = [zeros(6)]
        else
            xe_sampler = Product([
                Uniform(-dist, +dist),
                Uniform(-dist, +dist),
            ])
            xe_test = [vcat(rand(xe_sampler), zeros(4)) for i = 1:100]
        end

        perc = 0.8
        x0_sampler = Product([
            Uniform(-1.0*perc,1.0*perc),
            Uniform(-1.0*perc,1.0*perc),
            Uniform(-deg2rad(40*perc),deg2rad(40*perc)),
            Uniform(-0.5*perc,0.5*perc),
            Uniform(-0.5*perc,0.5*perc),
            Uniform(-0.25*perc,0.25*perc)
        ])

        x0_test = [rand(x0_sampler) for i = 1:100]


        xe_results = map(xe_test) do xe
            lqr_eDMD_projected = LQRController(
                model_eDMD_projected, Qlqr, Rlqr, xe, ue, dt, max_iters=10000)
            lqr_eDMD_projected_unreg = LQRController(
                model_eDMD_projected_unreg, Qlqr, Rlqr, xe, ue, dt, max_iters=10000)
            lqr_jDMD_projected = LQRController(
                model_jDMD_projected, Qlqr, Rlqr, xe, ue, dt, max_iters=10000)
            lqr_jDMD_projected2 = LQRController(
                model_jDMD_projected2, Qlqr, Rlqr, xe, ue, dt, max_iters=10000)
        
            error_eDMD_projected_x0s = mean(test_initial_conditions_offset(
                dmodel_real, lqr_eDMD_projected, xe, x0_test, t_sim, dt))
            error_eDMD_projected_unreg_x0s = mean(test_initial_conditions_offset(
                dmodel_real, lqr_eDMD_projected_unreg, xe, x0_test, t_sim, dt))
            error_jDMD_projected_x0s = mean(test_initial_conditions_offset(
                dmodel_real, lqr_jDMD_projected, xe, x0_test, t_sim, dt))
            error_jDMD_projected2_x0s = mean(test_initial_conditions_offset(
                dmodel_real, lqr_jDMD_projected2, xe, x0_test, t_sim, dt))

            if error_eDMD_projected_x0s > 1e3
                error_eDMD_projected_x0s = NaN
            end
            if error_eDMD_projected_unreg_x0s > 1e3
                error_eDMD_projected_unreg_x0s = NaN
            end
            if error_jDMD_projected_x0s > 1e3
                error_jDMD_projected_x0s = NaN
            end
            if error_jDMD_projected2_x0s > 1e3
                error_jDMD_projected2_x0s = NaN
            end
            (;error_eDMD_projected_x0s, error_eDMD_projected_unreg_x0s, error_jDMD_projected_x0s, error_jDMD_projected2_x0s)
        end
        
        error_eDMD_projected = mean(filter(isfinite, map(x->x.error_eDMD_projected_x0s, xe_results)))
        error_eDMD_projected_unreg = mean(filter(isfinite, map(x->x.error_eDMD_projected_unreg_x0s, xe_results)))
        error_jDMD_projected = mean(filter(isfinite, map(x->x.error_jDMD_projected_x0s, xe_results)))
        error_jDMD_projected2 = mean(filter(isfinite, map(x->x.error_jDMD_projected2_x0s, xe_results)))
        next!(prog)

        (;error_eDMD_projected, error_eDMD_projected_unreg, error_jDMD_projected, error_jDMD_projected2)

    end

    fields = keys(errors[1])
    res_equilibrium = Dict(Pair.(fields, map(x->getfield.(errors, x), fields)))
    res_equilibrium[:distances] = distances
    if save_to_file
        jldsave(PLANAR_QUAD_LQR_RESULTS; res_equilibrium)
    end
    return res_equilibrium
end


"""
    test_bilinear_mpc(model, bilinear_model, ics, tf, t_sim, dt)

Test the learned `bilinear_model` by tracking a straight line trajectory from each initial 
condition in `ics` to the origin. Simulates the system for `t_sim` seconds with a time step 
of `dt` seconds, assuming the goal is to arrive at the goal in `tf` seconds.
"""
function test_bilinear_mpc(model, bilinear_model, ics, tf, t_sim, dt)

    dmodel = RD.DiscretizedDynamics{RD.RK4}(model)

    N_tf = round(Int, tf/dt) + 1    
    N_sim = round(Int, t_sim/dt) + 1 
    Nt = 41
    T_ref = range(0,tf,step=dt)

    Qmpc = Diagonal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    Rmpc = Diagonal([1e-3, 1e-3])
    Qfmpc = 100*Qmpc

    results = map(ics) do x0

        X_ref = nominal_trajectory(x0,N_tf,dt)
        X_ref_full = [X_ref; [X_ref[end] for i = 1:N_sim - N_tf]]
        U_ref = [copy(BilinearControl.trim_controls(model)) for k = 1:N_tf]

        mpc = TrackingMPC(bilinear_model, 
            X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_sim,= simulatewithcontroller(dmodel, mpc, x0, t_sim, dt)

        err = norm(X_sim - X_ref_full) / N_sim

        (; err)
    end

    err_avg  = mean(filter(isfinite, map(x->x.err, results)))

    return err_avg
end

"""
    nominal_trajectory(x0, N, dt)

Generate a straight line trajectory from `x0` to the origin with `N` steps and a time step 
of `dt` seconds.
"""
function nominal_trajectory(x0,N,dt)
    Xref = [fill(NaN, length(x0)) for k = 1:N]
    
    pos_0 = x0[1:3]
    
    pos_ref = reshape(LinRange(pos_0, [0, 0, 0], N), N, 1)
    vel_ref = vcat([(pos_ref[2] - pos_ref[1]) ./ dt for k = 1:N-1], [[0, 0, 0]])
    
    for i = 1:N
        
        Xref[i] = vcat(pos_ref[i], vel_ref[i])
    
    end
    
    return Xref
end


"""
    train_planar_quadrotor_models(num_lqr, num_mpc; kwargs...)

Train eDMD and jDMD bilinear planar quadrotor models using `num_lqr` LQR training 
trajectories and `num_mpc` MPC training trajectories.
"""
function train_planar_quadrotor_models(num_lqr, num_mpc;  α=0.5, learnB=true, β=1.0, 
        reg_eDMD=1e-6, reg_jDMD=reg_eDMD)

    #############################################
    ## Define the Models
    #############################################

    # Define Nominal Simulated REx Planar Quadrotor Model
    model_nom = BilinearControl.NominalPlanarQuadrotor()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" REx Planar Quadrotor Model
    model_real = BilinearControl.SimulatedPlanarQuadrotor()  # this model has aero drag
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    #############################################  
    ## Load Training and Test Data
    #############################################  
    mpc_lqr_traj = load(joinpath(BilinearControl.DATADIR, "rex_planar_quadrotor_mpc_tracking_data.jld2"))

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
    eigfuns = ["state", "sine", "cosine", "chebyshev"]
    eigorders = [[0],[1],[1],[2,2]]

    t_train_eDMD = @elapsed model_eDMD = run_eDMD(X_train, U_train, dt, eigfuns, eigorders,
        reg=reg_eDMD, name="planar_quadrotor_eDMD")
    t_train_jDMD = @elapsed model_jDMD = run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom,
        reg=reg_jDMD, name="planar_quadrotor_jDMD"; α, β, learnB)

    return model_eDMD, model_jDMD
end

"""
    planar_quad_mpc_generalization()

Test the performance of a projected MPC controller on eDMD and jDMD models with both high 
and low levels or regularization. Tests performance as the initial conditions are sampled 
from a scaling of the training window.
"""
function planar_quad_mpc_generalization(;save_to_file=true)
    num_lqr = 0
    num_mpc = 50
    model_eDMD_1, model_jDMD_1 = train_planar_quadrotor_models(num_lqr, num_mpc, 
        α=0.5, β=1.0, learnB=true, reg_eDMD=0.0, reg_jDMD=1e-5)
    model_eDMD_2, model_jDMD_2 = train_planar_quadrotor_models(num_lqr, num_mpc, 
        α=0.5, β=1.0, learnB=true, reg_eDMD=1e-1, reg_jDMD=1e-1)

    # Projected Models
    model_eDMD_projected_1 = ProjectedEDMDModel(model_eDMD_1)
    model_jDMD_projected_1 = ProjectedEDMDModel(model_jDMD_1)
    model_eDMD_projected_2 = ProjectedEDMDModel(model_eDMD_2)
    model_jDMD_projected_2 = ProjectedEDMDModel(model_jDMD_2)

    # Generate models
    model_nom = BilinearControl.NominalPlanarQuadrotor()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    model_real = BilinearControl.SimulatedPlanarQuadrotor()  # this model has aero drag
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    Random.seed!(1)
    tf = 5.0
    dt = 0.05
    percentages = 0.1:0.1:2.5
    prog = Progress(length(percentages))
    errors = ThreadsX.map(percentages) do perc

        x0_sampler = Product([
            Uniform(-2.0*perc,2.0*perc),
            Uniform(-2.0*perc,2.0*perc),
            Uniform(-deg2rad(20*perc),deg2rad(20*perc)),
            Uniform(-0.5*perc,0.5*perc),
            Uniform(-0.5*perc,0.5*perc),
            Uniform(-0.2*perc,0.2*perc)
        ])

        x0_test = [rand(x0_sampler) for i = 1:50]

        error_eDMD_loreg = mean(test_bilinear_mpc(
            model_real, model_eDMD_projected_1, x0_test, tf, tf, dt))
        error_jDMD_loreg = mean(test_bilinear_mpc(
            model_real, model_jDMD_projected_1, x0_test, tf, tf, dt))
        error_eDMD_hireg = mean(test_bilinear_mpc(
            model_real, model_eDMD_projected_2, x0_test, tf, tf, dt))
        error_jDMD_hireg = mean(test_bilinear_mpc(
            model_real, model_jDMD_projected_2, x0_test, tf, tf, dt))
        next!(prog)

        (;error_eDMD_loreg, error_jDMD_loreg, error_eDMD_hireg, error_jDMD_hireg)
    end

    fields = keys(errors[1])
    res_training_range = Dict(Pair.(fields, map(x->getfield.(errors, x), fields)))
    if save_to_file
        jldsave(PLANAR_QUAD_MPC_RESULTS;
            percentages, 
            res_training_range
        )
    end

    res_training_range[:percentages] = percentages
    res_training_range
end