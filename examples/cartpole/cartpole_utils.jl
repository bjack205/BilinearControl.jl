using Altro
import TrajectoryOptimization as TO
using LinearAlgebra
import RobotDynamics as RD
using StaticArrays
using Statistics
using Distributions
using Random
using JLD2

const CARTPOLE_LQR_RESULTS_FILE = joinpath(BilinearControl.DATADIR, "cartpole_lqr_results.jld2")
const CARTPOLE_RESULTS = joinpath(BilinearControl.DATADIR, "cartpole_results.jld2")
const CARTPOLE_MISMATCH_RESULTS = joinpath(BilinearControl.DATADIR, "cartpole_mismatch_results.jld2")
const CARTPOLE_MPC_RESULTS = joinpath(BilinearControl.DATADIR, "cartpole_mpc_results.jld2")

"""
    gencartpoleproblem()

Generates the nonlinear trajectory optimization problem for the cartpole swingup. Passed 
ALTRO for solving.
"""
function gencartpoleproblem(x0=zeros(4), Qv=1e-2, Rv=1e-1, Qfv=1e2, u_bnd=3.0, tf=5.0; 
    dt=0.05, constrained=true, μ=0.0)

    # NOTE: this should exactly match RobotZoo.Cartpole() when μ = 0.0
    model = BilinearControl.NominalCartpole(; μ=μ)
    dmodel = RD.DiscretizedDynamics{RD.RK4}(model) 
    n,m = RD.dims(model)
    N = round(Int, tf/dt) + 1

    Q = Qv*Diagonal(@SVector ones(n)) * dt
    Qf = Qfv*Diagonal(@SVector ones(n))
    R = Rv*Diagonal(@SVector ones(m)) * dt
    xf = @SVector [0, pi, 0, 0]
    obj = TO.LQRObjective(Q,R,Qf,xf,N)

    conSet = TO.ConstraintList(n,m,N)
    bnd = TO.BoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd)
    goal = TO.GoalConstraint(xf)
    if constrained
    TO.add_constraint!(conSet, bnd, 1:N-1)
    TO.add_constraint!(conSet, goal, N:N)
    end

    X0 = [@SVector fill(NaN,n) for k = 1:N]
    u0 = @SVector fill(0.01,m)
    U0 = [u0 for k = 1:N-1]
    Z = TO.SampledTrajectory(X0,U0,dt=dt*ones(N-1))
    prob = TO.Problem(dmodel, obj, x0, tf, constraints=conSet, xf=xf) 
    TO.initial_trajectory!(prob, Z)
    TO.rollout!(prob)
    prob
end

"""
    generate_cartpole_data(;kwargs...)

Generate the training and test data for the cartpole. Includes both stabilization 
and swingup trajectories.
"""
function generate_cartpole_data(;num_lqr=50, num_swingup=50, save_to_file=true, 
        μ=0.1, μ_nom=0.0, max_lqr_samples=3*num_lqr,
        x_window = [0.7,deg2rad(45),0.2,0.2]
    )
    #############################################
    ## Define the Models
    #############################################
    # Define Nominal Simulated Cartpole Model
    model_nom = BilinearControl.NominalCartpole(;μ=μ_nom)
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" Cartpole Model
    model_real = BilinearControl.SimulatedCartpole(;μ=μ) # this model has damping
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
    num_train_lqr = num_lqr
    num_test_lqr = 10

    # Generate a stabilizing LQR controller about the top
    Qlqr = Diagonal([1.0,10.0,1e-2,1e-2])
    
    # Qlqr = Diagonal([0.2,10,1e-2,1e-2])  # this makes eDMD break...
    Rlqr = Diagonal([1e-3])
    xe = [0,pi,0,0]
    ue = [0.0]
    ctrl_lqr = LQRController(dmodel_nom, Qlqr, Rlqr, xe, ue, dt)
    T_lqr = range(0, t_sim, step=dt)

    # Sample a bunch of initial conditions for the LQR controller
    x0_sampler = Product(collect(Uniform(x-dx,x+dx) for (x,dx) in zip(xe,x_window)))
    Random.seed!(1)
    initial_conditions_train = [rand(x0_sampler) for _ in 1:num_train_lqr]
    Random.seed!(1)
    initial_conditions_test = [rand(x0_sampler) for _ in 1:num_test_lqr]

    # Create data set
    X_train_lqr, U_train_lqr = BilinearControl.create_data(dmodel_real, ctrl_lqr, 
        x0_sampler, num_train_lqr, xe, t_sim, dt, max_samples=max_lqr_samples
    )
    X_test_lqr, U_test_lqr = BilinearControl.create_data(dmodel_real, ctrl_lqr, 
        x0_sampler, num_test_lqr, xe, t_sim, dt, max_samples=max_lqr_samples
    );

    X_train = X_train_lqr
    U_train = U_train_lqr
    X_test = X_test_lqr
    U_test = U_test_lqr
    X_ref = []
    U_ref = []

    # Make sure they all stabilize
    num_train_stabilized = count(x->x<0.1, map(x->norm(x-xe), X_train_lqr[end,:]))
    num_test_stabilized = count(x->x<0.1, map(x->norm(x-xe), X_test_lqr[end,:]))
    if num_train_stabilized < num_train_lqr
        @warn "Not all of the LQR training trajectories succesfully stabilized. Got $num_train_stabilized / $num_train_lqr."
    end
    if num_test_stabilized < num_test_lqr
        @warn "Not all of the LQR test trajectories succesfully stabilized. Got $num_test_stabilized / $num_test_lqr."
    end

    #############################################
    ## ALTRO Training and Testing Data 
    #############################################
    if num_swingup > 0
        Random.seed!(1)
        num_train_swingup = num_swingup 
        num_test_swingup = 10

        train_params = map(1:num_train_swingup) do i
            Qv = 1e-2
            Rv = Qv * 10^rand(Uniform(-1,3.0))
            Qfv = Qv * 10^rand(Uniform(1,5.0)) 
            u_bnd = rand(Uniform(4.5, 8.0))
            (zeros(4), Qv, Rv, Qfv, u_bnd, tf)
        end

        Qmpc = Diagonal(fill(1e-0,4))
        Rmpc = Diagonal(fill(1e-3,1))
        Qfmpc = Diagonal(fill(1e2,4))

        train_trajectories = map(train_params) do params
            solver = Altro.solve!(ALTROSolver(gencartpoleproblem(params..., dt=dt), 
                show_summary=false, projected_newton=true))
            if Altro.status(solver) != Altro.SOLVE_SUCCEEDED
                @warn "ALTRO Solve failed"
            end
            X = Vector.(TO.states(solver))
            U = Vector.(TO.controls(solver))
            T = Vector(range(0,tf,step=dt))

            push!(U, zeros(RD.control_dim(solver)))

            mpc = TrackingMPC(dmodel_nom, X, U, T, Qmpc, Rmpc, Qfmpc; Nt=Nt)
            X_sim,U_sim,T_sim = simulatewithcontroller(dmodel_real, mpc, X[1], t_sim, T[2])
            
            Vector.(X), Vector.(U[1:end-1]), Vector.(X_sim), Vector.(U_sim)
        end

        X_train_swingup_ref = mapreduce(x->getindex(x,1), hcat, train_trajectories)
        U_train_swingup_ref = mapreduce(x->getindex(x,2), hcat, train_trajectories)
        X_train_swingup = mapreduce(x->getindex(x,3), hcat, train_trajectories)
        U_train_swingup = mapreduce(x->getindex(x,4), hcat, train_trajectories)

        test_params = [
            (zeros(4), 1e-2, 1e-1, 1e2,  3.0, tf)
            (zeros(4), 1e-0, 1e-1, 1e2,  5.0, tf)
            (zeros(4), 1e1,  1e-2, 1e2, 10.0, tf)
            (zeros(4), 1e-1, 1e-0, 1e2, 10.0, tf)
            (zeros(4), 1e-2, 1e-0, 1e1, 10.0, tf)
            (zeros(4), 1e-2, 1e-0, 1e1,  3.0, tf)
            (zeros(4), 1e1,  1e-3, 1e2, 10.0, tf)
            (zeros(4), 1e1,  1e-3, 1e2,  5.0, tf)
            (zeros(4), 1e3,  1e-3, 1e3, 10.0, tf)
            (zeros(4), 1e0,  1e-2, 1e2,  4.0, tf)
        ]
        test_trajectories = map(test_params) do params
            solver = Altro.solve!(ALTROSolver(gencartpoleproblem(params...; dt), show_summary=false))
            if Altro.status(solver) != Altro.SOLVE_SUCCEEDED
                @show params
                @warn "ALTRO Solve failed"
            end
            X = Vector.(TO.states(solver))
            U = Vector.(TO.controls(solver))
            T = Vector(range(0,tf,step=dt))

            push!(U, zeros(RD.control_dim(solver)))

            mpc = TrackingMPC(dmodel_nom, X, U, T, Qmpc, Rmpc, Qfmpc; Nt=Nt)
            X_sim,U_sim,T_sim = simulatewithcontroller(dmodel_real, mpc, X[1], t_sim, T[2])

            Vector.(X), Vector.(U[1:end-1]), Vector.(X_sim), Vector.(U_sim)
        end

        X_ref = mapreduce(x->getindex(x,1), hcat, test_trajectories)
        U_ref = mapreduce(x->getindex(x,2), hcat, test_trajectories)
        X_test_swingup = mapreduce(x->getindex(x,3), hcat, test_trajectories)
        U_test_swingup = mapreduce(x->getindex(x,4), hcat, test_trajectories)
        X_test = X_test_swingup
        U_test = U_test_swingup

        ## combine lqr and mpc training data
        X_train = [X_train_lqr X_train_swingup]
        U_train = [U_train_lqr U_train_swingup]
    end

    ## Save generated training and test data
    if save_to_file
        jldsave(joinpath(BilinearControl.DATADIR, "cartpole_swingup_data.jld2"); 
            X_train_lqr, U_train_lqr,
            X_train_swingup, U_train_swingup,
            X_test_swingup, U_test_swingup, 
            X_ref, U_ref,
            X_test_lqr, U_test_lqr, 
            tf, t_sim, dt
        )
    end
    metadata = (;t_train=t_sim, t_ref=tf, dt)
    return X_train, U_train, X_test, U_test, X_ref, U_ref, metadata
end

"""
    generate_stabilizing_mpc_controller(model, t_sim, dt)

Generate a linear MPC controller for stabilizing the cartpole about the upward equilibrium.
"""
function generate_stabilizing_mpc_controller(model, t_sim, dt; 
        Nt=41, ρ=1e-6, 
        Qmpc = Diagonal(fill(1e-0,4)),
        Rmpc = Diagonal(fill(1e-3,1)),
        Qfmpc = Diagonal([1e2,1e2,1e1,1e1]),
    )
    xe = [0,pi,0,0]
    ue = [0.]
    ye = BilinearControl.expandstate(model, xe)
    lifted_state_error(x,x0) = model.kf(x) - x0

    # Reference Trajectory
    T_sim = range(0,t_sim,step=dt)
    X_ref = [copy(xe) for t in T_sim]
    U_ref = [copy(ue) for t in T_sim]
    T_ref = copy(T_sim)
    Y_ref = map(x->BilinearControl.expandstate(model,x), X_ref)

    # Objective
    is_lifted_model = length(ye) > length(xe)
    if is_lifted_model
        Qmpc_lifted = Diagonal([ρ; diag(Qmpc); fill(ρ, length(ye)-5)])
        Qfmpc_lifted = Diagonal([ρ; diag(Qfmpc); fill(ρ, length(ye)-5)])
        state_error = lifted_state_error
    else
        Qmpc_lifted = Qmpc 
        Qfmpc_lifted = Qfmpc 
        state_error = (x,x0)->(x-x0)
    end

    # MPC controller
    TrackingMPC(model, 
        Y_ref, U_ref, Vector(T_ref), Qmpc_lifted, Rmpc, Qfmpc_lifted; Nt=Nt, state_error
    )
end

"""
    train_cartpole_models(num_lqr, num_swingup; kwargs...)

Trains eDMD and jDMD models with `num_lqr` LQR stabilization trajectories and 
`num_swingup` ALTRO swingup trajectories, loaded from `cartpole_swingup_data.jld2`.

After training, uses MPC to track the swingup reference trajectories in the data files,
and reports back statistics.
"""
function train_cartpole_models(num_lqr, num_swingup; α=0.5, learnB=true, β=1.0, reg=1e-6)

    #############################################
    ## Define the Models
    #############################################
    # Define Nominal Simulated Cartpole Model
    model_nom = BilinearControl.NominalCartpole()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" Cartpole Model
    model_real = BilinearControl.SimulatedCartpole() # this model has damping
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    #############################################  
    ## Load Training and Test Data
    #############################################  
    altro_lqr_traj = load(joinpath(BilinearControl.DATADIR, "cartpole_swingup_data.jld2"))

    # Training data
    X_train_lqr = altro_lqr_traj["X_train_lqr"][:,1:num_lqr]
    U_train_lqr = altro_lqr_traj["U_train_lqr"][:,1:num_lqr]
    X_train_swingup = altro_lqr_traj["X_train_swingup"][:,1:num_swingup]
    U_train_swingup = altro_lqr_traj["U_train_swingup"][:,1:num_swingup]
    X_train = [X_train_lqr X_train_swingup]
    U_train = [U_train_lqr U_train_swingup]

    # Test data
    X_test_swingup = altro_lqr_traj["X_test_swingup"]
    U_test_swingup = altro_lqr_traj["U_test_swingup"]
    X_test_swingup_ref = altro_lqr_traj["X_ref"]
    U_test_swingup_ref = altro_lqr_traj["U_ref"]

    # Metadata
    tf = altro_lqr_traj["tf"]
    t_sim = altro_lqr_traj["t_sim"]
    dt = altro_lqr_traj["dt"]

    T_ref = range(0,tf,step=dt)
    T_sim = range(0,t_sim,step=dt)

    #############################################
    ## Fit bilinear models 
    #############################################

    # Define basis functions
    eigfuns = ["state", "sine", "cosine", "sine", "sine", "chebyshev"]
    eigorders = [[0],[1],[1],[2],[4],[2, 4]]

    t_train_eDMD = @elapsed model_eDMD = BilinearControl.run_eDMD(X_train, U_train, dt, eigfuns, eigorders, 
        reg=reg, name="cartpole_eDMD")
    t_train_jDMD = @elapsed model_jDMD = BilinearControl.run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom, 
        reg=reg, name="cartpole_jDMD"; α, β, learnB)
    model_eDMD_projected = BilinearControl.ProjectedEDMDModel(model_eDMD)
    model_jDMD_projected = BilinearControl.ProjectedEDMDModel(model_jDMD)

    #############################################
    ## MPC Tracking
    #############################################

    xe = [0,pi,0,0.]
    ue = [0.]
    tf_sim = T_ref[end]*1.5
    Nt = 41  # MPC horizon
    T_sim = range(0,tf_sim,step=dt)
    N_sim = length(T_sim)

    Qmpc = Diagonal(fill(1e-0,4))
    Rmpc = Diagonal(fill(1e-3,1))
    Qfmpc = Diagonal([1e4,1e2,1e1,1e1])

    N_test = size(X_test_swingup,2)
    test_results = map(1:N_test) do i
        X_ref = deepcopy(X_test_swingup_ref[:,i])
        U_ref = deepcopy(U_test_swingup_ref[:,i])
        X_ref[end] .= xe
        push!(U_ref, ue)

        N_ref = length(T_ref)
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
        X_mpc_nom, U_mpc_nom, T_mpc = simulatewithcontroller(dmodel_real, mpc_nom,  X_ref[1], tf_sim, dt, printrate=false)
        X_mpc_eDMD,U_mpc_eDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_eDMD, X_ref[1], tf_sim, dt, printrate=false)
        X_mpc_jDMD,U_mpc_jDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_jDMD, X_ref[1], tf_sim, dt, printrate=false)

        err_nom = norm(X_mpc_nom - X_ref_full) / N_sim
        err_eDMD = norm(X_mpc_eDMD - X_ref_full) / N_sim
        err_jDMD = norm(X_mpc_jDMD - X_ref_full) / N_sim

        (; err_nom, err_eDMD, err_jDMD) #, t_train_eDMD, t_train_jDMD, num_lqr, num_swingup, nsamples=length(X_train)) end
    end

    nom_err_avg  = mean(filter(isfinite, map(x->x.err_nom, test_results)))
    eDMD_err_avg = mean(filter(isfinite, map(x->x.err_eDMD, test_results)))
    jDMD_err_avg = mean(filter(isfinite, map(x->x.err_jDMD, test_results)))
    eDMD_success = count(isfinite, map(x->x.err_eDMD, test_results))
    jDMD_success = count(isfinite, map(x->x.err_jDMD, test_results))

    (;nom_err_avg, eDMD_err_avg, eDMD_success, jDMD_err_avg, jDMD_success, 
        t_train_eDMD, t_train_jDMD, num_lqr, num_swingup, nsamples=length(X_train), 
        )
end

"""
    test_sample_size(;kwargs...)

Train a model with `num_train` LQR stabilization training trajectories, then test it on 
`num_test` initial conditions. Returns the success rate and average terminal error.
"""
function test_sample_size(;
    μ=0.1,
    μ_nom=μ,
    t_sim=4.0,
    num_train=20,
    num_test=10,
    err_thresh=0.1,
    alg=:eDMD,
    α=1e-1,
    reg=1e-6,
    x_window=[1, deg2rad(40), 0.5, 0.5],
    test_window_ratio=1.0,
    lifted=false,
    ρ=1e-6,
    Nt=21,
)

    ## Define the models
    model_nom = BilinearControl.NominalCartpole()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" Cartpole Model
    model_real = BilinearControl.SimulatedCartpole(; μ=μ) # this model has damping
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    # Generate data with the new damping term
    X_train, U_train, _, _, _, _, metadata = generate_cartpole_data(;
        save_to_file=false,
        num_swingup=0,
        num_lqr=num_train,
        μ=μ,
        μ_nom=μ_nom,
        max_lqr_samples=600,
        x_window,
    )
    dt = metadata.dt

    # Train new model
    eigfuns = ["state", "sine", "cosine", "sine", "sine", "chebyshev"]
    eigorders = [[0], [1], [1], [2], [4], [2, 4]]

    model = if alg == :eDMD
        run_eDMD(
            X_train,
            U_train,
            dt,
            eigfuns,
            eigorders;
            reg=reg,
            name="cartpole_eDMD",
            alg=:qr_rls,
        )
    elseif alg == :jDMD
        run_jDMD(
            X_train,
            U_train,
            dt,
            eigfuns,
            eigorders,
            dmodel_nom;
            reg=reg,
            name="cartpole_jDMD",
            learnB=true,
            α=α,
        )
    end

    ## Generate an MPC controller
    mpc = if lifted
        generate_stabilizing_mpc_controller(model, t_sim, dt; Nt, ρ)
    else
        model_projected = BilinearControl.ProjectedEDMDModel(model)
        generate_stabilizing_mpc_controller(model_projected, t_sim, dt; Nt, ρ)
    end
    return test_initial_conditions(
        dmodel_real, mpc, dt; t_sim, x_window, test_window_ratio, num_test, err_thresh
    )
end

"""
    test_initial_conditions(model, controller, dt; kwargs...)

Test the given controller on the "real" system. Runs `num_test` initial conditions sampled 
from `x_window`, scaled by `test_window_ratio`.
"""
function test_initial_conditions(
    model_real,
    ctrl,
    dt;
    t_sim=4.0,
    x_window=[1, deg2rad(30), 0.5, 0.5],
    test_window_ratio=1.0,
    num_test=50,
    err_thresh=0.1,
)

    # Test mpc controller
    # Set seed so that all are tested on the same conditions
    Random.seed!(100)

    # Generate initial conditions to test
    xe = [0, pi, 0, 0]
    x0_sampler = Product(
        collect(
            Uniform(x - dx, x + dx) for (x, dx) in zip(xe, x_window .* test_window_ratio)
        ),
    )
    x0_test = [rand(x0_sampler) for i in 1:num_test]

    # Run controller for each initial condition
    errors = map(x0_test) do x0
        X_sim, = simulatewithcontroller(model_real, ctrl, x0, t_sim, dt)
        return norm(X_sim[end] - xe)
    end
    average_error = mean(filter(x -> x < err_thresh, errors))
    success_rate = count(x -> x < err_thresh, errors) / num_test
    return success_rate, average_error
end

"""
    find_min_sample_to_stabilize(mu_vals, num_train)

Finds the minimum number of training samples (of theose given in `num_train`) to stabilize 
the system for each of the friction coefficients in `mu_vals`. Only runs for the algorithm 
given by `alg` (either `:eDMD`` or `:jDMD`).
"""
function find_min_sample_to_stabilize(
    mu_vals,
    num_train;
    err_thresh=0.1,
    success_rate_thresh=0.95,
    repeats_required=2,
    verbose=true,
    kwargs...,
)
    samples_required = Dict(Pair.(mu_vals, zeros(Int, length(mu_vals))))
    success_counts = Dict(Pair.(mu_vals, zeros(Int, length(mu_vals))))
    mu_vals_remaining = copy(collect(mu_vals))
    inds_lock = ReentrantLock()

    for N in num_train
        inds_to_delete = Int[]
        verbose && println("Testing with $N samples")
        Threads.@threads for i in 1:length(mu_vals_remaining)
            μ = mu_vals_remaining[i]
            success_rate, average_error = test_sample_size(;
                num_train=N, μ, err_thresh, kwargs...
            )

            did_successfully_stabilize =
                (success_rate >= success_rate_thresh) && (average_error < err_thresh)
            if did_successfully_stabilize
                verbose && println("  μ = $μ successfully stabilized with $N samples")
                success_counts[μ] += 1
            else
                samples_required[μ] = 0
                success_counts[μ] = 0
            end
            if success_counts[μ] == 1
                samples_required[μ] = N
            elseif success_counts[μ] > repeats_required
                verbose && println(
                    "  FINISHED: μ = $μ stabilized with $(samples_required[μ]) samples"
                )
                lock(inds_lock)
                push!(inds_to_delete, i)
                unlock(inds_lock)
            end
        end
        deleteat!(mu_vals_remaining, sort!(inds_to_delete))
        if isempty(mu_vals_remaining)
            break
        end
    end
    return samples_required
end

"""
    find_min_sample_to_beat_mpc(num_train, dt; kwargs)

Find the number of samples required be the nominal MPC controller at the cartpole 
stabilization task, using MPC controller.

setting `lifted = true`.

# Keyword Arguments
- `alg` algorithm to use (:eDMD, :jDMD)
- `lifted` used the lifted MPC controller
- `num_test` number of test initial conditions to test`
- `repeats_required` number of successive sequential sample sizes that need to succesfully
   stabilize the system for the sample size to be considered the minimum number of samples.
   Prevents counting extranneous samples sizes that stabilize but larger sample sizes don't.
- `x_window` Vector of widths from which to sample the initial conditions
- `test_window_ratio` Ratio of `x_window` over which to draw the test initial conditions. A 
   value of 1.0 samples from the same distribution. A lower value is more conservative, a 
   value greater than 1.0 means samples may be drawn that are not in the training 
   distribution.
- `t_sim` How long to run the simulation. Trajectories have a final state value less than 
   `error_thresh` to be considered successful.
- `Nt` MPC horizon
"""
function find_min_sample_to_beat_mpc(
    num_train,
    dt;
    num_test=50,
    repeats_required=2,
    success_rate_thresh=0.95,
    x_window=[1, deg2rad(40), 0.5, 0.5],
    test_window_ratio=0.9,  # chosen to give MPC controller a 98% success rate
    t_sim=4.0,
    Nt=21,
    verbose=true,
    kwargs...,
)
    samples_required = 0
    success_counts = 0 
    mu_vals_remaining = 0 

    ## Get the nominal MPC performance
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(BilinearControl.NominalCartpole())
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(BilinearControl.SimulatedCartpole())
    mpc_nom = generate_stabilizing_mpc_controller(dmodel_nom, t_sim, dt; Nt)
    success_rate, err_mpc_nom = test_initial_conditions(
        dmodel_real, mpc_nom, dt; x_window, test_window_ratio, num_test
    )
    verbose && println("Nominal MPC error: ", err_mpc_nom)
    verbose && println("Nominal MPC success rate: ", success_rate)

    for N in num_train
        inds_to_delete = Int[]
        verbose && println("Testing with $N samples. Count = ", success_counts, " / ", repeats_required + 1)
        success_rate, average_error = test_sample_size(; 
            num_train=N, num_test=50, μ_nom=0.0, x_window, test_window_ratio, t_sim, kwargs...
        )
        verbose && println("  got (", success_rate, ", ", average_error, ")")

        did_successfully_stabilize =
            (success_rate >= success_rate_thresh) && (average_error < err_mpc_nom)
        if did_successfully_stabilize
            verbose && println("  successfully stabilized with $N samples")
            success_counts += 1
        else
            samples_required = 0
            success_counts = 0
        end
        if success_counts == 1
            samples_required = N
        elseif success_counts > repeats_required
            verbose && println(
                "  FINISHED: stabilized with $(samples_required) samples"
            )
            break
        end
    end
    return samples_required
end