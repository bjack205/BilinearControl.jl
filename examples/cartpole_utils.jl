using Altro
import TrajectoryOptimization as TO

function gencartpoleproblem(x0=zeros(4), Qv=1e-2, Rv=1e-1, Qfv=1e2, u_bnd=3.0, tf=5.0; 
    dt=0.05, constrained=true, μ=0.0)

    # NOTE: this should exactly match RobotZoo.Cartpole() when μ = 0.0
    model = Problems.NominalCartpole(; μ=μ)
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

function generate_cartpole_data(;num_lqr=50, num_swingup=50, save_to_file=true, 
        μ=0.1, μ_nom=0.0, max_lqr_samples=3*num_lqr,
        x_window = [0.7,deg2rad(45),0.2,0.2]
    )
    #############################################
    ## Define the Models
    #############################################
    # Define Nominal Simulated Cartpole Model
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
    X_train_lqr, U_train_lqr = EDMD.create_data(dmodel_real, ctrl_lqr, 
        x0_sampler, num_train_lqr, xe, t_sim, dt, max_samples=max_lqr_samples
    )
    X_test_lqr, U_test_lqr = EDMD.create_data(dmodel_real, ctrl_lqr, 
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
        jldsave(joinpath(Problems.DATADIR, "cartpole_swingup_data.jld2"); 
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
