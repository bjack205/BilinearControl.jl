using ThreadsX
using Rotations
using Altro
using TrajectoryOptimization
const TO = TrajectoryOptimization

function gen_airplane_data(;num_train=30, num_test=10, dt=0.05, dp_window=[1.0,3.0,2.0], 
        save_to_file=true)
    ## Define nominal and true models
    model_nom = Problems.NominalAirplane()
    model_real = Problems.SimulatedAirplane()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    ## Get trajectories using ALTRO

    # General Parameters
    tf = 2.0
    pf = [5,0,1.5]  # final position

    # MPC Parameters
    Nt = 21
    Qk = Diagonal([fill(1e0, 3); fill(1e1, 3); fill(1e-1, 3); fill(2e-1, 3)])
    Rk = Diagonal(fill(1e-3,4))
    Qf = Diagonal([fill(1e-2, 3); fill(1e0, 3); fill(1e1, 3); fill(1e1, 3)]) * 10
    u_trim = [41.66667789082778, 105.99999999471807, 74.65179381344494, 106.00000124622453]
    xmax = [fill(0.5,3); fill(1.0, 3); fill(0.5, 3); fill(10.0, 3)]
    xmin = -xmax
    umin = fill(0.0, 4) - u_trim
    umax = fill(255.0, 4) - u_trim

    ## Sample ALTRO trajectories
    Random.seed!(2)
    dp_sampler = Product(collect(Uniform(-x,+x) for x in dp_window))
    max_attempts = 5
    plane_data = ThreadsX.map(1:num_train+num_test) do i
        println("Generating trajectory $i / $(num_train + num_test)")
        Xref = Vector{Float64}[]
        Uref = Vector{Float64}[]
        Tref = Vector{Float64}()
        Xsim = Vector{Float64}[]
        Usim = Vector{Float64}[]
        for i = 1:max_attempts
            dp = rand(dp_sampler)
            prob = AirplaneProblem(;tf, dt, Qv=15, Qw=5, dp, pf)
            solver = ALTROSolver(prob, verbose=0, show_summary=false)
            solve!(solver)

            if Altro.status(solver) != Altro.SOLVE_SUCCEEDED
                continue
            end

            Xref = Vector.(TO.states(solver))
            Uref = Vector.(TO.controls(solver))
            Tref = Vector(range(0,tf,step=dt))

            mpc = EDMD.LinearMPC(dmodel_nom, Xref, Uref, Tref, Qk, Rk, Qf; Nt=Nt,
                xmin,xmax,umin,umax
            )
            Xsim,Usim,Tsim = simulatewithcontroller(dmodel_real, mpc, Xref[1], Tref[end], Tref[2])
            if norm(Xsim[end][1:3] - pf) < 10.0
                break
            end
            if i == max_attempts
                @warn "Couldn't find a good trajectory in $max_attempts attempts"
            end
        end

        Vector.(Xsim), Vector.(Usim), Vector.(Xref), Vector.(Uref), Vector(Tref)
    end
    T_ref = range(0,tf,step=dt)

    # println("Running MPC controller")
    # mpc_trajectories = ThreadsX.map(1:num_train+num_test) do i
    #     X_ref,U_ref = reference_trajectories[i]

    #     mpc = EDMD.LinearMPC(dmodel_nom, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
    #         xmin,xmax,umin,umax
    #     )
    #     X_sim,U_sim,T_sim = simulatewithcontroller(dmodel_real, mpc, X_ref[1], T_ref[end], T_ref[2])
    #     X_sim,U_sim
    # end
    X_mpc = mapreduce(x->getindex(x,1), hcat, plane_data)
    U_mpc = mapreduce(x->getindex(x,2), hcat, plane_data)
    X_ref = mapreduce(x->getindex(x,3), hcat, plane_data)
    U_ref = mapreduce(x->getindex(x,4), hcat, plane_data)
    X_train = X_mpc[:,1:num_train]
    U_train = U_mpc[:,1:num_train]
    X_test = X_mpc[:,num_test .+ (1:num_test)]
    U_test = U_mpc[:,num_test .+ (1:num_test)]

    if save_to_file
        jldsave(AIRPLANE_DATAFILE; 
            X_train, U_train, X_test, U_test, X_ref, U_ref, T_ref,
            u_trim, pf, dp_window
        )
    end
    X_mpc,U_mpc, X_ref,U_ref
end

function train_airplane(num_train)
    # Get training data
    airplane_data = load(AIRPLANE_DATAFILE)
    good_cols = findall(x->isfinite(norm(x)), eachcol(airplane_data["X_train"]))
    X_train = airplane_data["X_train"][:,good_cols[1:num_train]]
    U_train = airplane_data["U_train"][:,good_cols[1:num_train]]
    T_ref = airplane_data["T_ref"]
    dt = T_ref[2]

    # Get nominal model
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(Problems.NominalAirplane())

    ## Train models
    model_eDMD = run_eDMD(X_train, U_train, dt, airplane_kf, nothing; 
        alg=:qr, showprog=false, reg=1e-6
    )
    model_jDMD = run_jDMD(X_train, U_train, dt, airplane_kf, nothing,
        dmodel_nom; showprog=false, verbose=false, reg=1e-6, alg=:qr, α=0.1
    )
    model_eDMD, model_jDMD
end

function test_airplane(model_eDMD, model_jDMD)
    # Models
    model_nom = Problems.NominalAirplane()
    model_real = Problems.SimulatedAirplane()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    # MPC parameters
    Nt = 21
    Qk = Diagonal([fill(1e0, 3); fill(1e1, 3); fill(1e-1, 3); fill(2e-1, 3)])
    Rk = Diagonal(fill(1e-3,4))
    Qf = Diagonal([fill(1e-2, 3); fill(1e0, 3); fill(1e1, 3); fill(1e1, 3)]) * 10
    u_trim = [41.66667789082778, 105.99999999471807, 74.65179381344494, 106.00000124622453]
    xmax = [fill(0.5,3); fill(1.0, 3); fill(0.5, 3); fill(10.0, 3)]
    xmin = -xmax
    umin = fill(0.0, 4) - u_trim
    umax = fill(255.0, 4) - u_trim

    # Get test data
    airplane_data = load(AIRPLANE_DATAFILE)
    X_test = airplane_data["X_test"]
    X_train = airplane_data["X_train"]
    num_train = size(X_train,2)
    num_test =  size(X_test,2)

    X_ref0 = airplane_data["X_ref"][:,num_train+1:end]
    U_ref0 = airplane_data["U_ref"][:,num_train+1:end]
    T_ref = airplane_data["T_ref"]
    dt = T_ref[2]
    t_ref = T_ref[end]

    # Allocate result vectors
    err_nom = zeros(num_test) 
    err_eDMD = zeros(num_test) 
    err_jDMD = zeros(num_test) 
    model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
    model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)

    # Run MPC on each trajectory
    Threads.@threads for i = 1:num_test
        X_ref = X_ref0[:,i]
        U_ref = U_ref0[:,i]
        N = length(X_ref)

        mpc_nom = EDMD.LinearMPC(dmodel_nom, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
            xmin,xmax,umin,umax
        )
        mpc_eDMD = EDMD.LinearMPC(model_eDMD_projected, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
            xmin,xmax,umin,umax
        )
        mpc_jDMD = EDMD.LinearMPC(model_jDMD_projected, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
            xmin,xmax,umin,umax
        )

        X_nom,  = simulatewithcontroller(dmodel_real, mpc_nom,  X_ref[1], t_ref, dt)
        X_eDMD, = simulatewithcontroller(dmodel_real, mpc_eDMD, X_ref[1], t_ref, dt)
        X_jDMD, = simulatewithcontroller(dmodel_real, mpc_jDMD, X_ref[1], t_ref, dt)
        err_nom[i] = norm(X_nom - X_ref) / N
        err_eDMD[i] = norm(X_eDMD - X_ref) / N
        err_jDMD[i] = norm(X_jDMD - X_ref) / N
    end
    Dict(:nominal=>err_nom, :eDMD=>err_eDMD, :jDMD=>err_jDMD)
end
