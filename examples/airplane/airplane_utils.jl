using ThreadsX
using Rotations
using Altro
using TrajectoryOptimization
using LinearAlgebra
using Random
using Distributions
using StaticArrays
using Rotations
using JLD2
using ProgressMeter
import RobotDynamics as RD
const TO = TrajectoryOptimization

const AIRPLANE_DATAFILE = joinpath(BilinearControl.DATADIR, "airplane_trajectory_data.jld2")
const AIRPLANE_MODELFILE = joinpath(BilinearControl.DATADIR, "airplane_trained_models.jld2")
const AIRPLANE_RESULTS = joinpath(BilinearControl.DATADIR, "airplane_results.jld2")

function AirplaneProblem(;dt=0.05, dp=zeros(3), tf=2.0, Qv=10.0, Qw=Qv, pf=[5,0,1.5])
    # Discretization
    model = BilinearControl.SimulatedAirplane()
    # model = BilinearControl.NominalAirplane()
    N = round(Int, tf/dt) + 1
    dt = tf / (N-1)

    # Initial condition
    p0 = MRP(0.997156, 0., 0.075366) # initial orientation
    x0     = [-5,0,1.5, Rotations.params(p0)..., 5,0,0, 0,0,0]
    u_trim = [41.66667789082778, 105.99999999471807, 74.65179381344494, 106.00000124622453]

    # Final condition
    xf = copy(x0)
    xf[1:3] .= pf
    xf[7] = 0.0

    # Shift initial position
    x0[1:3] .+= dp

    # Objective
    Qf = Diagonal([fill(1.0, 3); fill(1.0, 3); fill(Qv, 3); fill(Qw, 3)])
    Q  = Diagonal([fill(1e-2, 3); fill(1e-2, 3); fill(1e-1, 3); fill(1e-1, 3)])
    R = Diagonal(fill(1e-3,4))
    obj = TO.LQRObjective(Q,R,Qf,xf,N, uf=u_trim)

    # Constraint
    n,m = RD.dims(model)
    constraints = TO.ConstraintList(n,m,N)
    goalcon = GoalConstraint(xf, SA[1,2,3])
    add_constraint!(constraints, goalcon, N)

    U0 = [copy(u_trim) for k = 1:N-1]
    prob = Problem(model,obj,x0,tf; constraints, U0)
    rollout!(prob)
    prob
end

function airplane_kf(x)
    p = x[1:3]
    q = x[4:6]
    mrp = MRP(x[4], x[5], x[6])
    R = Matrix(mrp)
    v = x[7:9]
    w = x[10:12]
    α = atan(v[3],v[1])  # angle of attack
    β = atan(v[2],v[1])  # side slip
    vbody = R'v
    speed = vbody'vbody
    [1; x; vec(R); vbody; speed; sin.(p); α; β; α^2; β^2; α^3; β^3; p × v; p × w; 
        w × w; 
        BilinearControl.chebyshev(x, order=[3,4])]
end

function gen_airplane_data(;num_train=30, num_test=10, dt=0.05, dp_window=[1.0,3.0,2.0], 
        save_to_file=true)
    ## Define nominal and true models
    model_nom = BilinearControl.NominalAirplane()
    model_real = BilinearControl.SimulatedAirplane()
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
    prog = Progress(num_train + num_test, desc="Airplane Trajectory", showspeed=true, dt=0.01)
    plane_data = ThreadsX.map(1:num_train+num_test) do i
        # println("Generating trajectory $i / $(num_train + num_test)")
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

            mpc = BilinearControl.LinearMPC(dmodel_nom, Xref, Uref, Tref, Qk, Rk, Qf; Nt=Nt,
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

        next!(prog)
        Vector.(Xsim), Vector.(Usim), Vector.(Xref), Vector.(Uref), Vector(Tref)
    end
    T_ref = range(0,tf,step=dt)

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

function train_airplane(num_train; α=0.1)
    # Get training data
    airplane_data = load(AIRPLANE_DATAFILE)
    good_cols = findall(x->isfinite(norm(x)), eachcol(airplane_data["X_train"]))
    X_train = airplane_data["X_train"][:,good_cols[1:num_train]]
    U_train = airplane_data["U_train"][:,good_cols[1:num_train]]
    T_ref = airplane_data["T_ref"]
    dt = T_ref[2]

    # Get nominal model
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(BilinearControl.NominalAirplane())

    ## Train models
    model_eDMD = run_eDMD(X_train, U_train, dt, airplane_kf, nothing; 
        alg=:qr, showprog=false, reg=1e-6
    )
    model_jDMD = run_jDMD(X_train, U_train, dt, airplane_kf, nothing,
        dmodel_nom; showprog=false, verbose=false, reg=1e-6, alg=:qr, α=α
    )
    model_eDMD, model_jDMD
end

function jacobian_error(model, X, U, T)
    model_real = BilinearControl.SimulatedAirplane()
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)
    h = T[2] - T[1]

    n = length(X[1])
    m = length(U[1])

    jerr = map(zip(X,U,T)) do (x,u,t)
        xn = zero(x)
        z = RD.KnotPoint{n,m}(x,u,t,h)
        Jreal = zeros(n,n+m)
        Jnom = zeros(n,n+m)
        RD.jacobian!(
            RD.InPlace(), RD.ForwardAD(), model, Jnom, xn, z 
        )
        RD.jacobian!(
            RD.InPlace(), RD.ForwardAD(), dmodel_real, Jreal, xn, z 
        )
        Jnom - Jreal
    end
    norm(jerr) / length(jerr)
end

function test_airplane(model_eDMD, model_jDMD)
    # Models
    model_nom = BilinearControl.NominalAirplane()
    model_real = BilinearControl.SimulatedAirplane()
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
    jerr_nom = zeros(num_test) 
    jerr_eDMD = zeros(num_test) 
    jerr_jDMD = zeros(num_test) 
    model_eDMD_projected = BilinearControl.ProjectedEDMDModel(model_eDMD)
    model_jDMD_projected = BilinearControl.ProjectedEDMDModel(model_jDMD)

    # Run MPC on each trajectory
    for i = 1:num_test
        X_ref = X_ref0[:,i]
        U_ref = U_ref0[:,i]
        N = length(X_ref)

        mpc_nom = BilinearControl.LinearMPC(dmodel_nom, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
            xmin,xmax,umin,umax
        )
        mpc_eDMD = BilinearControl.LinearMPC(model_eDMD_projected, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
            xmin,xmax,umin,umax
        )
        mpc_jDMD = BilinearControl.LinearMPC(model_jDMD_projected, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
            xmin,xmax,umin,umax
        )

        # Tracking error
        X_nom,  = simulatewithcontroller(dmodel_real, mpc_nom,  X_ref[1], t_ref, dt)
        X_eDMD, = simulatewithcontroller(dmodel_real, mpc_eDMD, X_ref[1], t_ref, dt)
        X_jDMD, = simulatewithcontroller(dmodel_real, mpc_jDMD, X_ref[1], t_ref, dt)

        err_nom[i] = norm(X_nom - X_ref) / N
        err_eDMD[i] = norm(X_eDMD - X_ref) / N
        err_jDMD[i] = norm(X_jDMD - X_ref) / N

        # Evaluate Jacobian error
        jerr_nom[i] = jacobian_error(dmodel_nom, X_ref, U_ref, T_ref)
        jerr_eDMD[i] = jacobian_error(model_eDMD_projected, X_ref, U_ref, T_ref)
        jerr_jDMD[i] = jacobian_error(model_jDMD_projected, X_ref, U_ref, T_ref)
    end
    Dict(
        :nominal=>err_nom, :eDMD=>err_eDMD, :jDMD=>err_jDMD,
        :jerr_nominal=>jerr_nom, :jerr_eDMD=>jerr_eDMD, :jerr_jDMD=>jerr_jDMD,
    )
end

function test_airplane_open_loop(model_eDMD, model_jDMD)
    # Models
    model_nom = BilinearControl.NominalAirplane()
    model_real = BilinearControl.SimulatedAirplane()
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
    model_eDMD_projected = BilinearControl.ProjectedEDMDModel(model_eDMD)
    model_jDMD_projected = BilinearControl.ProjectedEDMDModel(model_jDMD)

    # Run MPC on each trajectory
    for i = 1:num_test
        X_ref = X_ref0[:,i]
        U_ref = U_ref0[:,i]
        N = length(X_ref)

        X_eDMD, = simulate(model_eDMD_projected, U_ref, X_ref[1], t_ref, dt)
        X_jDMD, = simulate(model_jDMD_projected, U_ref, X_ref[1], t_ref, dt)

        err_eDMD[i] = X_eDMD[end] - X_ref[end] / N
        err_jDMD[i] = X_jDMD[end] - X_ref[end] / N
    end
    Dict(:nominal=>err_nom, :eDMD=>err_eDMD, :jDMD=>err_jDMD)
end