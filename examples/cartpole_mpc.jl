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

include("learned_models/edmd_utils.jl")

## Create function for generating nominal cartpole problem for ALTRO
function gencartpoleproblem(x0=zeros(4), Qv=1e-2, Rv=1e-1, Qfv=1e2, u_bnd=3.0, tf=5.0; 
    dt=0.05, constrained=true)

    model = Problems.NominalCartpole()  # NOTE: this should exactly match RobotZoo.Cartpole()
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

function generate_cartpole_data()
    #############################################
    ## Define the Models
    #############################################
    # Define Nominal Simulated Cartpole Model
    model_nom = Problems.NominalCartpole()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" Cartpole Model
    model_real = Problems.SimulatedCartpole() # this model has damping
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
    num_test_lqr = 10

    # Generate a stabilizing LQR controller about the top
    Qlqr = Diagonal([1.0,10.0,1e-2,1e-2])
    Rlqr = Diagonal([1e-3])
    xe = [0,pi,0,0]
    ue = [0.0]
    ctrl_lqr = LQRController(dmodel_nom, Qlqr, Rlqr, xe, ue, dt)
    T_lqr = range(0, t_sim, step=dt)

    # Sample a bunch of initial conditions for the LQR controller
    x0_sampler = Product([
        Uniform(-0.7,0.7),
        Uniform(pi-pi/4,pi+pi/4),
        Uniform(-.2,.2),
        Uniform(-.2,.2),
    ])
    initial_conditions_train = [rand(x0_sampler) for _ in 1:num_train_lqr]
    initial_conditions_test = [rand(x0_sampler) for _ in 1:num_test_lqr]

    # Create data set
    X_train_lqr, U_train_lqr = create_data(dmodel_real, ctrl_lqr, initial_conditions_train, t_sim, dt)
    X_test_lqr, U_test_lqr = create_data(dmodel_real, ctrl_lqr, initial_conditions_test, t_sim, dt);

    # Make sure they all stabilize
    @test all(x->x<0.1, map(x->norm(x-xe), X_train_lqr[end,:]))
    @test all(x->x<0.1, map(x->norm(x-xe), X_test_lqr[end,:]))

    #############################################
    ## ALTRO Training and Testing Data 
    #############################################
    Random.seed!(1)
    num_train_swingup = 50
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

    X_test_swingup_ref = mapreduce(x->getindex(x,1), hcat, test_trajectories)
    U_test_swingup_ref = mapreduce(x->getindex(x,2), hcat, test_trajectories)
    X_test_swingup = mapreduce(x->getindex(x,3), hcat, test_trajectories)
    U_test_swingup = mapreduce(x->getindex(x,4), hcat, test_trajectories)
    X_test_swingup[end,:]

    ## combine lqr and mpc training data
    X_train = [X_train_lqr X_train_swingup]
    U_train = [U_train_lqr U_train_swingup]

    ## Save generated training and test data
    jldsave(joinpath(Problems.DATADIR, "cartpole_swingup_data.jld2"); 
        X_train_lqr, U_train_lqr,
        X_train_swingup, U_train_swingup,
        X_test_swingup, U_test_swingup, 
        X_test_swingup_ref, U_test_swingup_ref,
        X_test_lqr, U_test_lqr, 
        tf, t_sim, dt
    )
end

function train_cartpole_models(num_lqr, num_swingup)

    #############################################
    ## Define the Models
    #############################################
    # Define Nominal Simulated Cartpole Model
    model_nom = Problems.NominalCartpole()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" Cartpole Model
    model_real = Problems.SimulatedCartpole() # this model has damping
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    #############################################  
    ## Load Training and Test Data
    #############################################  
    altro_lqr_traj = load(joinpath(Problems.DATADIR, "cartpole_swingup_data.jld2"))

    # Training data
    X_train_lqr = altro_lqr_traj["X_train_lqr"][:,1:num_lqr]
    U_train_lqr = altro_lqr_traj["U_train_lqr"][:,1:num_lqr]
    X_train_swingup = altro_lqr_traj["X_train_swingup"][:,1:num_swingup]
    U_train_swingup = altro_lqr_traj["U_train_swingup"][:,1:num_swingup]
    X_train = [X_train_swingup X_train_swingup]
    U_train = [U_train_swingup U_train_swingup]

    # Test data
    X_test_swingup = altro_lqr_traj["X_test_swingup"]
    U_test_swingup = altro_lqr_traj["U_test_swingup"]
    X_test_swingup_ref = altro_lqr_traj["X_test_swingup_ref"]
    U_test_swingup_ref = altro_lqr_traj["U_test_swingup_ref"]

    # Metadata
    tf = altro_lqr_traj["tf"]
    t_sim = altro_lqr_traj["t_sim"]
    dt = altro_lqr_traj["dt"]

    T_ref = range(0,tf,step=dt)
    T_sim = range(0,t_sim,step=dt)

    #############################################
    ## Fit data using NOMINAL EDMD method
    #############################################

    # Define basis functions
    eigfuns = ["state", "sine", "cosine", "sine", "sine", "chebyshev"]
    eigorders = [[0],[1],[1],[2],[4],[2, 4]]

    # Build the data 
    Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders);

    # Learn nominal model
    t_train_eDMD = @elapsed A, B, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
        ["na", "na"]; 
        edmd_weights=[1.0], 
        mapping_weights=[0.0],
        algorithm=:qr
    )
    # Create a sparse version of the G Jacobian
    n0,m = RD.dims(model_nom)  # original dimensions
    n = length(Z_train[1])     # lifted dimension
    G = spdiagm(n0,n,1=>ones(n0)) 
    @test norm(G - g) < 1e-3

    # Create model
    eDMD_data = Dict(
        :A=>A, :B=>B, :C=>C, :g=>g, :t_train=>t_train_eDMD
    )
    model_bilinear_eDMD = EDMDModel(eDMD_data[:A],eDMD_data[:B],eDMD_data[:C],G,kf,dt,"cartpole_eDMD")
    model_bilinear_eDMD_projected = Problems.ProjectedEDMDModel(model_bilinear_eDMD)

    #############################################
    ## Fit data using Jacobian EDMD method
    #############################################

    # Generate Jacobians from nominal model
    n0,m = RD.dims(model_nom)  # original dimensions
    xn = zeros(n0)
    n = length(kf(xn))         # lifted state dimension
    jacobians = map(CartesianIndices(U_train)) do cind
        k = cind[1]
        x = X_train[cind]
        u = U_train[cind]
        z = RD.KnotPoint{n0,m}(x,u,T_sim[k],dt)
        J = zeros(n0,n0+m)
        RD.jacobian!(
            RD.InPlace(), RD.ForwardAD(), dmodel_nom, J, xn, z 
        )
        J
    end
    A_train = map(J->J[:,1:n0], jacobians)
    B_train = map(J->J[:,n0+1:end], jacobians)

    # Convert states to lifted Koopman states
    Y_train = map(kf, X_train)

    # Calculate Jacobian of Koopman transform
    F_train = map(@view X_train[1:end-1,:]) do x
        sparse(ForwardDiff.jacobian(kf, x))
    end

    # Create a sparse version of the G Jacobian
    G = spdiagm(n0,n,1=>ones(n0)) 
    xn .= randn(n0)
    @test G*kf(xn) ≈ xn

    # Build Least Squares Problem
    W,s = BilinearControl.EDMD.build_edmd_data(
        Z_train, U_train, A_train, B_train, F_train, G)

    # Create sparse LLS matrix
    @time Wsparse = sparse(W)
    @show BilinearControl.matdensity(Wsparse)

    # Solve with RLS
    t_train_jDMD = @elapsed x_rls = BilinearControl.EDMD.rls_qr(Vector(s), Wsparse; Q=1e-4)
    E = reshape(x_rls,n,:)

    # Extract out bilinear dynamics
    A = E[:,1:n]
    B = E[:,n .+ (1:m)]
    C = E[:,n+m .+ (1:n*m)]

    C_list = Matrix{Float64}[]
    for i in 1:m
        C_i = C[:, (i-1)*n+1:i*n]
        push!(C_list, C_i)
    end
    C = C_list

    # Create model
    jDMD_data = Dict(
        :A=>A, :B=>B, :C=>C, :g=>g, :t_train=>t_train_jDMD
    )
    model_bilinear_jDMD = EDMDModel(jDMD_data[:A],jDMD_data[:B],jDMD_data[:C],G,kf,dt,"cartpole_jDMD")
    model_bilinear_jDMD_projected = Problems.ProjectedEDMDModel(model_bilinear_jDMD)

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
        mpc_eDMD = TrackingMPC(model_bilinear_eDMD_projected, 
            X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
        )
        mpc_jDMD = TrackingMPC(model_bilinear_jDMD_projected, 
            X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
        )
        X_mpc_nom, U_mpc_nom, T_mpc = simulatewithcontroller(dmodel_real, mpc_nom,  X_ref[1], tf_sim, dt)
        X_mpc_eDMD,U_mpc_eDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_eDMD, X_ref[1], tf_sim, dt)
        X_mpc_jDMD,U_mpc_jDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_jDMD, X_ref[1], tf_sim, dt)

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
        t_train_eDMD, t_train_jDMD, num_lqr, num_swingup, nsamples=length(X_train))
end

generate_cartpole_data()
num_swingup = 6:3:30   # goes singular above 35?
results = map(num_swingup) do N
    println("\n\nRunning with N = $N")
    train_cartpole_models(0,N)
end
const CARTPOLE_RESULTS_FILE = joinpath(Problems.DATADIR, "cartpole_results.jld2")
jldsave(CARTPOLE_RESULTS_FILE; results)

## Process results
results = load(CARTPOLE_RESULTS_FILE)["results"]
fields = keys(results[1])
res = Dict(Pair.(fields, map(x->getfield.(results, x), fields)))
plot(res[:num_swingup], res[:t_train_eDMD])
plot!(res[:num_swingup], res[:t_train_jDMD])

pgfplotsx()
p = plot(res[:nsamples], res[:nom_err_avg], label="Nominal", lw=2,
    xlabel="Number of dynamics samples", ylabel="Tracking Error", legend=:topleft,
    ylims=(0,0.2)
)
plot!(res[:nsamples], res[:eDMD_err_avg], label="eDMD", lw=2)
plot!(p, res[:nsamples], res[:jDMD_err_avg], label="jDMD", lw=2)
res
savefig(p, joinpath(Problems.FIGDIR, "cartpole_mpc_test_error.tikz"))
