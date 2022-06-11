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
include("constants.jl")
const CARTPOLE_RESULTS_FILE = joinpath(Problems.DATADIR, "cartpole_results.jld2")

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

function train_cartpole_models(num_lqr, num_swingup; α=0.5, learnB=true, β=1.0, reg=1e-6)

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
    X_train = [X_train_lqr X_train_swingup]
    U_train = [U_train_lqr U_train_swingup]

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
    ## Fit bilinear models 
    #############################################

    # Define basis functions
    eigfuns = ["state", "sine", "cosine", "sine", "sine", "chebyshev"]
    eigorders = [[0],[1],[1],[2],[4],[2, 4]]

    t_train_eDMD = @elapsed model_eDMD = EDMD.run_eDMD(X_train, U_train, dt, eigfuns, eigorders, 
        reg=reg, name="cartpole_eDMD")
    t_train_jDMD = @elapsed model_jDMD = EDMD.run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom, 
        reg=reg, name="cartpole_jDMD"; α, β, learnB)
    model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
    model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)

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
        model_eDMD, model_jDMD)
end

##
generate_cartpole_data()
num_swingup = 2:2:50
2*120
β = 240 / (38 * 120)
res = train_cartpole_models(0, 50, α=0.5, β=1.0, learnB=true, reg=1e-3)
res = train_cartpole_models(0, 8, α=0.5, β=1.0, learnB=true, reg=1e-4)
# res = train_cartpole_models(0, 50, α=0.5, β=1.0, learnB=false, reg=1e-4)
res.jDMD_err_avg
res.eDMD_err_avg
num_swingup[4]
results = map(num_swingup) do N
    println("\nRunning with N = $N")
    res = train_cartpole_models(0,N, α=0.5, β=1.0, learnB=true, reg=1e-4)
    @show res.jDMD_err_avg
    @show res.eDMD_err_avg
    res
end
jldsave(CARTPOLE_RESULTS_FILE; results)
results

## Process results
using PGFPlotsX
results = load(CARTPOLE_RESULTS_FILE)["results"]
fields = keys(results[1])
res = Dict(Pair.(fields, map(x->getfield.(results, x), fields)))
res
good_inds = 1:18
plot(res[:nsamples][good_inds], res[:t_train_eDMD][good_inds])
plot!(res[:nsamples][good_inds], res[:t_train_jDMD][good_inds])
p = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Number of training samples",
        ylabel = "Training time (sec)",
        legend_pos = "north west",
    },
    PlotInc({no_marks, "very thick", "orange"}, Coordinates(res[:nsamples][good_inds], res[:t_train_eDMD][good_inds])),
    PlotInc({no_marks, "very thick", "cyan"}, Coordinates(res[:nsamples][good_inds], res[:t_train_jDMD][good_inds])),
    Legend(["eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "cartpole_mpc_train_time.tikz"), p, include_preamble=false)

p = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Number of training samples",
        ylabel = "Tracking error",
        ymax=0.2,
    },
    PlotInc({lineopts..., color=color_nominal}, Coordinates(res[:nsamples][good_inds], res[:nom_err_avg][good_inds])),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(res[:nsamples][good_inds], res[:eDMD_err_avg][good_inds])),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(res[:nsamples][good_inds], res[:jDMD_err_avg][good_inds])),
    Legend(["Nominal", "eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "cartpole_mpc_test_error.tikz"), p, include_preamble=false)
