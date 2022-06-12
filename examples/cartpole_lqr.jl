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
using Test
using PGFPlotsX
using StaticArrays 
import BilinearControl.Problems
using ProgressMeter

include("constants.jl")
include("cartpole_utils.jl")
const CARTPOLE_LQR_RESULTS_FILE = joinpath(Problems.DATADIR, "cartpole_lqr_results.jld2")

## Visualizer
model = RobotZoo.Cartpole()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
delete!(vis)
set_cartpole!(vis)
render(vis)


#############################################
## Generate Training and Test Data 
#############################################
tf = 2.0
dt = 0.02

# Generate Data From Mismatched Model
Random.seed!(1)

# Number of trajectories
num_test = 50
num_train = 50

# Generate a stabilizing LQR controller about the top
Qlqr = Diagonal([0.2,10,1e-2,1e-2])
Rlqr = Diagonal([1e-3])
xe = [0,pi,0,0]
ue = [0.0]
ctrl_lqr = LQRController(dmodel_real, Qlqr, Rlqr, xe, ue, dt)

# Sample a bunch of initial conditions for the LQR controller
x0_sampler = Product([
    Uniform(-0.7,0.7),
    Uniform(pi-pi/4,pi+pi/4),
    Uniform(-.2,.2),
    Uniform(-.2,.2),
])

initial_conditions_lqr = [rand(x0_sampler) for _ in 1:num_test]
initial_conditions_test = [rand(x0_sampler) for _ in 1:num_train]

# Create data set
X_train, U_train = create_data(dmodel_real, ctrl_lqr, initial_conditions_lqr, tf, dt)
X_test, U_test = create_data(dmodel_real, ctrl_lqr, initial_conditions_test, tf, dt)

#############################################
## Test Models 
#############################################

function get_success_rate(model, controller, xg, ics, tf, dt, thresh)
    num_success = 0
    avg_error = 0.0
    for x0 in ics
        X_sim, = simulatewithcontroller(model, controller, x0, tf, dt)
        err = norm(X_sim[end] - xg)
        is_success = err < thresh
        num_success += is_success 
        is_success && (avg_error += err)
    end
    return num_success / length(ics), avg_error / num_success
end

function generate_stabilizing_mpc_controller(model, t_sim, dt; 
        Nt=41, ρ=1e-6, 
        Qmpc = Diagonal(fill(1e-0,4)),
        Rmpc = Diagonal(fill(1e-3,1)),
        Qfmpc = Diagonal([1e2,1e2,1e1,1e1]),
    )
    xe = [0,pi,0,0]
    ue = [0.]
    ye = EDMD.expandstate(model, xe)
    lifted_state_error(x,x0) = model_eDMD.kf(x) - x0

    # Reference Trajectory
    T_sim = range(0,t_sim,step=dt)
    X_ref = [copy(xe) for t in T_sim]
    U_ref = [copy(ue) for t in T_sim]
    T_ref = copy(T_sim)
    Y_ref = map(x->EDMD.expandstate(model,x), X_ref)

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

function test_cartpole_controller(X_train, U_train, dt, num_train; 
        num_test=100, run_lqr=true, run_mpc=true, t_sim = 4.0
    )

    # Nominal Simulated Cartpole Model
    model_nom = RobotZoo.Cartpole(mc=1.0, mp=0.2, l=0.5)
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Mismatched "Real" Cartpole Model
    model_real = Cartpole2(mc=1.05, mp=0.19, l=0.52, b=0.02)  # this model has damping
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

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
    ## Train models 
    ############################################# 
    eigfuns = ["state", "sine", "cosine", "sine"]
    eigorders = [[0],[1],[1],[2],[4],[2, 4]]

    model_eDMD = run_eDMD(X_train[:,1:num_train], U_train[:,1:num_train], dt, eigfuns, eigorders, reg=1e-6, name="cartpole_eDMD")
    model_jDMD = run_jDMD(X_train[:,1:num_train], U_train[:,1:num_train], dt, eigfuns, eigorders, dmodel_nom, 
        reg=1e-6, name="cartpole_jDMD", α=0.5)

    println("New state dimension: ", RD.state_dim(model_eDMD))

    ############################################# 
    ## Generate LQR Controllers
    ############################################# 
    # Equilibrium position
    xe = [0,pi,0,0.]
    ue = [0.0]
    ye = EDMD.expandstate(model_eDMD, xe)
    ρ = 1e-6 

    # Cost function
    Qlqr = Diagonal([1.0,1.0,1e-2,1e-2])
    Rlqr = Diagonal([1e-3])
    Qlqr = Diagonal(fill(1e-0,4))
    Rlqr = Diagonal(fill(1e-3,1))

    lifted_state_error(x,x0) = model_eDMD.kf(x) - x0

    # Initial Conditions to test
    x0_sampler = Product([
        Uniform(-1.0,1.0),
        Uniform(pi-deg2rad(50),pi+deg2rad(50)),
        Uniform(-.5,.5),
        Uniform(-.5,.5),
    ])
    Random.seed!(100)
    x0_test = [rand(x0_sampler) for i = 1:num_test]

    if run_lqr

        Qlqr_lifted = Diagonal([ρ; diag(Qlqr); fill(ρ, length(ye) - 5)])

        # Nominal LQR Controller
        lqr_nominal = LQRController(dmodel_nom, Qlqr, Rlqr, xe, ue, dt, max_iters=10000)

        # Projected LQR Controllers
        model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
        model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)
        lqr_eDMD_projected = LQRController(model_eDMD_projected, Qlqr, Rlqr, xe, ue, dt, max_iters=10000)
        lqr_jDMD_projected = LQRController(model_jDMD_projected, Qlqr, Rlqr, xe, ue, dt, max_iters=10000)

        # Lifted LQR Controllers
        lqr_jDMD = LQRController(
            model_jDMD, Qlqr_lifted, Rlqr, ye, ue, dt, max_iters=20000,
            state_error=lifted_state_error
        )
        lqr_eDMD = LQRController(
            model_eDMD, Qlqr_lifted, Rlqr, ye, ue, dt, max_iters=10000,
            state_error=lifted_state_error
        )

        ## Run each controller on the same set of initial conditions

        println("  testing LQR controllers...")
        res_lqr_nom = get_success_rate(dmodel_real, lqr_nominal, xe, x0_test, t_sim, dt, 0.1)
        res_lqr_eDMD_projected = get_success_rate(dmodel_real, lqr_eDMD_projected, xe, x0_test, t_sim, dt, 0.1)
        res_lqr_jDMD_projected = get_success_rate(dmodel_real, lqr_jDMD_projected, xe, x0_test, t_sim, dt, 0.1)
        res_lqr_eDMD = get_success_rate(dmodel_real, lqr_eDMD, xe, x0_test, t_sim, dt, 0.1)
        res_lqr_jDMD = get_success_rate(dmodel_real, lqr_jDMD, xe, x0_test, t_sim, dt, 0.1)

        success_rate_lqr = (;
            nom = res_lqr_nom[1], 
            eDMD_projected = res_lqr_eDMD_projected[1],
            jDMD_projected = res_lqr_jDMD_projected[1],
            eDMD = res_lqr_eDMD[1],
            jDMD = res_lqr_jDMD[1],
        )
        average_error_lqr = (;
            nom = res_lqr_nom[2],
            eDMD_projected = res_lqr_eDMD_projected[2],
            jDMD_projected = res_lqr_jDMD_projected[2],
            eDMD = res_lqr_eDMD[2],
            jDMD = res_lqr_jDMD[2],
        )
    else
        success_rate_lqr = ()
        average_error_lqr = ()
    end

    #############################################
    ## MPC Controllers
    #############################################
    if run_mpc
        # Reference Trajectory
        T_sim = range(0,t_sim,step=dt)
        X_ref = [copy(xe) for t in T_sim]
        U_ref = [copy(ue) for t in T_sim]
        T_ref = copy(T_sim)
        Y_ref = model_eDMD.kf.(X_ref)
        Nt = 41

        # Objective
        Qmpc = copy(Qlqr)
        Rmpc = copy(Rlqr)
        Qfmpc = 100*Qmpc

        Qmpc = Diagonal(fill(1e-0,4))
        Rmpc = Diagonal(fill(1e-3,1))
        Qfmpc = Diagonal([1e2,1e2,1e1,1e1])
        Qmpc_lifted = Diagonal([ρ; diag(Qmpc); fill(ρ, length(ye)-5)])
        Qfmpc_lifted = Diagonal([ρ; diag(Qfmpc); fill(ρ, length(ye)-5)])

        # Nominal MPC controller
        mpc_nominal = TrackingMPC(dmodel_nom, 
            X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
        )

        # Projected MPC controllers
        # model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
        # model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)
        mpc_eDMD_projected, mpc_eDMD = generate_mpc_controllers(model_eDMD, t_sim, dt; Nt, ρ)
        mpc_jDMD_projected, mpc_jDMD = generate_mpc_controllers(model_jDMD, t_sim, dt; Nt, ρ)

        # mpc_eDMD_projected = TrackingMPC(model_eDMD_projected, 
        #     X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
        # )
        # mpc_jDMD_projected = TrackingMPC(model_jDMD_projected, 
        #     X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
        # )

        # # Lifted MPC controllers
        # mpc_eDMD = TrackingMPC(model_eDMD, 
        #     Y_ref, U_ref, Vector(T_ref), Qmpc_lifted, Rmpc, Qfmpc_lifted; Nt=Nt, state_error=lifted_state_error
        # )
        # mpc_jDMD = TrackingMPC(model_jDMD, 
        #     Y_ref, U_ref, Vector(T_ref), Qmpc_lifted, Rmpc, Qfmpc_lifted; Nt=Nt, state_error=lifted_state_error
        # )

        ## Run the controllers
        println("  testing nominal MPC...")
        res_mpc_nom = get_success_rate(dmodel_real, mpc_nominal, xe, x0_test, t_sim, dt, 0.1)
        println("  testing projected MPCcontrollers...")
        res_mpc_eDMD_projected = get_success_rate(dmodel_real, mpc_eDMD_projected, xe, x0_test, t_sim, dt, 0.1)
        res_mpc_jDMD_projected = get_success_rate(dmodel_real, mpc_jDMD_projected, xe, x0_test, t_sim, dt, 0.1)
        println("  testing lifted MPC controllers...")
        res_mpc_eDMD = get_success_rate(dmodel_real, mpc_eDMD, xe, x0_test, t_sim, dt, 0.1)
        res_mpc_jDMD = get_success_rate(dmodel_real, mpc_jDMD, xe, x0_test, t_sim, dt, 0.1)

        success_rate_mpc = (;
            nom = res_mpc_nom[1], 
            eDMD_projected = res_mpc_eDMD_projected[1],
            jDMD_projected = res_mpc_jDMD_projected[1],
            eDMD = res_mpc_eDMD[1],
            jDMD = res_mpc_jDMD[1],
        )
        average_error_mpc = (;
            nom = res_mpc_nom[2],
            eDMD_projected = res_mpc_eDMD_projected[2],
            jDMD_projected = res_mpc_jDMD_projected[2],
            eDMD = res_mpc_eDMD[2],
            jDMD = res_mpc_jDMD[2],
        )
    else
        success_rate_lqr = ()
        average_error_lqr = ()
    end

    success_rate_lqr, average_error_lqr, success_rate_mpc, average_error_mpc
end

## WARNING: This takes a long time to compute! (about 20 minutes)
num_train = [1:10; 20:30; 40:50]
results = map(num_train) do N
    println("Running test with N = $N")
    test_cartpole_controller(X_train, U_train, dt, N)
end
jldsave(CARTPOLE_LQR_RESULTS_FILE; results)
results
results_og = deepcopy(results)

## Generate plot
results = load(CARTPOLE_LQR_RESULTS_FILE)["results"]

err_lqr_nom = map(x->x[2].nom, results)
err_lqr_eDMD = map(x->x[2].eDMD, results)
err_lqr_jDMD = map(x->x[2].jDMD, results)
err_lqr_eDMD_projected = map(x->x[2].eDMD_projected, results)
err_lqr_jDMD_projected = map(x->x[2].jDMD_projected, results)

err_mpc_nom = map(x->x[4].nom, results)
err_mpc_eDMD = map(x->x[4].eDMD, results)
err_mpc_jDMD = map(x->x[4].jDMD, results)
err_mpc_eDMD_projected = map(x->x[4].eDMD_projected, results)
err_mpc_jDMD_projected = map(x->x[4].jDMD_projected, results)

eDMD_projected_samples = num_train[findfirst((t)->(t[1]<t[2]), collect(zip(err_mpc_eDMD_projected, err_mpc_nom)))]
jDMD_projected_samples = num_train[findfirst((t)->(t[1]<t[2]), collect(zip(err_mpc_jDMD_projected, err_mpc_nom)))]

eDMD_samples = num_train[findfirst((t)->(t[1]<t[2]), collect(zip(err_mpc_eDMD, err_mpc_nom)))]
jDMD_samples = num_train[findfirst((t)->(t[1]<t[2]), collect(zip(err_mpc_jDMD, err_mpc_nom)))]

plot(num_train, err_mpc_nom, c=:black)
plot!(num_train, err_mpc_eDMD_projected)
plot!(num_train, err_mpc_jDMD_projected)

plot(num_train, err_lqr_nom, c=:black)
plot!(num_train, err_lqr_eDMD_projected)
plot!(num_train, err_lqr_jDMD_projected)

p_bar = @pgf Axis(
    {
        reverse_legend,
        width="4in",
        height="4cm",
        xbar,
        ytick="data",
        yticklabels={"Projected", "Lifted"},
        enlarge_y_limits = 0.7,
        legend_pos = "south east",
        xlabel = "Training trajectories Rqd to Beat Nominal MPC",
        nodes_near_coords
    },
    PlotInc({no_marks, color=color_jDMD}, Coordinates([jDMD_projected_samples,jDMD_samples], [0,1])),
    PlotInc({no_marks, color=color_eDMD}, Coordinates([eDMD_projected_samples,eDMD_samples], [0,1])),
    Legend(["jDMD", "eDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "cartpole_lqr_samples.tikz"), p_bar, include_preamble=false)

#############################################
## Get controller rates
#############################################
eigfuns = ["state", "sine", "cosine", "sine"]
eigorders = [[0],[1],[1],[2],[4],[2, 4]]
model_eDMD = run_eDMD(X_train[:,1:10], U_train[:,1:10], dt, eigfuns, eigorders, reg=1e-6, name="cartpole_eDMD")
mpc_projected, mpc = generate_mpc_controllers(model_eDMD, 4, dt)
simulatewithcontroller(dmodel_real, mpc_projected, [0,pi-deg2rad(10),0,0], 4.0, dt, printrate=true)
simulatewithcontroller(dmodel_real, mpc, [0,pi-deg2rad(10),0,0], 4.0, dt, printrate=true)

#############################################
## Test MPC controllers
#############################################
function test_mpc_stabilization_controllers(X_train0, U_train0, t_train, dt; num_train=50, num_test=10)

# num_test=10
# num_train=50
    X_train = X_train0[:,1:num_train]
    U_train = U_train0[:,1:num_train]

    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(Problems.NominalCartpole())
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(Problems.SimulatedCartpole())

    eigfuns = ["state", "sine", "cosine", "sine"]
    # eigfuns = ["state", "sine", "cosine", "sine", "sine", "chebyshev"]
    eigorders = [[0],[1],[1],[2],[4],[2, 4]]

    model_eDMD = run_eDMD(X_train, U_train, dt, eigfuns, eigorders, reg=1e-6, name="cartpole_eDMD", alg=:qr_rls)
    model_jDMD = run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom,
        reg=1e-6, name="cartpole_jDMD", α=5e-1, learnB=true)
    model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
    model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)

    # Generate MPC controllers
    xe = [0,pi,0,0]
    Qmpc = Diagonal(fill(1e-0,4))
    Rmpc = Diagonal(fill(1e-3,1))
    Qfmpc = Diagonal([1e2,1e2,1e1,1e1])
    Nt = 21   # if horizon it too long the lifted controllers have numerical problems
    ρ = 0e-6
    t_sim = 4.0

    # Generate MPC Controllers 
    mpc_nominal = generate_stabilizing_mpc_controller(dmodel_nom, t_sim, dt; Nt, ρ)
    mpc_eDMD_projected = generate_stabilizing_mpc_controller(model_eDMD_projected, t_sim, dt; Nt, ρ)
    mpc_jDMD_projected = generate_stabilizing_mpc_controller(model_jDMD_projected, t_sim, dt; Nt, ρ)
    mpc_eDMD = generate_stabilizing_mpc_controller(model_eDMD, t_sim, dt; Nt, ρ)
    mpc_jDMD = generate_stabilizing_mpc_controller(model_jDMD, t_sim, dt; Nt, ρ)

    x0 = [0.1,pi,0,0]
    X_eDMD,_,T_sim = simulatewithcontroller(dmodel_real, mpc_eDMD_projected, x0, t_sim, dt)
    X_jDMD,_,T_sim = simulatewithcontroller(dmodel_real, mpc_jDMD_projected, x0, t_sim, dt)
    X_eDMD,_,T_sim = simulatewithcontroller(dmodel_real, mpc_eDMD, x0, t_sim, dt)
    X_jDMD,_,T_sim = simulatewithcontroller(dmodel_real, mpc_jDMD, x0, t_sim, dt)
    plotstates(T_sim, X_eDMD, inds=1:2)
    plotstates!(T_sim, X_jDMD, inds=1:2)

    # Test initial conditions
    x0_sampler = Product([
        Uniform(-1.0,1.0),
        Uniform(pi-deg2rad(30),pi+deg2rad(30)),
        Uniform(-.5,.5),
        Uniform(-.5,.5),
    ])
    Random.seed!(100)
    x0_test = [rand(x0_sampler) for i = 1:num_test]

    res_mpc_nom = get_success_rate(dmodel_real, mpc_nominal, xe, x0_test, t_sim, dt, 1e-1)
    res_mpc_eDMD_projected = get_success_rate(dmodel_real, mpc_eDMD_projected, xe, x0_test, t_sim, dt, 1e-1)
    res_mpc_jDMD_projected = get_success_rate(dmodel_real, mpc_jDMD_projected, xe, x0_test, t_sim, dt, 1e-1)
    res_mpc_eDMD = get_success_rate(dmodel_real, mpc_eDMD, xe, x0_test, t_sim, dt, 1e-1)
    res_mpc_jDMD = get_success_rate(dmodel_real, mpc_jDMD, xe, x0_test, t_sim, dt, 1e-1)

    success_rate = (;
        nom = res_mpc_nom[1], 
        eDMD_projected = res_mpc_eDMD_projected[1],
        jDMD_projected = res_mpc_jDMD_projected[1],
        eDMD = res_mpc_eDMD[1],
        jDMD = res_mpc_jDMD[1],
    )
    average_error = (;
        nom = res_mpc_nom[2],
        eDMD_projected = res_mpc_eDMD_projected[2],
        jDMD_projected = res_mpc_jDMD_projected[2],
        eDMD = res_mpc_eDMD[2],
        jDMD = res_mpc_jDMD[2],
    )
    success_rate, average_error
end
# num_train = [1:10; 20:30; 40:50]
X_train0, U_train0, _, _, _, _, metadata = generate_cartpole_data(
    save_to_file=false, num_swingup=0, num_lqr=100
)
t_train = metadata.t_train
dt = metadata.dt
num_train = [1:1:15; 15:2:50]
num_train = 1:10
@time results = @showprogress map(num_train) do N
    test_mpc_stabilization_controllers(X_train0, U_train0, t_train, dt, num_train=N, num_test=10)
end

num_train = 1:50
prog = Progress(length(num_train), dt=0.1, desc="Progress: ", showspeed=true)
@time results = map(num_train) do N
    next!(prog)
    test_mpc_stabilization_controllers(X_train0, U_train0, t_train, dt, num_train=N, num_test=10)
end
jldsave(CARTPOLE_LQR_RESULTS_FILE; results, num_train)

## Generate plot
results = load(CARTPOLE_LQR_RESULTS_FILE)["results"]
num_train = load(CARTPOLE_LQR_RESULTS_FILE)["num_train"]  
err_mpc_nom = map(x->x[2].nom, results)
err_mpc_eDMD = map(x->x[2].eDMD, results)
err_mpc_jDMD = map(x->x[2].jDMD, results)
err_mpc_eDMD_projected = map(x->x[2].eDMD_projected, results)
err_mpc_jDMD_projected = map(x->x[2].jDMD_projected, results)

plotly()
plot(num_train, err_mpc_nom, c=:black, legend=:outerright)
plot!(num_train, err_mpc_eDMD_projected, c=:orange, shape=:circle)
plot!(num_train, err_mpc_jDMD_projected, c=:cyan, shape=:circle)
plot!(num_train, err_mpc_eDMD, s=:dash, c=:orange, shape=:cross)
plot!(num_train, err_mpc_jDMD, s=:dash, c=:cyan, shape=:cross)

## Get index at which the method consistently beats nominal
num_results = length(results)
window_width = 3 
beats_nominal(nom,err,i) = all(j->err[j]<nom[j], i .+ (0:window_width))
eDMD_projected_samples = num_train[findfirst(
    i->beats_nominal(err_mpc_nom, err_mpc_eDMD_projected, i), 1:num_results - window_width
)]
jDMD_projected_samples = num_train[findfirst(
    i->beats_nominal(err_mpc_nom, err_mpc_jDMD_projected, i), 1:num_results - window_width
)]
eDMD_samples = num_train[findfirst(
    i->beats_nominal(err_mpc_nom, err_mpc_eDMD, i), 1:num_results - window_width
)]
jDMD_samples = num_train[findfirst(
    i->beats_nominal(err_mpc_nom, err_mpc_jDMD, i), 1:num_results - window_width
)]

p_bar = @pgf Axis(
    {
        reverse_legend,
        width="4in",
        height="4cm",
        xbar,
        ytick="data",
        yticklabels={"Projected", "Lifted"},
        xmax=27,
        enlarge_y_limits = 0.7,
        legend_pos = "south east",
        xlabel = "Training trajectories Rqd to Beat Nominal MPC",
        nodes_near_coords
    },
    PlotInc({no_marks, color=color_jDMD}, Coordinates([jDMD_projected_samples,jDMD_samples], [0,1])),
    PlotInc({no_marks, color=color_eDMD}, Coordinates([eDMD_projected_samples,eDMD_samples], [0,1])),
    Legend(["jDMD", "eDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "cartpole_lqr_samples.tikz"), p_bar, include_preamble=false)