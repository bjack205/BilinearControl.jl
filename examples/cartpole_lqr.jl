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

function test_mpc_stabilization_controllers(X_train0, U_train0, t_train, dt; 
        num_train=50, num_test=10, μ=0.1, α=0.5
    )

    X_train = X_train0[:,1:num_train]
    U_train = U_train0[:,1:num_train]

    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(Problems.NominalCartpole())
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(Problems.SimulatedCartpole(;μ))

    eigfuns = ["state", "sine", "cosine", "sine"]
    # eigfuns = ["state", "sine", "cosine", "sine", "sine", "chebyshev"]
    eigorders = [[0],[1],[1],[2],[4],[2, 4]]

    model_eDMD = run_eDMD(X_train, U_train, dt, eigfuns, eigorders, reg=1e-6, name="cartpole_eDMD", alg=:qr_rls)
    model_jDMD = run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom;
        reg=1e-6, name="cartpole_jDMD", learnB=true, α=α)
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

#################################################
## Get training samples rqd to beat nominal MPC 
#################################################

# Generation LQR training data
X_train0, U_train0, _, _, _, _, metadata = generate_cartpole_data(
    save_to_file=false, num_swingup=0, num_lqr=100
)
t_train = metadata.t_train
dt = metadata.dt

num_train = 1:50
prog = Progress(length(num_train), dt=0.1, desc="Progress: ", showspeed=true)

res = test_mpc_stabilization_controllers(X_train0, U_train0, t_train, dt, num_train=2, num_test=10)
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

#############################################
## Get controller rates
#############################################
eigfuns = ["state", "sine", "cosine", "sine"]
eigorders = [[0],[1],[1],[2],[4],[2, 4]]
model_jDMD = run_jDMD(X_train0[:,1:10], U_train0[:,1:10], dt, 
    eigfuns, eigorders, dmodel_nom, reg=1e-6, name="cartpole_jDMD", α=0.5)
mpc_projected = generate_stabilizing_mpc_controller(model_jDMD_projected, t_sim, dt; Nt, ρ)
mpc = generate_stabilizing_mpc_controller(model_jDMD, t_sim, dt; Nt, ρ)
simulatewithcontroller(dmodel_real, mpc_projected, [0,pi-deg2rad(10),0,0], 4.0, dt, printrate=true)
simulatewithcontroller(dmodel_real, mpc, [0,pi-deg2rad(10),0,0], 4.0, dt, printrate=true)

#############################################
## Analyze effect of model mismatch
#############################################
X_train0, U_train0, _, _, _, _, metadata = generate_cartpole_data(
    save_to_file=false, num_swingup=0, num_lqr=20,
)
t_train = metadata.t_train
dt = metadata.dt

_,err0 = test_mpc_stabilization_controllers(X_train0, U_train0, t_train, dt, num_train=20, num_test=10)
err0.nom
err0.eDMD_projected
err0.jDMD_projected


μ = 0.9
mu_vals = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results_mismatch = @showprogress map(mu_vals) do μ
    # Generate training data using the new friction value
    # NOTE: in order to get good training trajectories, we provide the LQR controller 
    #       for the data collection the correct friction coefficient 
    #       (otherwise it doesn't work)
    X_train, U_train, = generate_cartpole_data(
        save_to_file=false, num_swingup=0, num_lqr=100,
        μ=μ, μ_nom=μ, max_lqr_samples=600
    )

    # Test the controllers, using a "real" model with more and more friction
    _,err = test_mpc_stabilization_controllers(
        X_train, U_train, t_train, dt, num_train=20, num_test=10; μ, α=1e-3
    )
    err
end
err_nom = getfield.(results_mismatch, :nom)
err_eDMD = getfield.(results_mismatch, :eDMD_projected)
err_jDMD = getfield.(results_mismatch, :jDMD_projected)
plot(mu_vals, err_nom, c=:black, label="nominal")
plot!(mu_vals, err_eDMD, c=:orange, label="eDMD")
plot!(mu_vals, err_jDMD, c=:cyan, label="jDMD")