"""
    cartpole_mpc.jl

This script was used to generated Figure 2a in the paper
"""
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
using ProgressMeter

# include("learned_models/edmd_utils.jl")
include("constants.jl")
include("cartpole_utils.jl")
const CARTPOLE_RESULTS_FILE = joinpath(Problems.DATADIR, "cartpole_results.jld2")

##
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
        )
end

##
generate_cartpole_data()  # OPTIONAL (data file already exists)
train_cartpole_models(0,2, α=0.5, β=1.0, learnB=true, reg=1e-4)  # run once for JIT
num_swingup = 2:2:36
results = map(num_swingup) do N
    println("\nRunning with N = $N")
    res = train_cartpole_models(0,N, α=0.5, β=1.0, learnB=true, reg=1e-4)
    @show res.jDMD_err_avg
    @show res.eDMD_err_avg
    res
end
jldsave(CARTPOLE_RESULTS_FILE; results)

## Process results
using PGFPlotsX
results = load(CARTPOLE_RESULTS_FILE)["results"]
fields = keys(results[1])
res = Dict(Pair.(fields, map(x->getfield.(results, x), fields)))
res
p = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Number of training samples",
        ylabel = "Training time (sec)",
        legend_pos = "north west",
    },
    PlotInc({no_marks, "very thick", "orange"}, Coordinates(res[:nsamples], res[:t_train_eDMD])),
    PlotInc({no_marks, "very thick", "cyan"}, Coordinates(res[:nsamples], res[:t_train_jDMD])),
    Legend(["eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "cartpole_mpc_train_time.tikz"), p, include_preamble=false)

p = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Number of training trajectories",
        ylabel = "Tracking error",
        ymax=0.2,
    },
    PlotInc({lineopts..., color=color_nominal}, Coordinates(res[:num_swingup], res[:nom_err_avg])),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(res[:num_swingup], res[:eDMD_err_avg])),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(res[:num_swingup], res[:jDMD_err_avg])),
    # Legend(["Nominal", "eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "cartpole_mpc_test_error.tikz"), p, include_preamble=false)
