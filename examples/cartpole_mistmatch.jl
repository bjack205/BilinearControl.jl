using Pkg: Pkg;
Pkg.activate(joinpath(@__DIR__));
Pkg.instantiate();
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

include("constants.jl")
include("cartpole_utils.jl")
const CARTPOLE_MISMATCH_RESULTS = joinpath(Problems.DATADIR, "cartpole_mismatch_results.jld2")

function test_sample_size(;
    μ=0.1,
    t_sim=4.0,
    num_train=20,
    num_test=10,
    err_thresh=0.1,
    alg=:eDMD,
    α=1e-1,
    reg=1e-6,
    x_window=[1, deg2rad(30), 0.5, 0.5],
    test_window_ratio=1.0,
    lifted=false,
    ρ=1e-6,
)

    ## Define the models
    model_nom = Problems.NominalCartpole()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" Cartpole Model
    model_real = Problems.SimulatedCartpole(; μ=μ) # this model has damping
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    # Generate data with the new damping term
    X_train, U_train, _, _, _, _, metadata = generate_cartpole_data(;
        save_to_file=false,
        num_swingup=0,
        num_lqr=num_train,
        μ=μ,
        μ_nom=μ,
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
    Nt = 21
    xe = [0, pi, 0, 0]
    ue = [0.0]
    T_ref = range(0, t_sim; step=dt)
    X_ref = [copy(xe) for t in T_ref]
    U_ref = [copy(ue) for t in T_ref]
    Qmpc = Diagonal(fill(1e-0, 4))
    Rmpc = Diagonal(fill(1e-3, 1))
    Qfmpc = Diagonal([1e2, 1e2, 1e1, 1e1])
    model_projected = EDMD.ProjectedEDMDModel(model)
    @show typeof(model_projected)
    mpc = if lifted
        n0 = EDMD.originalstatedim(model)
        Y_ref = map(x->EDMD.expandstate(model,x), X_ref)
        lifted_state_error(x,x0) = model_eDMD.kf(x) - x0
        Qmpc_lifted = Diagonal([ρ; diag(Qmpc); fill(ρ, length(ye)-n0-1)])
        Qfmpc_lifted = Diagonal([ρ; diag(Qfmpc); fill(ρ, length(ye)-n0-1)])
        TrackingMPC(
            model, Y_ref, U_ref, Vector(T_ref), Qmpc_lifted, Rmpc, Qfmpc_lifted; 
            Nt=Nt, state_error=lifted_state_error,
        )
    else
        TrackingMPC(
            model_projected, X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
        )
    end

    # Test mpc controller
    # Set seed so that all are tested on the same conditions
    Random.seed!(100)

    # Generate initial conditions to test
    x0_sampler = Product(
        collect(
            Uniform(x - dx, x + dx) for (x, dx) in zip(xe, x_window .* test_window_ratio)
        ),
    )
    x0_test = [rand(x0_sampler) for i in 1:num_test]

    # Run controller for each initial condition
    errors = map(x0_test) do x0
        X_sim, = simulatewithcontroller(dmodel_real, mpc, x0, t_sim, dt)
        return norm(X_sim[end] - xe)
    end
    average_error = mean(filter(x -> x < err_thresh, errors))
    success_rate = count(x -> x < err_thresh, errors) / num_test
    return success_rate, average_error
end

#############################################
## Min Trajectories to Stabilize
#############################################
function find_min_sample_to_stabilize(mu_vals, num_train; 
        err_thresh=0.1, success_rate_thresh=0.95, repeats_required = 2, verbose=true, 
        kwargs...
    )
    samples_required = Dict(Pair.(mu_vals, zeros(Int,length(mu_vals))))
    success_counts = Dict(Pair.(mu_vals, zeros(Int,length(mu_vals))))
    mu_vals_remaining = copy(mu_vals)
    inds_lock = ReentrantLock()

    for N in num_train
        inds_to_delete = Int[]
        verbose && println("Testing with $N samples")
        Threads.@threads for i = 1:length(mu_vals_remaining) 
            μ = mu_vals_remaining[i]
            success_rate, average_error = test_sample_size(;num_train=N, μ, err_thresh, kwargs...)
            
            did_successfully_stabilize = (success_rate >= success_rate_thresh) && (average_error < err_thresh)
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
                verbose && println("  FINISHED: μ = $μ stabilized with $(samples_required[μ]) samples")
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

##
x_window = [1.0, deg2rad(40), 0.5, 0.5]
test_window_ratio = 0.5
err_thresh = 0.1
μ = 0.4
reg=1e-4
num_test = 20
mu_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
num_train = [2:15; 20:5:100]
num_test = 50

# Test jDMD with alpha = 0.01
α = 0.01
@time res_jDMD = find_min_sample_to_stabilize(mu_vals, num_train; 
    num_test, alg=:jDMD, x_window, test_window_ratio, reg, α
)

# Test jDMD with alpha = 0.5
α = 0.5
@time res_jDMD_2 = find_min_sample_to_stabilize(mu_vals, num_train; 
    num_test, alg=:jDMD, x_window, test_window_ratio, reg, α
)

# Test jDMD with alpha = 0.1
α = 0.1
@time res_jDMD_3 = find_min_sample_to_stabilize(mu_vals, num_train; 
    num_test, alg=:jDMD, x_window, test_window_ratio, reg, α
)

# Tset eDMD
@time res_eDMD = find_min_sample_to_stabilize(mu_vals, num_train; 
    num_test, alg=:eDMD, x_window, test_window_ratio, reg, α
)

res_jDMD_all = [
    merge(res_jDMD, Dict(:α=>0.01)),
    merge(res_jDMD_3, Dict(:α=>0.1)),
    merge(res_jDMD_2, Dict(:α=>0.5)),
]
jldsave(CARTPOLE_MISMATCH_RESULTS; res_jDMD = res_jDMD_all, res_eDMD, mu_vals, num_train)

## Plot the results
using PGFPlotsX
using LaTeXStrings
using Colors
colorant"dark green"
results_mismatch = load(CARTPOLE_MISMATCH_RESULTS)

mu_vals = results_mismatch["mu_vals"]
num_train = results_mismatch["num_train"]
setzerotonan(x) = x ≈ zero(x) ? 0 : x
getsamples(d) = map(k->setzerotonan(getindex(d, k)), mu_vals)

res_jDMD_0p5  = getsamples(results_mismatch["res_jDMD"][3])
res_jDMD_0p1  = getsamples(results_mismatch["res_jDMD"][2])
res_jDMD_0p01 = getsamples(results_mismatch["res_jDMD"][1])
res_eDMD      = getsamples(results_mismatch["res_eDMD"])

setzerotona(x) = x ≈ zero(x) ? "N/A" : x

mu_vals
res_jDMD_0p01

p_bar = @pgf Axis(
    {
        height="5cm",
        width="5in",
        bar_width = "7pt",
        reverse_legend,
        ybar,
        ymax=100,
        legend_pos = "north west",
        ylabel = "Training trajectories to Stabilize",
        xlabel = "Coloumb Friction Coefficient",
        nodes_near_coords
    },
    PlotInc({no_marks, color=colorant"forest green"}, Coordinates(mu_vals, res_jDMD_0p5)),
    PlotInc({no_marks, color=colorant"purple"}, Coordinates(mu_vals, res_jDMD_0p1)),
    PlotInc({no_marks, color=color_jDMD}, Coordinates(mu_vals, res_jDMD_0p01)),
    PlotInc({no_marks, color=color_eDMD}, Coordinates(mu_vals, res_eDMD)),
    # PlotInc({no_marks, color=color_eDMD}, Coordinates([eDMD_projected_samples,eDMD_samples], [0,1])),
    Legend(["jDMD " * L"(\alpha=0.5)", "jDMD " * L"(\alpha=0.1)", "jDMD " * L"(\alpha=0.01)", "eDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "cartpole_friction_mismatch.tikz"), p_bar, include_preamble=false)


#############################################
## Min Trajectories to beat MPC 
#############################################

x_window = [1.0, deg2rad(40), 0.5, 0.5]
test_window_ratio = 0.5
err_thresh = 0.1
μ = 0.1
reg=1e-4
num_test = 20
mu_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
num_train = [2,3,4,5,6,7,8]
test_sample_size(; num_train=10, alg=:jDMD, num_test=50)