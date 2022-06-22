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
using Infiltrator

include("constants.jl")
const REX_PLANAR_QUADROTOR_RESULTS_FILE = joinpath(Problems.DATADIR, "rex_planar_quadrotor_mpc_results.jld2")

#############################################
## Functions for generating data and training models
#############################################



#############################################
## MPC Tracking Study
#############################################

generate_planar_quadrotor_data()
num_traj = 2:2:36   
results = map(num_traj) do N
    println("\nRunning with N = $N")
    res = train_planar_quadrotor_models(0,N, α=0.5, β=1.0, learnB=true, reg=1e-5)
    @show res.jDMD_err_avg
    @show res.eDMD_err_avg
    res
end

jldsave(REX_PLANAR_QUADROTOR_RESULTS_FILE; results)
results

## Process results
using PGFPlotsX
results = load(REX_PLANAR_QUADROTOR_RESULTS_FILE)["results"]
fields = keys(results[1])
res = Dict(Pair.(fields, map(x->getfield.(results, x), fields)))
res
# plot(res[:nsamples][good_inds], res[:t_train_eDMD][good_inds])
# plot!(res[:nsamples][good_inds], res[:t_train_jDMD][good_inds])
p_time = @pgf Axis(
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
pgfsave(joinpath(Problems.FIGDIR, "rex_planar_quadrotor_mpc_train_time.tikz"), p_time, include_preamble=false)

p_ns = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Number of training trajectories", ylabel = "Tracking error",
        ymax=2.0,
    },
    PlotInc({lineopts..., color=color_nominal}, Coordinates(res[:num_mpc], res[:nom_err_avg])),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(res[:num_mpc], res[:eDMD_err_avg])),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(res[:num_mpc], res[:jDMD_err_avg])),
    Legend(["nominal", "eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "rex_planar_quadrotor_mpc_test_error.tikz"), p_ns, include_preamble=false)

#############################################
## Set up models for MPC Tracking
#############################################

# Define Nominal Simulated REx Planar Quadrotor Model
model_nom = Problems.NominalPlanarQuadrotor()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

# Define Mismatched "Real" REx Planar Quadrotor Model
model_real = Problems.SimulatedPlanarQuadrotor()  # this model has aero drag
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

res = train_planar_quadrotor_models(0, 50, α=0.5, β=1.0, learnB=true, reg=1e-5)
@show res.jDMD_err_avg
@show res.eDMD_err_avg

res_no_reg = train_planar_quadrotor_models_no_eDMD_reg(0, 50, α=0.5, β=1.0, learnB=true, reg=1e-1)

eDMD_data = res.eDMD_data
jDMD_data = res.jDMD_data
G = res.G
kf = res.kf
dt = res.dt

eDMD_data_no_reg = res_no_reg.eDMD_data
jDMD_data2 = res_no_reg.jDMD_data

model_eDMD = EDMDModel(eDMD_data[:A],eDMD_data[:B],eDMD_data[:C],G,kf,dt,"planar_quadrotor_jDMD")
model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
model_jDMD = EDMDModel(jDMD_data[:A],jDMD_data[:B],jDMD_data[:C],G,kf,dt,"planar_quadrotor_jDMD")
model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)
model_jDMD2 = EDMDModel(jDMD_data2[:A],jDMD_data2[:B],jDMD_data2[:C],G,kf,dt,"planar_quadrotor_jDMD")
model_jDMD_projected2 = EDMD.ProjectedEDMDModel(model_jDMD2)
model_eDMD_no_reg = EDMDModel(eDMD_data_no_reg[:A],eDMD_data_no_reg[:B],eDMD_data_no_reg[:C],G,kf,dt,"planar_quadrotor_jDMD")
model_eDMD_projected_no_reg = EDMD.ProjectedEDMDModel(model_eDMD_no_reg)

mpc_lqr_traj = load(joinpath(Problems.DATADIR, "rex_planar_quadrotor_mpc_tracking_data.jld2"))

# Test data
X_nom_mpc = mpc_lqr_traj["X_nom_mpc"]
U_nom_mpc = mpc_lqr_traj["U_nom_mpc"]
X_test_infeasible = mpc_lqr_traj["X_test_infeasible"]
U_test_infeasible = mpc_lqr_traj["U_test_infeasible"]

tf = mpc_lqr_traj["tf"]
t_sim = mpc_lqr_traj["t_sim"]
dt = mpc_lqr_traj["dt"]

#############################################
## Plot an MPC Trajectory
#############################################

# i = 1
# X_ref = deepcopy(X_test_infeasible[:,i])
# U_ref = deepcopy(U_test_infeasible[:,i])
# X_nom_mpc_traj = deepcopy(X_nom_mpc[:,i])
# push!(U_ref, Problems.trim_controls(model_real))
# T_ref = range(0,tf,step=dt)
# T_sim = range(0,t_sim,step=dt)
# x0 = X_ref[1]
# Nt = 41

# Qmpc = Diagonal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# Rmpc = Diagonal([1e-3, 1e-3])
# Qfmpc = 100*Qmpc

# mpc_eDMD_projected = TrackingMPC(model_eDMD_projected, 
#     X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt)
# mpc_jDMD_projected = TrackingMPC(model_jDMD_projected, 
#     X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt)

# X_mpc_eDMD_projected,= simulatewithcontroller(dmodel_real, mpc_eDMD_projected, x0, t_sim, dt)
# X_mpc_jDMD_projected,= simulatewithcontroller(dmodel_real, mpc_jDMD_projected, x0, t_sim, dt)

# plotstates(T_ref, X_ref, inds=1:3, xlabel="time (s)", ylabel="states",
#             label=["x (true MPC)" "y (true MPC)" "θ (true MPC)"], legend=:topright, lw=2,
#             linestyle=:dot, color=[1 2 3])
# plotstates!(T_sim, X_nom_mpc_traj, inds=1:3, xlabel="time (s)", ylabel="states",
#             label=["x (nominal MPC)" "y (nominal MPC)" "θ (nominal MPC)"], legend=:topright, lw=2,
#             linestyle=:dash, color=[1 2 3])
# plotstates!(T_sim, X_mpc_eDMD_projected, inds=1:3, xlabel="time (s)", ylabel="states",
#             label=["x (nominal EDMD)" "y (nominal EDMD)" "θ (nominal EDMD)"], legend=:topright, lw=2,
#             linestyle=:dashdot, color=[1 2 3])
# plotstates!(T_sim, X_mpc_jDMD_projected, inds=1:3, xlabel="time (s)", ylabel="states",
#             label=["x (JDMD)" "y (JDMD)" "θ (JDMD)"], legend=:topright, lw=2,
#             color=[1 2 3])

#############################################
## Plot Tracking Success vs. Training Window
#############################################

percentages = 0.1:0.1:2.5
errors = map(percentages) do perc

    println("percentage of training range = $perc")

    x0_sampler = Product([
        Uniform(-2.0*perc,2.0*perc),
        Uniform(-2.0*perc,2.0*perc),
        Uniform(-deg2rad(20*perc),deg2rad(20*perc)),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.2*perc,0.2*perc)
    ])

    x0_test = [rand(x0_sampler) for i = 1:50]

    error_eDMD_projected = mean(test_initial_conditions(model_real, model_eDMD_projected, x0_test, tf, tf, dt))
    error_jDMD_projected = mean(test_initial_conditions(model_real, model_jDMD_projected, x0_test, tf, tf, dt))
    error_jDMD_projected2 = mean(test_initial_conditions(model_real, model_jDMD_projected2, x0_test, tf, tf, dt))
    error_eDMD_projected_no_reg = mean(test_initial_conditions(model_real, model_eDMD_projected_no_reg, x0_test, tf, tf, dt))

    (;error_eDMD_projected, error_jDMD_projected, error_eDMD_projected_no_reg, error_jDMD_projected2)

end

fields = keys(errors[1])
res_training_range = Dict(Pair.(fields, map(x->getfield.(errors, x), fields)))
jldsave(joinpath(Problems.DATADIR, "rex_planar_quadrotor_mpc_training_range_results.jld2"); percentages, res_training_range)

##
results = load(joinpath(Problems.DATADIR, "rex_planar_quadrotor_mpc_training_range_results.jld2"))
percentages = results["percentages"]
res_training_range = results["res_training_range"]

p_tracking = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Fraction of training range",
        ylabel = "Tracking error",
        legend_pos = "north west"
    },
    PlotInc({no_marks, color=color_eDMD, thick}, Coordinates(percentages, res_training_range[:error_eDMD_projected])),
    PlotInc({lineopts..., color=color_eDMD, line_width=2.0}, Coordinates(percentages, res_training_range[:error_eDMD_projected_no_reg])),
    PlotInc({no_marks, color=color_jDMD, thick}, Coordinates(percentages, res_training_range[:error_jDMD_projected])),
    PlotInc({lineopts..., color=color_jDMD, line_width=2.0}, Coordinates(percentages, res_training_range[:error_jDMD_projected2])),
    Legend(["eDMD" * L"(\lambda = 0.0)", "eDMD" * L"(\lambda = 0.1)", "jDMD" * L"(\lambda = 10^{-5})", "jDMD" * L"(\lambda = 0.1)"])
)
pgfsave(joinpath(Problems.FIGDIR, "rex_planar_quadrotor_mpc_error_by_training_window.tikz"), p_tracking, include_preamble=false)