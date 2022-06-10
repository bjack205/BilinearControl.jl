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

include("constants.jl")

## Visualizer
model = Problems.RexPlanarQuadrotor()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
delete!(vis)
set_quadrotor!( vis, model)
render(vis)

#############################################
## Define the models 
#############################################

## Define Nominal Simulated REx Planar Quadrotor Model
model_nom = Problems.NominalPlanarQuadrotor()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

# Define Mismatched "Real" REx Planar Quadrotor Model
model_real = Problems.SimulatedPlanarQuadrotor()  # this model has aero drag
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

#############################################
## Generate Training and Test Data 
#############################################
tf = 5.0
dt = 0.05

## Generate Data From Mismatched Model
Random.seed!(1)

# Number of trajectories
num_test = 30
num_train = 30

# Generate a stabilizing LQR controller
Qlqr = Diagonal([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
Rlqr = Diagonal([1e-4, 1e-4])
xe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ue = Problems.trim_controls(model_real)
ctrl_lqr_nom = LQRController(dmodel_nom, Qlqr, Rlqr, xe, ue, dt)

# Sample a bunch of initial conditions for the LQR controller
x0_sampler = Product([
    Uniform(-1.0,1.0),
    Uniform(-1.0,1.0),
    Uniform(-deg2rad(60),deg2rad(60)),
    Uniform(-0.5,0.5),
    Uniform(-0.5,0.5),
    Uniform(-0.25,0.25)
])

initial_conditions_lqr = [rand(x0_sampler) for _ in 1:num_train]
initial_conditions_test = [rand(x0_sampler) for _ in 1:num_test]

# Create data set
X_train, U_train = create_data(dmodel_real, ctrl_lqr_nom, initial_conditions_lqr, tf, dt)
X_test, U_test = create_data(dmodel_real, ctrl_lqr_nom, initial_conditions_test, tf, dt)

#############################################
## Fit the training data
#############################################

## Define basis functions
eigfuns = ["state", "sine", "cosine", "monomial"]
eigorders = [[0],[1],[1],[2,2]]

model_eDMD = run_eDMD(X_train, U_train, dt, eigfuns, eigorders, reg=1e-4, name="planar_quadrotor_eDMD")
model_jDMD = run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom, reg=1e-4, name="planar_quadrotor_jDMD")

model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)

# Check test errors
EDMD.open_loop_error(model_eDMD, X_test, U_test)
EDMD.open_loop_error(model_jDMD, X_test, U_test)
BilinearControl.EDMD.fiterror(model_eDMD, X_test, U_test)
BilinearControl.EDMD.fiterror(model_jDMD, X_test, U_test)
n,m = RD.dims(model_eDMD)
n0, = RD.dims(model_real)

#############################################
## Plot an LQR Trajectory
#############################################

# ze = RD.KnotPoint{n0,m}(xe,ue,0.0,dt)
# ye = EDMD.expandstate(model_eDMD, xe)

# ## Create LQR controllers
# ctrl_lqr_real = EDMD.LQRController(dmodel_real, Qlqr, Rlqr, xe, ue, dt)
# ctrl_lqr_nom = EDMD.LQRController(dmodel_nom, Qlqr, Rlqr, xe, ue, dt)
# ctrl_lqr_eDMD = EDMD.LQRController(model_eDMD_projected, Qlqr, Rlqr, xe, ue, dt)
# ctrl_lqr_jDMD = EDMD.LQRController(model_jDMD_projected, Qlqr, Rlqr, xe, ue, dt)

# ## Simulate with specified initial condition
# tf_sim = 5.0
# Tsim_lqr = range(0,tf_sim,step=dt)
# x0 = [-0.5, 0.5, -deg2rad(20),-1.0,1.0,0.0]

# Xsim_lqr_real, = EDMD.simulatewithcontroller(dmodel_real, ctrl_lqr_real, x0, tf_sim, dt)
# Xsim_lqr_nom, = EDMD.simulatewithcontroller(dmodel_real, ctrl_lqr_nom, x0, tf_sim, dt)
# Xsim_lqr_eDMD, = EDMD.simulatewithcontroller(dmodel_real, ctrl_lqr_eDMD, x0, tf_sim, dt)
# Xsim_lqr_jDMD, = EDMD.simulatewithcontroller(dmodel_real, ctrl_lqr_jDMD, x0, tf_sim, dt)

# plotstates(Tsim_lqr, Xsim_lqr_real, inds=1:3, xlabel="time (s)", ylabel="states",
#             label=["x (true LQR)" "y (true LQR)" "θ (true LQR)"], legend=:topright, lw=2,
#             linestyle=:dot, color=[1 2 3])
# plotstates!(Tsim_lqr, Xsim_lqr_nom, inds=1:3, xlabel="time (s)", ylabel="states",
#             label=["x (nominal LQR)" "y (nominal LQR)" "θ (nominal LQR)"], legend=:topright, lw=2,
#             linestyle=:dash, color=[1 2 3])
# plotstates!(Tsim_lqr, Xsim_lqr_eDMD, inds=1:3, xlabel="time (s)", ylabel="states",
#             label=["x (nominal EDMD)" "y (nominal eDMD)" "θ (nominal eDMD)"], legend=:topright, lw=2,
#             linestyle=:dashdot, color=[1 2 3])
# plotstates!(Tsim_lqr, Xsim_lqr_jDMD, inds=1:3, xlabel="time (s)", ylabel="states",
#             label=["x (JDMD)" "y (JDMD)" "θ (JDMD)"], legend=:bottomright, lw=2,
#             color=[1 2 3])

# ylims!((-1.25,0.75))

#############################################
## Plot fit error vs regularization 
#############################################

regularizers = exp10.(-4:2)
errors = map(regularizers) do reg
    model_eDMD = run_eDMD(X_train, U_train, dt, eigfuns, eigorders, reg=reg, name="planar_quadrotor_eDMD")
    model_jDMD = run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom, reg=reg, name="planar_quadrotor_jDMD")

    # Check test errors
    olerr_eDMD = EDMD.open_loop_error(model_eDMD, X_test, U_test)
    olerr_jDMD = EDMD.open_loop_error(model_jDMD, X_test, U_test)
    fiterr_eDMD = BilinearControl.EDMD.fiterror(model_eDMD, X_test, U_test)
    fiterr_jDMD = BilinearControl.EDMD.fiterror(model_jDMD, X_test, U_test)
    (;olerr_eDMD, olerr_jDMD, fiterr_eDMD, fiterr_jDMD)
end
fields = keys(errors[1])
res = Dict(Pair.(fields, map(x->getfield.(errors, x), fields)))
p_ol = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xmode = "log",
        xlabel = "regularization value",
        ylabel = "open loop error",
        legend_pos = "north west"
    },
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(regularizers, res[:olerr_eDMD])),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(regularizers, res[:olerr_jDMD])),
    Legend(["eDMD", "jDMD"])
)
p_fit = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xmode = "log",
        xlabel = "regularization value",
        ylabel = "dynamics error",
        legend_pos = "north west"
    },
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(regularizers, res[:fiterr_eDMD])),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(regularizers, res[:fiterr_jDMD])),
    Legend(["eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "planar_quadrotor_openloop_error_by_reg.tikz"), p_ol, include_preamble=false)
pgfsave(joinpath(Problems.FIGDIR, "planar_quadrotor_fit_error_by_reg.tikz"), p_fit, include_preamble=false)




model_eDMD = run_eDMD(X_train, U_train, dt, eigfuns, eigorders, reg=1e-5, name="planar_quadrotor_eDMD")
model_jDMD = run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom, reg=1e-5, name="planar_quadrotor_jDMD")