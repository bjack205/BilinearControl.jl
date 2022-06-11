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
model = RobotZoo.Cartpole()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
delete!(vis)
set_cartpole!(vis)
render(vis)

#############################################
## Define the models 
#############################################

# Nominal Simulated Cartpole Model
model_nom = RobotZoo.Cartpole(mc=1.0, mp=0.2, l=0.5)
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

# Mismatched "Real" Cartpole Model
model_real = Cartpole2(mc=1.05, mp=0.19, l=0.52, b=0.02)  # this model has damping
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)


#############################################
## Generate Training and Test Data 
#############################################
tf = 2.0
dt = 0.02

## Generate Data From Mismatched Model
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
## Fit the training data
#############################################

## Define basis functions
eigfuns = ["state", "sine", "cosine", "sine"]
eigorders = [[0],[1],[1],[2],[4],[2, 4]]

model_eDMD = run_eDMD(X_train, U_train, dt, eigfuns, eigorders, reg=1e-6, name="cartpole_eDMD")
model_jDMD = run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom, 
    reg=1e-6, name="cartpole_jDMD", α=0.9)

# Check test errors
EDMD.open_loop_error(model_eDMD, X_test, U_test)
EDMD.open_loop_error(model_jDMD, X_test, U_test)
BilinearControl.EDMD.fiterror(model_eDMD, X_test, U_test)
BilinearControl.EDMD.fiterror(model_jDMD, X_test, U_test)

#############################################
## Plot fit error vs regularization 
#############################################

regularizers = exp10.(-4:2)
errors = map(regularizers) do reg
    model_eDMD = run_eDMD(X_train, U_train, dt, eigfuns, eigorders, reg=reg, name="cartpole_eDMD")
    model_jDMD = run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom, reg=reg, name="cartpole_jDMD")

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
pgfsave(joinpath(Problems.FIGDIR, "cartpole_openloop_error_by_reg.tikz"), p_ol, include_preamble=false)
pgfsave(joinpath(Problems.FIGDIR, "cartpole_fit_error_by_reg.tikz"), p_fit, include_preamble=false)

#############################################
## LQR Performance
#############################################

function test_initial_conditions(model, controller, xg, ics, tf, dt)
    map(ics) do x0
        X_sim, = simulatewithcontroller(model, controller, x0, tf, dt)
        norm(X_sim[end] - xg)
    end
end

# Equilibrium position
xe = [0,pi,0,0.]
ue = [0.0]
ye = EDMD.expandstate(model_eDMD, xe)

Qlqr = Diagonal([1.0,1.0,1e-2,1e-2])
Rlqr = Diagonal([1e-3])
Qlqr = Diagonal(fill(1e-0,4))
Rlqr = Diagonal(fill(1e-3,1))

ρ = 1e-6 
Qlqr_lifted = Diagonal([ρ; diag(Qlqr); fill(ρ, length(ye) - 5)])

# Nominal LQR Controller
lqr_nominal = LQRController(
    dmodel_nom, Qlqr, Rlqr, xe, ue, dt, max_iters=10000, verbose=true
)

# Projected LQR Controllers
model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)
lqr_eDMD_projected = LQRController(
    model_eDMD_projected, Qlqr, Rlqr, xe, ue, dt, max_iters=10000, verbose=true
)
lqr_jDMD_projected = LQRController(
    model_jDMD_projected, Qlqr, Rlqr, xe, ue, dt, max_iters=10000, verbose=true
)

# Lifted LQR Controllers
lifted_state_error(x,x0) = model_eDMD.kf(x) - x0
lqr_jDMD = LQRController(
    model_jDMD, Qlqr_lifted, Rlqr, ye, ue, dt, max_iters=20000, verbose=true,
    state_error=lifted_state_error
)
lqr_eDMD = LQRController(
    model_eDMD, Qlqr_lifted, Rlqr, ye, ue, dt, max_iters=10000, verbose=true,
    state_error=lifted_state_error
)

# Run each controller on the same set of initial conditions
Random.seed!(2)
x0_sampler = Product([
    Uniform(-1.5,1.5),
    Uniform(pi-deg2rad(70),pi+deg2rad(70)),
    Uniform(-1,1),
    Uniform(-1,1),
])
t_sim = 4.0
x0_test = [rand(x0_sampler) for i = 1:100]
errors_nominal = sort!(test_initial_conditions(dmodel_real, lqr_nominal, xe, x0_test, t_sim, dt))
errors_eDMD_projected = sort!(test_initial_conditions(dmodel_real, lqr_eDMD_projected, xe, x0_test, t_sim, dt))
errors_jDMD_projected = sort!(test_initial_conditions(dmodel_real, lqr_jDMD_projected, xe, x0_test, t_sim, dt))
errors_eDMD = sort!(test_initial_conditions(dmodel_real, lqr_eDMD, xe, x0_test, t_sim, dt))
errors_jDMD = sort!(test_initial_conditions(dmodel_real, lqr_jDMD, xe, x0_test, t_sim, dt))

p_lqr = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel="Percent of samples",
        ylabel="Tracking error",
        legend_cell_align={left},
        legend_pos="north west",
        ymax=15e-2,
        xmax=100,
    },
    PlotInc({lineopts..., color=color_nominal, style="solid"}, Coordinates(1:100, errors_nominal)),
    PlotInc({lineopts..., color=color_eDMD, style="solid"}, Coordinates(1:100, errors_eDMD_projected)),
    PlotInc({lineopts..., color=color_jDMD, style="solid"}, Coordinates(1:100, errors_jDMD_projected)),
    PlotInc({lineopts..., color=color_eDMD, style="dashed"}, Coordinates(1:100, errors_eDMD)),
    PlotInc({lineopts..., color=color_jDMD, style="dashed"}, Coordinates(1:100, errors_jDMD)),
    # Legend(["nominal", "eDMD (projected)", "jDMD (projected)", "eDMD (lifted)", "jDMD (lifted)"])
)
display(p_lqr)
pgfsave(joinpath(Problems.FIGDIR, "cartpole_lqr_stabilization_performance.tikz"), 
    p_lqr, include_preamble=false
)

#############################################
## MPC Stabilization Performance 
#############################################

# Reference Trajectory
X_ref = [copy(xe) for t in T_sim]
U_ref = [copy(ue) for t in T_sim]
T_ref = copy(T_sim)
Y_ref = kf.(X_ref)
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
model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)
mpc_eDMD_projected = TrackingMPC(model_eDMD_projected, 
    X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
)
mpc_jDMD_projected = TrackingMPC(model_jDMD_projected, 
    X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
)

# Lifted MPC controllers
mpc_eDMD = TrackingMPC(model_eDMD, 
    Y_ref, U_ref, Vector(T_ref), Qmpc_lifted, Rmpc, Qfmpc_lifted; Nt=Nt, state_error=lifted_state_error
)
mpc_jDMD = TrackingMPC(model_jDMD, 
    Y_ref, U_ref, Vector(T_ref), Qmpc_lifted, Rmpc, Qfmpc_lifted; Nt=Nt, state_error=lifted_state_error
)

# Run each controller on the same set of initial conditions
Random.seed!(2)
x0_sampler = Product([
    Uniform(-1.5,1.5),
    Uniform(pi-deg2rad(70),pi+deg2rad(70)),
    Uniform(-1,1),
    Uniform(-1,1),
])
x0_test = [rand(x0_sampler) for i = 1:100]
errors_nominal = sort!(test_initial_conditions(dmodel_real, mpc_nominal, xe, x0_test, t_sim, dt))
errors_eDMD_projected = sort!(test_initial_conditions(dmodel_real, mpc_eDMD_projected, xe, x0_test, t_sim, dt))
errors_jDMD_projected = sort!(test_initial_conditions(dmodel_real, mpc_jDMD_projected, xe, x0_test, t_sim, dt))
errors_eDMD = sort!(test_initial_conditions(dmodel_real, mpc_eDMD, xe, x0_test, t_sim, dt))
errors_jDMD = sort!(test_initial_conditions(dmodel_real, mpc_jDMD, xe, x0_test, t_sim, dt))

p_mpc = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel="Percent of samples",
        # ylabel="Tracking error",
        legend_cell_align={left},
        legend_pos="outer north east",
        ymax=15e-2,
        xmax=100,
    },
    PlotInc({lineopts..., color=color_nominal, style="solid"}, Coordinates(1:100, errors_nominal)),
    PlotInc({lineopts..., color=color_eDMD, style="solid"}, Coordinates(1:100, errors_eDMD_projected)),
    PlotInc({lineopts..., color=color_jDMD, style="solid"}, Coordinates(1:100, errors_jDMD_projected)),
    PlotInc({lineopts..., color=color_eDMD, style="dashed"}, Coordinates(1:100, errors_eDMD)),
    PlotInc({lineopts..., color=color_jDMD, style="dashed"}, Coordinates(1:100, errors_jDMD)),
    Legend(["nominal", "eDMD (projected)", "jDMD (projected)", "eDMD (lifted)", "jDMD (lifted)"])
)
display(p_mpc)
pgfsave(joinpath(Problems.FIGDIR, "cartpole_mpc_stabilization_performance.tikz"), 
    p_mpc, include_preamble=false
)