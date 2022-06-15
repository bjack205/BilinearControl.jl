using Pkg; Pkg.activate(joinpath(@__DIR__));
Pkg.instantiate();
using BilinearControl
using BilinearControl.Problems
using BilinearControl.EDMD
using Rotations
using StaticArrays
using Test
using LinearAlgebra 
using Altro
using RobotDynamics
using TrajectoryOptimization
const TO = TrajectoryOptimization
import RobotDynamics as RD
using BilinearControl: Problems
using JLD2
using Plots
using ProgressMeter
using Statistics
using Random

include("airplane_constants.jl")

## Get models
airplane_data = load(AIRPLANE_DATAFILE)
X_train = airplane_data["X_train"]
U_train = airplane_data["U_train"]
X_test = airplane_data["X_test"]
U_test = airplane_data["U_test"]
num_train = size(X_train,2)
num_test =  size(X_test,2)

X_ref0 = airplane_data["X_ref"][:,num_train+1:end]
U_ref0 = airplane_data["U_ref"][:,num_train+1:end]
T_ref = airplane_data["T_ref"]
dt = T_ref[2]
t_ref = T_ref[end]

## Learned models
airplane_models = load(AIRPLANE_MODELFILE)
model_eDMD = EDMDModel(airplane_models["eDMD"])
model_jDMD = EDMDModel(airplane_models["jDMD"])

# Analytical models
model_nom = Problems.NominalAirplane()
model_real = Problems.SimulatedAirplane()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

# Projected models
model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)

# Test fit errors
EDMD.fiterror(model_eDMD, X_train, U_train)
EDMD.fiterror(model_jDMD, X_train, U_train)
EDMD.fiterror(model_eDMD, X_test, U_test)
EDMD.fiterror(model_jDMD, X_test, U_test)

## MPC Parameters
Nt = 21
Qk = Diagonal([fill(1e0, 3); fill(1e1, 3); fill(1e-1, 3); fill(2e-1, 3)])
Rk = Diagonal(fill(1e-3,4))
Qf = Diagonal([fill(1e-2, 3); fill(1e0, 3); fill(1e1, 3); fill(1e1, 3)]) * 10
u_trim = [41.66667789082778, 105.99999999471807, 74.65179381344494, 106.00000124622453]
xmax = [fill(0.5,3); fill(1.0, 3); fill(0.5, 3); fill(10.0, 3)]
xmin = -xmax
umin = fill(0.0, 4) - u_trim
umax = fill(255.0, 4) - u_trim

##
i = 5
X_ref = X_ref0[:,i]
U_ref = U_ref0[:,i]

mpc_nom = EDMD.LinearMPC(dmodel_nom, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
    xmin,xmax,umin,umax
)
mpc_eDMD = EDMD.LinearMPC(model_eDMD_projected, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
    xmin,xmax,umin,umax
)
mpc_jDMD = EDMD.LinearMPC(model_jDMD_projected, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
    xmin,xmax,umin,umax
)

X_nom,  = simulatewithcontroller(dmodel_real, mpc_nom,  X_ref[1], t_ref, dt)
X_eDMD, = simulatewithcontroller(dmodel_real, mpc_eDMD, X_ref[1], t_ref, dt)
X_jDMD, = simulatewithcontroller(dmodel_real, mpc_jDMD, X_ref[1], t_ref, dt)

plotstates(T_ref, X_ref,inds=[1,3,4,7], lw=3, label=["x" "z" "roll" "vx"], ylim=(-10,10))
plotstates!(T_ref,X_nom,inds=[1,3,4,7], label="", s=:solid, lw=:1, c=[1 2 3 4])
plotstates!(T_ref,X_eDMD,inds=[1,3,4,7], label="", s=:dash, lw=:2, c=[1 2 3 4])
plotstates!(T_ref,X_jDMD,inds=[1,3,4,7], label="", s=:dot, lw=:2, c=[1 2 3 4])

visualize!(vis, model_nom, t_ref, X_jDMD)
visualize!(vis, model_nom, t_ref, X_ref)

## Run Test
airplane_data = load(AIRPLANE_DATAFILE)
X_test = airplane_data["X_test"]
U_test = airplane_data["U_test"]
num_train = size(X_train,2)
num_test =  size(X_test,2)

X_ref0 = airplane_data["X_ref"][:,num_train+1:end]
U_ref0 = airplane_data["U_ref"][:,num_train+1:end]
T_ref = airplane_data["T_ref"]
dt = T_ref[2]
t_ref = T_ref[end]

err_nom = zeros(num_test) 
err_eDMD = zeros(num_test) 
err_jDMD = zeros(num_test) 
model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)
prog = Progress(num_test)
Threads.@threads for i = 1:num_test
    X_ref = X_ref0[:,i]
    U_ref = U_ref0[:,i]
    N = length(X_ref)

    mpc_nom = EDMD.LinearMPC(dmodel_nom, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
        xmin,xmax,umin,umax
    )
    mpc_eDMD = EDMD.LinearMPC(model_eDMD_projected, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
        xmin,xmax,umin,umax
    )
    mpc_jDMD = EDMD.LinearMPC(model_jDMD_projected, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
        xmin,xmax,umin,umax
    )

    X_nom,  = simulatewithcontroller(dmodel_real, mpc_nom,  X_ref[1], t_ref, dt)
    X_eDMD, = simulatewithcontroller(dmodel_real, mpc_eDMD, X_ref[1], t_ref, dt)
    X_jDMD, = simulatewithcontroller(dmodel_real, mpc_jDMD, X_ref[1], t_ref, dt)
    err_nom[i] = norm(X_nom - X_ref) / N
    err_eDMD[i] = norm(X_eDMD - X_ref) / N
    err_jDMD[i] = norm(X_jDMD - X_ref) / N
    next!(prog)
end

##
did_track(x) = x<1e1
sr_nom = count(did_track, err_nom) / num_test
sr_eDMD = count(did_track, err_eDMD) / num_test
sr_jDMD = count(did_track, err_jDMD) / num_test

ae_nom = mean(filter(did_track, err_nom))
ae_eDMD = mean(filter(did_track, err_eDMD))
ae_jDMD = mean(filter(did_track, err_jDMD))
