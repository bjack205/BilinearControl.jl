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

include("airplane_constants.jl")

##
airplane_data = load(AIRPLANE_DATAFILE)
good_cols = findall(x->isfinite(norm(x)), eachcol(airplane_data["X_train"]))
num_train = 15
X_train = airplane_data["X_train"][:,good_cols[1:num_train]]
U_train = airplane_data["U_train"][:,good_cols[1:num_train]]
T_ref = airplane_data["T_ref"]

dt = T_ref[2]
t_ref = T_ref[end]


dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(Problems.NominalAirplane())
##
t_train_eDMD = @elapsed model_eDMD = run_eDMD(X_train, U_train, dt, airplane_kf, nothing; 
    alg=:qr, showprog=true, reg=1e-6
)
model_eDMD.dt
t_train_jDMD = @elapsed model_jDMD = run_jDMD(X_train, U_train, dt, airplane_kf, nothing,
    dmodel_nom; showprog=true, verbose=true, reg=1e-6, alg=:qr, α=0.5
)
model_eDMD.dt
jldsave(AIRPLANE_MODELFILE; 
    eDMD=EDMD.getmodeldata(model_eDMD), 
    jDMD=EDMD.getmodeldata(model_jDMD),
    t_train_eDMD, t_train_jDMD, kf=airplane_kf
)

x,u = rand(dmodel_nom)
RD.discrete_dynamics(model_eDMD_projected, x, u, 0.0, dt)
RD.discrete_dynamics(model_jDMD_projected, x, u, 0.0, dt)