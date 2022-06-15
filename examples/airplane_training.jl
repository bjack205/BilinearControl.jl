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
X_train = airplane_data["X_train"]
U_train = airplane_data["U_train"]
T_ref = airplane_data["T_ref"]
dt = T_ref[2]
t_ref = T_ref[end]

dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(Problems.NominalAirplane())

t_train_eDMD = @elapsed model_eDMD = run_eDMD(X_train, U_train, dt, airplane_kf, nothing; 
    alg=:qr_rls, showprog=true
)
t_train_jDMD = @elapsed model_jDMD = run_jDMD(X_train, U_train, dt, airplane_kf, nothing, 
    dmodel_nom; showprog=true, verbose=true, reg=1e-6
)
jldsave(AIRPLANE_MODELFILE; 
    eDMD=EDMD.getmodeldata(model_eDMD), 
    jDMD=EDMD.getmodeldata(model_jDMD),
    t_train_eDMD, t_train_jDMD, kf=airplane_kf
)