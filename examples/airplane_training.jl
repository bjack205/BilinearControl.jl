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

const AIRPLANE_DATAFILE = joinpath(Problems.DATADIR, "airplane_trajectory_data.jld2")
const AIRPLANE_MODELFILE = joinpath(Problems.DATADIR, "airplane_trained_models.jld2")
##
airplane_data = load(AIRPLANE_DATAFILE)
X_train = airplane_data["X_train"]
U_train = airplane_data["U_train"]
T_ref = airplane_data["T_ref"]
dt = T_ref[2]
t_ref = T_ref[end]

dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(Problems.NominalAirplane())

function airplane_kf(x)
    p = x[1:3]
    q = x[4:6]
    mrp = MRP(x[4], x[5], x[6])
    R = Matrix(mrp)
    v = x[7:9]
    w = x[10:12]
    α = atan(v[3],v[1])  # angle of attack
    β = atan(v[2],v[1])  # side slip
    vbody = R'v
    speed = vbody'vbody
    [1; x; vec(R); vbody; speed; sin.(p); α; β; α^2; β^2; α^3; β^3; p × v; p × w; EDMD.chebyshev(x, order=[3,3])]
end

t_train_eDMD = @elapsed model_eDMD = run_eDMD(X_train, U_train, dt, airplane_kf, nothing; 
    alg=:qr_rls, showprog=true
)
t_train_jDMD = @elapsed model_jDMD = run_jDMD(X_train, U_train, dt, airplane_kf, nothing, 
    dmodel_nom; showprog=true, verbose=true
)
jldsave(AIRPLANE_MODELFILE; 
    eDMD=EDMD.getmodeldata(model_eDMD), 
    jDMD=EDMD.getmodeldata(model_jDMD),
    t_train_eDMD, t_train_jDMD
)