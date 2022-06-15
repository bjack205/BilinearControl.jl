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
using PGFPlotsX

include("airplane_constants.jl")
include("constants.jl")

function train_airplane(num_train)
    # Get training data
    airplane_data = load(AIRPLANE_DATAFILE)
    good_cols = findall(x->isfinite(norm(x)), eachcol(airplane_data["X_train"]))
    X_train = airplane_data["X_train"][:,good_cols[1:num_train]]
    U_train = airplane_data["U_train"][:,good_cols[1:num_train]]
    T_ref = airplane_data["T_ref"]
    dt = T_ref[2]

    # Get nominal model
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(Problems.NominalAirplane())

    ## Train models
    model_eDMD = run_eDMD(X_train, U_train, dt, airplane_kf, nothing; 
        alg=:qr, showprog=false, reg=1e-6
    )
    model_jDMD = run_jDMD(X_train, U_train, dt, airplane_kf, nothing,
        dmodel_nom; showprog=false, verbose=false, reg=1e-6, alg=:qr, α=0.1
    )
    model_eDMD, model_jDMD
end

function test_airplane(model_eDMD, model_jDMD)
    # Models
    model_nom = Problems.NominalAirplane()
    model_real = Problems.SimulatedAirplane()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    # MPC parameters
    Nt = 21
    Qk = Diagonal([fill(1e0, 3); fill(1e1, 3); fill(1e-1, 3); fill(2e-1, 3)])
    Rk = Diagonal(fill(1e-3,4))
    Qf = Diagonal([fill(1e-2, 3); fill(1e0, 3); fill(1e1, 3); fill(1e1, 3)]) * 10
    u_trim = [41.66667789082778, 105.99999999471807, 74.65179381344494, 106.00000124622453]
    xmax = [fill(0.5,3); fill(1.0, 3); fill(0.5, 3); fill(10.0, 3)]
    xmin = -xmax
    umin = fill(0.0, 4) - u_trim
    umax = fill(255.0, 4) - u_trim

    # Get test data
    airplane_data = load(AIRPLANE_DATAFILE)
    X_test = airplane_data["X_test"]
    X_train = airplane_data["X_train"]
    num_train = size(X_train,2)
    num_test =  size(X_test,2)

    X_ref0 = airplane_data["X_ref"][:,num_train+1:end]
    U_ref0 = airplane_data["U_ref"][:,num_train+1:end]
    T_ref = airplane_data["T_ref"]
    dt = T_ref[2]
    t_ref = T_ref[end]

    # Allocate result vectors
    err_nom = zeros(num_test) 
    err_eDMD = zeros(num_test) 
    err_jDMD = zeros(num_test) 
    model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
    model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)

    # Run MPC on each trajectory
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
    end
    Dict(:nominal=>err_nom, :eDMD=>err_eDMD, :jDMD=>err_jDMD)
end

##
num_train = [2; 5:5:50]
results = @showprogress map(num_train) do N
    test_airplane(train_airplane(N)...)
end
jldsave(AIRPLANE_RESULTS; results)

##
results = load(AIRPLANE_RESULTS)["results"]
airplane_data = load(AIRPLANE_DATAFILE)
num_test =  size(airplane_data["X_test"],2)

did_track(x) = x<1e1
function get_average_error(results, method)
    map(x->mean(filter(did_track, x[method])), results)
end
function get_success_rate(results, method)
    map(x->count(did_track, x[method]) / num_test, results)
end

err_nom  = get_average_error(results, :nominal) 
err_eDMD = get_average_error(results, :eDMD) 
err_jDMD = get_average_error(results, :jDMD) 
sr_nom  = get_success_rate(results, :nominal) 
sr_eDMD = get_success_rate(results, :eDMD) 
sr_jDMD = get_success_rate(results, :jDMD) 

p_err = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Number of Training Trajectories",
        ylabel = "Average Tracking Error",
    },
    PlotInc({lineopts..., color=color_nominal}, Coordinates(num_train, err_nom)),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(num_train, err_eDMD)),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(num_train, err_jDMD)),
    Legend(["Nominal MPC", "eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "airplane_error_by_num_train.tikz"), p_err, 
    include_preamble=false)

p_sr = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Number of Training Trajectories",
        ylabel = "Success Rate",
    },
    PlotInc({lineopts..., color=color_nominal}, Coordinates(num_train, sr_nom)),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(num_train, sr_eDMD)),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(num_train, sr_jDMD)),
    Legend(["Nominal MPC", "eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "airplane_success_by_num_train.tikz"), p_sr, 
    include_preamble=false)
