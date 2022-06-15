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
using Distributions
using Random

include("airplane_problem.jl")
const AIRPLANE_DATAFILE = joinpath(Problems.DATADIR, "airplane_trajectory_data.jld2")

function gen_airplane_data()
    ## Define nominal and true models
    model_nom = Problems.NominalAirplane()
    model_real = Problems.SimulatedAirplane()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    ## Get trajectories using ALTRO

    # General Parameters
    tf = 2.0
    dt = 0.05
    dp_window = [1.0,3,2.]
    num_train = 30 
    num_test = 10 
    pf = [5,0,1.5]  # final position

    # MPC Parameters
    Nt = 21
    Qk = Diagonal([fill(1e0, 3); fill(1e1, 3); fill(1e-1, 3); fill(2e-1, 3)])
    Rk = Diagonal(fill(1e-3,4))
    Qf = Diagonal([fill(1e-2, 3); fill(1e0, 3); fill(1e1, 3); fill(1e1, 3)]) * 10
    u_trim = [41.66667789082778, 105.99999999471807, 74.65179381344494, 106.00000124622453]
    xmax = [fill(0.5,3); fill(1.0, 3); fill(0.5, 3); fill(10.0, 3)]
    xmin = -xmax
    umin = fill(0.0, 4) - u_trim
    umax = fill(255.0, 4) - u_trim

    ## Sample ALTRO trajectories
    Random.seed!(2)
    dp_sampler = Product(collect(Uniform(-x,+x) for x in dp_window))
    max_attempts = 5
    reference_trajectories = map(1:num_train+num_test) do i
        println("Generating trajectory $i / $(num_train + num_test)")
        local solver
        for i = 1:max_attempts
            dp = rand(dp_sampler)
            prob = AirplaneProblem(;tf, dt, Qv=10, Qw=5, dp, pf)
            solver = ALTROSolver(prob, verbose=0, show_summary=false)
            solve!(solver)
            if Altro.status(solver) == Altro.SOLVE_SUCCEEDED
                break
            elseif i == max_attempts
                @warn "Couldn't find a good trajectory in $max_attempts attempts"
            end
        end

        X = Vector.(TO.states(solver))
        U = Vector.(TO.controls(solver))
        T = Vector(range(0,tf,step=dt))
        Vector.(X), Vector.(U)
    end
    X_ref = mapreduce(x->getindex(x,1), hcat, reference_trajectories[1:num_train])
    U_ref = mapreduce(x->getindex(x,2), hcat, reference_trajectories[1:num_train])
    T_ref = range(0,tf,step=dt)

    println("Running MPC controller")
    mpc_trajectories = map(1:num_train+num_test) do i
        X_ref,U_ref = reference_trajectories[i]

        mpc = EDMD.LinearMPC(dmodel_nom, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
            xmin,xmax,umin,umax
        )
        X_sim,U_sim,T_sim = simulatewithcontroller(dmodel_real, mpc, X_ref[1], T_ref[end], T_ref[2])
        X_sim,U_sim
    end
    X_mpc = mapreduce(x->getindex(x,1), hcat, mpc_trajectories)
    U_mpc = mapreduce(x->getindex(x,2), hcat, mpc_trajectories)
    X_train = X_mpc[:,1:num_train]
    U_train = X_mpc[:,1:num_train]
    X_test = X_mpc[:,num_test .+ (1:num_test)]
    U_test = X_mpc[:,num_test .+ (1:num_test)]

    jldsave(AIRPLANE_DATAFILE; 
        X_train, U_train, X_test, U_test, X_ref, U_ref, T_ref,
        u_trim, pf, dp_window
    )
    X_mpc,U_mpc, X_ref,U_ref
end