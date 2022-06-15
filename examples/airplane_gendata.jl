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
using ThreadsX

include("airplane_problem.jl")
include("airplane_constants.jl")

function gen_airplane_data(;num_train=30, num_test=10, dt=0.05, dp_window=[1.0,3.0,2.0])
    ## Define nominal and true models
    model_nom = Problems.NominalAirplane()
    model_real = Problems.SimulatedAirplane()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    ## Get trajectories using ALTRO

    # General Parameters
    tf = 2.0
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
    plane_data = ThreadsX.map(1:num_train+num_test) do i
        println("Generating trajectory $i / $(num_train + num_test)")
        Xref = Vector{Float64}[]
        Uref = Vector{Float64}[]
        Tref = Vector{Float64}()
        Xsim = Vector{Float64}[]
        Usim = Vector{Float64}[]
        for i = 1:max_attempts
            dp = rand(dp_sampler)
            prob = AirplaneProblem(;tf, dt, Qv=15, Qw=5, dp, pf)
            solver = ALTROSolver(prob, verbose=0, show_summary=false)
            solve!(solver)

            if Altro.status(solver) != Altro.SOLVE_SUCCEEDED
                continue
            end

            Xref = Vector.(TO.states(solver))
            Uref = Vector.(TO.controls(solver))
            Tref = Vector(range(0,tf,step=dt))

            mpc = EDMD.LinearMPC(dmodel_nom, Xref, Uref, Tref, Qk, Rk, Qf; Nt=Nt,
                xmin,xmax,umin,umax
            )
            Xsim,Usim,Tsim = simulatewithcontroller(dmodel_real, mpc, Xref[1], Tref[end], Tref[2])
            if norm(Xsim[end][1:3] - pf) < 10.0
                break
            end
            if i == max_attempts
                @warn "Couldn't find a good trajectory in $max_attempts attempts"
            end
        end

        Vector.(Xsim), Vector.(Usim), Vector.(Xref), Vector.(Uref), Vector(Tref)
    end
    T_ref = range(0,tf,step=dt)

    # println("Running MPC controller")
    # mpc_trajectories = ThreadsX.map(1:num_train+num_test) do i
    #     X_ref,U_ref = reference_trajectories[i]

    #     mpc = EDMD.LinearMPC(dmodel_nom, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
    #         xmin,xmax,umin,umax
    #     )
    #     X_sim,U_sim,T_sim = simulatewithcontroller(dmodel_real, mpc, X_ref[1], T_ref[end], T_ref[2])
    #     X_sim,U_sim
    # end
    X_mpc = mapreduce(x->getindex(x,1), hcat, plane_data)
    U_mpc = mapreduce(x->getindex(x,2), hcat, plane_data)
    X_ref = mapreduce(x->getindex(x,3), hcat, plane_data)
    U_ref = mapreduce(x->getindex(x,4), hcat, plane_data)
    X_train = X_mpc[:,1:num_train]
    U_train = U_mpc[:,1:num_train]
    X_test = X_mpc[:,num_test .+ (1:num_test)]
    U_test = U_mpc[:,num_test .+ (1:num_test)]

    jldsave(AIRPLANE_DATAFILE; 
        X_train, U_train, X_test, U_test, X_ref, U_ref, T_ref,
        u_trim, pf, dp_window
    )
    X_mpc,U_mpc, X_ref,U_ref
end