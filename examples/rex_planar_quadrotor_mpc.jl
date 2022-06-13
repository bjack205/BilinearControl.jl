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

function test_initial_conditions(model, bilinear_model, ics, tf, t_sim, dt)

    dmodel = RD.DiscretizedDynamics{RD.RK4}(model)

    N_tf = round(Int, tf/dt) + 1    
    N_sim = round(Int, t_sim/dt) + 1 
    Nt = 41
    T_ref = range(0,tf,step=dt)

    Qmpc = Diagonal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    Rmpc = Diagonal([1e-3, 1e-3])
    Qfmpc = 100*Qmpc

    results = map(ics) do x0

        X_ref = nominal_trajectory(x0,N_tf,dt)
        X_ref_full = [X_ref; [X_ref[end] for i = 1:N_sim - N_tf]]
        U_ref = [copy(Problems.trim_controls(model)) for k = 1:N_tf]

        mpc = TrackingMPC(bilinear_model, 
            X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_sim,= simulatewithcontroller(dmodel, mpc, x0, t_sim, dt)

        err = norm(X_sim - X_ref_full) / N_sim

        (; err)
    end

    err_avg  = mean(filter(isfinite, map(x->x.err, results)))

    return err_avg
end

function nominal_trajectory(x0,N,dt)
    Xref = [fill(NaN, length(x0)) for k = 1:N]
    
    # TODO: Design a trajectory that linearly interpolates from x0 to the origin
    
    pos_0 = x0[1:3]
    
    pos_ref = reshape(LinRange(pos_0, [0, 0, 0], N), N, 1)
    vel_ref = vcat([(pos_ref[2] - pos_ref[1]) ./ dt for k = 1:N-1], [[0, 0, 0]])
    
    for i = 1:N
        
        Xref[i] = vcat(pos_ref[i], vel_ref[i])
    
    end
    
    return Xref
end

function generate_planar_quadrotor_data()
    #############################################
    ## Define the Models
    #############################################

    ## Define Nominal Simulated REx Planar Quadrotor Model
    model_nom = Problems.NominalPlanarQuadrotor()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" REx Planar Quadrotor Model
    model_real = Problems.SimulatedPlanarQuadrotor()  # this model has aero drag
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    # Time parameters
    tf = 5.0
    dt = 0.05
    Nt = 41  # MPC Horizon
    t_sim = tf*1.2  # length of simulation (to capture steady-state behavior) 

    #############################################
    ## LQR Training and Testing Data 
    #############################################

    ## Stabilization trajectories 
    Random.seed!(1)
    num_train_lqr = 50
    num_test_lqr = 50

    # Generate a stabilizing LQR controller
    Qlqr = Diagonal([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    Rlqr = Diagonal([1e-4, 1e-4])
    xe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ue = Problems.trim_controls(model_real)
    ctrl_lqr_nom = LQRController(dmodel_nom, Qlqr, Rlqr, xe, ue, dt)

    # Sample a bunch of initial conditions for the LQR controller
    x0_train_sampler = Product([
        Uniform(-1.0,1.0),
        Uniform(-1.0,1.0),
        Uniform(-deg2rad(40),deg2rad(40)),
        Uniform(-0.5,0.5),
        Uniform(-0.5,0.5),
        Uniform(-0.25,0.25)
    ])

    perc = 2.0
    x0_test_sampler = Product([
        Uniform(-1.0*perc,1.0*perc),
        Uniform(-1.0*perc,1.0*perc),
        Uniform(-deg2rad(40*perc),deg2rad(40*perc)),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.25*perc,0.25*perc)
    ])

    initial_conditions_train = [rand(x0_train_sampler) for _ in 1:num_train_lqr]
    initial_conditions_test = [rand(x0_test_sampler) for _ in 1:num_test_lqr]

    # Create data set
    X_train_lqr, U_train_lqr = create_data(dmodel_real, ctrl_lqr_nom, initial_conditions_train, t_sim, dt)
    X_test_lqr, U_test_lqr = create_data(dmodel_real, ctrl_lqr_nom, initial_conditions_test, t_sim, dt);

    # Make sure they all stabilize
    @test all(x->x<0.1, map(x->norm(x-xe), X_train_lqr[end,:]))
    @test all(x->x<0.1, map(x->norm(x-xe), X_test_lqr[end,:]))

    #############################################
    ## MPC Training and Testing Data 
    #############################################
    Random.seed!(1)
    num_train_mpc = 50
    num_test_mpc = 35

    x0_train_sampler = Product([
        Uniform(-2.0,2.0),
        Uniform(-2.0,2.0),
        Uniform(-deg2rad(20),deg2rad(20)),
        Uniform(-0.5,0.5),
        Uniform(-0.5,0.5),
        Uniform(-0.2,0.2)
    ])

    perc = 3.0
    x0_test_sampler = Product([
        Uniform(-2.0*perc,2.0*perc),
        Uniform(-2.0*perc,2.0*perc),
        Uniform(-deg2rad(20*perc),deg2rad(20*perc)),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.5*perc,0.5*perc),
        Uniform(-0.2*perc,0.2*perc)
    ])

    initial_conditions_mpc_train = [rand(x0_train_sampler) for _ in 1:num_train_mpc]
    initial_conditions_mpc_test = [rand(x0_test_sampler) for _ in 1:num_test_mpc]

    Random.seed!(1)

    Qmpc = Diagonal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    Rmpc = Diagonal([1e-3, 1e-3])
    Qfmpc = 100*Qmpc

    N_tf = round(Int, tf/dt) + 1
    N_tsim = round(Int, t_sim/dt) + 1

    X_train_mpc = Matrix{Vector{Float64}}(undef, N_tsim, num_train_mpc)
    U_train_mpc = Matrix{Vector{Float64}}(undef, N_tsim-1, num_train_mpc)

    for i = 1:num_train_mpc

        x0 = initial_conditions_mpc_train[i]
        X = nominal_trajectory(x0,N_tf,dt)
        U = [copy(Problems.trim_controls(model_real)) for k = 1:N_tf]
        T = range(0,tf,step=dt)

        mpc = TrackingMPC(dmodel_nom, X, U, Vector(T), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_sim,U_sim,T_sim = simulatewithcontroller(dmodel_real, mpc, X[1], t_sim, T[2])
            
        X_train_mpc[:,i] = X_sim
        U_train_mpc[:,i] = U_sim
    end

    # i = 5
    # T_ref = range(0,tf,step=dt)
    # X_ref = nominal_trajectory(initial_conditions_mpc_train[i],N,dt,final_conditions_mpc_train[i])

    # plotstates(T_ref, X_ref, inds=1:3, xlabel="time (s)", ylabel="states",
    #             label=["x (ref)" "y (ref)" "θ (ref)"], legend=:right, lw=2,
    #             linestyle=:dot, color=[1 2 3])
    # plotstates!(T_ref, X_train_mpc[:, i], inds=1:3, xlabel="time (s)", ylabel="states",
    #             label=["x (mpc)" "θ (mpc)"], legend=:right, lw=2,
    #             linestyle=:dot, color=[1 2 3])

    # Generate test data

    X_test_infeasible = Matrix{Vector{Float64}}(undef, N_tf, num_test_mpc)
    X_nom_mpc = Matrix{Vector{Float64}}(undef, N_tsim, num_test_mpc)

    U_test_infeasible = Matrix{Vector{Float64}}(undef, N_tf-1, num_test_mpc)
    U_nom_mpc = Matrix{Vector{Float64}}(undef, N_tsim-1, num_test_mpc)

    for i = 1:num_test_mpc

        x0 = initial_conditions_mpc_test[i]
        X = nominal_trajectory(x0,N_tf,dt)
        U = [copy(Problems.trim_controls(model_real)) for k = 1:N_tf]
        T = range(0,tf,step=dt)
        
        mpc_nom = TrackingMPC(dmodel_nom, X, U, Vector(T), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_nom,U_nom,T_nom = simulatewithcontroller(dmodel_real, mpc_nom, X[1], t_sim, T[2])

        X_test_infeasible[:,i] = X
        U_test_infeasible[:,i] = U[1:end-1]

        X_nom_mpc[:,i] = X_nom
        U_nom_mpc[:,i] = U_nom
    end

    # i = 1
    # T_sim = range(0,t_sim,step=dt)
    # T_ref = range(0,tf,step=dt)
    # X_ref = nominal_trajectory(initial_conditions_mpc_test[i],N,dt,final_conditions_mpc_test[i])

    # plotstates(T_ref, X_ref, inds=1:3, xlabel="time (s)", ylabel="states",
    #             label=["x (ref)" "y (ref)" "θ (ref)"], legend=:right, lw=2,
    #             linestyle=:dot, color=[1 2 3])
    # plotstates!(T_sim, X_test_nom_mpc[:, i], inds=1:3, xlabel="time (s)", ylabel="states",
    #             label=["x (mpc)" "θ (mpc)"], legend=:right, lw=2,
    #             linestyle=:dot, color=[1 2 3])

    ## Save generated training and test data
    jldsave(joinpath(Problems.DATADIR, "rex_planar_quadrotor_mpc_tracking_data.jld2"); 
        X_train_lqr, U_train_lqr,
        X_train_mpc, U_train_mpc,
        X_nom_mpc, U_nom_mpc, 
        X_test_infeasible, U_test_infeasible,
        X_test_lqr, U_test_lqr, 
        tf, t_sim, dt
    )
end

function train_planar_quadrotor_models(num_lqr, num_mpc;  α=0.5, learnB=true, β=1.0, reg=1e-6)

    #############################################
    ## Define the Models
    #############################################

    # Define Nominal Simulated REx Planar Quadrotor Model
    model_nom = Problems.NominalPlanarQuadrotor()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" REx Planar Quadrotor Model
    model_real = Problems.SimulatedPlanarQuadrotor()  # this model has aero drag
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    #############################################  
    ## Load Training and Test Data
    #############################################  
    mpc_lqr_traj = load(joinpath(Problems.DATADIR, "rex_planar_quadrotor_mpc_tracking_data.jld2"))

    # Training data
    X_train_lqr = mpc_lqr_traj["X_train_lqr"][:,1:num_lqr]
    U_train_lqr = mpc_lqr_traj["U_train_lqr"][:,1:num_lqr]
    X_train_mpc = mpc_lqr_traj["X_train_mpc"][:,1:num_mpc]
    U_train_mpc = mpc_lqr_traj["U_train_mpc"][:,1:num_mpc]

    ## combine lqr and mpc training data
    X_train = [X_train_lqr X_train_mpc]
    U_train = [U_train_lqr U_train_mpc]

    # Test data
    X_nom_mpc = mpc_lqr_traj["X_nom_mpc"]
    U_nom_mpc = mpc_lqr_traj["U_nom_mpc"]
    X_test_infeasible = mpc_lqr_traj["X_test_infeasible"]
    U_test_infeasible = mpc_lqr_traj["U_test_infeasible"]

    # Metadata
    tf = mpc_lqr_traj["tf"]
    t_sim = mpc_lqr_traj["t_sim"]
    dt = mpc_lqr_traj["dt"]

    T_ref = range(0,tf,step=dt)
    T_sim = range(0,t_sim,step=dt)

    #############################################
    ## Fit the training data
    #############################################

    # Define basis functions
    eigfuns = ["state", "sine", "cosine", "chebyshev"]
    eigorders = [[0],[1],[1],[2,2]]

    t_train_eDMD = @elapsed model_eDMD = run_eDMD(X_train, U_train, dt, eigfuns, eigorders,
        reg=0.0, name="planar_quadrotor_eDMD")
    t_train_jDMD = @elapsed model_jDMD = run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom,
        reg=reg, name="planar_quadrotor_jDMD"; α, β, learnB)

    model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
    model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)

    eDMD_data = Dict(
        :A=>model_eDMD.A, :B=>model_eDMD.B, :C=>model_eDMD.C, :g=>model_eDMD.g, :t_train=>t_train_eDMD
    )
    jDMD_data = Dict(
        :A=>model_jDMD.A, :B=>model_jDMD.B, :C=>model_jDMD.C, :g=>model_jDMD.g, :t_train=>t_train_jDMD
    )

    #############################################
    ## MPC Tracking
    #############################################

    xe = zeros(6)
    ue = Problems.trim_controls(model_real)
    Nt = 41  # MPC horizon
    N_sim = length(T_sim)
    N_ref = length(T_ref)

    Qmpc = Diagonal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    Rmpc = Diagonal([1e-3, 1e-3])
    Qfmpc = 100*Qmpc

    N_test = size(X_nom_mpc,2)
    test_results = map(1:N_test) do i
        X_ref = deepcopy(X_test_infeasible[:,i])
        U_ref = deepcopy(U_test_infeasible[:,i])
        X_ref[end] .= xe
        push!(U_ref, ue)

        X_ref_full = [X_ref; [copy(xe) for i = 1:N_sim - N_ref]]
        mpc_nom = TrackingMPC(dmodel_nom, 
            X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
        )
        mpc_eDMD = TrackingMPC(model_eDMD_projected, 
            X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
        )
        mpc_jDMD = TrackingMPC(model_jDMD_projected, 
            X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
        )
        X_mpc_nom, U_mpc_nom, T_mpc = simulatewithcontroller(dmodel_real, mpc_nom,  X_ref[1], t_sim, dt)
        X_mpc_eDMD,U_mpc_eDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_eDMD, X_ref[1], t_sim, dt)
        X_mpc_jDMD,U_mpc_jDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_jDMD, X_ref[1], t_sim, dt)

        err_nom = norm(X_mpc_nom - X_ref_full) / N_sim
        err_eDMD = norm(X_mpc_eDMD - X_ref_full) / N_sim
        err_jDMD = norm(X_mpc_jDMD - X_ref_full) / N_sim

        (; err_nom, err_eDMD, err_jDMD) #, t_train_eDMD, t_train_jDMD, num_lqr, num_swingup, nsamples=length(X_train)) end
    end

    nom_err_avg  = mean(filter(isfinite, map(x->x.err_nom, test_results)))
    eDMD_err_avg = mean(filter(isfinite, map(x->x.err_eDMD, test_results)))
    jDMD_err_avg = mean(filter(isfinite, map(x->x.err_jDMD, test_results)))
    eDMD_success = count(isfinite, map(x->x.err_eDMD, test_results))
    jDMD_success = count(isfinite, map(x->x.err_jDMD, test_results))

    G = model_jDMD.g
    kf = model_jDMD.kf

    (;nom_err_avg, eDMD_err_avg, eDMD_success, jDMD_err_avg, jDMD_success, 
        t_train_eDMD, t_train_jDMD, num_lqr, num_mpc, nsamples=length(X_train), 
        eDMD_data, jDMD_data, G, kf, dt)
end

#############################################
## MPC Tracking Study
#############################################

generate_planar_quadrotor_data()
num_traj = 2:2:50

β = 240 / (38 * 120)
res = train_planar_quadrotor_models(0, 50, α=0.5, β=1.0, learnB=true, reg=1e-5);
# res = train_planar_quadrotor_models(0, 10, α=0.5, β=1.0, learnB=true, reg=1e-4)
# res = train_cartpole_models(0, 5, α=0.5, β=1.0, learnB=false, reg=1e-4)
@show res.jDMD_err_avg
@show res.eDMD_err_avg
num_traj[4]
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
good_inds = 1:length(res[:nsamples])
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
        xlabel = "Number of training samples",
        ylabel = "Tracking error",
        ymax=2.0,
    },
    PlotInc({lineopts..., color=color_nominal}, Coordinates(res[:nsamples][good_inds], res[:nom_err_avg][good_inds])),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(res[:nsamples][good_inds], res[:eDMD_err_avg][good_inds])),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(res[:nsamples][good_inds], res[:jDMD_err_avg][good_inds])),
    Legend(["Nominal", "eDMD", "jDMD"])
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

eDMD_data = res.eDMD_data
jDMD_data = res.jDMD_data
G = res.G
kf = res.kf
dt = res.dt

model_eDMD = EDMDModel(eDMD_data[:A],eDMD_data[:B],eDMD_data[:C],G,kf,dt,"planar_quadrotor_jDMD")
model_eDMD_projected = EDMD.ProjectedEDMDModel(model_eDMD)
model_jDMD = EDMDModel(jDMD_data[:A],jDMD_data[:B],jDMD_data[:C],G,kf,dt,"planar_quadrotor_jDMD")
model_jDMD_projected = EDMD.ProjectedEDMDModel(model_jDMD)

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

    println("percentage of training window = $perc")

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

    (;error_eDMD_projected, error_jDMD_projected)

end

fields = keys(errors[1])
res = Dict(Pair.(fields, map(x->getfield.(errors, x), fields)))

p_tracking = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Fraction of training range",
        ylabel = "Tracking error",
        legend_pos = "north west"
    },
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(percentages, res[:error_eDMD_projected])),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(percentages, res[:error_jDMD_projected])),
    Legend(["eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "rex_planar_quadrotor_mpc_error_by_training_window.tikz"), p_tracking, include_preamble=false)