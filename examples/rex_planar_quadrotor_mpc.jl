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

include("learned_models/edmd_utils.jl")

#############################################
## Functions for generating data and training models
#############################################

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
    num_train_lqr = 15
    num_test_lqr = 15

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

    initial_conditions_train = [rand(x0_sampler) for _ in 1:num_train_lqr]
    initial_conditions_test = [rand(x0_sampler) for _ in 1:num_test_lqr]

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
    num_train_mpc = 15
    num_test_mpc = 15

    x0_sampler = Product([
        Uniform(-5.0,5.0),
        Uniform(-5.0,5.0),
        Uniform(-deg2rad(60),deg2rad(60)),
        Uniform(-0.5,0.5),
        Uniform(-0.5,0.5),
        Uniform(-0.25,0.25)
    ])

    initial_conditions_mpc_train = [rand(x0_sampler) for _ in 1:num_train_mpc]
    initial_conditions_mpc_test = [rand(x0_sampler) for _ in 1:num_test_mpc]

    Random.seed!(1)

    Qmpc = Diagonal([10, 100, 10, 1e-4, 1e-4, 1e-4])
    Rmpc = Diagonal(fill(1e-4,2))
    Qfmpc = Diagonal(fill(1e2,6))

    N = round(Int, t_sim/dt) + 1
    X_train_mpc = Matrix{Vector{Float64}}(undef, N, num_train_mpc)
    U_train_mpc = Matrix{Vector{Float64}}(undef, N-1, num_test_mpc)

    for i = 1:num_train_mpc

        x0 = initial_conditions_mpc_train[i]
        X = nominal_trajectory(x0,N,dt)
        U = [copy(Problems.trim_controls(model_real)) for k = 1:N]
        T = range(0,t_sim,step=dt)

        mpc = TrackingMPC(dmodel_nom, X, U, Vector(T), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_sim,U_sim,T_sim = simulatewithcontroller(dmodel_real, mpc, X[1], T[end], T[2])
            
        X_train_mpc[:,i] = X_sim
        U_train_mpc[:,i] = U_sim
    end

    # i = 5
    # T_sim = range(0,t_sim,step=dt)
    # X_ref = nominal_trajectory(initial_conditions_mpc_train[i],N,dt,final_conditions_mpc_train[i])

    # plotstates(T_sim, X_ref, inds=1:3, xlabel="time (s)", ylabel="states",
    #             label=["x (ref)" "y (ref)" "θ (ref)"], legend=:right, lw=2,
    #             linestyle=:dot, color=[1 2 3])
    # plotstates!(T_sim, X_train_mpc[:, i], inds=1:3, xlabel="time (s)", ylabel="states",
    #             label=["x (mpc)" "θ (mpc)"], legend=:right, lw=2,
    #             linestyle=:dot, color=[1 2 3])

    # Generate test data

    X_test_real_mpc = Matrix{Vector{Float64}}(undef, N, num_train_mpc)
    X_test_nom_mpc = Matrix{Vector{Float64}}(undef, N, num_train_mpc)

    U_test_real_mpc = Matrix{Vector{Float64}}(undef, N-1, num_test_mpc)
    U_test_nom_mpc = Matrix{Vector{Float64}}(undef, N-1, num_test_mpc)

    for i = 1:num_test_mpc

        x0 = initial_conditions_mpc_test[i]
        X = nominal_trajectory(x0,N,dt)
        U = [copy(Problems.trim_controls(model_real)) for k = 1:N]
        T = range(0,t_sim,step=dt)
        
        mpc_nom = TrackingMPC(dmodel_nom, X, U, Vector(T), Qmpc, Rmpc, Qfmpc; Nt=Nt)
        X_nom,U_nom,T_nom = simulatewithcontroller(dmodel_real, mpc_nom, X[1], T[end], T[2])

        X_test_real_mpc[:,i] = X
        U_test_real_mpc[:,i] = U[1:end-1]

        X_test_nom_mpc[:,i] = X_nom
        U_test_nom_mpc[:,i] = U_nom
    end

    # i = 1
    # T_sim = range(0,t_sim,step=dt)
    # X_ref = nominal_trajectory(initial_conditions_mpc_test[i],N,dt,final_conditions_mpc_test[i])

    # plotstates(T_sim, X_ref, inds=1:3, xlabel="time (s)", ylabel="states",
    #             label=["x (ref)" "y (ref)" "θ (ref)"], legend=:right, lw=2,
    #             linestyle=:dot, color=[1 2 3])
    # plotstates!(T_sim, X_test_nom_mpc[:, i], inds=1:3, xlabel="time (s)", ylabel="states",
    #             label=["x (mpc)" "θ (mpc)"], legend=:right, lw=2,
    #             linestyle=:dot, color=[1 2 3])

    ## Save generated training and test data
    jldsave(joinpath(Problems.DATADIR, "rex_planar_quadrotor_mpc_tracking_data.jld2"); 
        X_train_lqr, U_train_lqr,
        X_train_mpc, U_train_mpc,
        X_test_mpc=X_test_nom_mpc, U_test_mpc=U_test_nom_mpc, 
        X_test_mpc_ref=X_test_real_mpc, U_test_mpc_ref=U_test_real_mpc,
        X_test_lqr, U_test_lqr, 
        tf, t_sim, dt
    )
end

function train_planar_quadrotor_models(num_lqr, num_mpc)

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
    X_test_mpc = mpc_lqr_traj["X_test_mpc"]
    U_test_mpc = mpc_lqr_traj["U_test_mpc"]
    X_test_mpc_ref = mpc_lqr_traj["X_test_mpc_ref"]
    U_test_mpc_ref = mpc_lqr_traj["U_test_mpc_ref"]

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
    eigfuns = ["state", "sine", "cosine", "monomial"]
    eigorders = [[0],[1],[1],[2,2]]

    t_train_eDMD = @elapsed model_eDMD = run_eDMD(X_train, U_train, dt, eigfuns, eigorders, reg=0.0, name="planar_quadrotor_eDMD")
    t_train_jDMD = @elapsed model_jDMD = run_jDMD(X_train, U_train, dt, eigfuns, eigorders, dmodel_nom, reg=1e-4, name="planar_quadrotor_jDMD")

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
    ue = zeros(2)
    tf_sim = T_sim[end]
    Nt = 41  # MPC horizon
    N_sim = length(T_sim)

    Qmpc = Diagonal(fill(1e0,6))
    Rmpc = Diagonal(fill(1e-4,2))
    Qfmpc = Diagonal(fill(1e2,6))

    N_test = size(X_test_mpc,2)
    test_results = map(1:N_test) do i
        X_ref = deepcopy(X_test_mpc_ref[:,i])
        U_ref = deepcopy(U_test_mpc_ref[:,i])
        X_ref[end] .= xe
        push!(U_ref, ue)

        N_ref = length(T_sim)
        X_ref_full = [X_ref; [copy(xe) for i = 1:N_sim - N_ref]]
        mpc_nom = TrackingMPC(dmodel_nom, 
            X_ref, U_ref, Vector(T_sim), Qmpc, Rmpc, Qfmpc; Nt=Nt
        )
        mpc_eDMD = TrackingMPC(model_eDMD_projected, 
            X_ref, U_ref, Vector(T_sim), Qmpc, Rmpc, Qfmpc; Nt=Nt
        )
        mpc_jDMD = TrackingMPC(model_jDMD_projected, 
            X_ref, U_ref, Vector(T_sim), Qmpc, Rmpc, Qfmpc; Nt=Nt
        )
        X_mpc_nom, U_mpc_nom, T_mpc = simulatewithcontroller(dmodel_real, mpc_nom,  X_ref[1], tf_sim, dt, printrate=false)
        X_mpc_eDMD,U_mpc_eDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_eDMD, X_ref[1], tf_sim, dt, printrate=false)
        X_mpc_jDMD,U_mpc_jDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_jDMD, X_ref[1], tf_sim, dt, printrate=false)

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
## 
#############################################

generate_planar_quadrotor_data()
train_planar_quadrotor_models(10,6)
num_traj = 2:1:6   # goes singular above 35?
results = map(num_traj) do N
    println("\n\nRunning with N = $N")
    train_planar_quadrotor_models(0,N)
end
const PLANAR_QUADROTOR_RESULTS_FILE = joinpath(Problems.DATADIR, "rex_planar_quadrotor_results.jld2")
jldsave(PLANAR_QUADROTOR_RESULTS_FILE; results)

#############################################
## Process results
#############################################

using PGFPlotsX
results = load(PLANAR_QUADROTOR_RESULTS_FILE)["results"]
fields = keys(results[1])
res = Dict(Pair.(fields, map(x->getfield.(results, x), fields)))
plot(res[:num_mpc], res[:t_train_eDMD])
plot!(res[:num_mpc], res[:t_train_jDMD])

p = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Number of training samples",
        ylabel = "Tracking error",
    },
    PlotInc({no_marks, "very thick", "black"}, Coordinates(res[:nsamples], res[:nom_err_avg])),
    PlotInc({no_marks, "very thick", "orange"}, Coordinates(res[:nsamples], res[:eDMD_err_avg])),
    PlotInc({no_marks, "very thick", "cyan"}, Coordinates(res[:nsamples], res[:jDMD_err_avg])),
    Legend(["Nominal", "eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "rex_planar_quadrotor_mpc_test_error.tikz"), p, include_preamble=false)

#############################################
## Stabilization analysis
#############################################

# Define Nominal Simulated REx Planar Quadrotor Model
model_nom = Problems.NominalPlanarQuadrotor()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

# Define Mismatched "Real" REx Planar Quadrotor Model
model_real = Problems.SimulatedPlanarQuadrotor()  # this model has aero drag
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

results = train_planar_quadrotor_models(10,6)
eDMD_data = results.eDMD_data
jDMD_data = results.jDMD_data
G = results.G
kf = results.kf
dt = results.dt
eDMD_data

model_bilinear_eDMD = EDMDModel(eDMD_data[:A],eDMD_data[:B],eDMD_data[:C],G,kf,dt,"planar_quadrotor_jDMD")
model_eDMD_projected = EDMD.ProjectedEDMDModel(model_bilinear_eDMD)
model_bilinear_jDMD = EDMDModel(jDMD_data[:A],jDMD_data[:B],jDMD_data[:C],G,kf,dt,"planar_quadrotor_jDMD")
model_jDMD_projected = EDMD.ProjectedEDMDModel(model_bilinear_jDMD)

# Generate LQR Controllers
xe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ue = Problems.trim_controls(model_real)
ye = EDMD.expandstate(model_bilinear_eDMD, xe)

Qlqr = Diagonal([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
Rlqr = Diagonal([1e-4, 1e-4])

ρ = 1e-6 
Qlqr_expanded = Diagonal([ρ; diag(Qlqr); fill(ρ, length(ye) - 7)])

lifted_state_error(x,x0) = kf(x) - x0
lqr_eDMD = LQRController(
    model_bilinear_eDMD, Qlqr_expanded, Rlqr, ye, ue, dt, max_iters=10000, verbose=true,
    state_error=lifted_state_error
)
lqr_eDMD_projected = LQRController(
    model_eDMD_projected, Qlqr, Rlqr, xe, ue, dt, max_iters=10000, verbose=true
)
lqr_jDMD = LQRController(
    model_bilinear_jDMD, Qlqr_expanded, Rlqr, ye, ue, dt, max_iters=20000, verbose=true,
    state_error=lifted_state_error
)
lqr_jDMD_projected = LQRController(
    model_jDMD_projected, Qlqr, Rlqr, xe, ue, dt, max_iters=10000, verbose=true
)

# Run LQR controllers
x0_sampler = Product([
    Uniform(-1.0,1.0),
    Uniform(-1.0,1.0),
    Uniform(-deg2rad(60),deg2rad(60)),
    Uniform(-0.5,0.5),
    Uniform(-0.5,0.5),
    Uniform(-0.25,0.25)
])

x0_test = [rand(x0_sampler) for _ in 1:100]
x0 = x0_test[1]
t_sim = 5.0
T_sim = range(0,t_sim,step=dt)
X_lqr_eDMD_projected,= simulatewithcontroller(dmodel_real, lqr_eDMD_projected, x0, t_sim, dt)
X_lqr_jDMD_projected,= simulatewithcontroller(dmodel_real, lqr_jDMD_projected, x0, t_sim, dt)
X_lqr_eDMD,= simulatewithcontroller(dmodel_real, lqr_eDMD, x0, t_sim, dt)
X_lqr_jDMD,= simulatewithcontroller(dmodel_real, lqr_jDMD, x0, t_sim, dt)

plotstates(T_sim, X_lqr_eDMD_projected, inds=1:2)
plotstates(T_sim, X_lqr_jDMD_projected, inds=1:2)
plotstates(T_sim, X_lqr_eDMD, inds=1:2)
plotstates(T_sim, X_lqr_jDMD, inds=1:2)

#############################################
## MPC Tracking Analysis
#############################################

mpc_lqr_traj = load(joinpath(Problems.DATADIR, "rex_planar_quadrotor_mpc_tracking_data.jld2"))

# Test data
X_test_mpc = mpc_lqr_traj["X_test_mpc"]
U_test_mpc = mpc_lqr_traj["U_test_mpc"]
X_test_mpc_ref = mpc_lqr_traj["X_test_mpc_ref"]
U_test_mpc_ref = mpc_lqr_traj["U_test_mpc_ref"]

tf = mpc_lqr_traj["tf"]
t_sim = mpc_lqr_traj["t_sim"]
dt = mpc_lqr_traj["dt"]

i = 1
X_ref = deepcopy(X_test_mpc[:,i])
U_ref = deepcopy(U_test_mpc[:,i])
push!(U_ref, zeros(RD.dims(model_eDMD_projected)[2]))
T_ref = range(0,t_sim,step=dt)
Y_ref = kf.(X_ref)
Nt = 41

Qmpc = Diagonal([10, 100, 10, 1e-4, 1e-4, 1e-4])
Rmpc = Diagonal(fill(1e-4,2))
Qfmpc = Diagonal(fill(1e2,6))

Qmpc_lifted = Diagonal([ρ; diag(Qmpc); fill(ρ, length(ye)-7)])
Qfmpc_lifted = Diagonal([ρ; diag(Qfmpc); fill(ρ, length(ye)-7)])

mpc_eDMD_projected = TrackingMPC(model_eDMD_projected, 
    X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
)
mpc_jDMD_projected = TrackingMPC(model_jDMD_projected, 
    X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
)
Nt = 21
mpc_eDMD = TrackingMPC(model_bilinear_eDMD, 
    Y_ref, U_ref, Vector(T_ref), Qmpc_lifted, Rmpc, Qfmpc_lifted; Nt=Nt, state_error=lifted_state_error
)
mpc_jDMD = TrackingMPC(model_bilinear_jDMD, 
    Y_ref, U_ref, Vector(T_ref), Qmpc_lifted, Rmpc, Qfmpc_lifted; Nt=Nt, state_error=lifted_state_error
)
solve!(mpc_jDMD, x0, 1)
map(x->G*x, mpc_jDMD.X)

X_mpc_eDMD_projected,= simulatewithcontroller(dmodel_real, mpc_eDMD_projected, x0, t_sim, dt)
X_mpc_jDMD_projected,= simulatewithcontroller(dmodel_real, mpc_jDMD_projected, x0, t_sim, dt)
X_mpc_eDMD, = simulatewithcontroller(dmodel_real, mpc_eDMD, x0, t_sim, dt)
X_mpc_jDMD, = simulatewithcontroller(dmodel_real, mpc_jDMD, x0, t_sim, dt)

plotstates(T_ref, X_mpc_eDMD_projected, inds=1:2)
plotstates(T_ref, X_mpc_jDMD_projected, inds=1:2)
plotstates(T_ref, X_mpc_eDMD, inds=1:2)
plotstates(T_ref, X_mpc_jDMD, inds=1:2)

mpc_lqr_traj = load(joinpath(Problems.DATADIR, "rex_planar_quadrotor_mpc_tracking_data.jld2"))

# Compare open-loop simulations

model_nom = Problems.NominalPlanarQuadrotor()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)
model_real = Problems.SimulatedPlanarQuadrotor()  # this model has aero drag
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

X_test_lqr = mpc_lqr_traj["X_test_lqr"]
U_test_lqr = mpc_lqr_traj["U_test_lqr"]
tf = mpc_lqr_traj["t_sim"]
dt = mpc_lqr_traj["dt"]
T_sim = range(0,tf,step=dt)

X, = simulate(dmodel_real, U_test_lqr[:,1], X_test_lqr[1], tf, dt)
X, = simulate(dmodel_nom, U_test_lqr[:,1], X_test_lqr[1], tf, dt)
X, = simulate(model_eDMD_projected, U_test_lqr[:,1], X_test_lqr[1], tf, dt)
X, = simulate(model_jDMD_projected, U_test_lqr[:,1], X_test_lqr[1], tf, dt)

Y, = simulate(model_bilinear_eDMD, U_test_lqr[:,1], kf(X_test_lqr[1]), tf, dt)
Y, = simulate(model_bilinear_jDMD, U_test_lqr[:,1], kf(X_test_lqr[1]), tf, dt)
X = map(x->EDMD.originalstate(model_bilinear_eDMD, x), Y)
norm(X - X_test_lqr[:,1])
plotstates(T_sim, X, inds=1:2)

#############################################
## Plot an MPC Trajectory
#############################################

i = 1
X_ref = deepcopy(X_test_mpc_ref[:,i])
U_ref = deepcopy(U_test_mpc_ref[:,i])
X_nom_mpc = deepcopy(X_test_mpc[:,i])
push!(U_ref, ue)
T_ref = range(0,t_sim,step=dt)
T_sim = range(0,t_sim*1.5,step=dt)
x0 = X_ref[1]

Qmpc = Diagonal([10, 100, 10, 1e-4, 1e-4, 1e-4])
Rmpc = Diagonal(fill(1e-4,2))
Qfmpc = Diagonal(fill(1e2,6))

mpc_eDMD_projected = TrackingMPC(model_eDMD_projected, 
    X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt)
mpc_jDMD_projected = TrackingMPC(model_jDMD_projected, 
    X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt)

X_mpc_eDMD_projected,= simulatewithcontroller(dmodel_real, mpc_eDMD_projected, x0, t_sim*1.5, dt)
X_mpc_jDMD_projected,= simulatewithcontroller(dmodel_real, mpc_jDMD_projected, x0, t_sim*1.5, dt)

plotstates(T_ref, X_ref, inds=1:3, xlabel="time (s)", ylabel="states",
            label=["x (true MPC)" "y (true MPC)" "θ (true MPC)"], legend=:bottomright, lw=2,
            linestyle=:dot, color=[1 2 3])
plotstates!(T_ref, X_nom_mpc, inds=1:3, xlabel="time (s)", ylabel="states",
            label=["x (nominal MPC)" "y (true MPC)" "θ (nominal MPC)"], legend=:bottomright, lw=2,
            linestyle=:dash, color=[1 2 3])
plotstates!(T_sim, X_mpc_eDMD_projected, inds=1:3, xlabel="time (s)", ylabel="states",
            label=["x (nominal EDMD)" "y (nominal EDMD)" "θ (nominal EDMD)"], legend=:bottomright, lw=2,
            linestyle=:dashdot, color=[1 2 3])
plotstates!(T_sim, X_mpc_jDMD_projected, inds=1:3, xlabel="time (s)", ylabel="states",
            label=["x (JDMD)" "y (JDMD)" "θ (JDMD)"], legend=:bottomright, lw=2,
            color=[1 2 3])
ylims!((-2.5,1.75))