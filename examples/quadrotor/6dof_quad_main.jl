using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using BilinearControl
using SparseArrays
using PGFPlotsX
using LaTeXTabulars

include("6dof_quad_utils.jl")
include("../plotting_constants.jl")


#############################################
## Generate quadrotor data
#############################################
generate_quadrotor_data()

#############################################
## Train bilinear models
#############################################

num_lqr = 10
num_mpc = 20

res = train_quadrotor_models(num_lqr, num_mpc, α=0.5, β=1.0, learnB=true, reg=1e-6)

# Save model information to file
eDMD_data = res.eDMD_data
jDMD_data = res.jDMD_data
kf = jDMD_data[:kf]
G = sparse(jDMD_data[:g])
dt = res.dt
model_info = (; eDMD_data, jDMD_data, G, kf, dt)
jldsave(FULL_QUAD_MODEL_DATA; model_info)

#############################################
## Load models 
#############################################

model_info = load(FULL_QUAD_MODEL_DATA)["model_info"]
eDMD_data = model_info.eDMD_data
jDMD_data = model_info.jDMD_data
G = model_info.G
kf = model_info.kf
dt = model_info.dt

model_eDMD = EDMDModel(eDMD_data[:A],eDMD_data[:B],eDMD_data[:C],G,kf,dt,"quadrotor_eDMD")
model_eDMD_projected = ProjectedEDMDModel(model_eDMD)
model_jDMD = EDMDModel(jDMD_data[:A],jDMD_data[:B],jDMD_data[:C],G,kf,dt,"quadrotor_jDMD")
model_jDMD_projected = ProjectedEDMDModel(model_jDMD)

#############################################
## Run Test Trajectories
#############################################

# Define Nominal Simulated Quadrotor Model
model_nom = BilinearControl.NominalRexQuadrotor()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

# Define Mismatched "Real" Quadrotor Model
model_real = BilinearControl.SimulatedRexQuadrotor()  # this model has aero drag
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

# Load Test Trajectories
mpc_lqr_traj = load(FULL_QUAD_DATA)
X_test_infeasible = mpc_lqr_traj["X_test_infeasible"]
U_test_infeasible = mpc_lqr_traj["U_test_infeasible"]
N_test = size(X_test_infeasible,2)

# Metadata
tf = mpc_lqr_traj["tf"]
t_sim = 10.0
dt = mpc_lqr_traj["dt"]
T_ref = range(0,tf,step=dt)
T_sim = range(0,t_sim,step=dt)

# MPC Params
xe = zeros(12)
ue = BilinearControl.trim_controls(model_real)
Nt = 20  # MPC horizon
N_sim = length(T_sim)
N_ref = length(T_ref)

Qmpc = Diagonal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
Rmpc = Diagonal(fill(1e-3, 4))
Qfmpc = Qmpc*100

test_results = ThreadsX.map(1:N_test) do i
    # Get reference trajectory (a line back to the origin)
    X_ref = deepcopy(X_test_infeasible[:,i])
    U_ref = deepcopy(U_test_infeasible[:,i])
    X_ref[end] .= xe
    push!(U_ref, ue)

    # Use MPC controller to get it back to the origin
    X_ref_full = [X_ref; [copy(xe) for i = 1:N_sim - N_ref]]
    mpc_nom = BilinearControl.TrackingMPC_no_OSQP(dmodel_nom, 
        X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
    )
    mpc_eDMD = BilinearControl.TrackingMPC_no_OSQP(model_eDMD_projected, 
        X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
    )
    mpc_jDMD = BilinearControl.TrackingMPC_no_OSQP(model_jDMD_projected, 
        X_ref, U_ref, Vector(T_ref), Qmpc, Rmpc, Qfmpc; Nt=Nt
    )
    X_mpc_nom, U_mpc_nom, T_mpc = simulatewithcontroller(dmodel_real, mpc_nom,  X_ref[1], t_sim, dt)
    X_mpc_eDMD,U_mpc_eDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_eDMD, X_ref[1], t_sim, dt)
    X_mpc_jDMD,U_mpc_jDMD,T_mpc = simulatewithcontroller(dmodel_real, mpc_jDMD, X_ref[1], t_sim, dt)

    # Calculate average error over trajectory
    err_nom = norm(X_mpc_nom - X_ref_full) / N_sim
    err_eDMD = norm(X_mpc_eDMD - X_ref_full) / N_sim
    err_jDMD = norm(X_mpc_jDMD - X_ref_full) / N_sim

    (; err_nom, err_eDMD, err_jDMD, X_ref, X_mpc_nom, X_mpc_eDMD, X_mpc_jDMD, T_mpc)
end

# Post-process the results
nom_err_avg  = mean(filter(isfinite, map(x->x.err_nom, test_results)))
eDMD_err_avg = mean(filter(isfinite, map(x->x.err_eDMD, test_results)))
jDMD_err_avg = mean(filter(isfinite, map(x->x.err_jDMD, test_results)))
nom_success = count(x -> norm(x[end]-xe)<=10, map(x->x.X_mpc_nom, test_results)) / N_test
eDMD_success = count(x -> norm(x[end]-xe)<=10, map(x->x.X_mpc_eDMD, test_results)) / N_test
jDMD_success = count(x -> norm(x[end]-xe)<=10, map(x->x.X_mpc_jDMD, test_results)) / N_test

nom_errs  = map(x->x.err_nom, test_results)
eDMD_errs = map(x->x.err_eDMD, test_results)
jDMD_errs = map(x->x.err_jDMD, test_results)

X_ref = map(x->x.X_ref, test_results)
X_mpc_nom = map(x->x.X_mpc_nom, test_results)
X_mpc_eDMD = map(x->x.X_mpc_eDMD, test_results)
X_mpc_jDMD = map(x->x.X_mpc_jDMD, test_results)
T_mpc = map(x->x.T_mpc, test_results)

G = model_jDMD.g
kf = model_jDMD.kf

# Save data to file
MPC_test_results = (;X_ref, X_mpc_nom, X_mpc_eDMD, X_mpc_jDMD, T_mpc, nom_err_avg,
    nom_errs, eDMD_errs, jDMD_errs, nom_success, eDMD_err_avg,
    eDMD_success, jDMD_err_avg, jDMD_success)

jldsave(FULL_QUAD_RESULTS; MPC_test_results)

## Save results to file
MPC_test_results = load(FULL_QUAD_RESULTS)["MPC_test_results"]
latex_tabular(joinpath(BilinearControl.FIGDIR, "tables", "full_quad_mpc.tex"),
    Tabular("cccc"),
    [
        Rule(:top),
        ["", 
            "{\\color{black} \\textbf{Nominal}}",
            "{\\color{orange} \\textbf{EDMD}}",
            "{\\color{cyan} \\textbf{JDMD}}",
        ],
        Rule(:mid),
        ["Tracking Err.", 
            round(MPC_test_results[:nom_err_avg], digits=2), 
            round(MPC_test_results[:eDMD_err_avg], digits=2), 
            round(MPC_test_results[:jDMD_err_avg], digits=2), 
        ],
        ["Success Rate", 
            string(MPC_test_results[:nom_success] * 100) * "\\%", 
            string(MPC_test_results[:eDMD_success] * 100) * "\\%", 
            string(MPC_test_results[:jDMD_success] * 100) * "\\%", 
        ],
        Rule(:bottom)
    ]
)
