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
## Test bilinear models
#############################################
## Run test trajectories
MPC_test_results = test_full_quadrotor()

## Print Summary
println("Test Summary:")
println("  Model  |  Success Rate ")
println("---------|-------------------")
println(" nom MPC |  ", MPC_test_results[:nom_success])
println("  eDMD   |  ", MPC_test_results[:eDMD_success])
println("  jDMD   |  ", MPC_test_results[:jDMD_success])
println("")
println("Test Summary:")
println("  Model  |  Avg Tracking Err ")
println("---------|-------------------")
println(" nom MPC |  ", MPC_test_results[:nom_err_avg])
println("  eDMD   |  ", MPC_test_results[:eDMD_err_avg])
println("  jDMD   |  ", MPC_test_results[:jDMD_err_avg])

## Save results to latex table 
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
            string(round(MPC_test_results[:nom_success] * 100, digits=2)) * "\\%", 
            string(round(MPC_test_results[:eDMD_success] * 100, digits=2)) * "\\%", 
            string(round(MPC_test_results[:jDMD_success] * 100, digits=2)) * "\\%", 
        ],
        Rule(:bottom)
    ]
)

