
using BilinearControl
using BilinearControl.EDMD
using Test
include(joinpath(BilinearControl.EXAMPLES_DIR,"cartpole_utils.jl"))

const CARTPOLE_RESULTS_FILE = joinpath(Problems.DATADIR, "cartpole_results.jld2")
res = train_cartpole_models(0,2, α=0.5, β=1.0, learnB=true, reg=1e-4)
@test abs(res.nom_err_avg - 0.1563) < 1e-4
@test abs(res.eDMD_err_avg - 2.15667) < 1e-4
@test abs(res.jDMD_err_avg - 0.041855) < 1e-4
@test res.num_swingup == 2
@test res.jDMD_success == 7

res = train_cartpole_models(0,10, α=0.5, β=1.0, learnB=true, reg=1e-4)
@test abs(res.nom_err_avg - 0.1563) < 1e-4
@test abs(res.eDMD_err_avg - 0.096449) < 1e-4
@test abs(res.jDMD_err_avg - 0.044179) < 1e-4
@test res.num_swingup == 10 
@test res.jDMD_success == 7

