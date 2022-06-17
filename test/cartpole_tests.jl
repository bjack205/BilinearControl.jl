
using BilinearControl
using BilinearControl.EDMD
using BilinearControl.Problems
using BilinearControl: Problems
using Test
include(joinpath(BilinearControl.EXAMPLES_DIR,"cartpole_utils.jl"))

## Test Cartpole controller at specific sample size
println("\nTESTING CARTPOLE MPC CONTROLLER WITH N = ", 2)
const CARTPOLE_RESULTS_FILE = joinpath(Problems.DATADIR, "cartpole_results.jld2")
res = train_cartpole_models(0,2, α=0.5, β=1.0, learnB=true, reg=1e-4)
@test abs(res.nom_err_avg - 0.1563) < 1e-4
@test abs(res.eDMD_err_avg - 2.15667) < 1e-4
@test abs(res.jDMD_err_avg - 0.041855) < 1e-4
@test res.num_swingup == 2
@test res.jDMD_success == 7

println("\nTESTING CARTPOLE MPC CONTROLLER WITH N = ", 10)
res = train_cartpole_models(0,10, α=0.5, β=1.0, learnB=true, reg=1e-4)
@test abs(res.nom_err_avg - 0.1563) < 1e-4
@test abs(res.eDMD_err_avg - 0.096449) < 1e-4
@test abs(res.jDMD_err_avg - 0.044179) < 1e-4
@test res.num_swingup == 10 
@test res.jDMD_success == 7

## Test cartpole mismatch
test_window_ratio = 0.5
reg = 1e-4
num_train = [2:25; 25:5:100]
mu_vals = 0:0.1:0.6
num_test = 50
repeats_required = 4

α = 0.01
println("\nTESTING CARTPOLE MISMATCH WITH α = ", α)
@time res_jDMD = find_min_sample_to_stabilize(
    mu_vals, num_train; num_test, alg=:jDMD, test_window_ratio, reg, α,
    repeats_required
)
vals = map(mu->res_jDMD[mu], mu_vals)
@test vals == [2,2,2,2,3,7,12]

## Compare projected and lifted MPC
dt = 0.05
repeates_required = 4
println("FINDING MIN SAMPLES FOR PROJECTED MPC")
res_jDMD = find_min_sample_to_beat_mpc(2:10, dt; alg=:jDMD, lifted=false, 
    repeats_required, α=0.1
)
@test res_jDMD == 2

println("FINDING MIN SAMPLES FOR LIFTED MPC")
res_jDMD_lifted = find_min_sample_to_beat_mpc(13:25, dt; alg=:jDMD, lifted=true, 
    repeats_required, α=0.1
)
res_jDMD_lifted == 15