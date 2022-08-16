using BilinearControl
using BilinearControl.EDMD
using BilinearControl.Problems
using BilinearControl: Problems
using Test

include(joinpath(BilinearControl.EXAMPLES_DIR,"airplane_problem.jl"))
include(joinpath(BilinearControl.EXAMPLES_DIR,"airplane_constants.jl"))
include(joinpath(BilinearControl.EXAMPLES_DIR,"airplane_utils.jl"))

airplane_data = load(AIRPLANE_DATAFILE)
airplane_data["dp_window"]
airplane_data["X_train"]
airplane_data["X_test"]
airplane_data["X_ref"]
airplane_data["T_ref"]

num_test = 50
X_mpc,U_mpc, X_ref,U_ref = gen_airplane_data(num_train=50, num_test=num_test, 
    dp_window=fill(0.5,3), dt=0.04, save_to_file=false)

# Test final velocity of reference
@test all(x->norm(x[7:end]) < 2, X_ref[end,:])

# Test final position of the reference
@test all(x->norm(x[1:3] - [5,0,1.5]) < 1e-3, X_ref[end,:])

# Test MPC trajectories
@test median(map(x->norm(x[1:3] - [5,0,1.5]), X_mpc[end,:])) < 1.0
@test maximum(map(x->norm(x[1:3] - [5,0,1.5]), X_mpc[end,:])) < 1.0

## Test airplane training
res = test_airplane(train_airplane(5)...)

did_track(x) = x<1e1
@test count(did_track, res[:jDMD]) / num_test == 1.0
@test count(did_track, res[:eDMD]) / num_test == 0.66
count(did_track, res[:eDMD]) / num_test

res = test_airplane(train_airplane(10)...)
@test count(did_track, res[:jDMD]) / num_test == 1.0
@test count(did_track, res[:eDMD]) / num_test == 1.0