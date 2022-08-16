using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using BilinearControl
using PGFPlotsX

include("cartpole_utils.jl")
include("../plotting_constants.jl")

#############################################
## Model Mismatch Study (Table 3) 
#############################################

## Set params
test_window_ratio = 0.5
reg = 1e-4
num_train = [2:25; 25:5:100]
mu_vals = 0:0.1:0.6
num_test = 50
repeats_required = 4

# Test jDMD with alpha = 0.01
α = 0.01
@time res_jDMD = find_min_sample_to_stabilize(
    mu_vals, num_train; num_test, alg=:jDMD, test_window_ratio, reg, α,
    repeats_required
)

# Test eDMD
@time res_eDMD = find_min_sample_to_stabilize(
    mu_vals, num_train; num_test, alg=:eDMD, test_window_ratio, reg=1e-6, α,
    repeats_required
)

jldsave(CARTPOLE_MISMATCH_RESULTS; res_jDMD, res_eDMD, mu_vals, num_train)

## Process Results
results_mismatch = load(CARTPOLE_MISMATCH_RESULTS)
mu_vals = results_mismatch["mu_vals"]
num_train = results_mismatch["num_train"]
getsamples(d) = map(k -> getindex(d, k), mu_vals)
res_jDMD = getsamples(results_mismatch["res_jDMD"])
res_eDMD = getsamples(results_mismatch["res_eDMD"])

#############################################
## Projected vs Lifted Comparison (Table 2)
#############################################

dt = 0.05
num_train = 2:15 
repeats_required = 4

# stabilized with 2
res_jDMD = find_min_sample_to_beat_mpc(2:10, dt; alg=:jDMD, lifted=false, 
    repeats_required, α=0.1
)

# stabilized with 15 @ alpha = 0.1
res_jDMD_lifted = find_min_sample_to_beat_mpc(2:1:100, dt; alg=:jDMD, lifted=true, 
    repeats_required, α=0.1
)

# Stabilized with 18 samples
# - bumping up regularization to 1e-4 doesn't improve it
res_eDMD = find_min_sample_to_beat_mpc(2:40, dt; alg=:eDMD, lifted=false, repeats_required=4)

# Stabilized with 17 
res_eDMD_lifted = find_min_sample_to_beat_mpc(2:60, dt; alg=:eDMD, lifted=true, 
    repeats_required=4,
)

#############################################
## MPC Performance Sample Efficient (Fig. 2a)
#############################################
generate_cartpole_data()  # OPTIONAL (data file already exists)
train_cartpole_models(0,2, α=0.5, β=1.0, learnB=true, reg=1e-4)  # run once for JIT
num_swingup = 2:2:36
results = map(num_swingup) do N
    println("\nRunning with N = $N")
    res = train_cartpole_models(0,N, α=0.5, β=1.0, learnB=true, reg=1e-4)
    @show res.jDMD_err_avg
    @show res.eDMD_err_avg
    res
end
jldsave(CARTPOLE_MPC_RESULTS; results)

## Plot Results  (Fig. 2a)
results = load(CARTPOLE_MPC_RESULTS)["results"]
fields = keys(results[1])
res = Dict(Pair.(fields, map(x->getfield.(results, x), fields)))

p = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Number of training trajectories",
        ylabel = "Tracking error",
        ymax=0.2,
    },
    PlotInc({lineopts..., color=color_nominal}, Coordinates(res[:num_swingup], res[:nom_err_avg])),
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(res[:num_swingup], res[:eDMD_err_avg])),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(res[:num_swingup], res[:jDMD_err_avg])),
    # Legend(["Nominal", "eDMD", "jDMD"])
)
pgfsave(joinpath(Problems.FIGDIR, "cartpole_mpc_test_error.tikz"), p, include_preamble=false)
