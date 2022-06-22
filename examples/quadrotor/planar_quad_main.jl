using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using BilinearControl
using PGFPlotsX
using LaTeXStrings

include("planar_quad_utils.jl")
include("../plotting_constants.jl")

#####################################################
## LQR Stabilization w/ Equilibrium Offset (Fig 3a)
#####################################################
res_equilibrium = planar_quad_lqr_offset()

## Plot Results
res_equilibrium = load(PLANAR_QUAD_LQR_RESULTS)["res_equilibrium"]
distances = res_equilibrium[:distances]

p_lqr_equilibrium = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Equilibrium offset",
        ylabel = "Stabilization error",
        legend_pos = "north west",
        ymax = 200,
        
    },
    PlotInc({no_marks, color=color_eDMD, thick}, Coordinates(distances, res_equilibrium[:error_eDMD_projected_unreg])),
    PlotInc({lineopts..., color=color_eDMD, line_width=2}, Coordinates(distances, res_equilibrium[:error_eDMD_projected])),
    PlotInc({no_marks, color=color_jDMD, thick}, Coordinates(distances, res_equilibrium[:error_jDMD_projected])),
    PlotInc({lineopts..., color=color_jDMD, line_width=2}, Coordinates(distances, res_equilibrium[:error_jDMD_projected2])),

    # Legend(["eDMD" * L"(\lambda = 0.0)", "eDMD" * L"(\lambda = 0.1)", "jDMD" * L"(\lambda = 10^{-5})", "jDMD" * L"(\lambda = 0.1)"])
)
pgfsave(joinpath(BilinearControl.FIGDIR, 
    "rex_planar_quadrotor_lqr_error_by_equilibrium_change.tikz"), 
    p_lqr_equilibrium, 
    include_preamble=false
)

#####################################################
## MPC Tracking vs Window Size (Fig 3b) 
#####################################################
generate_planar_quadrotor_data()

planar_quad_mpc_generalization()

results = load(joinpath(BilinearControl.DATADIR, "rex_planar_quadrotor_mpc_training_range_results.jld2"))
percentages = results["percentages"]
res_training_range = results["res_training_range"]

p_tracking = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Fraction of training range",
        ylabel = "Tracking error",
        legend_pos = "north west"
    },
    PlotInc({no_marks, color=color_eDMD, thick}, Coordinates(percentages, res_training_range[:error_eDMD_loreg])),
    PlotInc({lineopts..., color=color_eDMD, line_width=2.0}, Coordinates(percentages, res_training_range[:error_eDMD_hireg])),
    PlotInc({no_marks, color=color_jDMD, thick}, Coordinates(percentages, res_training_range[:error_jDMD_loreg])),
    PlotInc({lineopts..., color=color_jDMD, line_width=2.0}, Coordinates(percentages, res_training_range[:error_jDMD_hireg])),
    Legend(["eDMD" * L"(\lambda = 0.0)", "eDMD" * L"(\lambda = 0.1)", "jDMD" * L"(\lambda = 10^{-5})", "jDMD" * L"(\lambda = 0.1)"])
)
pgfsave(joinpath(BilinearControl.FIGDIR, "rex_planar_quadrotor_mpc_error_by_training_window.tikz"), p_tracking, include_preamble=false)