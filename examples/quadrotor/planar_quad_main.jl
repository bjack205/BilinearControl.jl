using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using BilinearControl
using PGFPlotsX

include("planar_quad_utils.jl")
include("../plotting_constants.jl")

#####################################################
## LQR Stabilization w/ Equilibrium Offset (Fig 3a)
#####################################################
res_equilibrium = equilibrium_offset_test()
distances = res_equilibrium[:distances]

## Plot Results
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