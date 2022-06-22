using LinearAlgebra
using Distributions
using Statistics
using Random
using ProgressMeter
using ThreadsX
using JLD2
import RobotDynamics as RD

const PLANAR_QUAD_LQR_RESULTS = joinpath(BilinearControl.DATADIR, "planar_quad_lqr_results.jld2")

function test_initial_conditions_offset(model, controller, xg, ics, tf, dt)
    map(ics) do x0
        X_sim, = simulatewithcontroller(model, controller, x0+xg, tf, dt)
        norm(X_sim[end] - xg)
    end
end

function equilibrium_offset_test(; num_train=30, verbose=true, save_to_file=true)
    Random.seed!(1)

    ############################################# 
    ## Define the Nominal and True Models
    ############################################# 
    model_nom = BilinearControl.NominalPlanarQuadrotor()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    model_real = BilinearControl.SimulatedPlanarQuadrotor()  # this model has aero drag
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)


    ############################################# 
    ## Generate Training Data
    ############################################# 
    verbose && println("Generating Training Data...")
    tf = 5.0
    dt = 0.05

    # Generate a stabilizing LQR controller
    Qlqr = Diagonal([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    Rlqr = Diagonal([1e-4, 1e-4])
    xe = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ue = BilinearControl.trim_controls(model_real)
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
    initial_conditions_lqr = [rand(x0_train_sampler) for _ in 1:num_train]

    # Create data set
    X_train, U_train = BilinearControl.create_data(dmodel_real, ctrl_lqr_nom, initial_conditions_lqr, tf, dt)

    ############################################# 
    ## Train the Bilinear Models 
    ############################################# 
    verbose && println("Training Bilinear Models...")
    eigfuns = ["state", "sine", "cosine", "chebyshev"]
    eigorders = [[0],[1],[1],[2,2]]

    model_eDMD = run_eDMD(
        X_train, U_train, dt, eigfuns, eigorders, reg=1e-1, name="planar_quadrotor_eDMD"
    )
    model_eDMD_unreg = run_eDMD(
        X_train, U_train, dt, eigfuns, eigorders, reg=0.0, name="planar_quadrotor_eDMD"
    )
    model_jDMD = run_jDMD(
        X_train, U_train, dt, eigfuns, eigorders, dmodel_nom, reg=1e-5, 
        name="planar_quadrotor_jDMD"
    )
    model_jDMD2 = run_jDMD(
        X_train, U_train, dt, eigfuns, eigorders, dmodel_nom, reg=1e-1, 
        name="planar_quadrotor_jDMD"
    )

    # Generate Projected Bilinear Models
    model_eDMD_projected = BilinearControl.ProjectedEDMDModel(model_eDMD)
    model_eDMD_projected_unreg = BilinearControl.ProjectedEDMDModel(model_eDMD_unreg)
    model_jDMD_projected = BilinearControl.ProjectedEDMDModel(model_jDMD)
    model_jDMD_projected2 = BilinearControl.ProjectedEDMDModel(model_jDMD2)

    ####################################################
    ## Test Performance as Equilibrium Position Changes
    ####################################################
    verbose && println("Testing Performance vs Equilibrium offset...")
    distances = 0:0.1:4
    prog = Progress(length(distances))
    errors = ThreadsX.map(distances) do dist

        # println("equilibrium offset = $dist")
        t_sim = 5.0

        if dist == 0
            xe_test = [zeros(6)]
        else
            xe_sampler = Product([
                Uniform(-dist, +dist),
                Uniform(-dist, +dist),
            ])
            xe_test = [vcat(rand(xe_sampler), zeros(4)) for i = 1:100]
        end

        perc = 0.8
        x0_sampler = Product([
            Uniform(-1.0*perc,1.0*perc),
            Uniform(-1.0*perc,1.0*perc),
            Uniform(-deg2rad(40*perc),deg2rad(40*perc)),
            Uniform(-0.5*perc,0.5*perc),
            Uniform(-0.5*perc,0.5*perc),
            Uniform(-0.25*perc,0.25*perc)
        ])

        x0_test = [rand(x0_sampler) for i = 1:100]


        xe_results = map(xe_test) do xe
            lqr_eDMD_projected = LQRController(
                model_eDMD_projected, Qlqr, Rlqr, xe, ue, dt, max_iters=10000)
            lqr_eDMD_projected_unreg = LQRController(
                model_eDMD_projected_unreg, Qlqr, Rlqr, xe, ue, dt, max_iters=10000)
            lqr_jDMD_projected = LQRController(
                model_jDMD_projected, Qlqr, Rlqr, xe, ue, dt, max_iters=10000)
            lqr_jDMD_projected2 = LQRController(
                model_jDMD_projected2, Qlqr, Rlqr, xe, ue, dt, max_iters=10000)
        
            error_eDMD_projected_x0s = mean(test_initial_conditions_offset(
                dmodel_real, lqr_eDMD_projected, xe, x0_test, t_sim, dt))
            error_eDMD_projected_unreg_x0s = mean(test_initial_conditions_offset(
                dmodel_real, lqr_eDMD_projected_unreg, xe, x0_test, t_sim, dt))
            error_jDMD_projected_x0s = mean(test_initial_conditions_offset(
                dmodel_real, lqr_jDMD_projected, xe, x0_test, t_sim, dt))
            error_jDMD_projected2_x0s = mean(test_initial_conditions_offset(
                dmodel_real, lqr_jDMD_projected2, xe, x0_test, t_sim, dt))

            if error_eDMD_projected_x0s > 1e3
                error_eDMD_projected_x0s = NaN
            end
            if error_eDMD_projected_unreg_x0s > 1e3
                error_eDMD_projected_unreg_x0s = NaN
            end
            if error_jDMD_projected_x0s > 1e3
                error_jDMD_projected_x0s = NaN
            end
            if error_jDMD_projected2_x0s > 1e3
                error_jDMD_projected2_x0s = NaN
            end
            (;error_eDMD_projected_x0s, error_eDMD_projected_unreg_x0s, error_jDMD_projected_x0s, error_jDMD_projected2_x0s)
        end
        
        error_eDMD_projected = mean(filter(isfinite, map(x->x.error_eDMD_projected_x0s, xe_results)))
        error_eDMD_projected_unreg = mean(filter(isfinite, map(x->x.error_eDMD_projected_unreg_x0s, xe_results)))
        error_jDMD_projected = mean(filter(isfinite, map(x->x.error_jDMD_projected_x0s, xe_results)))
        error_jDMD_projected2 = mean(filter(isfinite, map(x->x.error_jDMD_projected2_x0s, xe_results)))
        next!(prog)

        (;error_eDMD_projected, error_eDMD_projected_unreg, error_jDMD_projected, error_jDMD_projected2)

    end

    fields = keys(errors[1])
    res_equilibrium = Dict(Pair.(fields, map(x->getfield.(errors, x), fields)))
    res_equilibrium[:distances] = distances
    if save_to_file
        jldsave(PLANAR_QUAD_LQR_RESULTS; res_equilibrium)
    end
    return res_equilibrium
end