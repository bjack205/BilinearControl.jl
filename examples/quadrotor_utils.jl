using ProgressMeter
using Rotations
using ThreadsX
import TrajectoryOptimization as TO
using Altro

function heading2mrp(ψ)
    Rotations.params(Rotations.MRP(expm(ψ * [0,0,1])))
end

function nominal_trajectory(x0,N,dt; xf=zeros(3))
    Xref = [fill(NaN, length(x0)) for k = 1:N]
    
    pos_0 = x0[1:3]
    mrp_0 = x0[4:6]
    
    pos_ref = reshape(LinRange(pos_0, xf[1:3], N), N, 1)
    mrp_ref = reshape(LinRange(mrp_0, zeros(3), N), N, 1)
    angle_ref = map((x) -> Vector(Rotations.params(RotXYZ(MRP(x[1], x[2], x[3])))), mrp_ref)

    vel_pos_ref = vcat([(pos_ref[2] - pos_ref[1]) ./ dt for k = 1:N-1], [zeros(3)])
    vel_attitude_ref = vcat([(angle_ref[2] - angle_ref[1]) ./ dt for k = 1:N-1], [zeros(3)])

    for i = 1:N
        Xref[i] = vcat(pos_ref[i], mrp_ref[i], vel_pos_ref[i], vel_attitude_ref[i])
    end

    # Set initial and final velocities to zero
    Xref[1][7:12] .= 0
    Xref[end][7:12] .= 0
    
    return Xref
end

function build_mpc_problem(model, x0, Xref, Uref, Tref; Nmpc=21)
    if !(model isa RD.DiscreteDynamics)
        dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
    else
        dmodel = model
    end
    h = Tref[2] - Tref[1] 
    tf = Tref[Nmpc]

    # Create (Lie) Tracking Objective
    Q = Diagonal(SA[1, 1, 10, 0.5, 0.5, 0.5, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])
    R = Diagonal(@SVector fill(1e-0, 4))
    costs = map(1:Nmpc) do k
        s = k == Nmpc ? 1 : 1
        TO.LQRCost(Q * s, R, SVector{12}(Xref[k]), Uref[k])
    end
    obj = TO.Objective(costs)

    # Add constraints
    n, m = 12, 4
    constraints = TO.ConstraintList(n, m, Nmpc)
    # TO.add_constraint!(constraints, con, 2:Nmpc)

    # Create the Problem
    prob = TO.Problem(dmodel, obj, x0, tf; constraints)
    TO.initial_states!(prob, Xref)
    TO.initial_controls!(prob, Uref)
    return prob
end


function generate_quadrotor_data(; tf=5.0, dt=0.05, Nt=21, 
        num_train=50,
        num_test=20,
        workspace_size=[20,20,2], workspace_origin=[0,0,1]
    )
    #############################################
    ## Define the Models
    #############################################

    # Define Nominal Simulated Quadrotor Model
    model_nom = Problems.NominalRexQuadrotor()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" Quadrotor Model
    model_real = Problems.SimulatedRexQuadrotor()  # this model has aero drag
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    T = range(0, tf, step=dt)

    #############################################
    ## Generate initial and final conditions 
    #############################################
    println("Generating initial and final conditions...")
    pose_sampler = Product([[
        Uniform(
            workspace_origin[i] - workspace_size[i] / 2, 
            workspace_origin[i] + workspace_size[i] / 2
        ) for i = 1:3
    ]; Uniform(-pi,pi)])
    sample_state() = begin
        pose = rand(pose_sampler)
        [pose[1:3]; heading2mrp(pose[4]); zeros(6)]
    end

    Random.seed!(3)
    x0 = [sample_state() for i = 1:num_train + num_test]
    Random.seed!(4)
    xf = [sample_state() for i = 1:num_train + num_test]

    #############################################
    ## Generate reference trajectories 
    #############################################
    println("Setting up data...")
    Tref = range(0,tf,step=dt)
    Uref = [Problems.trim_controls(model_nom) for t in Tref]

    # Generate simulated trajectories
    T_sim = range(0,t_sim,step=dt)
    Nsim = length(T_sim)
    X_ref = Matrix{Vector{Float64}}(undef, length(Tref), num_train + num_test)
    opts = Altro.SolverOptions(show_summary=false)
    prog = Progress(num_train + num_test)

    println("Running MPC...")
    map(1:num_train + num_test) do i
        Xref = nominal_trajectory(x0[i], length(T), dt, xf=xf[i])
        X_ref[:,i] = Xref
    end
    X_sim, U_sim = run_mpc(X_ref; tf, t_sim, dt, opts)
    X_train = X_sim[:,1:num_train]
    U_train = U_sim[:,1:num_train]
    X_test = X_sim[:,num_train .+ (1:num_test)]
    U_test = U_sim[:,num_train .+ (1:num_test)]
    return X_train, U_train, X_test, U_test, X_ref
end

function run_mpc(X_ref; dt=0.05, tf=5.0, t_sim=tf*1.2, Nt=41,
        opts = Altro.SolverOptions(show_summary=true, verbose=4,
            cost_tolerance=1e-3, cost_tolerance_intermediate=1e-2,
            expected_decrease_tolerance=1e-4,
        )
    )
    model_nom = Problems.NominalQuadrotor()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)

    # Define Mismatched "Real" Cartpole Model
    model_real = Problems.SimulatedQuadrotor()
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    Q = Diagonal([1, 1, 10, 0.5, 0.5, 0.5, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])
    R = Diagonal(fill(1e-0, 4))
    Qf = copy(Q)
    xbnd = [fill(2.0,3); fill(0.2,3); fill(3.0,6)]
    ubnd = fill(5.0, 4)

    Tsim = range(0,t_sim,step=dt)
    Tref = range(0,tf,step=dt)
    Uref = [Problems.trim_controls(model_nom) for t in Tref]
    Nref,num_sims = size(X_ref)
    Nsim = length(Tsim)
    X_sim = [zeros(12) for ci in CartesianIndices((Nsim, num_sims))]
    U_sim = [zeros(4) for ci in CartesianIndices((Nsim-1, num_sims))]
    prog = Progress(num_sims)
    # for i = 1:num_sims
    # ThreadsX.foreach(1:num_sims) do i
    ThreadsX.map(1:num_sims) do i
        Tmpc = range(0,step=dt,length=Nt)
        Xref = X_ref[:,i]
        # prob = build_mpc_problem(model_nom, Xref[1], Xref, Uref, Tref; Nmpc=Nt)
        # mpc = AltroController(prob, Xref, Uref, Tref; opts)
        mpc = EDMD.LinearMPC(dmodel_nom, Xref, Uref, collect(Tref), Q, R, Qf; Nt,
            # xmax=xbnd, xmin=-xbnd, 
            # umax=ubnd, umin=-ubnd
        )
        X,U,T = simulatewithcontroller(dmodel_real, mpc, Xref[1], t_sim, dt)
        X_sim[:,i] = X
        U_sim[:,i] = U
        next!(prog)
        nothing
    end
    return X_sim, U_sim
end

function mpc_step(dmodel, mpc, i, x)
    Tmpc = TO.gettimes(mpc.solver)
    dt = Tmpc[2] - Tmpc[1]
    t = (i-1)*dt
    u = EDMD.getcontrol(mpc, x, t)
    x .= RD.discrete_dynamics(dmodel, x, u, t, dt)
    p = plotstates(mpc.Tref, mpc.Xref, inds=1:3, s=:dash)
    plotstates!(Tmpc .+ t, TO.states(mpc.solver), inds=1:3, lw=2, c=[1 2 3])
    display(p)
end