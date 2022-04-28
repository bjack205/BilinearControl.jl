
function DubinsProblem(model=BilinearDubins(); 
        scenario=:turn90, N=101, ubnd=1.5
    )
    # model
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)
    n,m = RD.dims(model)

    tf = 3.
    dt = tf / (N-1)

    # cost
    d = 1.5
    x0 = @SVector [0., 0., 0.]
    if scenario == :turn90
        xf = @SVector [d, d,  deg2rad(90)]
    else
        xf = @SVector [0, d, 0.]
    end
    Qf = 100.0*Diagonal(@SVector ones(n))
    Q = (1e-2)*Diagonal(@SVector ones(n))
    R = (1e-2)*Diagonal(@SVector ones(m))

    if model isa BilinearDubins
        x0 = expandstate(model, x0)
        xf = expandstate(model, xf)
        Q = Diagonal([diag(Q)[1:2]; fill(Q[3,3]*1e-3, 2)]) 
        Qf = Diagonal([diag(Qf)[1:2]; fill(Qf[3,3]*1e-3, 2)]) 
    end

    # objective 
    obj = LQRObjective(Q*dt,R*dt,Qf,xf,N)

    # Initial Guess
    U = [@SVector fill(0.1,m) for k = 1:N-1]

    # constraints
    cons = ConstraintList(n,m,N)
    add_constraint!(cons, GoalConstraint(xf), N)
    add_constraint!(cons, BoundConstraint(n,m, u_min=-ubnd, u_max=ubnd), 1:N-1)

    if scenario == :parallelpark
        x_min = @SVector [-0.25, -0.1, -Inf]
        x_max = @SVector [0.25, d + 0.1, Inf]
        if model isa BilinearDubins
            x_min = push(x_min, -Inf) 
            x_max = push(x_max, +Inf) 
        end
        bnd_x = BoundConstraint(n,m, x_min=x_min, x_max=x_max)
        add_constraint!(cons, bnd_x, 2:N-1)
    end

    prob = Problem(dmodel, obj, x0, tf, xf=xf, U0=U, constraints=cons)
    rollout!(prob)

    return prob
end

function DubinsMPCProblem(Zref; N=51, tf=2.0, kstart=1)
    model = BilinearDubins()
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)
    nx,nu = RD.dims(model)

    x0 = RD.state(Zref[1])
    dt = tf / (N-1)

    # cost
    Q = Diagonal(SA[1.0, 1.0, 1e-2, 1e-2])
    R = Diagonal(@SVector fill(1e-4, nu))
    Z = RD.SampledTrajectory(Zref[kstart - 1 .+ (1:N)])
    Z[end].dt = 0.0
    obj = TO.TrackingObjective(Q, R, Z)

    prob = TO.Problem(dmodel, obj, x0, tf)
    TO.initial_trajectory!(prob, Z)
    prob
end

function SE3Problem()
    # Model
    model = SE3Kinematics()
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

    # Discretization
    tf = 3.0
    N = 301

    # Dimensions
    nx = RD.state_dim(model)
    nu = RD.control_dim(model)

    # Initial and final conditions
    x0 = [zeros(3); vec(I(3))]
    xf = [5; 0; 1; vec(RotZ(deg2rad(90)) * RotX(deg2rad(90)))]

    # Objective
    Q = Diagonal([fill(1e-1, 3); fill(1e-2, 9)])
    R = Diagonal([fill(1e-2, 3); fill(1e-2, 3)])
    Qf = Q*10
    obj = LQRObjective(Q,R,Qf,xf,N)

    # Goal state
    cons = ConstraintList(nx, nu, N)
    goalcon = GoalConstraint(xf)
    add_constraint!(cons, goalcon, N)

    # Initial Guess
    U0 = [fill(0.1,nu) for k = 1:N-1] 

    # Build the problem
    Problem(dmodel, obj, x0, tf, xf=xf, constraints=cons, U0=U0)
end

function AttitudeProblem(::Val{Nu}) where Nu
    # Model
    model = AttitudeDynamics{Nu}()  # i.e. quaternion
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

    # Discretization
    tf = 3.0
    N = 301

    # Dimensions
    nx = RD.state_dim(model)
    nu = RD.control_dim(model)

    # Initial and final conditions
    x0 = [1,0,0,0]
    if Nu == 3
        xf = [0.382683, 0.412759, 0.825518, 0.0412759]
    else
        xf = normalize([1,0,0,1.0])  # flip 90° around unactuated axis
    end

    # Objective
    Q = Diagonal(fill(1e-1, nx))
    R = Diagonal(fill(2e-2, nu))
    Qf = Diagonal(fill(100.0, nx))
    obj = LQRObjective(Q,R,Qf,xf,N)

    # Goal state
    cons = ConstraintList(nx, nu, N)
    goalcon = GoalConstraint(xf)  # only constraint the original states
    add_constraint!(cons, goalcon, N)

    # Initial Guess
    U0 = [fill(0.1,Nu) for k = 1:N-1] 

    # Build the problem
    Problem(dmodel, obj, x0, tf, xf=xf, constraints=cons, U0=U0)
end

function SO3Problem(::Val{Nu}; Rf::Rotation{3}=RotZ(deg2rad(90)), ubnd=Inf) where Nu
    # Model
    model = SO3Dynamics{Nu}()
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

    # Discretization
    tf = 3.0
    N = 301

    # Dimensions
    nx = RD.state_dim(model)
    nu = RD.control_dim(model)

    # Initial and final conditions
    x0 = vec(I(3))
    xf = vec(Rf)

    # Objective
    Q = Diagonal(fill(0.0, nx))
    R = Diagonal(fill(2e-2, nu))
    Qf = Diagonal(fill(100.0, nx))
    # costs = map(1:N) do k
    #     q = -xf  # tr(Rf'R)
    #     r = zeros(nu)
    #     TO.DiagonalCost(Q,R,q,r,0.0)
    # end
    # obj = TO.Objective(costs)
    obj = LQRObjective(Q,R,Qf,xf,N)

    # Goal state
    cons = ConstraintList(nx, nu, N)
    goalcon = GoalConstraint(xf)
    add_constraint!(cons, goalcon, N)

    # Control bound constraints
    add_constraint!(cons, BoundConstraint(nx,nu, u_min=-ubnd, u_max=ubnd), 1:N-1)

    # Initial Guess
    U0 = [fill(0.1,nu) for k = 1:N-1] 

    # Build the problem
    Problem(dmodel, obj, x0, tf, xf=xf, constraints=cons, U0=U0)
end

function SE3ForceProblem()
    # Model
    mass = 2.9
    model = Se3ForceDynamics(mass)
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

    # Discretization
    tf = 3.0
    N = 101

    # Dimensions
    nx = RD.state_dim(model)
    nu = RD.control_dim(model)
    nb = base_state_dim(model)
    ns = nx - nb 

    # Initial and final conditions
    x0_ = [0;0;0; vec(RotZ(deg2rad(0))); zeros(3)]
    xf_ = [3;0;1; vec(RotZ(deg2rad(90)) * RotX(deg2rad(150))); zeros(3)]
    x0 = expandstate(model, x0_)
    xf = expandstate(model, xf_)

    # Objective
    Q = Diagonal([fill(1e-1, 3); fill(1e-2, 9); fill(1e-2, 3); fill(0.0, ns)])
    R = Diagonal([fill(1e-2, 3); fill(1e-2, 3)])
    Qf = 100 * Q 
    obj = LQRObjective(Q,R,Qf,xf,N)

    # Goal state
    cons = ConstraintList(nx, nu, N)
    goalcon = GoalConstraint(xf, 1:nb)  # only constraint the original states
    add_constraint!(cons, goalcon, N)

    # Initial Guess
    U0 = [fill(0.1,nu) for k = 1:N-1] 

    # Build the problem
    prob = Problem(dmodel, obj, x0, tf, xf=xf, constraints=cons, U0=U0)
    rollout!(RD.InPlace(), prob)
    prob
end

function QuadrotorProblem()
    model = QuadrotorSE23()
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

    # Discretization
    tf = 3.0
    N = 101

    # Dimensions
    nx = RD.state_dim(model)
    nu = RD.control_dim(model)

    # Initial and final state
    x0 = [0; 0; 1.0; vec(I(3)); zeros(3)]
    xf = [5; 0; 2.0; vec(RotZ(deg2rad(90))); zeros(3)]

    # Objective
    Q = Diagonal([fill(1e-2, 3); fill(1e-2, 9); fill(1e-2, 3)])
    R = Diagonal([fill(1e-2,3); 1e-2])
    Qf = (N-1)*Q
    uhover = SA[0,0,0,model.mass*model.gravity]
    obj = LQRObjective(Q,R,Qf,xf,N, uf=uhover)

    # Goal state
    cons = ConstraintList(nx, nu, N)
    goalcon = GoalConstraint(xf)

    # Initial Guess
    U0 = [copy(uhover) for i = 1:N-1]

    prob = Problem(dmodel, obj, x0, tf, constraints=cons, U0=U0, xf=xf)
    rollout!(prob)
    prob
end

function QuadrotorRateLimitedSolver()
    model = QuadrotorRateLimited()

    # Discretization
    tf = 3.0
    N = 101
    h = tf / (N-1)

    # Initial and Final states
    x0 = [0; 0; 1.0; vec(I(3)); zeros(3); zeros(3)]
    xf = [5; 0; 2.0; vec(RotZ(deg2rad(90))); zeros(3); zeros(3)]
    uhover = [0,0,0,model.mass*model.gravity]

    # Build bilinear constraint matrices
    Abar,Bbar,Cbar,Dbar = BilinearControl.buildbilinearconstraintmatrices(
        model, x0, xf, h, N
    )

    # Build cost
    Q = Diagonal([fill(1e-2, 3); fill(1e-2, 9); fill(1e-2, 3); fill(1e-1, 3)])
    Qf = Q*(N-1)
    R = Diagonal([fill(1e-2,3); 1e-2])
    Qbar = Diagonal(vcat([diag(Q) for i = 1:N-1]...))
    Qbar = Diagonal([diag(Qbar); diag(Qf)])
    Rbar = Diagonal(vcat([diag(R) for i = 1:N]...))
    q = repeat(-Q*xf, N)
    r = repeat(-R*uhover, N)
    c = 0.5*sum(dot(xf,Q,xf) for k = 1:N-1) + 0.5*dot(xf,Qf,xf) + 
        0.5*sum(dot(uhover,R,uhover) for k = 1:N)

    X = repeat(x0, N)
    U = repeat(uhover, N)
    admm = BilinearADMM(Abar,Bbar,Cbar,Dbar, Qbar,q,Rbar,r,c)
    admm.x .= X
    admm.z .= U
    admm.opts.penalty_threshold = 1e2
    BilinearControl.setpenalty!(admm, 1e4)
    admm
end

function QuadrotorLanding(; tf=3.0, N=101, θ_glideslope=NaN)
    model = QuadrotorRateLimited()
    n = RD.state_dim(model)

    # Discretization
    h = tf / (N-1)

    # Initial and Final states
    x0 = [3; 3; 5.0; vec(I(3)); 0; 0; -3; zeros(3)]
    xf = [0; 0; 0.0; vec(I(3)); zeros(3); zeros(3)]
    uhover = [0,0,0,model.mass*model.gravity]

    # Build bilinear constraint matrices
    Abar,Bbar,Cbar,Dbar = BilinearControl.buildbilinearconstraintmatrices(
        model, x0, xf, h, N
    )

    # Build cost
    Q = Diagonal([fill(1e-2, 3); fill(1e-2, 9); fill(1e-2, 3); fill(1e-1, 3)])
    Qf = Q*(N-1)
    R = Diagonal([fill(1e-2,3); 1e-2])
    Qbar = Diagonal(vcat([diag(Q) for i = 1:N-1]...))
    Qbar = Diagonal([diag(Qbar); diag(Qf)])
    Rbar = Diagonal(vcat([diag(R) for i = 1:N]...))
    q = repeat(-Q*xf, N)
    r = repeat(-R*uhover, N)
    c = 0.5*sum(dot(xf,Q,xf) for k = 1:N-1) + 0.5*dot(xf,Qf,xf) + 
        0.5*sum(dot(uhover,R,uhover) for k = 1:N)

    # Initial guess
    X = repeat(x0, N)
    U = repeat(uhover, N)

    # Constraints
    Nx = length(X)
    if !isnan(θ_glideslope)
        α = tan(θ_glideslope)
        b = zeros(3)
        constraints = map(1:N-1) do k
            A = spzeros(3, Nx)
            for i = 1:3
                A[1,(k-1)*n + 3] = α
                A[2,(k-1)*n + 1] = 1.0
                A[3,(k-1)*n + 1] = 1.0
            end
            COSMO.Constraint(A, b, COSMO.SecondOrderCone)
        end
    else
        constraints = COSMO.Constraint{Float64}[]
    end

    admm = BilinearADMM(Abar,Bbar,Cbar,Dbar, Qbar,q,Rbar,r,c, constraints=constraints)
    admm.x .= X
    admm.z .= U
    admm.opts.penalty_threshold = 1e2
    BilinearControl.setpenalty!(admm, 1e4)
    admm
end

function SE3TorqueProblem(;N=11, tf=2.0, 
    xf = [vec(RotX(deg2rad(45)) * RotZ(deg2rad(180))); zeros(3)]
)
    J = Diagonal(SA[1,1,2.])
    dmodel = ConsensusDynamics(J)
    n,m = RD.dims(dmodel)

    x0 = [vec(I(3)); zeros(3)]
    u0 = zeros(m)
    h = tf / (N-1)

    # Build bilinear constraint
    A,B,C,D = BilinearControl.buildbilinearconstraintmatrices(dmodel, x0, xf, h, N)

    # Cost Objective
    Q = Diagonal([fill(1e-1, 9); fill(1e-1, 3)])
    Qf = Q*(N-1)
    R = Diagonal([fill(1e-1,3); fill(1e-3,3)])
    Qbar = Diagonal(vcat([diag(Q) for i = 1:N-1]...))
    Qbar = Diagonal([diag(Qbar); diag(Qf)])
    Rbar = Diagonal(vcat([diag(R) for i = 1:N]...))
    q = repeat(-Q*xf, N)
    r = repeat(-R*u0, N)
    c = 0.5*sum(dot(xf,Q,xf) for k = 1:N-1) + 0.5*dot(xf,Qf,xf) + 
        0.5*sum(dot(u0,R,u0) for k = 1:N)

    # Initial trajectory
    X = repeat(x0, N)
    U = repeat(u0, N)

    # Solver
    admm = BilinearADMM(A,B,C,D, Qbar,q,Rbar,r,c)
    admm.x .= X
    admm.z .= U
    admm.opts.penalty_threshold = 1e2
    BilinearControl.setpenalty!(admm, 1e4)

    admm
end

function generate_linear_models(;h=0.02)
    datadir = joinpath(@__DIR__, "..", "data")

    function linearizeaboutzero(model, h)
        n,m = RD.dims(model)
        x = zeros(n)
        u = zeros(m)
        DiscreteLinearModel(model, copy(x), x, u, h)
    end

    function savelinearmodel(data, model, name)
        A = BilinearControl.getA(model)
        B = BilinearControl.getB(model)
        C = BilinearControl.getC(model)
        d = BilinearControl.getD(model)
        data[name] = Dict(
            "A" => A,
            "B" => B,
            "C" => C,
            "d" => d,
        )
    end

    data = Dict{String, Dict{String, AbstractVecOrMat}}()

    # Swarm models
    di2 = DoubleIntegrator{2}(gravity=zeros(2))
    swarm24 = Swarm{4}(di2)
    swarm220 = Swarm{20}(di2)
    swarm24_linear = linearizeaboutzero(swarm24, h) 
    swarm220_linear = linearizeaboutzero(swarm220, h) 
    savelinearmodel(data, swarm24_linear, "swarm4")
    savelinearmodel(data, swarm220_linear, "swarm20")
    
    # Dubins
    dd = BilinearDubins()
    x0 = [0,0,1,0.]
    u0 = zeros(2)
    dubins_linear = DiscreteLinearModel(dd, copy(x0), x0, u0, h)
    G = SA[1 0 0; 0 1 0; 0 0 -x0[3]; 0 0 x0[4]]
    dubins_linear = DiscreteLinearModel(
        G'dubins_linear.A*G, 
        G'dubins_linear.B, 
        G'dubins_linear.C*G, 
        G'dubins_linear.d
    )
    savelinearmodel(data, dubins_linear, "dubins")

    # SO3
    so33 = SO3Dynamics{3}()
    x0 = vec(I(3))
    u0 = zeros(3)
    so33_linear = DiscreteLinearModel(so33, copy(x0), x0, u0, h)
    savelinearmodel(data, so33_linear, "so3_fullyactuated")

    so32 = SO3Dynamics{2}()
    x0 = vec(I(3))
    u0 = zeros(2)
    so32_linear = DiscreteLinearModel(so32, copy(x0), x0, u0, h)
    savelinearmodel(data, so32_linear, "so3_underactuated")
    
    # Quadrotor-SE2(3)
    quad = QuadrotorSE23()
    x0 = [zeros(3); vec(I(3)); zeros(3)]
    u0 = [zeros(3); quad.mass * quad.gravity] 
    quad_linear = DiscreteLinearModel(quad, copy(x0), x0, u0, h)
    savelinearmodel(data, quad_linear, "quad")

    jldsave(joinpath(datadir, "linear_models.jld2"); data)
    data
end

expandstate(::RobotZoo.Cartpole, x) = x

function Cartpole(model = RobotZoo.Cartpole(); constrained::Bool=true, N=101, 
        Qv=1e-2, Rv=1e-1, Qfv=1e2, u_bnd=3.0, tf=5.0)
    n,m = RD.dims(model)
    if model isa BilinearCartpole
        N = round(Int, tf / model.dt) + 1
        dt = tf / (N-1)
    else
        dt = tf/(N-1)
    end
    n0,m0 = 4,1    
    nd = n - n0

    Q = Qv*Diagonal([(@SVector ones(n0)); @SVector zeros(nd)]) * dt
    Qf = Qfv*Diagonal([(@SVector ones(n0)); @SVector zeros(nd)])
    R = Rv*Diagonal(@SVector ones(m)) * dt
    x0 = expandstate(model, @SVector zeros(n0))
    xf = expandstate(model, @SVector [0, pi, 0, 0])
    obj = LQRObjective(Q,R,Qf,xf,N)

    u_bnd = u_bnd 
    conSet = ConstraintList(n,m,N)
    bnd = ControlBound(m, u_min=-u_bnd, u_max=u_bnd)
    goal = GoalConstraint(xf)
    if constrained
        add_constraint!(conSet, bnd, 1:N-1)
        add_constraint!(conSet, goal, N:N)
    end

    # X0 = [@SVector fill(NaN,n) for k = 1:N]
    u0 = @SVector fill(0.01,m)
    U0 = [copy(u0) for k = 1:N-1]
    prob = Problem(model, obj, x0, tf, constraints=conSet, U0=U0)
    rollout!(prob)
    prob
end

function BilinearCartpoleProblem(; constrained::Bool=true, N=101, 
        Qv=1e-2, Rv=1e-1, Qfv=1e2, u_bnd=3.0, tf=5.0)
    model = BilinearCartpole()
    n,m = RD.dims(model)
    if model isa BilinearCartpole
        N = round(Int, tf / model.dt) + 1
        dt = tf / (N-1)
    else
        dt = tf/(N-1)
    end
    n0,m0 = 4,1    
    nd = n - n0

    Qd = fill(Qv, n0)
    Qfd = fill(Qfv, n0)
    Q = Diagonal([0; fill(Qv, n0); zeros(n-n0-1)])
    Qf = Diagonal([0; fill(Qfv, n0); zeros(n-n0-1)])
    # Q = Diagonal(fill(Qv, n))
    # Qf = Diagonal(fill(Qfv, n))
    # Q = Diagonal(expandstate(model, Qd))
    # Qf = Diagonal(expandstate(model, Qfd))
    # Q = Qv*Diagonal([(@SVector ones(n0)); @SVector zeros(nd)]) * dt
    # Qf = Qfv*Diagonal([(@SVector ones(n0)); @SVector zeros(nd)])
    R = Rv*Diagonal(@SVector ones(m)) * dt
    x0 = expandstate(model, @SVector zeros(n0))
    xf = @SVector [0, pi, 0, 0]
    zf = expandstate(model, xf)
    obj = LQRObjective(Q,R,Qf,zf,N)

    u_bnd = u_bnd 
    conSet = ConstraintList(n,m,N)
    bnd = ControlBound(m, u_min=-u_bnd, u_max=u_bnd)
    goal = GoalConstraint(zf, 1 .+ (1:n0))
    if constrained
        # add_constraint!(conSet, bnd, 1:N-1)
        add_constraint!(conSet, goal, N:N)
    end

    # X0 = [@SVector fill(NaN,n) for k = 1:N]
    u0 = @SVector fill(0.01,m)
    U0 = [copy(u0) for k = 1:N-1]
    prob = Problem(model, obj, x0, tf, constraints=conSet, U0=U0)
    rollout!(prob)
    prob
end

function PendulumProblem()

    model = RobotZoo.Pendulum()
    n,m = RD.dims(model)
    tf = 3.0
    N = 51
    dt = tf / (N-1)

    # cost
    Q = 1e-3*Diagonal(@SVector ones(n))
    R = 1e-3*Diagonal(@SVector ones(m))
    Qf = 1e-0*Diagonal(@SVector ones(n))
    x0 = @SVector zeros(n)
    xf = @SVector [pi, 0.0]  # i.e. swing up
    obj = LQRObjective(Q*dt,R*dt,Qf,xf,N)

    # constraints
    conSet = ConstraintList(n,m,N)
    u_bnd = 3.
    bnd = BoundConstraint(n,m,u_min=-u_bnd,u_max=u_bnd)
    goal = GoalConstraint(xf)
    add_constraint!(conSet, bnd, 1:N-1)
    add_constraint!(conSet, goal, N:N)

    # problem
    times = range(0,tf,length=N)
    U = [SA[cos(t/2)] for t in times]
    pendulum_static = Problem(model, obj, x0, tf, constraints=conSet, xf=xf)
    initial_controls!(pendulum_static, U)
    rollout!(pendulum_static)
    return pendulum_static
end

function BilinearPendulumProblem(;constraints=true, u_bnd=7.0)
    model = BilinearPendulum()
    n,m = RD.dims(model)
    tf = 3.0
    dt = model.dt
    N = floor(Int,tf/dt) + 1
    dt = tf / (N-1)

    # cost
    n0 = 2
    Q = 1e-3*Diagonal([1e-3; ones(n0); fill(1e-3,n-n0-1)])
    R = 1e-3*Diagonal(ones(m))
    Qf = Q*(N-1)
    x0 = expandstate(model, zeros(n0))
    xf = expandstate(model, [pi, 0.0])  # i.e. swing up
    obj = LQRObjective(Q*dt,R*dt,Qf,xf,N)

    # constraints
    conSet = ConstraintList(n,m,N)
    bnd = BoundConstraint(n,m,u_min=-u_bnd,u_max=u_bnd)
    goal = GoalConstraint(xf, 1 .+ (1:n0))
    if constraints
        add_constraint!(conSet, bnd, 1:N-1)
        add_constraint!(conSet, goal, N:N)
    end

    # problem
    times = range(0,tf,length=N)
    U = [SA[cos(t/2)] for t in times]
    pendulum_static = Problem(model, obj, x0, tf, constraints=conSet, xf=xf)
    initial_controls!(pendulum_static, U)
    rollout!(pendulum_static)
    return pendulum_static
end
