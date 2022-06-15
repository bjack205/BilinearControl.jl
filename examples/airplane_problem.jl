function AirplaneProblem(;dt=0.05, dp=zeros(3), tf=2.0, Qv=10.0, Qw=Qv, pf=[5,0,1.5])
    # Discretization
    model = Problems.SimulatedAirplane()
    # model = Problems.NominalAirplane()
    N = round(Int, tf/dt) + 1
    dt = tf / (N-1)

    # Initial condition
    p0 = MRP(0.997156, 0., 0.075366) # initial orientation
    x0     = [-5,0,1.5, Rotations.params(p0)..., 5,0,0, 0,0,0]
    u_trim = [41.66667789082778, 105.99999999471807, 74.65179381344494, 106.00000124622453]

    # Final condition
    xf = copy(x0)
    xf[1:3] .= pf
    xf[7] = 0.0

    # Shift initial position
    x0[1:3] .+= dp

    # Objective
    Qf = Diagonal([fill(1.0, 3); fill(1.0, 3); fill(Qv, 3); fill(Qw, 3)])
    Q  = Diagonal([fill(1e-2, 3); fill(1e-2, 3); fill(1e-1, 3); fill(1e-1, 3)])
    R = Diagonal(fill(1e-3,4))
    obj = TO.LQRObjective(Q,R,Qf,xf,N, uf=u_trim)

    # Constraint
    n,m = RD.dims(model)
    constraints = TO.ConstraintList(n,m,N)
    goalcon = GoalConstraint(xf, SA[1,2,3])
    add_constraint!(constraints, goalcon, N)

    U0 = [copy(u_trim) for k = 1:N-1]
    prob = Problem(model,obj,x0,tf; constraints, U0)
    rollout!(prob)
    prob
end