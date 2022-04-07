import Pkg; Pkg.activate(@__DIR__)
using BilinearControl
using BilinearControl.RD
using BilinearControl.TO
import BilinearControl.RD
import BilinearControl.TO

include(joinpath(@__DIR__, "..", "test", "models", "dubins_model.jl"))
using Altro
using TrajectoryOptimization
using LinearAlgebra
using RobotZoo
using StaticArrays
using Test
using Plots
# const TO = TrajectoryOptimization

function testdynamics()
    # Initialize both normal and lifted bilinear model
    model0 = RobotZoo.DubinsCar()
    model = BilinearDubins()
    n,m = RD.dims(model0)
    ny = RD.state_dim(model)
    x,u = rand(model0)
    y = SA[x[1], x[2], cos(x[3]), sin(x[3])]

    # Test that the dynamics match
    ydot = RD.dynamics(model, y, u)
    xdot = RD.dynamics(model0, x, u)
    A,B,C,D = getA(model), getB(model), getC(model), getD(model)
    @test RD.dynamics(model, y, u) ≈ A*y + B*u + sum(u[i]*C[i]*y for i = 1:m) + D
end

function builddubinsproblem(;N=101, ubnd=1.5)
    # model
    model = RobotZoo.DubinsCar()
    n,m = RD.dims(model)
    tf = 3.
    dt = tf / (N-1)

    # cost
    d = 1.5
    x0 = @SVector [0., 0., 0.]
    xf = @SVector [d, d,  deg2rad(90)]
    Qf = 100.0*Diagonal(@SVector ones(n))
    Q = (1e-2)*Diagonal(@SVector ones(n))
    R = (1e-2)*Diagonal(@SVector ones(m))

    # problem
    U = [@SVector fill(0.1,m) for k = 1:N-1]
    obj = LQRObjective(Q*dt,R*dt,Qf,xf,N)

    # constraints
    cons = ConstraintList(n,m,N)
    add_constraint!(cons, GoalConstraint(xf), N)
    add_constraint!(cons, BoundConstraint(n,m, u_min=-ubnd, u_max=ubnd), 1:N-1)

    prob = Problem(model, obj, x0, tf, xf=xf, U0=U, constraints=cons)
    rollout!(prob)

    return prob
end

function buildliftedproblem(prob0)
    model = BilinearDubins()
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)
    # dmodel = RD.DiscretizedDynamics{RD.RK4}(model)

    # Dimensions
    nx = RD.state_dim(prob0, 1)
    ny = RD.state_dim(model)
    nu = RD.control_dim(model)
    N = prob0.N

    # Initial and final conditions
    y0 = expand(model, prob0.x0)
    yf = expand(model, prob0.xf)

    # Objective
    # Sets cost for extra costs to 0
    obj = Objective(map(prob0.obj.cost) do cst
        Q = Diagonal([diag(cst.Q)[1:2]; fill(cst.Q[3,3]*1e-3, 2)])
        R = copy(cst.R)
        LQRCost(Q, R, yf)
    end)

    # Initial trajectory
    U0 = controls(prob0)

    # Goal state
    cons = ConstraintList(ny, nu, N)
    goalcon = GoalConstraint(yf)  # only constraint the original states
    add_constraint!(cons, goalcon, N)

    # Control bounds
    ubnd = get_constraints(prob0)[2]
    umin = TO.lower_bound(ubnd)[nx+1:end]
    umax = TO.upper_bound(ubnd)[nx+1:end]
    ubnd = BoundConstraint(ny, nu, u_min=umin, u_max=umax)
    add_constraint!(cons, ubnd, 1:N-1)

    # Build the problem
    Problem(dmodel, obj, y0, prob0.tf, xf=yf, constraints=cons)
end

function expansion_errors(model, model0, X)
    nx = RD.state_dim(model0)
    ny = RD.state_dim(model0)
    [expand(model, x[1:nx]) - x for x in X]
end

# Solve original problem with ALTRO
prob0 = builddubinsproblem(ubnd=1.5)
U0 = deepcopy(controls(prob0))
altro0 = ALTROSolver(prob0)
solve!(altro0)
RD.traj2(states(altro0))
plot(controls(altro0))

# Solve lifted problem with ALTRO with implicit midpoint
initial_controls!(prob0, U0)
prob = buildliftedproblem(prob0)

altro = ALTROSolver(prob)
altro.opts.dynamics_diffmethod = RD.ImplicitFunctionTheorem(RD.ForwardAD())
# altro.opts.dynamics_diffmethod = RD.ForwardAD()
Altro.usestatic(altro)
solve!(altro)
RD.traj2(states(altro0))
RD.traj2!(states(altro))
TO.cost(altro)
TO.cost(altro0)
plot(controls(altro))

## Solve with ADMM
prob = buildliftedproblem(prob0)
rollout!(prob)
model = prob.model[1].continuous_dynamics
n,m = RD.dims(model)
A,B,C,D = BilinearControl.buildbilinearconstraintmatrices(prob.model[1].continuous_dynamics, prob.x0, prob.xf, prob.Z[1].dt, prob.N)
X = vcat(Vector.(states(prob))...)
U = vcat(Vector.(controls(prob))...)
c1 = A*X + B*U + sum(U[i] * C[i] * X for i = 1:length(U)) + D
c2 = BilinearControl.evaluatebilinearconstraint(prob)
@test c1 ≈ c2

Q,q,R,r,c = BilinearControl.buildcostmatrices(prob)
admm = BilinearADMM(A,B,C,D, Q,q,R,r,c, umin=-0.9, umax=0.9)
admm.opts.penalty_threshold = 1e4
BilinearControl.setpenalty!(admm, 1e3)
admm.ulo
admm.uhi
Xsol, Usol = BilinearControl.solve(admm, X, U)
v,ω = collect(eachrow(reshape(Usol, m, :)))
xtraj = reshape(Xsol,n,:)[1,:]
ytraj = reshape(Xsol,n,:)[2,:]
norm([norm(x[3:4]) - 1 for x in eachcol(reshape(Xsol,n,:))], Inf)

RD.traj2(states(altro0), label="ALTRO (RK4)")
RD.traj2!(states(altro), label="ALTRO (Implicit Midpoint)")
RD.traj2!(xtraj, ytraj, label="ADMM", legend=:topleft)

plot(v)
plot!(ω)

## MPC
function liftedmpcproblem(x0, Zref, kstart=1; N=51)
    model0 = RobotZoo.DubinsCar()
    model = BilinearDubins()
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

    # Dimensions
    nx = RD.state_dim(model0)
    ny = RD.state_dim(model)
    nu = RD.control_dim(model)

    # Initial state 
    y0 = expand(model, x0)

    # Objective
    # Sets cost for extra costs to 0
    Q = Diagonal([10; 10; 0.1; fill(1e-3, ny-nx)])
    R = Diagonal(fill(0.01, nu))
    dt = Zref[1].dt
    Zlifted = TO.SampledTrajectory(map(kstart .+ (1:N)) do k
        k = min(k, length(Zref))
        z = Zref[k]
        x = RD.state(z)
        u = RD.control(z)
        y = expand(model, x)
        KnotPoint{ny,nu}(ny, nu, [y; u], z.t, dt)
    end)
    Zlifted[end].dt = 0
    RD.set_dt!(Zlifted, Zref[1].dt)
    obj = TO.TrackingObjective(Q, R, Zlifted)

    # Initial trajectory
    U0 = controls(Zlifted)

    # Build the problem
    t0 = Zlifted[1].t
    tf = Zlifted[end].t
    # @show t0 tf
    Problem(dmodel, obj, y0, tf, t0=t0, U0=U0), Zlifted
end

function mpcproblem(x0, Zref, kstart=1; N=51)
    model = RobotZoo.DubinsCar()
    dmodel = RD.DiscretizedDynamics{RD.RK4}(model)

    # Dimensions
    nx = RD.state_dim(model)
    nu = RD.control_dim(model)

    # Objective
    # Sets cost for extra costs to 0
    Q = Diagonal([10; 10; 0.1])
    R = Diagonal(fill(0.01, nu))
    dt = Zref[1].dt
    Zmpc = TO.SampledTrajectory(map(kstart .+ (1:N)) do k
        k = min(k, length(Zref))
        z = Zref[k]
        x = RD.state(z)
        u = RD.control(z)
        # y = expand(model, x)
        KnotPoint{nx,nu}(nx, nu, [x; u], z.t, dt)
    end)
    Zmpc[end].dt = 0
    RD.set_dt!(Zmpc, Zref[1].dt)
    obj = TO.TrackingObjective(Q, R, Zmpc)

    # Initial trajectory
    U0 = controls(Zmpc)

    # Build the problem
    t0 = Zmpc[1].t
    tf = Zmpc[end].t
    # @show t0 tf
    Problem(dmodel, obj, x0, tf, t0=t0, U0=U0), Zmpc
end

function simulate(x0, Zref; dt_sim=Zref[1].dt, Nmpc=51, lifted=false)
    tf = Zref[end].t
    h = Zref[1].dt
    tsim = tf - (Nmpc-1)*h
    tsim = tf
    times = range(0, tsim, step=dt_sim)
    m = RD.control_dim(Zref,1)

    model = RobotZoo.DubinsCar()
    dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
    X = [copy(x0) for k = 1:length(times) + Nmpc]
    U = [zeros(m) for x in X]
    for (i,t) in enumerate(times)
        if lifted
            prob, = liftedmpcproblem(X[i], Zref, i, N=Nmpc)
            diffmethod = RD.ImplicitFunctionTheorem(RD.ForwardAD())
        else
            prob, = mpcproblem(X[i], Zref, i, N=Nmpc)
            diffmethod = RD.ForwardAD()
        end
        altro = ALTROSolver(prob)
        altro.opts.verbose = 0
        altro.opts.show_summary = 0
        altro.opts.dynamics_diffmethod = diffmethod 
        solve!(altro)
        Zsol = get_trajectory(altro)
        u = RD.control(Zsol[1])
        U[i] .= u
        X[i+1] = RD.discrete_dynamics(dmodel, X[i], u, t, dt_sim)
    end
    @show length(times)
    Nsim = length(times)
    return X[1:Nsim], U[1:Nsim-1]
end

# Solve original problem with ALTRO
prob0,opts = Problems.DubinsCar(:turn90, N=301)
U0 = deepcopy(controls(prob0))
altro0 = ALTROSolver(prob0, opts)
solve!(altro0)
RD.traj2(states(altro0))

get_trajectory(altro0)
Zref = get_trajectory(altro0) 
kref = 1 
x0 = copy(RD.state(Zref[kref])) + [0,0.1,0]
probmpc, Zlifted = liftedmpcproblem(x0, Zref, kref)
altro = ALTROSolver(probmpc)
altro.opts.dynamics_diffmethod = RD.ImplicitFunctionTheorem(RD.ForwardAD())
solve!(altro)
RD.traj2(states(Zref))
RD.traj2!(states(get_trajectory(altro)))
controls(altro)[1]

x0 = copy(prob0.x0) + [0.0,0.1,0]
Xsim, Usim = simulate(x0, get_trajectory(altro0), lifted=false)
Xsim_lifted, Usim_lifted = simulate(x0, get_trajectory(altro0), lifted=true)
RD.traj2(states(Zref), label="reference", c=:black)
RD.traj2!(Xsim, label="MPC")
RD.traj2!(Xsim_lifted, label="MPC (lifted)")
