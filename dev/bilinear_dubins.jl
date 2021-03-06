import Pkg; Pkg.activate(@__DIR__)
using BilinearControl
using BilinearControl.RD
using BilinearControl.TO
import BilinearControl.RD
import BilinearControl.TO

include(joinpath(@__DIR__, "..", "test", "models", "dubins_model.jl"))
include(joinpath(@__DIR__, "problems.jl"))
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

expandstate(::RobotZoo.DubinsCar, x) = x


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

## Solve original problem with ALTRO
opts = SolverOptions(
    dynamics_diffmethod=RD.ImplicitFunctionTheorem(RD.UserDefined()),
    penalty_initial=1e-2,
    penalty_scaling=1e4,
    projected_newton=false,
    constraint_tolerance=1e-4,
)
prob0 = builddubinsproblem(scenario=:parallelpark, ubnd=1.15)
U0 = deepcopy(controls(prob0))
altro0 = ALTROSolver(prob0, opts) 
solve!(altro0)
RD.traj2(states(altro0))
plot(controls(altro0))

# Solve lifted problem with ALTRO with implicit midpoint
prob = builddubinsproblem(BilinearDubins(), scenario=:parallelpark, ubnd=1.15)

altro = ALTROSolver(prob, opts)
solve!(altro)
RD.traj2(states(altro0))
RD.traj2!(states(altro))
TO.cost(altro)
TO.cost(altro0)
plot(controls(altro))

## Solve with ADMM
ubnd = 1.15
# prob = builddubinsproblem(BilinearDubins(), scenario=:parallelpark, ubnd=ubnd)
prob = builddubinsproblem(BilinearDubins(), ubnd=Inf)
rollout!(prob)
model = prob.model[1].continuous_dynamics
n,m = RD.dims(model)
admm = BilinearADMM(prob)
BilinearControl.hasstateconstraints(admm)
BilinearControl.hascontrolconstraints(admm)
X = extractstatevec(prob)
U = extractcontrolvec(prob)
admm.opts.ϵ_abs_primal = 1e-4
admm.opts.penalty_threshold = 1e2
BilinearControl.setpenalty!(admm, 1e2)
Xsol, Usol = BilinearControl.solve(admm, X, U, max_iters=200)
v,ω = collect(eachrow(reshape(Usol, m, :)))
xtraj = reshape(Xsol,n,:)[1,:]
ytraj = reshape(Xsol,n,:)[2,:]
norm([norm(x[3:4]) - 1 for x in eachcol(reshape(Xsol,n,:))], Inf)

RD.traj2(states(altro0), label="ALTRO")
RD.traj2!(states(altro), label="ALTRO (Bilinear)")
RD.traj2!(xtraj, ytraj, label="ADMM", legend=:topleft)

t = TO.gettimes(prob)
plot(t[1:end-1], v, label="linear velocity")
plot!(t[1:end-1], ω, label="angular velocity", legend=:bottom)
hline!([ubnd], c=:black, s=:dash, label="control bound")


# Iterative solvers


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
