import Pkg; Pkg.activate(@__DIR__)
include("bilinear_dubins_model.jl")
include("bilinear_constraint.jl")
using Altro
using TrajectoryOptimization
using LinearAlgebra
using RobotZoo
using StaticArrays
using Test
using Plots
using BilinearControl
const TO = TrajectoryOptimization

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

    # Build the problem
    Problem(dmodel, obj, y0, prob0.tf, xf=yf, constraints=cons)
end

function expansion_errors(model, model0, X)
    nx = RD.state_dim(model0)
    ny = RD.state_dim(model0)
    [expand(model, x[1:nx]) - x for x in X]
end

# Solve original problem with ALTRO
prob0,opts = Problems.DubinsCar(:turn90, N=301)
U0 = deepcopy(controls(prob0))
altro0 = ALTROSolver(prob0, opts)
solve!(altro0)
RD.traj2(states(altro0))

# Solve lifted problem with ALTRO with implicit midpoint
initial_controls!(prob0, U0)
prob = buildliftedproblem(prob0)

altro = ALTROSolver(prob, opts)
altro.opts.dynamics_diffmethod = RD.ImplicitFunctionTheorem(RD.ForwardAD())
# altro.opts.dynamics_diffmethod = RD.ForwardAD()
Altro.usestatic(altro)
solve!(altro)
RD.traj2(states(altro0))
RD.traj2!(states(altro))
cost(altro)
cost(altro0)
states(altro)[end]

## Solve with ADMM
prob = buildliftedproblem(prob0)
rollout!(prob)
model = prob.model[1].continuous_dynamics
n,m = RD.dims(model)
A,B,C,D = buildbilinearconstraintmatrices(prob.model[1].continuous_dynamics, prob.x0, prob.xf, prob.Z[1].dt, prob.N)
X = vcat(Vector.(states(prob))...)
U = vcat(Vector.(controls(prob))...)
c1 = A*X + B*U + sum(U[i] * C[i] * X for i = 1:length(U)) + D
c2 = evaluatebilinearconstraint(prob)
@test c1 ≈ c2

Q,q,R,r,c = buildcostmatrices(prob)
admm = BilinearADMM(A,B,C,D, Q,q,R,r,c)
admm.opts.penalty_threshold = 1e4
BilinearControl.setpenalty!(admm, 1e3)
Xsol, Usol = BilinearControl.solve(admm, X, U)
xtraj = reshape(Xsol,n,:)[1,:]
ytraj = reshape(Xsol,n,:)[2,:]
[norm(x[3:4]) for x in eachcol(reshape(Xsol,n,:))]
RD.traj2(xtraj, ytraj)

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
