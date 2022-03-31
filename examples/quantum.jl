import Pkg; Pkg.activate(@__DIR__)
using SparseArrays
using LinearAlgebra
using BilinearControl
using Test
using RobotZoo 
using Altro
using FiniteDiff
using StaticArrays
import RobotDynamics as RD
using TrajectoryOptimization
const TO = TrajectoryOptimization
include("QOC/QOC.jl")
using .QOC
include("bilinear_constraint.jl")

const SA_C64 = SA{ComplexF64}

function TwoQubitProblem(Model=TwoQubit;
    tf=70.0,  # evolution time (nsec)
    dt=0.1,   # time step (nsec)
)
    # Params
    # ω0 = 0.04  # GHz 
    # ω1 = 0.06  # GHz 
    ω0 = 1.0
    ω1 = 1.0

    # Model
    if Model == ControlDerivative
        model = TwoQubitBase(2pi*ω0, 2pi*ω1)
        dmodel0 = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model) 
        dmodel = QOC.ControlDerivative(dmodel0)
    else
        model = Model(2pi*ω0, 2pi*ω1)  # convert frequencies to rad/ns
        dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model) 
    end
    # dmodel = QTrajOpt.DiscreteTwoQubit(model)
    n,m = RD.dims(dmodel)
    if Model == TwoQubit
        nquantum = n - 3  # number of state elements for the quantum information 
    elseif Model == ControlDerivative
        nquantum = n - 2
    else
        nquantum = n
    end
    N = round(Int, tf / dt) + 1

    # Initial and Final states
    q0 = [
        SA_C64[1,0,0,0],  # i.e. 0b00
        SA_C64[0,1,0,0],
        SA_C64[0,0,1,0],
        SA_C64[0,0,0,1],
    ]
    x0 = zeros(n)
    for i = 1:4
        QOC.setqstate!(model, x0, q0[i], i)
    end

    U = sqrtiSWAP()
    xf = zeros(n)
    for i = 1:4
        QOC.setqstate!(model, xf, U*q0[i], i)
    end

    # Cost Function
    #      x    ∫a   a    da    dda
    qs = [1e0, 1e0, 1e0, 1e-1, 1e-1]
    nqubits = QOC.nqubits(model)     # number of qubits
    qubitsize = QOC.qubitdim(model)  # size of a single qubit
    qudim = nqubits * qubitsize 
    ncontrol = RD.control_dim(model)
    
    Qd = fill(qs[1], qudim)
    if Model == TwoQubit
        Qd = [
            Qd; 
            fill(qs[2], ncontrol); # ∫a
            fill(qs[3], ncontrol); # a
            fill(qs[4], ncontrol); # ∂a
        ]
        R = Diagonal(SA[qs[5]])
    elseif Model == ControlDerivative
        Qd = [
            Qd;
            fill(qs[3]*0, ncontrol)  # uprev
            fill(1e2, ncontrol)  # du
        ]
        R = Diagonal(SA[qs[3]] * 1e-1)
    else
        R = Diagonal(SA[qs[3]] * 1e-1) 
    end
    Q = Diagonal(SVector{n}(Qd))
    Qdf = Qd * N 
    # Qdf[nquantum + 1] *= 10
    Qf = Diagonal(SVector{n}(Qdf))
    obj = LQRObjective(Q, R, Qf, xf, N)

    cons = ConstraintList(n, m, N)

    # Control amplitude constraint 
    if Model == TwoQubit
        control_idx = qudim + 2
        x_max = fill(Inf, n)
        x_min = fill(-Inf, n)
        x_max[control_idx] = 0.5
        x_min[control_idx] = -0.5
        control_bound = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
        # add_constraint!(cons, control_bound, 1:N-1)
    end

    # Goal constraint
    goal = GoalConstraint(xf, 1:nquantum)
    # goal = GoalConstraint(xf, nquantum .+ (1:1))
    add_constraint!(cons, goal, N)

    # Initial guess
    u0 = @SVector fill(1e-4, 1)
    U0 = [copy(u0) for k = 1:N-1]
    X0 = [copy(x0) for k = 1:N]

    # Build the problem
    prob = Problem(dmodel, obj, SVector{n}(x0), tf, xf=xf, X0=X0, U0=U0, constraints=cons)

    return prob
end

# prob = TwoQubitProblem(TwoQubitBase, tf=20.)
prob = TwoQubitProblem(ControlDerivative, tf=20.)
n,m = RD.dims(prob.model[1])
t = RD.gettimes(prob.Z)
U0 = reshape((@. 10*sin(5t/pi)),1,:)
initial_controls!(prob, U0)
xn = zeros(n)
RD.discrete_dynamics!(prob.model[1], xn, prob.Z[1])

solver = ALTROSolver(prob, infeasible=true)
solver.opts.dynamics_funsig = RD.InPlace()
solver.opts.dynamics_diffmethod = RD.ImplicitFunctionTheorem(RD.UserDefined())
solver.opts.penalty_initial = 1e-3
solver.opts.penalty_scaling = 10
solver.opts.cost_tolerance_intermediate = 1e-2
solver.opts.iterations = 2000
solver.opts.cost_tolerance = 1e-3
solver.opts.verbose = 4
# solver.opts.expected_decrease_tolerance = 1e-5
solve!(solver)

# Plot controls
using Plots
U = hcat(controls(solver)...)
a = RD.states(get_trajectory(solver), n-2)
plot(t, a)
plot!(t, @. sin(4*t/pi) * 4*sin(t/2pi))

Z = get_trajectory(solver)
a = RD.controls(get_trajectory(solver), 1)
t = RD.gettimes(solver)[1:end-1]
plot(t, a)

## Bilinear version
prob = twospinproblem()
dmodel = prob.model[1]
model = dmodel.continuous_dynamics

# Check continuous dynamics
ψ0 = ComplexF64[1,0,0,0]
u0 = [0.1]
ψ = let ω1 = 1.0, ω2 = 1.0
    # Drift Hamiltonian
    I2 = I(2) 
    σz = paulimat(:z)
    σz_1 = kron(σz, I2)
    σz_2 = kron(I2, σz)
    Hdrift = σz_1 * ω1 / 2 + σz_2 * ω2 / 2 

    # Drive Hamiltonian
    σx = paulimat(:x)
    σx_1 = kron(σx, I2)
    σx_2 = kron(I2, σx)
    Hdrive = σx_1 * σx_2

    (Hdrift * ψ0 + u0[1] * Hdrive * ψ0) / 1im
end
Hdrift_real,Hdrive_real = twospinhamiltonian()
x0 = Vector(prob.x0)
xdot = complex2real(ψ)
@test xdot ≈ Hdrift_real*x0 + u0[1]*Hdrive_real * x0
@test xdot ≈ RD.dynamics(model, x0, u0)

# Discrete Dynamics
n,m = RD.dims(model)
dt = 0.01
H = Hdrift_real + Hdrive_real * u0[1]
xn_exp = exp(Matrix(H)*dt) * x0
@test norm(xn_exp) ≈ 1.0

z = KnotPoint{8,1}(x0,u0,0,dt)
xn = RD.discrete_dynamics(dmodel, z)
@test norm(xn) ≈ 1.0

# Check dynamics derivatives
fc(z) = RD.dynamics(model, z[1:8], z[9:9])
fc([x0;u0])
J = zeros(n, n+m)
y = zeros(n)
RD.jacobian!(model, J, y, z)
@test J ≈ FiniteDiff.finite_difference_jacobian(fc, [x0;u0])

# Try solving with ALTRO
prob = twospinproblem()
TO.initial_controls!(prob, fill(0.1,m,prob.N-1))
altro = ALTROSolver(prob)
altro.opts.dynamics_diffmethod = RD.ImplicitFunctionTheorem(RD.UserDefined())
altro.opts.verbose = 3
altro.opts.cost_tolerance = 1e-3
altro.opts.cost_tolerance_intermediate = 1e-2
solve!(altro)
states(altro)[end] - prob.xf
Altro.cost(altro)

using Plots
plot(controls(altro))

## Test bilinear constraint function 
prob = twospinproblem()
rollout!(prob)
model = prob.model[1].continuous_dynamics
Abar,Bbar,Cbar,Dbar = buildbilinearconstraintmatrices(model, prob.x0, prob.xf, prob.Z[1].dt, prob.N)
Xvec = vcat(states(prob)...)
Uvec = vcat(controls(prob)...)
Zvec = vcat([z.z for z in prob.Z]...)

c1 = Abar*Xvec + Bbar*Uvec + sum(Uvec[i] * Cbar[i] * Xvec for i = 1:length(Uvec)) + Dbar
c2 = evaluatebilinearconstraint(model, prob.x0, prob.xf, prob.Z[1].dt, prob.N, Zvec)
c3 = evaluatebilinearconstraint(prob)
@test c1 ≈ c2 ≈ c3

## Solve with ADMM
Q = Diagonal(vcat([Vector(diag(cst.Q)) for cst in prob.obj]...))
R = Diagonal(vcat([Vector(diag(prob.obj[k].R)) for k = 1:prob.N-1]...))
q = vcat([Vector(cst.q) for cst in prob.obj]...)
r = vcat([Vector(prob.obj[k].r) for k = 1:prob.N-1]...)
c = sum(cst.c for cst in prob.obj)
admm = BilinearControl.BilinearADMM(Abar, Bbar, Cbar, Dbar, Q,q,R,r,c)
Xvec = vcat(states(prob)...)
Uvec = vcat(controls(prob)...)
admm.opts.penalty_threshold = 1e6
BilinearControl.setpenalty!(admm, 500)
Xsol, Usol = BilinearControl.solve(admm, Xvec, Uvec, max_iters=2000)
Xs = [x for x in eachcol(reshape(Xsol,n,:))]
@test maximum(norm.(Xs)) - 1 < 1e-4   # test maximum norm
@test norm(Xs[end] - prob.xf) < 1e-4  # test final state
@test norm(Xs[1] - prob.x0) < 1e-4    # test initial state

Hdrive, Hdrift = twospinhamiltonian()
blockdiag([Hdrive for i = 1:4]...)
blockdiag([Hdrift for i = 1:4]...)


## Solve full gate problem with ALTRO
prob = twospingateproblem()
rollout!(prob)
model = prob.model[1].continuous_dynamics
x0 = Vector(states(prob)[5])
u0 = Vector(controls(prob)[5])
x = let x = x0, u = u0
    gate = sqrtiSWAP()
    n = length(x) 
    q = n ÷ 8 
    ψ1 = x[0q+1:1q] + x[4q+1:5q]*1im
    ψ2 = x[1q+1:2q] + x[5q+1:6q]*1im
    ψ3 = x[2q+1:3q] + x[6q+1:7q]*1im
    ψ4 = x[3q+1:4q] + x[7q+1:8q]*1im

    Hdrift0, Hdrive0 = twospinhamiltonian()

    H = (Hdrift0 + Hdrive0*u[1])
    ψdot = [
        H*ψ1;
        H*ψ2;
        H*ψ3;
        H*ψ4;
    ]
    xdot = complex2real(ψdot)

    Hbar = blockdiag([H for i = 1:4]...)
    ψ = vcat(ψ1, ψ2, ψ3, ψ4)
    ψdot2 = Hbar * ψ
    xdot2 = complex2real(ψdot2)

    Abar = blockdiag([Hdrift0 for i = 1:4]...)
    Cbar = blockdiag([Hdrive0 for i = 1:4]...)
    ψdot3 = Abar * ψ + u[1] * Cbar*ψ
    xdot3 = complex2real(ψdot3)

    Abar_real = complex2real(Abar)
    Cbar_real = complex2real(Cbar)
    xdot4 = Abar_real * x + u[1]*Cbar_real * x

    xdot ≈ xdot2 ≈ xdot3 ≈ xdot4
end

prob = twospingateproblem()
initial_controls!(prob, fill(1.4, 1, prob.N-1))
initial_controls!(prob, Usol) 
rollout!(prob)
norm(states(prob)[end] - prob.xf, Inf)
altro = ALTROSolver(prob)
max_violation(altro)
norm(get_constraints(altro)[1].vals[1], Inf)
altro.opts.dynamics_diffmethod = RD.ImplicitFunctionTheorem(RD.UserDefined())
altro.opts.cost_tolerance = 1e-3
altro.opts.cost_tolerance_intermediate = 1e-3
altro.opts.penalty_initial = 1e-5
altro.opts.iterations_outer = 50
altro.opts.projected_newton = false
altro.opts.verbose = 3
solve!(altro)
Usol = deepcopy(controls(altro))
Vector(states(prob)[end]) - xf1 
Altro.findmax_violation(get_constraints(altro))
x = states(altro)[101]
[norm(q) for q in eachcol(reshape(x,8,4))]
controls(altro)