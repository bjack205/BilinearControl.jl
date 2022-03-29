import Pkg; Pkg.activate(@__DIR__)
using SparseArrays
using LinearAlgebra
using BilinearControl
using Test
using RobotZoo 
using Altro
using FiniteDiff
import RobotDynamics as RD
using TrajectoryOptimization
const TO = TrajectoryOptimization
include("quantum_hamiltonions.jl")
include("bilinear_constraint.jl")

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
xn_exp = exp(H*dt) * x0
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