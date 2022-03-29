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
J ≈ FiniteDiff.finite_difference_jacobian(fc, [x0;u0])

# Try solving with ALTRO
prob = twospinproblem()
TO.initial_controls!(prob, fill(0.1,m,prob.N-1))
altro = ALTROSolver(prob)
altro.opts.dynamics_diffmethod = RD.ImplicitFunctionTheorem(RD.UserDefined())
altro.opts.verbose = 3
altro.opts.cost_tolerance = 1e-3
solve!(altro)
states(altro)[end] - prob.xf