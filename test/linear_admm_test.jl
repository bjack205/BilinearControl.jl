using BilinearControl
using LinearAlgebra
using FiniteDiff
using Test
using Random
Random.seed!(1)

N = 11
n,m = 3,2
prob = rand(TOQP{n,m}, N)
ρ = 10.0
solver = BilinearControl.TrajOptADMM(prob)
BilinearControl.setpenalty!(solver, ρ)
x = [randn(n) for k = 1:N]
u = [randn(m) for k = 1:N-1]
y = [randn(n) for k = 1:N]
λ = [yk * ρ for yk in y]
X = vcat(x...)
U = vcat(u...)

@test BilinearControl.eval_f(solver, x) ≈ sum(1:N) do k
    dot(x[k], prob.Q[k], x[k]) / 2 + dot(prob.q[k], x[k])
end

@test BilinearControl.eval_g(solver, u) ≈ sum(1:N-1) do k
    dot(u[k], prob.R[k], u[k]) / 2 + dot(prob.r[k], u[k])
end

Lρ = sum(1:N) do k
    J = dot(x[k], prob.Q[k], x[k]) / 2 + dot(prob.q[k], x[k])
    if k == 1
        J += λ[1]'*(prob.x0 - x[k])
        J += ρ * norm(prob.x0 - x[k])^2 / 2
        J += sum(prob.c)
    end
    if k < N
        J += dot(u[k], prob.R[k], u[k]) / 2 + dot(prob.r[k], u[k])
        J += λ[k+1]'*(prob.A[k]*x[k] + prob.B[k]*u[k] + prob.d[k] + prob.C[k]*x[k+1]) 
        J += ρ*norm(prob.A[k]*x[k] + prob.B[k]*u[k] + prob.d[k] + prob.C[k]*x[k+1])^2 / 2
    end
    J
end
@test BilinearControl.auglag(solver, x, u, y, ρ) ≈ Lρ

grad_x = FiniteDiff.finite_difference_gradient(
    x->BilinearControl.auglag(solver, collect(eachcol(reshape(x,:,N))), u, y, ρ),
    X
)

grad_u = FiniteDiff.finite_difference_gradient(
    u->BilinearControl.auglag(solver, x, collect(eachcol(reshape(u,:,N-1))), y, ρ),
    U
)

A,b = BilinearControl.buildstatesystem(solver, u, y, ρ)
@test A*X + b ≈ grad_x

A,b = BilinearControl.buildcontrolsystem(solver, x, y, ρ)
@test A*U + b ≈ grad_u

# Constraints
c = zeros(BilinearControl.num_duals(solver))
BilinearControl.eval_c!(solver, c, x, u)
Â,B̂,ĉ = BilinearControl.buildadmmconstraint(solver)
@test Â*X + B̂*U - ĉ ≈ c

@test BilinearControl.primal_residual(solver, x, u) ≈ norm(c,Inf)

uprev = [randn(m) for k = 1:N-1]
Uprev = vcat(uprev...)
@test BilinearControl.dual_residual(solver, u, uprev) ≈ 
    norm(ρ*Â'B̂*(U-Uprev), Inf)

# Test solve
BilinearControl.setpenalty!(solver, 10.0)
solver.opts.penalty_threshold = 1e2
solver.opts.ϵ_abs_primal = 1e-6
solver.opts.ϵ_abs_dual = 1e-3
xsol, usol, ysol = BilinearControl.solve(solver, x, u, verbose=0, max_iters=200)
ρ = BilinearControl.getpenalty(solver)
BilinearControl.primal_residual(solver, xsol, usol)
λsol = [y*ρ for y in ysol]

N = BilinearControl.nhorizon(prob)
μ = [zeros(0) for k=1:N, j=1:2]
ν = [zeros(0) for k=1:N, j=1:2]
@test BilinearControl.primal_feasibility(prob, xsol, usol) ≈ 
    BilinearControl.primal_residual(solver, xsol, usol)
@test BilinearControl.primal_feasibility(prob, xsol, usol) < 1e-6

@test BilinearControl.stationarity(prob, xsol, usol, λsol, μ, ν) < 1e-3
@test BilinearControl.dual_feasibility(prob, λsol, μ, ν) < 1e-3
iters0 = iterations(solver)

## Test with Acceleration
using COSMOAccelerators
acceleration = AndersonAccelerator{Float64, Type2{QRDecomp}, RestartedMemory, NoRegularizer}
solver2 = BilinearControl.TrajOptADMM(prob, acceleration=acceleration)
solver2.opts.penalty_threshold = 1e2
solver2.opts.ϵ_abs_primal = 1e-6
solver2.opts.ϵ_abs_dual = 1e-3
xsol, usol, ysol = BilinearControl.solve(solver2, x, u, verbose=0, max_iters=200)
iterations(solver2)

@test BilinearControl.primal_feasibility(prob, xsol, usol) ≈ 
    BilinearControl.primal_residual(solver2, xsol, usol)
@test BilinearControl.primal_feasibility(prob, xsol, usol) < 1e-6

@test BilinearControl.stationarity(prob, xsol, usol, λsol, μ, ν) < 1e-3
@test BilinearControl.dual_feasibility(prob, λsol, μ, ν) < 1e-3

@test iterations(solver2) < iters0