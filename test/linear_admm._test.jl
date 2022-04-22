using BilinearControl
using LinearAlgebra
using FiniteDiff
using Test

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
