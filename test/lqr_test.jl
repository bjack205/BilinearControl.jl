using BilinearControl
using LinearAlgebra
using Test

data = rand(TOQP{10,5}, 11)
lqrsolver = RiccatiSolver(data)
BilinearControl.solve!(lqrsolver)
lqrsolver.X
A,b = BilinearControl.build_Ab(data)
Y = A\b
X,U,λ = BilinearControl.unpackY(data, Y)
@test X ≈ lqrsolver.X
@test U ≈ lqrsolver.U
@test λ ≈ lqrsolver.λ
@test BilinearControl.primal_residual(data, X, U) < 1e-10
@test BilinearControl.dual_residual(data, X, U, λ) < 1e-10
