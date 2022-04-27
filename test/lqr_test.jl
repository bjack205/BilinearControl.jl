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
μ = [zeros(0) for k=1:11, j=1:2]
ν = [zeros(0) for k=1:11, j=1:2]
@test BilinearControl.primal_feasibility(data, X, U) < 1e-10
@test BilinearControl.stationarity(data, X, U, λ, μ, ν) < 1e-10
