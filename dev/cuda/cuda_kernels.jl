import Pkg; Pkg.activate(@__DIR__)
using CUDA
using BilinearControl
using BilinearControl.Problems
using BenchmarkTools
using IterativeSolvers

## Build Sparse Array
using BilinearControl: getA, getB, getC, getD
prob = Problems.QuadrotorProblem()
solver = BilinearADMM(prob)
A,C = getA(solver), getC(solver)

Ahat = solver.Ahat
x,z = solver.x, solver.z
BilinearControl.updateAhat!(solver, Ahat, z)

ρ = BilinearControl.getpenalty(solver)
P̂ = BilinearControl.updatePhat!(solver)
a = vec(BilinearControl.geta(solver, solver.z))
q̂ = solver.q + ρ * solver.Ahat'*(a + solver.w)
xn1 = P̂\(-q̂)
xn2 = BilinearControl.solvex(solver, solver.z, solver.w)
xn3 = cg(P̂, -q̂)
norm(xn1 - xn2)
norm(xn1 - xn3)

using CUDA.CUSPARSE
P_d = CuSparseMatrixCSC(P̂)
q_d = CuVector(q̂)
xn_d = cg(P_d, -q_d)
norm(xn1 - Vector(xn_d))

@btime cg($P̂, $q̂)
@btime cg($P_d, $q_d)

@btime mul!($xn_d, $P_d, $q_d)
@btime mul!($xn1, $P̂, $q̂)