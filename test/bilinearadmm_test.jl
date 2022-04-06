using Test
using LinearAlgebra
using Random
using BilinearControl
include("gen_controllable.jl")
using Main.RandomLinearModels

Random.seed!(1)

# Size
n,m = 10,5

# Constraint
A,B = RandomLinearModels.gencontrollable(n, m)
C = [randn(n,n) for k = 1:m]
d = randn(n,1)

# Objective
Q = Diagonal(fill(1.0, n))
q = zeros(n)
R = Diagonal(fill(0.1, m))
r = zeros(m)

# Create solver
solver = BilinearADMM(A,B,C,d, Q,q,R,r)
x0 = randn(n)
z0 = randn(m)

# Test update Ahat and Bhat
Ahat = A + sum(C[i]*z0[i] for i = 1:m)
@test Ahat ≈ BilinearControl.getAhat(solver, z0)
Bhat = B + hcat([C[i]*x0 for i = 1:m]...)
@test Bhat ≈ BilinearControl.getBhat(solver, x0)
@test Ahat ≈ BilinearControl.updateAhat!(solver, solver.Ahat, z0)
@test Bhat ≈ BilinearControl.updateBhat!(solver, solver.Bhat, x0)

BilinearControl.setpenalty!(solver, 10.)
x,z,w = BilinearControl.solve(solver, x0, z0, max_iters=1000)

# Test optimality conditions
@test norm(A*x + B*z + sum(z[i] * C[i] * x for i = 1:m) + d, Inf) < 2e-3
ρ = BilinearControl.getpenalty(solver)
Ahat = BilinearControl.getAhat(solver, z)
Bhat = BilinearControl.getBhat(solver, x)
@test norm(Q*x + q + ρ * Ahat'w, Inf) < 1e-2
@test norm(R*z + r + ρ * Bhat'w, Inf) < 1e-3


## Try with sparse data
Random.seed!(2)
using SparseArrays
n,m = 100,6
A = sprandn(n,n,0.1)
B = sprandn(n,m,0.1)
C = [sprandn(n,n,0.2) for i = 1:m]
D = sprandn(n,1,0.05)

# Objective
Q = Diagonal(fill(1.0, n))
q = zeros(n)
R = Diagonal(fill(0.1, m))
r = zeros(m)

# Create solver
solver = BilinearADMM(A,B,C,D, Q,q,R,r)
x0 = randn(n)
z0 = randn(m)

# Test update Ahat and Bhat
Ahat = A + sum(C[i]*z0[i] for i = 1:m)
Ahat
@test Ahat ≈ BilinearControl.getAhat(solver, z0)
Bhat = B + hcat([C[i]*x0 for i = 1:m]...)
@test Bhat ≈ BilinearControl.getBhat(solver, x0)
@test Ahat ≈ BilinearControl.updateAhat!(solver, solver.Ahat, z0)
@test Bhat ≈ BilinearControl.updateBhat!(solver, solver.Bhat, x0)

BilinearControl.setpenalty!(solver, 10000.)
solver.opts.penalty_threshold = 1e3
x,z,w = BilinearControl.solve(solver, x0, z0, max_iters=1000)

perr = norm(A*x + B*z + sum(z[i] * C[i] * x for i = 1:m) + D, Inf)
@test perr < 1e-3
ρ = BilinearControl.getpenalty(solver)
Ahat = BilinearControl.getAhat(solver, z)
Bhat = BilinearControl.getBhat(solver, x)
derrx = norm(Q*x + q + ρ * Ahat'w, Inf)
derrz = norm(R*z + r + ρ * Bhat'w, Inf)
@test derrx < 1e-2
@test derrz < 1e-8