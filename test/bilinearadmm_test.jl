using LinearAlgebra
using Random
include(joinpath(@__DIR__, "../src/admm.jl"))
include("gen_controllable.jl")
using Main.RandomLinearModels

Random.seed!(1)

# Size
n,m = 10,5

# Constraint
A,B = RandomLinearModels.gencontrollable(n, m)
C = [randn(n,n) for k = 1:m]
d = randn(n)

# Objective
Q = Diagonal(fill(1.0, n))
q = zeros(n)
R = Diagonal(fill(0.1, m))
r = zeros(m)

# Create solver
solver = BilinearADMM(A,B,C,d, Q,q,R,r)
x0 = randn(n)
z0 = randn(m)

setpenalty!(solver, 10.1)
x,z = solve(solver, x0, z0, max_iters=1000)