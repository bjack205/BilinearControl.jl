using Test
using LinearAlgebra
using Random
using BilinearControl
include("gen_controllable.jl")
using Main.RandomLinearModels

Random.seed!(1)

function gendense()
    n,m = 10,5
    A,B = RandomLinearModels.gencontrollable(n, m)
    C = [randn(n,n) for k = 1:m]
    d = randn(n,1)
    return A,B,C,d
end

function gensparse()
    n,m = 100,6
    A = sprandn(n,n,0.1)
    B = sprandn(n,m,0.1)
    C = [sprandn(n,n,0.2) for i = 1:m]
    d = sprandn(n,1,0.05)
    return A,B,C,d
end

function testadmmsolve(genmats)
    
    # Constraint
    A,B,C,d = genmats()
    n,m = size(B)
    
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
    
    BilinearControl.setpenalty!(solver, 10000.)
    solver.opts.penalty_threshold = 1e3
    x,z,w = BilinearControl.solve(solver, x0, z0, max_iters=2000)
    
    # Test optimality conditions
    perr = norm(A*x + B*z + sum(z[i] * C[i] * x for i = 1:m) + d, Inf)
    @test perr < BilinearControl.get_primal_tolerance(solver)
    ρ = BilinearControl.getpenalty(solver)
    Ahat = BilinearControl.getAhat(solver, z)
    Bhat = BilinearControl.getBhat(solver, x)
    derrx = norm(Q*x + q + ρ * Ahat'w, Inf)
    derrz = norm(R*z + r + ρ * Bhat'w, Inf)
    @test derrx < BilinearControl.get_dual_tolerance(solver)
    @test derrz < BilinearControl.get_dual_tolerance(solver)

end

@testset "ADMM Constructor ($name)" for (name,genmats) in (
    ("Dense", gendense),
    ("Sparse", gensparse),
) 
# Constraint
A,B,C,d = genmats()
p,m = size(B)
n = size(A,2)

# Objective
Q = Diagonal(fill(1.0, n))
q = zeros(n)
R = Diagonal(fill(0.1, m))
r = zeros(m)

# Create solver
solver = BilinearADMM(A,B,C,d, Q,q,R,r)
@test length(solver.x) == n
@test length(solver.z) == m
@test length(solver.w) == p
@test BilinearControl.hascontrolconstraints(solver) == false
@test BilinearControl.hasstateconstraints(solver) == false
@test solver.xlo == fill(-Inf, n)
@test solver.xhi == fill(+Inf, n)
@test solver.ulo == fill(-Inf, m)
@test solver.uhi == fill(+Inf, m)

solver = BilinearADMM(A,B,C,d, Q,q,R,r, umin=0)
@test BilinearControl.hascontrolconstraints(solver) == true 
@test BilinearControl.hasstateconstraints(solver) == false
@test solver.ulo == fill(0, m)
@test solver.xhi == fill(+Inf, n)

solver = BilinearADMM(A,B,C,d, Q,q,R,r, xmax=100.)
@test BilinearControl.hascontrolconstraints(solver) == false 
@test BilinearControl.hasstateconstraints(solver) == true 
@test solver.ulo == fill(-Inf, m)
@test solver.xhi == fill(100, n)
end

@testset "Dense Solve test" begin
    Random.seed!(1)
    testadmmsolve(gendense)
end

@testset "Sparse Solve test" begin
    Random.seed!(1)
    testadmmsolve(gensparse)
end