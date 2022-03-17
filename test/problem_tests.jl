import Pkg; Pkg.activate(@__DIR__)
using BilinearControl
using LinearAlgebra
using Random
using Test
using FiniteDiff

function gentestdynamics(n, m)
    A,B = RandomLinearModels.gencontrollable(n,m)
    C = [randn(n,n) for k = 1:m]
    return BiLinearDynamics(A, B, C)
end

function gentestproblem()
    n,m = 5,3
    N = 11

    # Initial and final conditions
    x0 = zeros(n)
    xf = fill(10.0, n)

    # Cost matrices
    Q = Diagonal(fill(1.0, n)) 
    R = Diagonal(fill(0.1, m)) 
    Qf = Q * N

    # Build model 
    model = gentestdynamics(n, m)

    # Build problem
    BiLinearProblem(model, Q, R, Qf, x0, xf, N)
end

@testset "Test Problem" begin
prob = gentestproblem()
n,m,N = BilinearControl.dims(prob)

local X,U,Z
@testset "Basic methods" begin
    @test n == 5
    @test m == 3
    @test N == 11 
    @test BilinearControl.num_primals(prob) == N*n + (N-1) * m
    @test BilinearControl.num_duals(prob) == N*n
    Np = BilinearControl.num_primals(prob)
    Z = randn(Np)
    X,U = BilinearControl.unpackZ(prob, Z)
    @test Z ≈ [X; U]
end

Xs = [Vector(x) for x in eachcol(reshape(X, n, :))]
Us = [Vector(u) for u in eachcol(reshape(U, m, :))]
@testset "Cost" begin
    Jx = mapreduce(+, enumerate(Xs)) do (k,x)
        xf = prob.xf
        Qk = k == N ? prob.Qf : prob.Q
        0.5 * (x - xf)'Qk*(x - xf)
    end
    @test statecost(prob, X) ≈ Jx

    Ju = mapreduce(+, enumerate(Us)) do (k,u)
        0.5 * u'prob.R*u
    end
    @test controlcost(prob, U) ≈ Ju
end

@testset "Cost gradients" begin
    gx = zero(X)
    statecost_grad!(prob, gx, X)
    @test gx ≈ FiniteDiff.finite_difference_gradient(x->statecost(prob, x), X) rtol=1e-8

    gu = zero(U)
    controlcost_grad!(prob, gu, U)
    @test gu ≈ FiniteDiff.finite_difference_gradient(u->controlcost(prob, u), U) rtol=1e-8
end

# Constraints
Nd = BilinearControl.num_duals(prob)
@testset "Constraints" begin
    c = zeros(Nd)
    constraints!(prob, c, X, U)
    model = getmodel(prob)
    A, B, C = model.A, model.B, model.C 
    f(x,u) = A * x + B * u + sum(u[i] * C[i] * x for i = 1:m)
    @test c[1:n] ≈ prob.x0 - Xs[1]
    cdyn = vcat([f(Xs[k],Us[k]) - Xs[k+1] for k = 1:N-1]...)
    @test c[n+1:end] ≈ cdyn
end

# Constraint Jacobians
@testset "Constraint Jacobian" begin
    function con(x,u) 
        c = zero(x)
        constraints!(prob, c, x, u)
        return c
    end
    jx = zeros(Nd, length(X))
    constraint_state_jacobian!(prob, jx, X, U)
    @test jx ≈ FiniteDiff.finite_difference_jacobian(x->con(x,U), X) rtol=1e-6

    ju = zeros(Nd, length(U))
    constraint_control_jacobian!(prob, ju, X, U)
    @test ju ≈ FiniteDiff.finite_difference_jacobian(u->con(X,u), U) rtol=1e-6
end
end