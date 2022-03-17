
@testset "Bilinear Dynamics" begin
Random.seed!(1)
n,m = 5,3
A,B = RandomLinearModels.gencontrollable(n,m)
C = [randn(n,n) for k = 1:m]
local model

# Test public API
@testset "Public API" begin
    @test_nowarn model = BiLinearDynamics(A, B, C)
    @test state_dim(model) == n
    @test control_dim(model) == m
end

# Internal API tests
@testset "Internal API" begin
    x = randn(n)
    u = randn(m)
    xn = zero(x)
    BilinearControl.discrete_dynamics!(model, xn, x, u) 
    f(x,u) = A * x + B * u + sum(u[i] * C[i] * x for i = 1:m)
    @test xn ≈ f(x,u) 

    @test BilinearControl.getAhat(model, u) ≈ FiniteDiff.finite_difference_jacobian(x->f(x,u), x) atol=1e-6
    @test BilinearControl.getBhat(model, x) ≈ FiniteDiff.finite_difference_jacobian(u->f(x,u), u) atol=1e-6
end
end