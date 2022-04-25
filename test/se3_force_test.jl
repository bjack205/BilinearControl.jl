
using BilinearControl: getA, getB, getC, getD
function test_se3_force_dynamics()
    model = Se3ForceDynamics(2.0)
    n,m = RD.dims(model)
    @test n == 42
    @test m == 6
    x,u = rand(model)
    r = x[1:3]
    R = SMatrix{3,3}(x[4:12]...) 
    v = x[13:15]
    F = u[1:3]
    ω = u[4:6]

    @test det(reshape(x[4:12], 3, 3)) ≈ 1.0
    @test length(x) == n

    xdot = zeros(n)
    RD.dynamics!(model, xdot, x, u)
    @test xdot[1:3] ≈ R*v
    @test xdot[4:12] ≈ vec(R*skew(ω))
    @test xdot[13:15] ≈ F/model.m - ω × v

    # Test dynamics match bilinear dynamics
    A,B,C,D = getA(model), getB(model), getC(model), getD(model)
    @test xdot ≈ A*x + B*u + sum(u[i]*C[i]*x for i = 1:m) + D

    # Test Jacobian
    J = zeros(n, n+m)
    RD.jacobian!(model, J, xdot, x, u)
    Jfd = zero(J)
    FiniteDiff.finite_difference_jacobian!(
        Jfd, (y,z)->RD.dynamics!(model, y, z[1:n], z[n+1:end]), Vector([x;u])
    )
    @test Jfd ≈ J
end

function testse3forceproblem()
    # prob = buildse3forceproblem()
    prob = Problems.SE3ForceProblem()
    admm = BilinearADMM(prob)
    X = extractstatevec(prob)
    U = extractcontrolvec(prob)

    BilinearControl.setpenalty!(admm, 1e4)
    admm.opts.penalty_threshold = 1e2
    Xsol, Usol = BilinearControl.solve(admm, X, U, max_iters=800)
    n,m = RD.dims(prob.model[1])
    Xs = collect(eachcol(reshape(Xsol, n, :)))
    Us = collect(eachcol(reshape(Usol, m, :)))

    # Check it reaches the goal
    @test norm(Xs[end] - prob.xf) < BilinearControl.get_primal_tolerance(admm) 

    # Check the rotation matrices
    @test norm([det(reshape(x[4:12],3,3)) - 1 for x in Xs], Inf) < 1e-1

    # Test that the controls are smooth
    @test norm(mean(diff(Us)), Inf) < 0.1
end

@testset "SE(3) Force Dynamics" begin 
    test_se3_force_dynamics()
    testse3forceproblem()
end