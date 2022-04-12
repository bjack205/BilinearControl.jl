using .Problems: skew

function test_se3_kinematics()
    ## Test kinematics
    model = SE3Kinematics()
    n,m = RD.dims(model)
    x,u = rand(model)
    @test length(x) == 12
    @test length(u) == 6

    v = u[1:3]
    ω = u[4:end]
    R = RotMatrix{3}(x[4:end])
    @test RD.dynamics(model, x, u) ≈ [R*v; vec(R*skew(ω))]

    # Test custom Jacobian
    J = zeros(n, n+m)
    xdot = zeros(n)
    RD.jacobian!(model, J, xdot, x, u)
    Jfd = zero(J)
    FiniteDiff.finite_difference_jacobian!(
        Jfd, (y,z)->RD.dynamics!(model, y, z[1:12], z[13:end]), Vector([x;u])
    )
    @test Jfd ≈  J
end

function solve_se3_kinematics()
    ## Try solving with ADMM
    # prob = buildse3problem()
    prob = Problems.SE3Problem()
    admm = BilinearADMM(prob)
    X = extractstatevec(prob)
    U = extractcontrolvec(prob)
    Xsol, Usol = BilinearControl.solve(admm, X, U, max_iters=200)

    n,m = RD.dims(prob.model[1])
    Xs = collect(eachcol(reshape(Xsol, n, :)))
    Us = collect(eachcol(reshape(Usol, m, :)))

    # Check it reaches the goal
    @test norm(Xs[end] - prob.xf) < 1e-3

    # Check the rotation matrices
    @test norm([det(reshape(x[4:end],3,3)) - 1 for x in Xs], Inf) < 1e-1

    # Test that the controls are smooth
    @test norm(mean(diff(Us)), Inf) < 0.1
end

@testset "SE(3) Kinematics" begin
    test_se3_kinematics()
    solve_se3_kinematics()
end

