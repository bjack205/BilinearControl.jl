using BilinearControl.Problems: qrot, skew
using BilinearControl: getA, getB, getC, getD

function test_quadrotor_dynamics()
    model = QuadrotorSE23()
    r = randn(3)
    R = qrot(normalize(randn(4)))
    v = randn(3)
    ω = randn(3)
    F = rand()*10

    x = [r; vec(R); v]
    u = [ω; F]
    xdot = [
        v;
        vec(R*skew(ω));
        (R*[0,0,F] + [0,0,-model.mass*model.gravity]) / model.mass
    ]
    @test xdot ≈ RD.dynamics(model, x, u)

    # Test custom Jacobian
    n,m = RD.dims(model)
    J = zeros(n, n+m)
    RD.jacobian!(model, J, xdot, x, u)
    Jfd = zero(J)
    z = [x;u]
    FiniteDiff.finite_difference_jacobian!(
        Jfd, (y,z)->RD.dynamics!(model, y, z[1:15], z[16:end]), Vector([x;u])
    )
    @test Jfd ≈ J

    # Test dynamics match bilinear dynamics
    A,B,C,D = getA(model), getB(model), getC(model), getD(model)
    @test xdot ≈ A*x + B*u + sum(u[i]*C[i]*x for i = 1:length(u)) + D
end

function test_quadrotor_solve()
    ## Solve with ADMM 
    prob = Problems.QuadrotorProblem()
    model = prob.model[1].continuous_dynamics
    admm = BilinearADMM(prob)
    X = extractstatevec(prob)
    U = extractcontrolvec(prob)
    admm.opts.penalty_threshold = 1e2
    BilinearControl.setpenalty!(admm, 1e4)
    Xsol, Usol = BilinearControl.solve(admm, X, U, verbose=false)
    @test admm.stats.iterations < 100

    Xs = collect(eachcol(reshape(Xsol, RD.state_dim(model), :)))
    norm(Xs[end] - prob.xf) < BilinearControl.get_primal_tolerance(admm)
    deterr = norm([det(reshape(x[4:12],3,3))-1 for x in Xs], Inf)
    @test deterr < 0.1
    
    Us = collect(eachrow(reshape(Usol, RD.control_dim(model), :)))
    @test norm(diff(Us[1]), Inf) < 1.0
    @test norm(diff(Us[2]), Inf) < 1.0
    @test norm(diff(Us[3]), Inf) < 1.0
    @test norm(diff(Us[4]), Inf) < 1.0
end

@testset "Quadrotor" begin
    test_quadrotor_dynamics()
    test_quadrotor_solve()
end