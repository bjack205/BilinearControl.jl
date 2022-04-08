include("models/se3_models.jl")

function buildse3problem()
    # Model
    model = SE3Kinematics()
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

    # Discretization
    tf = 3.0
    N = 301

    # Dimensions
    nx = RD.state_dim(model)
    nu = RD.control_dim(model)

    # Initial and final conditions
    x0 = [zeros(3); vec(I(3))]
    xf = [5; 0; 1; vec(RotZ(deg2rad(90)) * RotX(deg2rad(90)))]

    # Objective
    Q = Diagonal([fill(1e-1, 3); fill(1e-2, 9)])
    R = Diagonal([fill(1e-2, 3); fill(1e-2, 3)])
    Qf = Q*10
    obj = LQRObjective(Q,R,Qf,xf,N)

    # Goal state
    cons = ConstraintList(nx, nu, N)
    goalcon = GoalConstraint(xf)
    add_constraint!(cons, goalcon, N)

    # Initial Guess
    U0 = [fill(0.1,nu) for k = 1:N-1] 

    # Build the problem
    Problem(dmodel, obj, x0, tf, xf=xf, constraints=cons, U0=U0)
end

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
    prob = buildse3problem()
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

