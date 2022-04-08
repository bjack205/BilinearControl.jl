include("models/se3_force_model.jl")

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

function buildse3forceproblem()
    # Model
    mass = 2.9
    model = Se3ForceDynamics(mass)
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

    # Discretization
    tf = 3.0
    N = 101

    # Dimensions
    nx = RD.state_dim(model)
    nu = RD.control_dim(model)
    nb = base_state_dim(model)
    ns = nx - nb 

    # Initial and final conditions
    x0_ = [0;0;0; vec(RotZ(deg2rad(0))); zeros(3)]
    xf_ = [3;0;1; vec(RotZ(deg2rad(90)) * RotX(deg2rad(150))); zeros(3)]
    x0 = expandstate(model, x0_)
    xf = expandstate(model, xf_)

    # Objective
    Q = Diagonal([fill(1e-1, 3); fill(1e-2, 9); fill(1e-2, 3); fill(0.0, ns)])
    R = Diagonal([fill(1e-2, 3); fill(1e-2, 3)])
    Qf = 100 * Q 
    obj = LQRObjective(Q,R,Qf,xf,N)

    # Goal state
    cons = ConstraintList(nx, nu, N)
    goalcon = GoalConstraint(xf, 1:nb)  # only constraint the original states
    add_constraint!(cons, goalcon, N)

    # Initial Guess
    U0 = [fill(0.1,nu) for k = 1:N-1] 

    # Build the problem
    prob = Problem(dmodel, obj, x0, tf, xf=xf, constraints=cons, U0=U0)
    rollout!(prob)
    prob
end

function testse3forceproblem()
    prob = buildse3forceproblem()
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