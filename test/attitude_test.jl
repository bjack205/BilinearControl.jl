using BilinearControl
using BilinearControl.RD
using BilinearControl.TO
import BilinearControl.TO
using Test
using FiniteDiff
using LinearAlgebra
using Statistics
import BilinearControl.RD
include("models/attitude_model.jl")

using BilinearControl: getA, getB, getC, getD
const Nu = 2
function attitude_dynamics_test(::Val{Nu}) where Nu
    model = AttitudeDynamics{Nu}()
    n,m = RD.dims(model)
    @test n == 4
    @test m == Nu 
    x,u = rand(model) 
    @test norm(x) ≈ 1
   
    # Test dynamics match expeccted
    xdot = zeros(4)
    RD.dynamics!(model, xdot, x, u)
    if Nu == 3
        @test xdot ≈ 0.5*[
            -x[2]*u[1] - x[3]*u[2] - x[4]*u[3]
            x[1]*u[1] - x[4]*u[2] + x[3]*u[3]
            x[4]*u[1] + x[1]*u[2] - x[2]*u[3]
            -x[3]*u[1] + x[2]*u[2] + x[1]*u[3]
        ]
    else
        @test xdot ≈ 0.5*[
            -x[2]*u[1] - x[3]*u[2]
            x[1]*u[1] - x[4]*u[2]
            x[4]*u[1] + x[1]*u[2]
            -x[3]*u[1] + x[2]*u[2]
        ]
    end

    # Test dynamics match bilinear dynamics
    A,B,C,D = getA(model), getB(model), getC(model), getD(model)
    @test xdot ≈ A*x + B*u + sum(u[i]*C[i]*x for i = 1:m) + D

    # Test custom Jacobian
    J = zeros(n, n+m)
    RD.jacobian!(model, J, xdot, x, u)
    Jfd = zero(J)
    FiniteDiff.finite_difference_jacobian!(
        Jfd, (y,z)->RD.dynamics!(model, y, z[1:4], z[5:end]), Vector([x;u])
    )
    @test Jfd ≈ J
end

function so3_dynamics_test(::Val{Nu}) where Nu
    model = SO3Dynamics{Nu}()
    n,m = RD.dims(model)
    @test n == 9
    @test m == Nu 
    x,u = rand(model)
    R = SMatrix{3,3}(x)
    @test det(R) ≈ 1

    # Test dynamics match expected
    xdot = zeros(9)
    RD.dynamics!(model, xdot, x, u)
    ω = getangularvelocity(model, u)
    if Nu == 3
        @test xdot ≈ vec(R*skew(u))
    else
        @test xdot ≈ vec(R*skew([u[1],u[2],0.0]))
    end

    # Test bilinear dynamics
    A,B,C,D = getA(model), getB(model), getC(model), getD(model)
    @test xdot ≈ A*x + B*u + sum(u[i]*C[i]*x for i = 1:m) + D

    # Test custom Jacobian
    J = zeros(n, n+m)
    RD.jacobian!(model, J, xdot, x, u)
    Jfd = zero(J)
    FiniteDiff.finite_difference_jacobian!(
        Jfd, (y,z)->RD.dynamics!(model, y, z[1:9], z[10:end]), Vector([x;u])
    )
    @test Jfd ≈ J
end

function buildattitudeproblem(::Val{Nu}) where Nu
    # Model
    model = AttitudeDynamics{Nu}()
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

    # Discretization
    tf = 3.0
    N = 301

    # Dimensions
    nx = RD.state_dim(model)
    nu = RD.control_dim(model)

    # Initial and final conditions
    x0 = [1,0,0,0]
    if Nu == 3
        xf = [0.382683, 0.412759, 0.825518, 0.0412759]
    else
        xf = normalize([1,0,0,1.0])  # flip 90° around unactuated axis
    end

    # Objective
    Q = Diagonal(fill(1e-1, nx))
    R = Diagonal(fill(2e-2, nu))
    Qf = Diagonal(fill(100.0, nx))
    obj = LQRObjective(Q,R,Qf,xf,N)

    # Goal state
    cons = ConstraintList(nx, nu, N)
    goalcon = GoalConstraint(xf)  # only constraint the original states
    add_constraint!(cons, goalcon, N)

    # Initial Guess
    U0 = [fill(0.1,Nu) for k = 1:N-1] 

    # Build the problem
    Problem(dmodel, obj, x0, tf, xf=xf, constraints=cons, U0=U0)
end

function buildso3problem(::Val{Nu}) where Nu
    # Model
    model = SO3Dynamics{Nu}()
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

    # Discretization
    tf = 3.0
    N = 301

    # Dimensions
    nx = RD.state_dim(model)
    nu = RD.control_dim(model)

    # Initial and final conditions
    x0 = vec(I(3))
    xf = vec(RotZ(deg2rad(90)))

    # Objective
    Q = Diagonal(fill(0.0, nx))
    R = Diagonal(fill(2e-2, nu))
    Qf = Diagonal(fill(100.0, nx))
    # costs = map(1:N) do k
    #     q = -xf  # tr(Rf'R)
    #     r = zeros(nu)
    #     TO.DiagonalCost(Q,R,q,r,0.0)
    # end
    # obj = TO.Objective(costs)
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

function testattitudeproblem(Nu)
    attitude_dynamics_test(Nu)
    prob = buildattitudeproblem(Nu)
    rollout!(prob)

    A,B,C,D = BilinearControl.buildbilinearconstraintmatrices(
        prob.model[1].continuous_dynamics, prob.x0, prob.xf, prob.Z[1].dt, prob.N
    )
    X = vcat(Vector.(states(prob))...)
    U = vcat(Vector.(controls(prob))...)
    c1 = A*X + B*U + sum(U[i] * C[i] * X for i = 1:length(U)) + D
    c2 = BilinearControl.evaluatebilinearconstraint(prob)
    @test c1 ≈ c2

    # Solve use ADMM
    Q,q,R,r,c = BilinearControl.buildcostmatrices(prob)
    admm = BilinearADMM(A,B,C,D, Q,q,R,r,c)
    admm.opts.penalty_threshold = 1e4
    admm.opts.ϵ_abs_dual = 1e-4
    admm.opts.ϵ_rel_dual = 1e-4
    BilinearControl.setpenalty!(admm, 1e3)
    Xsol, Usol = BilinearControl.solve(admm, X, U, max_iters=300)

    # Reshape the solution vectors
    n,m = RD.dims(prob.model[1])
    Xs = collect(eachcol(reshape(Xsol, n, :)))
    Us = collect(eachcol(reshape(Usol, m, :)))
    Zsol = SampledTrajectory(Xs,Us, tf=prob.tf)

    # Test that it got to the goal
    @test abs(Xs[end]'prob.xf - 1.0) < BilinearControl.get_primal_tolerance(admm)  

    # Test that the quaternion norms are preserved
    norm_error = norm(norm.(Xs) .- 1, Inf)
    @test norm_error < 1e-2 

    # Check that the control signals are smooth 
    Us = reshape(Usol, m, :)
    @test all(x->x< 2e-2, mean(diff(Us, dims=2), dims=2))
end

function testso3problem(Nu; x_solver=:ldl, z_solver=:cholesky)
    prob = buildso3problem(Nu)
    rollout!(prob)
    admm = BilinearADMM(prob)
    X = extractstatevec(prob)
    U = extractcontrolvec(prob)
    admm.opts.ϵ_abs_primal = 1e-5
    admm.opts.ϵ_rel_primal = 1e-5
    admm.opts.ϵ_abs_dual = 1e-4
    admm.opts.ϵ_rel_dual = 1e-4
    admm.opts.x_solver = x_solver
    admm.opts.z_solver = z_solver
    Xsol, Usol = BilinearControl.solve(admm, X, U, max_iters=50)

    n,m = RD.dims(prob.model[1])
    Xs = collect(eachcol(reshape(Xsol, n, :)))
    Us = collect(eachcol(reshape(Usol, m, :)))

    # Test that it got to the goal
    Rgoal = SMatrix{3,3}(prob.xf)
    Rf = SMatrix{3,3}(Xs[end])
    @test abs(tr(Rgoal'Rf) - 3) < BilinearControl.get_primal_tolerance(admm) 

    # Test that the quaternion norms are preserved
    det_error = norm([det(SMatrix{3,3}(x)) .- 1 for x in Xs], Inf)
    @test det_error < 1e-2 

    # Check that the control signals are smooth 
    Us = reshape(Usol, m, :)
    @test all(x->x< 2e-2, mean(diff(Us, dims=2), dims=2))
    Xsol, Usol, admm
end


@testset "Attitude with $Nu controls" for Nu in (3,2)
    testattitudeproblem(Val(Nu))
end

@testset "SO(3) with $Nu controls" for Nu in (2,)
    Xsol, Usol, admm = testso3problem(Val(Nu))
    if Nu == 2
        @testset "Cholesky" begin
            X2, U2, admm2 = testso3problem(Val(Nu), x_solver=:cholesky, z_solver=:cholesky)
            @test X2 ≈ Xsol rtol=1e-5
            @test U2 ≈ Usol rtol=1e-5
            @test admm.stats.iterations == admm2.stats.iterations
        end
        @testset "OSQP" begin
            X2, U2, admm2 = testso3problem(Val(Nu), x_solver=:osqp, z_solver=:osqp)
            @test X2 ≈ Xsol rtol=1e-5
            @test U2 ≈ Usol rtol=1e-5
            @test admm.stats.iterations == admm2.stats.iterations
        end
        @testset "CG" begin
            X2, U2, admm2 = testso3problem(Val(Nu), x_solver=:cg, z_solver=:cg)
            @test X2 ≈ Xsol rtol=1e-5
            @test U2 ≈ Usol rtol=1e-5
            @test admm.stats.iterations == admm2.stats.iterations
        end
        @testset "MINRES/CG" begin
            X2, U2, admm2 = testso3problem(Val(Nu), x_solver=:minres, z_solver=:cg)
            @test X2 ≈ Xsol rtol=1e-5
            @test U2 ≈ Usol rtol=1e-5
            @test admm.stats.iterations == admm2.stats.iterations
        end
    end
end

