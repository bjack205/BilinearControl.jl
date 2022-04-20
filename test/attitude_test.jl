using BilinearControl
using BilinearControl.RD
using BilinearControl.TO
import BilinearControl.TO
using Test
using FiniteDiff
using LinearAlgebra
using Statistics
import BilinearControl.RD

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

function testattitudeproblem(Nu)
    attitude_dynamics_test(Nu)
    # prob = buildattitudeproblem(Nu)
    prob = Problems.AttitudeProblem(Nu)
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
    # prob = buildso3problem(Nu)
    prob = Problems.SO3Problem(Nu)
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

function test_so3_torque_dynamics()
    J = Diagonal(SA[1,2,1.])
    model = FullAttitudeDynamics(J)
    n,m = RD.dims(model)
    R = qrot(randn(4))
    ω = randn(3)
    T = randn(3)
    w = copy(ω)

    x = [vec(R); ω]
    u = [T; w]

    xdot = RD.dynamics(model, x, u)
    @test xdot ≈ [vec(R*skew(ω)); J\(T - cross(ω, J*ω))]
    B,C = getB(model), getC(model)
    @test xdot ≈ B*u + sum(u[i] * C[i] * x for i = 1:length(u))

    ## Discrete model
    dmodel = ConsensusDynamics(J)
    R1 = qrot(randn(4))
    ω1 = randn(3)
    T1 = randn(3)
    w1 = copy(ω1)

    R2 = qrot(randn(4))
    ω2 = randn(3)
    T2 = randn(3)
    w2 = copy(ω2)

    x1 = [vec(R1); ω1]
    x2 = [vec(R2); ω2]
    u1 = [T1; w1]
    u2 = [T2; w2]
    h = 0.1

    z1 = RD.KnotPoint{n,m}(n,m, [x1; u1], 0, h)
    z2 = RD.KnotPoint{n,m}(n,m, [x2; u2], h, h)

    err = RD.dynamics_error(dmodel, z2, z1)
    xdot_mid = RD.dynamics(model, (x1+x2)/2, u1)
    @test err[1:n] ≈ h*xdot_mid + x1 - x2
    @test err[n+1:end] ≈ ω1 - w1

    Ad,Bd,Cd,Dd = getA(dmodel,h), getB(dmodel,h), getC(dmodel,h), getD(dmodel,h)
    x12 = [x1;x2]
    u12 = [u1;u2]

    @test Ad*x12 ≈ [x1 - x2;  ω1]
    @test Bd*u12 ≈ [zeros(9); h*(J\T1); -w1]
    tmp = sum(u1[i]*Cd[i]*x12 for i = 1:m)
    @test tmp[1:9] ≈ vec(h*(R1 + R2)/2 * skew(w1))
    @test tmp[10:12] ≈ -h*inv(J)*(w1 × (J*(ω1+ω2)/2))
    @test tmp[13:15] ≈ zeros(3)
    @test Ad*x12 + Bd*u12 + tmp ≈ err
end

function test_se3_torque_bilinear_constraint()
    J = Diagonal(SA[1,2,1.])
    dmodel = ConsensusDynamics(J)
    n,m = RD.dims(dmodel)
    x0 = [vec(I(3)); zeros(3)]
    xf = [vec(RotX(deg2rad(45)) * RotZ(deg2rad(180))); zeros(3)]
    N = 11
    tf = 2.0
    h = tf / (N-1)

    Ad,Bd,Cd,Dd = getA(dmodel,h), getB(dmodel,h), getC(dmodel,h), getD(dmodel,h)
    A,B,C,D = BilinearControl.buildbilinearconstraintmatrices(dmodel, x0, xf, h, N)
    Xs = [Vector(rand(dmodel)[1]) for k = 1:N]
    Us = [Vector(rand(dmodel)[2]) for k = 1:N]
    X = vcat(Xs...)
    U = vcat(Us...)
    c = A*X + B*U + sum(U[i] * C[i] * X for i = 1:length(U)) + D
    p = size(Ad,1)
    @test c[1:n] ≈ x0 - Xs[1]
    @test all(1:N-1) do k
        c[n+1+p*(k-1):n+p*k] ≈ Ad * [Xs[k]; Xs[k+1]] + Bd * [Us[k]; Us[k+1]] + 
            sum(Us[k][i] * Cd[i] * [Xs[k]; Xs[k+1]] for i = 1:m) + Dd
    end
    @test c[end-n+1:end] ≈ xf - Xs[end]
end


@testset "Attitude with $Nu controls" for Nu in (3,2)
    testattitudeproblem(Val(Nu))
end

@testset "SO(3) with $Nu controls" for Nu in (2,3)
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

@testset "SO(3) w/ Torque Inputs" begin
    test_so3_torque_dynamics()
    test_se3_torque_bilinear_constraint()
end