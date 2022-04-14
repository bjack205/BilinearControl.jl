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

    Us
end

# Function to test the formation of the rate-limited bilinear constraint
function eval_dynamics_constraint(model, x0,xf,h,  X, U)
    n,m = RD.dims(model)
    Xs = reshape(X, n, :)
    Us = reshape(U, m, :)
    N = size(Xs,2)

    # Initialize some useful ranges
    ic = 1:n
    ix12 = 1:2n
    iu12 = 1:2m

    # Initialize
    c = zeros(n*(N+1))

    # Initial condition
    c[ic] = x0 - Xs[:,1] 
    ic = ic .+ n

    # Dynamics
    A,B,C,D = getA(model,h), getB(model,h), getC(model,h), getD(model,h)
    for k = 1:N-1
        x12 = X[ix12]
        u12 = U[iu12]
        c[ic] .= A*x12 + B*u12 + sum(u12[i] * C[i] * x12 for i = 1:2m) + D

        ix12 = ix12 .+ n
        iu12 = iu12 .+ m
        ic = ic .+ n
    end

    # Terminal constraint
    c[ic] .= xf .- Xs[:,end]

    c
end

function test_rate_limited_dynamics()
    model = QuadrotorRateLimited()
    n,m = RD.dims(model)
    r1,r2 = randn(3), randn(3) 
    R1,R2 = qrot(normalize(randn(4))), qrot(normalize(randn(4)))
    v1,v2 = randn(3), randn(3) 
    α1,α2 = randn(3), randn(3) 
    ω1,ω2 = randn(3), randn(3) 
    F1,F2 = rand(), rand()

    x1 = [r1; vec(R1); v1; α1]
    x2 = [r2; vec(R2); v2; α2]
    u1 = [ω1; F1]
    u2 = [ω2; F2]

    h = 0.1
    z1 = RD.KnotPoint{n,m}(n,m,[x1;u1],0.0,h)
    z2 = RD.KnotPoint{n,m}(n,m,[x2;u2],h,h)
    err = RD.dynamics_error(model, z2, z1)

    @test err[1:3] ≈ h * (v1 + v2) / 2 + r1 - r2
    @test err[4:12] ≈ vec(h * (R1 + R2) /2 * skew(ω1) + R1 - R2)
    @test err[13:15] ≈ h*( (R1 + R2) /2 * [0,0,F1]) / model.mass - 
        h*[0,0,model.gravity] + v1 - v2
    @test err[16:18] ≈ h*α2 - (ω2 - ω1)

    # Test dynamics match bilinear dynamics
    A,B,C,D = getA(model,h), getB(model,h), getC(model,h), getD(model,h)
    x12 = [x1;x2]
    u12 = [u1;u2]
    err2 = A*x12 + B*u12 + sum(u12[i]*C[i]*x12 for i = 1:length(u12)) + D
    @test err ≈ err2

    ## Test the formation of the entire bilinear constraint
    tf = 3.0
    N = 51
    h = tf / (N-1)
    x0 = [0; 0; 1.0; vec(I(3)); zeros(3); zeros(3)]
    xf = [5; 0; 2.0; vec(RotZ(deg2rad(90))); zeros(3); zeros(3)]
    
    A,B,C,D = getA(model,h), getB(model,h), getC(model,h), getD(model,h)
    Xs = [rand(model)[1] for k = 1:N]
    Us = [rand(model)[2] for k = 1:N]
    X = vcat(Vector.(Xs)...)
    U = vcat(Vector.(Us)...)
    x0 = [zeros(3); vec(I(3)); zeros(6)]
    c1 = eval_dynamics_constraint(model, x0,xf,h, X, U)
    @test c1[1:n] ≈ x0 - Xs[1]
    @test c1[n+1:2n] ≈ A*[Xs[1]; Xs[2]] + B* [Us[1]; Us[2]] + 
        sum([Us[1]; Us[2]][i] * C[i]*[Xs[1]; Xs[2]] for i = 1:2m) + D
    @test c1[2n+1:3n] ≈ A*[Xs[2]; Xs[3]] + B* [Us[2]; Us[3]] + 
        sum([Us[2]; Us[3]][i] * C[i]*[Xs[2]; Xs[3]] for i = 1:2m) + D
    @test c1[end-n+1:end] ≈ xf - Xs[end]
    
    
    Abar,Bbar,Cbar,Dbar = BilinearControl.buildbilinearconstraintmatrices(model, x0, xf, h, N) 
    c2 = Abar*X + Bbar*U + sum(U[i] * Cbar[i]*X for i = 1:length(U)) + Dbar
    @test c1 ≈ c2

end

function test_quadrotor_soc()
    tf = 3.0
    N = 101
    model = QuadrotorRateLimited()
    θ_glideslope = deg2rad(45.0)
    admm = Problems.QuadrotorLanding(tf=tf, N=N, θ_glideslope=θ_glideslope*NaN)
    BilinearControl.setpenalty!(admm, 1e4)
    X = copy(admm.x)
    U = copy(admm.z)
    admm.opts.x_solver = :osqp
    Xsol, Usol = BilinearControl.solve(admm, X, U, verbose=false)
    @test admm.stats.iterations < 100

    # Solve with glideslope
    admm2 = Problems.QuadrotorLanding(tf=tf, N=N, θ_glideslope=θ_glideslope)
    @test length(admm2.constraints) == N-1
    admm2.opts.x_solver = :cosmo
    BilinearControl.setpenalty!(admm2, 1e4)
    Xsol2, Usol2 = BilinearControl.solve(admm2, X, U, verbose=false)
    @test admm2.stats.iterations < 100

    Xs = collect(eachcol(reshape(Xsol, RD.state_dim(model), :)))
    X2s = collect(eachcol(reshape(Xsol2, RD.state_dim(model), :)))
    α = tan(θ_glideslope)
    socerr = map(Xs) do x
        norm(SA[x[1], x[2]]) - α*x[3]
    end
    @test maximum(socerr) > 0.1
    socerr2 = map(X2s) do x
        norm(SA[x[1], x[2]]) - α*x[3]
    end
    @test maximum(socerr2) < 0.1
end

function test_rate_limited_solve(U0s)
    model = QuadrotorRateLimited()
    n,m = RD.dims(model)
    admm = Problems.QuadrotorRateLimitedSolver()
    BilinearControl.setpenalty!(admm, 1e4)
    Xsol, Usol = BilinearControl.solve(admm, verbose=false, max_iters=100)
    xf = admm.d[end-n+1:end]
    @test admm.stats.iterations < 100

    Xs = collect(eachcol(reshape(Xsol, RD.state_dim(model), :)))
    norm(Xs[end] - xf) < BilinearControl.get_primal_tolerance(admm)
    deterr = norm([det(reshape(x[4:12],3,3))-1 for x in Xs], Inf)
    @test deterr < 0.1
    
    Us = collect(eachrow(reshape(Usol, RD.control_dim(model), :)))
    @test norm(diff(Us[1]), Inf) < norm(diff(U0s[1]), Inf)
    @test norm(diff(Us[2]), Inf) < norm(diff(U0s[2]), Inf)
    @test norm(diff(Us[3]), Inf) < norm(diff(U0s[3]), Inf)

end

@testset "Quadrotor" begin
    Us = Vector{Float64}[]
    @testset "SE2(3)" begin
        test_quadrotor_dynamics()
        Us = test_quadrotor_solve()
    end
    @testset "Rate-limited" begin
        test_rate_limited_dynamics()
        test_rate_limited_solve(Us)
    end
    @testset "Second-Order Cone" begin
        test_quadrotor_soc()
    end
end