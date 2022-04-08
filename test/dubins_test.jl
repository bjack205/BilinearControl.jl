
using BilinearControl: getA, getB, getC, getD
function testdynamics()
    # Initialize both normal and lifted bilinear model
    model = BilinearDubins()
    ny,nu = RD.dims(model)
    y,u = rand(model)

    # Test that the dynamics match
    A,B,C,D = getA(model), getB(model), getC(model), getD(model)
    @test RD.dynamics(model, y, u) ≈ A*y + B*u + sum(u[i]*C[i]*y for i = 1:nu) + D
end

function builddubinsproblem(model=BilinearDubins(); 
        scenario=:turn90, N=101, ubnd=Inf
    )
    # model
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)
    n,m = RD.dims(model)

    tf = 3.
    dt = tf / (N-1)

    # cost
    d = 1.5
    x0 = @SVector [0., 0., 0.]
    if scenario == :turn90
        xf = @SVector [d, d,  deg2rad(90)]
    else
        xf = @SVector [0, d, 0.]
    end
    Qf = 100.0*Diagonal(@SVector ones(n))
    Q = (1e-2)*Diagonal(@SVector ones(n))
    R = (1e-2)*Diagonal(@SVector ones(m))

    if model isa BilinearDubins
        x0 = expandstate(model, x0)
        xf = expandstate(model, xf)
        Q = Diagonal([diag(Q)[1:2]; fill(Q[3,3]*1e-3, 2)]) 
        Qf = Diagonal([diag(Qf)[1:2]; fill(Qf[3,3]*1e-3, 2)]) 
    end

    # objective 
    obj = LQRObjective(Q*dt,R*dt,Qf,xf,N)

    # Initial Guess
    U = [@SVector fill(0.1,m) for k = 1:N-1]

    # constraints
    cons = ConstraintList(n,m,N)
    add_constraint!(cons, GoalConstraint(xf), N)
    add_constraint!(cons, BoundConstraint(n,m, u_min=-ubnd, u_max=ubnd), 1:N-1)

    if scenario == :parallelpark
        x_min = @SVector [-0.25, -0.1, -Inf]
        x_max = @SVector [0.25, d + 0.1, Inf]
        if model isa BilinearDubins
            x_min = push(x_min, -Inf) 
            x_max = push(x_max, +Inf) 
        end
        bnd_x = BoundConstraint(n,m, x_min=x_min, x_max=x_max)
        add_constraint!(cons, bnd_x, 2:N-1)
    end

    prob = Problem(dmodel, obj, x0, tf, xf=xf, U0=U, constraints=cons)
    rollout!(prob)

    return prob
end

@testset "Dubins Dynamics" begin
testdynamics()
prob = builddubinsproblem()
A,B,C,D = BilinearControl.buildbilinearconstraintmatrices(
    prob.model[1].continuous_dynamics, prob.x0, prob.xf, prob.Z[1].dt, prob.N
)
X = vcat(Vector.(states(prob))...)
U = vcat(Vector.(controls(prob))...)
c1 = A*X + B*U + sum(U[i] * C[i] * X for i = 1:length(U)) + D
c2 = BilinearControl.evaluatebilinearconstraint(prob)
@test c1 ≈ c2
end

## Run ADMM
@testset "Dubins (unconstrained)" begin
prob = builddubinsproblem()
admm = BilinearADMM(prob)
admm.opts.penalty_threshold = 1e2
BilinearControl.setpenalty!(admm, 1e4)
X = extractstatevec(prob)
U = extractcontrolvec(prob)
Xsol, Usol = BilinearControl.solve(admm, X, U, max_iters=300)
n,m = RD.dims(prob.model[1])

# Make sure it made it to the goal
xtraj = reshape(Xsol,n,:)[1,:]
ytraj = reshape(Xsol,n,:)[2,:]
@test abs(xtraj[end] - prob.xf[1]) < BilinearControl.get_primal_tolerance(admm) 
@test abs(ytraj[end] - prob.xf[2]) < BilinearControl.get_primal_tolerance(admm) 

# Check the terminal heading
zterm = Xsol[end-1:end]
@test abs(zterm'*[0,1] - 1.0) < BilinearControl.get_primal_tolerance(admm) 

# Check that the norm is preserved
normerr = maximum([norm(x[3:4]) - 1 for x in eachcol(reshape(Xsol,n,:))])
@test normerr < 1e-2

# Check that the control signals are smooth 
Us = reshape(Usol, m, :)
@test all(x->x< 1e-2, mean(diff(Us, dims=2), dims=2))

global umax0 = norm(Us,Inf)
end

## Check with constraints
@testset "Dubins w/ Control Constraints" begin
ubnd = 0.8
prob = builddubinsproblem(ubnd=ubnd)
admm = BilinearADMM(prob)
@test admm.ulo == fill(-ubnd, length(admm.z))
@test admm.uhi == fill(+ubnd, length(admm.z))
admm.opts.penalty_threshold = 1e2
BilinearControl.setpenalty!(admm, 1e3)
X = extractstatevec(prob)
U = extractcontrolvec(prob)
Xsol, Usol = BilinearControl.solve(admm,X,U, max_iters=100)

# Make sure it made it to the goal
n,m = RD.dims(prob.model[1])
xtraj = reshape(Xsol,n,:)[1,:]
ytraj = reshape(Xsol,n,:)[2,:]
@test abs(xtraj[end] - prob.xf[1]) < BilinearControl.get_primal_tolerance(admm) 
@test abs(ytraj[end] - prob.xf[2]) < BilinearControl.get_primal_tolerance(admm) 

# Check the terminal heading
zterm = Xsol[end-1:end]
@test abs(zterm'*[0,1] - 1.0) < BilinearControl.get_primal_tolerance(admm) 

# Check that the norm is preserved
normerr = maximum([norm(x[3:4]) - 1 for x in eachcol(reshape(Xsol,n,:))])
@test normerr < 1e-2

# Check that the control signals are smooth 
Us = reshape(Usol, m, :)
@test all(x->x< 1e-2, mean(diff(Us, dims=2), dims=2))
Us = reshape(Usol, m, :)

# Check maximum control
umax = norm(Us,Inf)
@test umax < umax0*0.99
@test umax - ubnd < 1e-3

end

@testset "Dubins (parallel park)" begin
Random.seed!(1)
ubnd = 1.15
prob = builddubinsproblem(scenario=:parallelpark, ubnd=ubnd)
rollout!(prob)
model = prob.model[1].continuous_dynamics
n,m = RD.dims(model)
admm = BilinearADMM(prob)
X = extractstatevec(prob)
U = extractcontrolvec(prob)
admm.opts.ϵ_abs_primal = 1e-5
admm.opts.ϵ_rel_primal = 1e-5
admm.opts.ϵ_abs_dual = 1e-3
admm.opts.ϵ_rel_dual = 1e-3
admm.opts.penalty_threshold = Inf 
BilinearControl.setpenalty!(admm, 1e3)
Xsol, Usol = BilinearControl.solve(admm, X, U, max_iters=400)
v,ω = collect(eachrow(reshape(Usol, m, :)))
xtraj = reshape(Xsol,n,:)[1,:]
ytraj = reshape(Xsol,n,:)[2,:]

@test norm([norm(x[3:4]) - 1 for x in eachcol(reshape(Xsol,n,:))], Inf) < 1e-2 

# Make sure it made it to the goal
n,m = RD.dims(prob.model[1])
xtraj = reshape(Xsol,n,:)[1,:]
ytraj = reshape(Xsol,n,:)[2,:]
@test abs(xtraj[end] - prob.xf[1]) < BilinearControl.get_primal_tolerance(admm) 
@test abs(ytraj[end] - prob.xf[2]) < BilinearControl.get_primal_tolerance(admm) 

# Check the terminal heading
zterm = Xsol[end-1:end]
@test abs(zterm'*[1,0] - 1.0) < BilinearControl.get_primal_tolerance(admm) 

# Check that the norm is preserved
normerr = maximum([norm(x[3:4]) - 1 for x in eachcol(reshape(Xsol,n,:))])
@test normerr < 1e-2

# Check that the control signals are smooth 
Us = reshape(Usol, m, :)
@test all(x->x< 1e-2, mean(diff(Us, dims=2), dims=2))

# Check maximum control
umax = norm(Us,Inf)
@test umax - ubnd < BilinearControl.get_primal_tolerance(admm) 
end
