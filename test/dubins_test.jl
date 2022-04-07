
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

function builddubinsproblem(ubnd=Inf)
    # Model
    model = BilinearDubins()
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

    # Discretization
    tf = 3.0
    N = 301

    # Dimensions
    nx = RD.state_dim(model)
    nu = RD.control_dim(model)

    # Initial and final conditions
    x0 = [0,0,1,0]
    xf = [1,1,0,1]

    # Objective
    Q = Diagonal([1e-2,1e-2, 1e-4, 1e-4])
    R = Diagonal(fill(1e-2, nu))
    Qf = Diagonal(fill(100.0, nx))
    obj = LQRObjective(Q,R,Qf,xf,N)

    # Goal state
    cons = ConstraintList(nx, nu, N)
    goalcon = GoalConstraint(xf)  # only constraint the original states
    add_constraint!(cons, goalcon, N)
    
    # Control bound constraints
    add_constraint!(cons, BoundConstraint(nx, nu, u_min=-ubnd, u_max=ubnd), 1:N-1)

    # Initial Guess
    U0 = [[0.1,0] for k = 1:N-1] 

    # Build the problem
    prob = Problem(dmodel, obj, x0, tf, xf=xf, constraints=cons, U0=U0)
    rollout!(prob)
    prob
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
@test abs(xtraj[end] - prob.xf[1]) < 1e-4
@test abs(ytraj[end] - prob.xf[1]) < 1e-4

# Check the terminal heading
zterm = Xsol[end-1:end]
@test abs(zterm'*[0,1] - 1.0) < 1e-4

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
ubnd = 1.0
prob = builddubinsproblem(ubnd)
admm = BilinearADMM(prob)
@test admm.ulo == fill(-ubnd, length(admm.z))
@test admm.uhi == fill(+ubnd, length(admm.z))
admm.opts.penalty_threshold = 1e4
BilinearControl.setpenalty!(admm, 1e3)
X = extractstatevec(prob)
U = extractcontrolvec(prob)
Xsol, Usol = BilinearControl.solve(admm,X,U, max_iters=100)

# Make sure it made it to the goal
n,m = RD.dims(prob.model[1])
xtraj = reshape(Xsol,n,:)[1,:]
ytraj = reshape(Xsol,n,:)[2,:]
@test abs(xtraj[end] - prob.xf[1]) < 1e-4
@test abs(ytraj[end] - prob.xf[1]) < 1e-4

# Check the terminal heading
zterm = Xsol[end-1:end]
@test abs(zterm'*[0,1] - 1.0) < 1e-4

# Check that the norm is preserved
normerr = maximum([norm(x[3:4]) - 1 for x in eachcol(reshape(Xsol,n,:))])
@test normerr < 1e-2

# Check that the control signals are smooth 
Us = reshape(Usol, m, :)
@test all(x->x< 1e-2, mean(diff(Us, dims=2), dims=2))
Us = reshape(Usol, m, :)

# Check maximum control
umax = norm(Us,Inf)
@test umax < umax0*0.9
@test abs(umax - ubnd) < 1e-3
end