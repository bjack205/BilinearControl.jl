
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

@testset "Dubins Dynamics" begin
testdynamics()
# prob = builddubinsproblem()
prob = Problems.DubinsProblem()
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
# prob = builddubinsproblem()
prob = Problems.DubinsProblem(ubnd=Inf)
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
ubnd = 0.9
# prob = builddubinsproblem(ubnd=ubnd)
prob = Problems.DubinsProblem(ubnd=ubnd)
admm = BilinearADMM(prob)
@test admm.ulo == fill(-ubnd, length(admm.z))
@test admm.uhi == fill(+ubnd, length(admm.z))
admm.opts.penalty_threshold = 1e2
admm.opts.z_solver = :osqp
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
prob = Problems.DubinsProblem(scenario=:parallelpark, ubnd=ubnd)
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
admm.opts.x_solver = :ldl
admm.opts.z_solver = :cholesky
BilinearControl.setpenalty!(admm, 1e3)

# Test the constrained solver warnings
Y = admm.w
@test_logs (:warn, r"Can't use ldl") BilinearControl.solvex(admm, U, Y)
admm.opts.x_solver = :osqp
@test_logs (:warn, r"Cannot solve with control bounds") BilinearControl.solvez(admm, X, Y)
admm.opts.z_solver = :osqp

# Solve problem w/ OSQP
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
