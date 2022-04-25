using BilinearControl
using BilinearControl.Problems
using LinearAlgebra
using FiniteDiff
using SparseArrays
import BilinearControl.MOI
using Test

prob = Problems.DubinsProblem()

# Build bilinear matrices
A,B,C,D = BilinearControl.buildbilinearconstraintmatrices(
    prob.model[1].continuous_dynamics, prob.x0, prob.xf, prob.Z[1].dt, prob.N
)

# Extract quadratic cost
Q,q,R,r,c = BilinearControl.buildcostmatrices(prob)

# Create NLP evaluator
nlp = BilinearControl.BilinearMOI(A,B,C,D, Q,q,R,r,c)

X = extractstatevec(prob)
U = extractcontrolvec(prob)
z = [X; U]

# Test objective
@test BilinearControl.num_primals(nlp) == length(z)
@test MOI.eval_objective(nlp, z) ≈ 
    0.5*(dot(X, Q, X) + dot(U, R, U)) + dot(X, q) + dot(U, r) + c

# Test objective gradient
df = zero(z)
MOI.eval_objective_gradient(nlp, df, z)
@test df ≈ FiniteDiff.finite_difference_gradient(x->MOI.eval_objective(nlp, x), z)


# Test constraint
@test BilinearControl.num_duals(nlp) == length(D)
c = zeros(length(D))
MOI.eval_constraint(nlp, c, z)
@test c ≈ A*X + B*U + sum(U[i] * C[i] * X for i = 1:length(U)) + D

# Test constraint Jacobian
rc = MOI.jacobian_structure(nlp)
r = getindex.(rc, 1) 
c = getindex.(rc, 2) 
jvec = zeros(length(rc)) 
MOI.eval_constraint_jacobian(nlp, jvec, z)
J = sparse(r,c,jvec)

Jfd = zeros(length(D), length(z))
FiniteDiff.finite_difference_jacobian!(Jfd, (y,x)->MOI.eval_constraint(nlp, y, x), z)
@test Jfd ≈ J

# Test solve
zsol, solver = BilinearControl.solve(nlp, z)
@test MOI.get(solver, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED