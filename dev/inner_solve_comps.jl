import Pkg; Pkg.activate(@__DIR__)
using BilinearControl
using BilinearControl.RD
using BilinearControl.TO
import BilinearControl.RD
import BilinearControl.TO

# using Altro
using TrajectoryOptimization
using LinearAlgebra
using RobotZoo
using StaticArrays
using Test
using Plots

include("problems.jl")

## Build Solver
ubnd = Inf
# prob = builddubinsproblem(BilinearDubins(), scenario=:turn90, ubnd=ubnd)
prob = buildse3problem()
rollout!(prob)
admm = BilinearADMM(prob)

## Solve with different methods
X = extractstatevec(prob)
U = extractcontrolvec(prob)
norm(X)
norm(U)
# admm.opts.x_solver = :cholesky
admm.opts.x_solver = :cholesky
admm.opts.z_solver = :cholesky
admm.opts.calc_x_residual = true
admm.opts.calc_z_residual = true
BilinearControl.setpenalty!(admm, 1e2)
# admm.Ahat .= 0
# admm.Bhat .= 0

Xsol, Usol = BilinearControl.solve(admm, X, U)
admm.stats.x_solve_iters
admm.stats.x_solve_residual
admm.stats.z_solve_iters
admm.stats.z_solve_residual

## Solve for states 
using OSQP
using IterativeSolvers
Ahat = admm.Ahat
Bhat = admm.Bhat
X = extractstatevec(prob)
U = extractcontrolvec(prob)
Y = admm.w
a = BilinearControl.geta(admm, U)
b = BilinearControl.getb(admm, X)
BilinearControl.updateAhat!(admm, Ahat, U)
BilinearControl.updateBhat!(admm, Bhat, X)
ρ = BilinearControl.getpenalty(admm)
p = length(Y)

# Primal
Hp = admm.Q + ρ*Ahat'Ahat
gp = admm.q + ρ*Ahat'*(a + Y)
x_primal = - (Hp\gp) 
 
# Primal-Dual
Hd = [admm.Q Ahat'; Ahat -I(p)/ρ]
gd = [admm.q; a + Y]
x_dual = -(Hd\gd)[1:length(X)]

# OSQP
osqp = OSQP.Model()
OSQP.setup!(osqp, P=Hp, q=vec(gp), verbose=false)
osqp_res = OSQP.solve!(osqp)
osqp_res.info.status
x_osqp = osqp_res.x 

# CG
x_cg,ch = cg(Hp, -gp, log=true, abstol=1e-6)
ch

# MINRES
z_minres,ch = minres(Hd, -gd, log=true)
x_minres = z_minres[1:length(X)]
ch

# Compare
norm(x_primal - x_dual)
norm(x_primal - x_osqp)
norm(x_primal - x_cg)
norm(x_primal - x_minres)


# Solve for controls