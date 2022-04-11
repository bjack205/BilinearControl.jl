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
prob = builddubinsproblem(BilinearDubins(), scenario=:turn90, ubnd=ubnd)
admm = BilinearADMM(prob)

## Solve with different methods
X = extractstatevec(prob)
U = extractcontrolvec(prob)
norm(X)
norm(U)
admm.opts.x_solver = :cholesky
admm.opts.z_solver = :cg
BilinearControl.setpenalty!(admm, 1e2)
# admm.Ahat .= 0
# admm.Bhat .= 0
Xsol, Usol = BilinearControl.solve(admm, X, U)

## Solve for states 
using OSQP
using IterativeSolvers
Ahat = admm.Ahat
Bhat = admm.Bhat
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