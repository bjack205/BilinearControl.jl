import Pkg; Pkg.activate(@__DIR__)

using BilinearControl
using COSMOAccelerators
import RobotDynamics as RD
import TrajectoryOptimization as TO
BilinearControl.loadexamples()
import BilinearControl.Problems
include("visualization/visualization.jl")

## Initialize Visualization
using MeshCat
vis = Visualizer()
open(vis)
setendurance!(vis)

## Solve Problem
prob = Problems.SO3Problem(Val(2), Rf=RotZ(deg2rad(180)))
n,m = RD.dims(prob,1)
X = extractstatevec(prob)
U = extractcontrolvec(prob)
# aa = AndersonAccelerator{Float64, Type2{QRDecomp}, RestartedMemory, NoRegularizer}
aa = EmptyAccelerator
admm = BilinearADMM(prob, acceleration=aa)
admm.opts.penalty_threshold = 1e2
# admm.opts.z_solver = :osqp
BilinearControl.setpenalty!(admm, 1e4)
Xsol, Usol, Ysol = BilinearControl.solve(admm, X, U, verbose=true)

## Solve with bound constraints
prob = Problems.SO3Problem(Val(2), Rf=RotZ(deg2rad(180)), ubnd=1.5)
admm = BilinearADMM(prob, acceleration=aa)
admm.opts.penalty_threshold = 1e4
admm.opts.z_solver = :osqp
BilinearControl.setpenalty!(admm, 1e3)
Xsol2, Usol2 = BilinearControl.solve(admm, Xsol, Usol, Ysol*0, verbose=true)

## Visualization
Xs = collect(eachcol(reshape(Xsol2, n, :)))
Us = reshape(Usol2, m, :)
times = TO.gettimes(prob)
visualize!(vis, prob.model[1].continuous_dynamics, TO.get_final_time(prob), Xs)

# Plots 
using Plots, LaTeXStrings
plot(
    times[1:end-1], Us', 
    label=[L"\omega_x" L"\omega_y"], legend=:top, 
    xlabel="time (s)", ylabel="angular velociy (rad/s)"
)
