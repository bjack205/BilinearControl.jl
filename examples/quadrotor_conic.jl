import Pkg; Pkg.activate(@__DIR__)

using BilinearControl
using BilinearControl.Problems
using COSMO
import RobotDynamics as RD
import TrajectoryOptimization as TO
using LinearAlgebra
using StaticArrays
using Statistics
using Rotations
using BilinearControl.Problems: qrot, skew
using SparseArrays
using Test

using BilinearControl: getA, getB, getC, getD

## Visualization 
using MeshCat
vis = Visualizer()
open(vis)
visdir = joinpath(@__DIR__, "../examples/visualization")
include(joinpath(visdir, "visualization.jl"))
setquadrotor!(vis)

# Solve without glideslope
tf = 3.0
N = 101
model = QuadrotorRateLimited()
θ_glideslope = deg2rad(45.0)
admm = Problems.QuadrotorLanding(tf=tf, N=N, θ_glideslope=θ_glideslope*NaN)
BilinearControl.setpenalty!(admm, 1e4)
X = copy(admm.x)
U = copy(admm.z)
admm.opts.x_solver = :osqp
Xsol, Usol = BilinearControl.solve(admm, X, U, verbose=true)

# Solve with glideslope
admm2 = Problems.QuadrotorLanding(tf=tf, N=N, θ_glideslope=θ_glideslope)
length(admm2.constraints) == N-1
admm2.opts.x_solver = :cosmo
BilinearControl.setpenalty!(admm2, 1e4)
Xsol2, Usol2 = BilinearControl.solve(admm2, X, U, verbose=true)

Xs = collect(eachcol(reshape(Xsol, RD.state_dim(model), :)))
X2s = collect(eachcol(reshape(Xsol2, RD.state_dim(model), :)))
α = tan(θ_glideslope)
socerr = map(Xs) do x
    norm(SA[x[1], x[2]]) - α*x[3]
end
maximum(socerr)
socerr2 = map(X2s) do x
    norm(SA[x[1], x[2]]) - α*x[3]
end
maximum(socerr2) < 0.1


# Compare both trajectories 
#   Blue is the solve without SOC constraint
function comparison(vis, model, tf, X1, X2)
    delete!(vis["robot"])
    setquadrotor!(vis["quad1"])
    setquadrotor!(vis["quad2"], color=RGBA(0,0,1,0.5))
    N = length(X1)
    fps = Int(floor((N-1)/tf))
    anim = MeshCat.Animation(fps)
    for k = 1:N
        atframe(anim, k) do 
            visualize!(vis["quad1"], model, X1[k])
            visualize!(vis["quad2"], model, X2[k])
        end
    end
    setanimation!(vis, anim)
end
coneheight = 4.5
r = tan(θ_glideslope) * coneheight
soc = Cone(Point(0,0,coneheight), Point(0,0,0.), r)
mat = MeshPhongMaterial(color=RGBA(1.0,0,0,0.2))
setobject!(vis["soc"], soc, mat)
comparison(vis, model, tf, X2s, Xs)
