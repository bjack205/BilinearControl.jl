import Pkg; Pkg.activate(@__DIR__)
using BilinearControl
using BilinearControl.RD
using BilinearControl.TO
import BilinearControl.RD
import BilinearControl.TO
# using Altro
using BilinearControl.Problems
using BilinearControl: Problems
using BilinearControl.Problems: expandstate
using MeshCat
using Plots
using RobotZoo
using Distributions
using LinearAlgebra
using GeometryBasics

## Visualization
vis = Visualizer()
open(vis)
delete!(vis)
let
    w = 0.2
    h = 0.05
    t = 0.1
    setobject!(vis["robot"]["geometry"]["body"], Rect(Vec(-w/2,-h/2,-h/2), Vec(w,h,h)))
    setobject!(vis["robot"]["geometry"]["left"], Rect(Vec(-w/2,-h/2,-h/2), Vec(h,h,t)))
    setobject!(vis["robot"]["geometry"]["right"], Rect(Vec(+w/2,-h/2,-h/2), Vec(h,h,t)))
end

## StraightLineController 
model = RobotZoo.PlanarQuadrotor()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
dx = Product(Uniform.(
    [-1; -1; fill(-eps(), 4)],
    [+1; +1; fill(+eps(), 4)]
))
Q = Diagonal([1e2,1e2,1e2,1e1,1e1,1e1])
R = Diagonal(fill(1e-4,2))
tf = 3.0
dt = 0.05
times = range(0,tf,step=dt)
u0 = fill(model.mass * model.g / 2, 2)

# Generate reference trajectory
xf = [1,1,0,0,0,0.]
x0 = zeros(6)
Xref = map(range(0,1,length(times))) do t
    x0 .+ t .* (xf .- x0)
end
Uref = [copy(u0) for t in times]

ctrl_tvlqr = TVLQRController(dmodel, Q,R, Xref,Uref,times)

Problems.resetcontroller!(ctrl, x0, 0.0)
ctrl.tvlqr.xref[end]

t_sim = 5.0
times_sim = range(0,t_sim,step=dt)
Xsim,Usim = Problems.simulatewithcontroller(dmodel, ctrl, x0, t_sim, dt)

using Plots
plotstates(times_sim, Xsim, inds=1:3)
include(joinpath(Problems.VISDIR, "visualization.jl"))