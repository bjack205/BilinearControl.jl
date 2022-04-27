import Pkg; Pkg.activate(@__DIR__)
using BilinearControl
using BilinearControl.RD
using BilinearControl.TO
import BilinearControl.RD
import BilinearControl.TO
using Altro
using BilinearControl.Problems
using BilinearControl: Problems
using BilinearControl.Problems: expandstate
using MeshCat
using Plots
using RobotZoo

## Visualization
model = RobotZoo.Cartpole()
visdir = joinpath(@__DIR__, "../examples/visualization/")
include(joinpath(visdir, "visualization.jl"))
vis = Visualizer()
open(vis)
delete!(vis)
set_cartpole!(vis)

##
prob = Problems.Cartpole(u_bnd=10, N=1001)
altro = ALTROSolver(prob)
solve!(altro)
X0 = TO.states(prob)
X = TO.states(altro)
U = TO.controls(altro)
t = TO.gettimes(altro)
plot(t, X, inds=1:2)
plot(t[1:end-1], U)
visualize!(vis, model, TO.get_final_time(prob), X)
visualize!(vis, model, TO.get_final_time(prob), X0)

## Rollout with Bilinear model
function simulate_bilinear(model::BilinearCartpole, x0, U)
    N = length(U) + 1
    z0 = expandstate(model, x0)
    Z = [copy(z0) for k = 1:N]
    X = [copy(x0) for k = 1:N]
    dt = model.dt
    t = range(0, length=N, step=dt)
    for k = 1:N-1
        Z[k+1] = RD.discrete_dynamics(model, Z[k], U[k], t[k], dt)
        X[k+1] = model.g * Z[k+1]
    end
    t,X
end

model_bl = Problems.BilinearCartpole()
t_bl, X_bl = simulate_bilinear(model_bl, prob.x0, TO.controls(altro))
plot(t, X, inds=1:2, color=[1 2], label="true")
plot!(t, X_bl, inds=1:2, color=[1 2], label="bilinear", s=:dash, legend=:bottomleft)

## Bilinear Problems
prob_bilinear = Problems.BilinearCartpoleProblem(constrained=true)
X0_bl = map(z->model_bl.g*z, TO.states(prob_bilinear))
visualize!(vis, model, TO.get_final_time(prob), X0_bl)


altro_bilinear = ALTROSolver(prob_bilinear, verbose=4)
altro_bilinear.opts.dynamics_diffmethod = RD.UserDefined()
solve!(altro_bilinear)
X_bl = map(z->model_bl.g*z, TO.states(altro_bilinear))
U_bl = TO.controls(altro_bilinear)
t_bl = TO.gettimes(altro_bilinear)
X_mat = reduce(hcat, X_bl)
U_mat = reduce(hcat, U_bl)

plot(t_bl, X_mat[1:2,:]')
plot(t_bl[1:end-1], U_mat')

