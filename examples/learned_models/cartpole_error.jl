import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using BilinearControl
using BilinearControl.Problems
using BilinearControl.EDMD
import RobotDynamics as RD
import TrajectoryOptimization as TO
using RobotZoo
using LinearAlgebra
using StaticArrays
using SparseArrays
# using MeshCat, GeometryBasics, Colors, CoordinateTransformations, Rotations
using Plots
using Distributions
using Distributions: Normal
using Random
using JLD2
using Altro
using BilinearControl: Problems
using QDLDL
using Test

include("edmd_utils.jl")
include("cartpole_model.jl")

## Visualization 
visdir = Problems.VISDIR
include(joinpath(visdir, "visualization.jl"))
vis = Visualizer()
open(vis)
delete!(vis)
set_cartpole!(vis)

## Simulate the two models
model0 = RobotZoo.Cartpole()
b = 0.02  # damping 
model_nominal = RD.DiscretizedDynamics{RD.RK4}(model0)
model_true = RD.DiscretizedDynamics{RD.RK4}(Cartpole2(model0.mc, model0.mp, model0.l, model0.g, b))
n,m = RD.dims(model_nominal)
dt = 0.01

x0 = [0, pi-0.1, 0.1, 0.0]
t_sim = 4.0
times_sim = range(0,t_sim,step=dt)
U = [zeros(m) for t in times_sim]

Xsim_nom = simulate(model_nominal, U, x0, t_sim, dt)
Xsim_tru = simulate(model_true, U, x0, t_sim, dt)

plot(times_sim, reduce(hcat, Xsim_nom)[1:2,:]', c=[1 2], label="nominal")
plot!(times_sim, reduce(hcat, Xsim_tru)[1:2,:]', c=[1 2], s=:dash, label="true")

visualize!(vis, RobotZoo.Cartpole(), t_sim, Xsim_nom)
visualize!(vis, RobotZoo.Cartpole(), t_sim, Xsim_tru)

## Generate some trajectories from the "true" model
Random.seed!(1)
num_train = 100
num_test = 50 
Qlqr = Diagonal([0.2,10,1e-2,1e-2])
Rlqr = Diagonal([1e-3])
xe = [0,pi,0,0]
ue = [0.0]
ctrl_rand = RandomController(model_nominal, Uniform(-2,2.))
# ctrl_lqr = LQRController(model_nominal, Qlqr, Rlqr, xe, ue, dt)  # generate controller using nominal model
x0_sampler = Product([
    Uniform(-1.0,1.0),
    Uniform(pi-pi/3,pi+pi/3),
    Uniform(-.5,.5),
    Uniform(-.5,.5),
])
initial_conditions_train = [rand(x0_sampler) for _ in 1:num_train]
initial_conditions_test = [rand(x0_sampler) for _ in 1:num_test]

# Generate data using "true" model
t_train = 2.0
X_train, U_train = create_data(model_true, ctrl_rand, initial_conditions_train, t_train, dt)
X_test, U_test = create_data(model_true, ctrl_rand, initial_conditions_test, t_train, dt)
visualize!(vis, model0, t_train, X_train[:,4])

# Calculate "error" output
eigfuns = ["state", "sine", "cosine", "sine", "sine", "chebyshev"]
eigorders = [0,0,0,2,4,4]
kf(x) = koopman_transform(x, eigfuns, eigorders)
n = length(kf(X_train[1]))
m = RD.control_dim(model_nominal)

Xerr_train = calc_error(model_nominal, X_train, U_train, dt)
W,s,Wg,sg = let U = U_train, X = X_train, Xe = Xerr_train
    P = length(U)
    Uj = reduce(hcat, U)
    Yj = mapreduce(kf, hcat, X[1:end-1,:])
    Ye = mapreduce(kf, hcat, Xe)
    Xerr = reduce(hcat, Xe)
    n0 = length(X[1])
    n = size(Yj,1) 
    
    Z = mapreduce(hcat, 1:P) do j
        y = Yj[:,j]
        u = Uj[:,j]
        [y; u; vec(y*u')]
    end

    W = kron(sparse(Z'), sparse(I,n,n))
    s = vec(Ye)
    W,s

    Wg = kron(sparse(Ye'), sparse(I,n0,n0))
    sg = vec(Xerr)
    W,s, Wg,sg
end
W
rho = 1e1
p = size(W,2)
@time F = qr([W; sqrt(rho) * sparse(I,p,p)]);
@time x = F\[s; zeros(p)]
norm(W*x - s)

@time Fg = qr(Wg)
@time g = Fg \ sg
norm(Wg*g - sg)
G = reshape(g, :, n)

E = reshape(x, n, :)
A = E[:,1:n]
B = E[:,n .+ (1:m)]
C = E[:,n+m .+ (1:n*m)]

BilinearControl.EDMD.fiterror(A,B,C,G,kf, X_train, U_train, X_err)

model_bilinear = EDMDModel(A,B,[C],G,kf,dt,"cartpole")
model_learned = EDMDErrorModel(model_nominal, model_bilinear)

BilinearControl.EDMD.fiterror(model_learned, dt, X_train, U_train)
BilinearControl.EDMD.fiterror(model_learned, dt, X_test, U_test)
BilinearControl.EDMD.fiterror(model_nominal, dt, X_train, U_train)

# Simulate systems
Xsim_nom = simulate(model_nominal, U, x0, t_sim, dt)
Xsim_tru = simulate(model_true, U, x0, t_sim, dt)
Xsim_lrn = simulate(model_learned, U, x0, t_sim, dt)

# plot(times_sim, reduce(hcat, Xsim_nom)[1:2,:]', c=[1 2], label="nominal")
plot(times_sim, reduce(hcat, Xsim_tru)[1:2,:]', c=[1 2], s=:solid, label="true",
    xlabel="time (s)", ylabel="states", legend=:outerright
)
plot!(times_sim, reduce(hcat, Xsim_lrn)[1:2,:]', c=[1 2], s=:dash, label="learned")
plot!(times_sim, reduce(hcat, Xsim_nom)[1:2,:]', c=[1 2], s=:dot, w=2, label="nominal")