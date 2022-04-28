import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using RobotZoo
import RobotDynamics as RD
using LinearAlgebra
using StaticArrays
using SparseArrays
# using MeshCat, GeometryBasics, Colors, CoordinateTransformations, Rotations
using Plots
using BilinearControl
using BilinearControl.Problems
using BilinearControl.EDMD
using Distributions
using Random
using JLD2

## Define some useful controllers
abstract type AbstractController end

struct RandomController{D} <: AbstractController
    m::Int
    distribution::D
    function RandomController(model::RD.AbstractModel, 
            distribution::D=Normal()) where D 
        new{D}(RD.control_dim(model), distribution)
    end
end

function getcontrol(ctrl::RandomController, x, t)
    u = rand(ctrl.distribution, ctrl.m)
    return u
end

struct ZeroController <: AbstractController 
    m::Int
    ZeroController(model::RD.AbstractModel) = new(RD.control_dim(model))
end
getcontrol(ctrl::ZeroController, x, t) = zeros(ctrl.m)


function simulatewithcontroller(sig::RD.FunctionSignature, 
                                model::RD.DiscreteDynamics, ctrl::AbstractController, x0, 
                                tf, dt)
    times = range(0, tf, step=dt)
    m = RD.control_dim(model)
    N = length(times)
    X = [copy(x0) for k = 1:N]
    U = [zeros(m) for k = 1:N-1]
    for k = 1:N-1 
        t = times[k]
        dt = times[k+1] - times[k]
        u = getcontrol(ctrl, X[k], t)
        U[k] = u
        RD.discrete_dynamics!(sig, model, X[k+1], X[k], u, times[k], dt)
    end
    X,U
end

function compare_models(sig::RD.FunctionSignature, model::EDMDModel,
                        model0::RD.DiscreteDynamics, x0, tf, U; 
                        doplot=false, inds=1:RD.state_dim(model0))
    times = range(0, tf, step=dt)
    N = length(times) 
    m = RD.control_dim(model)
    @assert m == RD.control_dim(model0)
    z0 = expandstate(model, x0) 
    Z = [copy(z0) for k = 1:N]
    X = [copy(x0) for k = 1:N]
    for k = 1:N-1
        RD.discrete_dynamics!(sig, model, Z[k+1], Z[k], U[k], times[k], dt)
        RD.discrete_dynamics!(sig, model0, X[k+1], X[k], U[k], times[k], dt)
    end
    X_bl = map(z->model.g * z, Z)
    X_bl, X
    if doplot
        X_mat = reduce(hcat, X_bl)
        X0_mat = reduce(hcat, X)
        p = plot(times, X0_mat[inds,:]', label="original", c=[1 2])
        plot!(p, times, X_mat[inds,:]', label="bilinear", c=[1 2], s=:dash)
        display(p)
    end
    sse = norm(X_bl - X)^2
    println("SSE: $sse")
    X_bl, X
end

function create_data(model::RD.DiscreteDynamics, ctrl::AbstractController, 
                              initial_conditions, tf, dt; sig=RD.InPlace())
    num_traj = length(initial_conditions)
    N = round(Int, tf/dt) + 1
    X_sim = Matrix{Vector{Float64}}(undef, N, num_traj)
    U_sim = Matrix{Vector{Float64}}(undef, N-1, num_traj)
    for i = 1:num_traj
        X,U = simulatewithcontroller(sig, model, ctrl, initial_conditions[i], tf, dt)
        X_sim[:,i] = X
        U_sim[:,i] = U
    end
    X_sim, U_sim
end


## Generate training data
model = RobotZoo.Pendulum()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
num_traj = 600
tf = 2.0
dt = 0.05
ctrl = RandomController(model, Uniform(-5.,5.))
x0_sampler = product_distribution([Uniform(-pi/2,pi/2), Normal(0.0, 1.0)])
initial_conditions = tovecs(rand(x0_sampler, num_traj), length(x0_sampler))
# initial_conditions = [zeros(2) for i = 1:num_traj]
X_train, U_train = create_data(dmodel, ctrl, initial_conditions, tf, dt)
length(X_train)
length(U_train)

## Generate test data
Random.seed!(1)
num_traj_test = 8
tf_test = tf
initial_conditions = tovecs(rand(x0_sampler, num_traj_test), length(x0_sampler))
# initial_conditions = [zeros(2) for i = 1:num_traj_test]
X_test, U_test = create_data(dmodel, ctrl, initial_conditions, tf_test, dt)
length(X_test)

## Learn Bilinear Model
eigfuns = ["state", "sine", "cosine"]
eigorders = [0, 0, 0]
Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders)

# learn bilinear model
F, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
    ["lasso", "lasso"]; edmd_weights=[0.0], mapping_weights=[0.0]);

model_bilinear = EDMDModel(F,C,g,kf,dt,"pendulum")

let i = 6
    compare_models(RD.InPlace(), model_bilinear, dmodel, initial_conditions[i], tf_test, 
        U_test[:,i], doplot=true)
end

const datadir = joinpath(dirname(pathof(BilinearControl)), "../data")
jldsave(joinpath(datadir, "pendulum_eDMD_data.jld2"); A=F, C, g, dt=dt, eigfuns, eigorders)
