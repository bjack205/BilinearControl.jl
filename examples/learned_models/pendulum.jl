import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using BilinearControl
using BilinearControl.Problems
using BilinearControl.EDMD
import RobotDynamics as RD
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

## Define some useful controllers
abstract type AbstractController end

resetcontroller!(::AbstractController) = nothing

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

"""
Picks a random control value each time the controller is reset, 
and applies it as a constant.
"""
struct RandConstController{D} <: AbstractController
    distribution::D
    u::Vector{Float64}
    function RandConstController(distribution::D) where D
        u = rand(distribution)
        new{D}(distribution, u)
    end
end

resetcontroller!(ctrl::RandConstController) = rand!(ctrl.distribution, ctrl.u)

getcontrol(ctrl::RandConstController, x, t) = ctrl.u

struct ZeroController <: AbstractController 
    m::Int
    ZeroController(model::RD.AbstractModel) = new(RD.control_dim(model))
end
getcontrol(ctrl::ZeroController, x, t) = zeros(ctrl.m)

struct LQRController <: AbstractController
    K::Matrix{Float64}
    xeq::Vector{Float64}
    ueq::Vector{Float64}
end

function LQRController(A,B,Q,R, xeq, ueq)
    K = dlqr(A,B,Q,R)
    LQRController(Matrix(K), Vector(xeq), Vector(ueq))
end

function LQRController(model::RD.DiscreteDynamics, Q, R, xeq, ueq, dt)
    n,m = RD.dims(model)
    n != length(xeq) && throw(DimensionMismatch("Expected a state vector of length $n, got $(length(xeq))."))
    m != length(ueq) && throw(DimensionMismatch("Expected a control vector of length $m, got $(length(ueq))."))
    zeq = RD.KnotPoint{n,m}(xeq, ueq, 0.0, dt)
    J = zeros(n, n+m)
    xn = zeros(n)
    RD.jacobian!(RD.default_signature(model), RD.default_diffmethod(model), model, J, xn, zeq)
    A = J[:,1:n]
    B = J[:,n+1:n+m]
    LQRController(A,B,Q,R,xeq,ueq)
end

function dlqr(A,B,Q,R; max_iters=200, tol=1e-6)
    P = Matrix(copy(Q))
    n,m = size(B)
    K = zeros(m,n)
    K_prev = copy(K)
    
    for k = 1:max_iters
        K .= (R + B'P*B) \ (B'P*A)
        P .= Q + A'P*A - A'P*B*K
        if norm(K-K_prev,Inf) < tol
            println("Converged in $k iterations")
            return K
        end
        K_prev .= K
    end
    return K * NaN
end

function getcontrol(ctrl::LQRController, x, t)
    dx = x - ctrl.xeq
    ctrl.ueq - ctrl.K*dx
end

function simulatewithcontroller(model::RD.DiscreteDynamics, ctrl::AbstractController, x0, 
                                tf, dt)
    simulatewithcontroller(RD.default_signature(model), model, ctrl, x0, tf, dt)
end

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
    N = length(U) + 1
    times = range(0, tf, length=N)
    dt = times[2]
    @show dt
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
        resetcontroller!(ctrl)
        X,U = simulatewithcontroller(sig, model, ctrl, initial_conditions[i], tf, dt)
        X_sim[:,i] = X
        U_sim[:,i] = U
    end
    X_sim, U_sim
end


## Generate training data
model = RobotZoo.Pendulum()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
num_traj = 200
tf = 2.0
dt = 0.05
ctrl_1 = RandomController(model, Uniform(-5.,5.))
ctrl_2 = RandConstController(Product([Uniform(-7,7)]))
Q = Diagonal([1.0, 0.1])
R = Diagonal(fill(1e-4, 1))
xeq = [pi,0]
ueq = [0.]
ctrl_3 = LQRController(dmodel, Q, R, xeq, ueq, dt)

x0_sampler_1 = Product([Uniform(-eps(),0), Normal(0.0, 0.0)])
x0_sampler_2 = Product([Uniform(-pi/4,pi/4), Normal(0.0, 2.0)])
x0_sampler_3 = Product([Uniform(pi-pi, pi+pi), Normal(0.0, 4.0)])
initial_conditions_1 = tovecs(rand(x0_sampler_1, num_traj), length(x0_sampler_1))
initial_conditions_2 = tovecs(rand(x0_sampler_2, num_traj), length(x0_sampler_2))
initial_conditions_3 = tovecs(rand(x0_sampler_3, num_traj), length(x0_sampler_3))
X_train_1, U_train_1 = create_data(dmodel, ctrl_1, initial_conditions_1, tf, dt)
X_train_2, U_train_2 = create_data(dmodel, ctrl_2, initial_conditions_1, tf, dt)
X_train_3, U_train_3 = create_data(dmodel, ctrl_2, initial_conditions_3, tf, dt)
X_train = hcat(X_train_1, X_train_2, X_train_3)
U_train = hcat(U_train_1, U_train_2, U_train_3)
X_train = X_train_1
U_train = U_train_1

## Generate test data
Random.seed!(1)
num_traj_test = 8
tf_test = tf
initial_conditions = tovecs(rand(x0_sampler_1, num_traj_test), length(x0_sampler_1))
# initial_conditions = [zeros(2) for i = 1:num_traj_test]
X_test, U_test = create_data(dmodel, ctrl_1, initial_conditions, tf_test, dt)

## Learn Bilinear Model
eigfuns = ["state", "sine", "cosine"]
eigorders = [0, 0, 0]
Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders)

# learn bilinear model
F, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
    ["lasso", "lasso"]; edmd_weights=[0.0], mapping_weights=[0.0], algorithm=:cholesky);

model_bilinear = EDMDModel(F,C,g,kf,dt,"pendulum")

let i = 3
    compare_models(RD.InPlace(), model_bilinear, dmodel, initial_conditions[i], tf_test, 
        U_test[:,i], doplot=true)
end

const datadir = joinpath(dirname(pathof(BilinearControl)), "../data")
jldsave(joinpath(datadir, "pendulum_eDMD_data.jld2"); A=F, C, g, dt=dt, eigfuns, eigorders)
