import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using JSON
using LinearAlgebra
using FiniteDiff
using ForwardDiff
using Distributions
using BilinearControl

include("mlp.jl")

function discrete_double_integrator!(h, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T <: Real
    n,m = size(B)
    A .= zero(T) 
    B .= zero(T) 
    for i = 1:n
        A[i, i] = one(T)
    end
    b = h * h / 2
    for i = 1:m
        A[i, i + m] = h
        B[i, i] = b
        B[i + m, i] = h
    end
    nothing
end

function discrete_double_integrator(h::T, dim) where T
    A = zeros(T, 2dim, 2dim)
    B = zeros(T, 2dim, dim)
    discrete_double_integrator!(h, A, B)
    return A,B
end

function simulate(x0, h, tf)
    m = 2
    A,B = discrete_double_integrator(h, m)
    times = range(0,tf,step=h)
    X = [copy(x0) for t in times]
    U = [zeros(m) for t in 1:length(times)-1]
    kp = 100.0
    kd = 20.0
    for k in 1:length(times)-1 
        pos = X[k][1:2]
        vel = X[k][3:4]
        U[k] = -kp * pos - kd * vel
        X[k+1] = A*X[k] + B * U[k]
    end
    X,U
end

function double_integrator_gendata(;h=0.05, tf=2.0)
    n,m = 4,2
    N = round(Int, tf/h)
    num_traj = 20
    P = N * num_traj

    X = [zeros(n) for k = 1:N+1, i = 1:num_traj]
    U = [zeros(m) for k = 1:N, i = 1:num_traj]

    x_window = [5,5,2,2.0]
    x0_sampler = Product(collect(Uniform(-dx,+dx) for dx in x_window))
    x0 = [rand(x0_sampler) for i = 1:num_traj]
    for i = 1:num_traj
        Xsim,Usim = simulate(x0[i], h, tf)
        X[:,i] .= Xsim
        U[:,i] .= Usim
    end

    states = reduce(hcat, @view X[1:end-1,:])
    inputs = reduce(hcat, U)
    nextstates = reduce(hcat, @view X[2:end,:])

    A,B = discrete_double_integrator(h, m)
    jacobians = zeros(n, n + m, P)
    for i = 1:P
        jacobians[:,:,i] .= [A B]
    end

    data = Dict(
        "name"=>"double integrator",
        "h"=>h,
        "states"=>states,
        "inputs"=>inputs,
        "nextstates"=>nextstates,
        "jacobians"=>jacobians
    )
    open(joinpath(@__DIR__, "double_integrator_data.json"), "w") do f
        JSON.print(f, data, 2)
    end
end


function simulate_learned_model(mlp::MLP, x0, h, tf)
    m = 2
    times = range(0,tf,step=h)
    X = [copy(x0) for t in times]
    U = [zeros(m) for t in 1:length(times)-1]
    kp = 100.0
    kd = 20.0
    for k in 1:length(times)-1 
        pos = X[k][1:2]
        vel = X[k][3:4]
        U[k] = -kp * pos - kd * vel
        X[k+1] = mlp(X[k], U[k]) 
    end
    X,U,times
end

##
h = 0.05
tf = 2.0
double_integrator_gendata(;h, tf)

modelfile = joinpath(@__DIR__, "double_integrator_model.json")
modelfile_jac = joinpath(@__DIR__, "double_integrator_model_jacobian.json")
h = 0.05
mlp = MLP(modelfile)
mlp_jac = MLP(modelfile_jac)

Q = Diagonal([fill(1e0,2); fill(1e-1,2)])
R = Diagonal(fill(1e-2,2))
xe = zeros(4)
ue = zeros(2)
ctrl = BilinearControl.LQRController(mlp, Q, R, xe, ue, h)
ctrl_jac = BilinearControl.LQRController(mlp_jac, Q, R, xe, ue, h)

using RobotZoo
using Plots
model = RobotZoo.DoubleIntegrator(2)
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
ctrl0 = BilinearControl.LQRController(dmodel, Q, R, xe, ue, h)

x0 = [2,3,0,0.]
Xsim,Usim,Tsim = BilinearControl.simulatewithcontroller(dmodel, ctrl, x0, 2.0, h)
Xjac,Ujac,Tjac = BilinearControl.simulatewithcontroller(dmodel, ctrl_jac, x0, 2.0, h)
Xref,Uref,Tref = BilinearControl.simulatewithcontroller(dmodel, ctrl0, x0, 2.0, h)

plotstates(Tref,Xref, inds=1:2, c=[1 2], s=:solid, lw=2.0, label=["nom"  ""])
plotstates!(Tsim,Xsim, inds=1:2, c=[1 2], s=:dash, lw=2.0, label=["MLP"])
plotstates!(Tjac,Xjac, inds=1:2, c=[1 2], s=:dot, lw=2.0, label=["jacMLP"])
BilinearControl.getcontrol(ctrl, x0, 0.0)
Xsim