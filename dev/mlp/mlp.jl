import RobotDynamics
import RobotDynamics as RD

RD.@autodiff struct MLP{T} <: RD.DiscreteDynamics
    n::Int
    m::Int
    W::Vector{Matrix{T}}
    b::Vector{Vector{T}}
end

function MLP(filename)
    data = JSON.parsefile(filename)
    W1 = Matrix{Float64}(reduce(hcat, data["W1"])')
    W2 = Matrix{Float64}(reduce(hcat, data["W2"])')
    W3 = Matrix{Float64}(reduce(hcat, data["W3"])')
    b1 = Vector{Float64}(data["b1"])
    b2 = Vector{Float64}(data["b2"])
    b3 = Vector{Float64}(data["b3"])
    n = length(b3)
    nm = size(W1,2) 
    m = nm - n
    W = [W1,W2,W3]
    b = [b1,b2,b3]
    MLP{Float64}(n,m,W,b)
end

RD.state_dim(model::MLP) = model.n
RD.control_dim(model::MLP) = model.m
RD.default_diffmethod(::MLP) = RD.ForwardAD()

function RD.discrete_dynamics(mlp::MLP, x, u, t, h)
    mlp(x, u)
end

function (mlp::MLP)(x, u)
    mlp([x;u])
end

function (mlp::MLP)(z)
    y1 = tanh.(mlp.W[1] * z .+ mlp.b[1])
    y2 = tanh.(mlp.W[2] * y1 .+ mlp.b[2])
    y = mlp.W[3] * y2 .+ mlp.b[3]
    y 
end

function jacobian(mlp::MLP, x, u)
    y = [x;u]
    ForwardDiff.jacobian(mlp, y)
end
