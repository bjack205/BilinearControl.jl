import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using FiniteDiff
using ForwardDiff
using LinearAlgebra

struct MLP{T}
    p::Vector{Int}           # (L+1,) dimensions
    A::Vector{Matrix{T}}     # (L,) coeffs
    b::Vector{Vector{T}}     # (L,) biases
    a::Vector{Vector{T}}     # (L,) temp before activation
    y::Vector{Vector{T}}     # (L,) temp after activatation. y[end] is the ouput
    phi::Vector{Function}    # (L,) activation functions
    dphi::Vector{Function}   # (L,) activation function first derivatives
    ddphi::Vector{Function}  # (L,) activation function second derivatives
    X::Matrix{T}             # (p0,P) Input data
    Y::Matrix{T}             # (p3,P) Output data
    J::Vector{Matrix{T}}     # (P,)   Jacobian data
    alpha::T
end

numsamples(mlp::MLP) = length(mlp.J)

function (mlp::MLP)(theta, x)
    a,y = mlp.a, mlp.y
    yprev = x
    L = 3

    for i = 1:L
        A = getA(mlp, theta, i)
        b = getb(mlp, theta, i)
        a[i] .= A * yprev + b
        y[i] .= mlp.phi[i].(a[i])
        yprev = y[i]
    end

    y[L]
end

function loss(mlp::MLP{T}, theta) where T
    ℒ = zero(T) 
    α = mlp.alpha
    P = numsamples(mlp)
    inputdim = mlp.p[1]
    for i = 1:P
        x = @view mlp.X[:,i]
        y = @view mlp.Y[:,i]
        δf = mlp(theta, x) - y 
        ℒ += (1-α) / 2 * dot(δf, δf)

        # TODO: pre-calculate the full Jacobian to avoid wasted compute
        for j = 1:inputdim
            Fj = model_jacobian(mlp, theta, i, j)
            F̂j = @view mlp.J[i][:,j]
            δF = Fj - F̂j
            ℒ += α / 2 * dot(δF, δF)
        end
    end
    ℒ
end

function grad(mlp::MLP{T}, theta) where T

end

function model_jacobian(mlp::MLP, theta, i, j)
    x = @view mlp.X[:,i]
    mlp(theta, x)  # update a,y
    a = mlp.a
    y = mlp.y

    # TODO: pre-calculate the full Jacobian to avoid wasted compute
    Y3 = Diagonal(mlp.dphi.(a[3])) * getA(mlp, theta, 3) 
    Y2 = Diagonal(mlp.dphi.(a[2])) * getA(mlp, theta, 2) 
    Y1 = Diagonal(mlp.dphi.(a[1])) * getA(mlp, theta, 1) 

    F = Y3*Y2*Y1
    return F[:,j]
end

function getoffset(mlp::MLP, i)
    off = 0
    for j = 1:i-1
        off += mlp.p[j] * mlp.p[j+1] + mlp.p[j+1]
    end
    off
end

function getA(mlp::MLP, theta, i)
    off = getoffset(mlp, i)
    len = mlp.p[i] * mlp.p[i+1]
    view(theta, off .+ (1:len))
end

function getb(mlp::MLP, theta, i)
    off = getoffset(mlp, i) + mlp.p[i] * mlp.p[i+1]
    len = mlp.p[i+1]
    view(theta, off .+ (1:len))
end




p0 = 2
p1 = 4
P = 10
x = randn(p0,P)

layer(A,b) = A*x .+ b 
layer(theta) = layer(reshape(view(theta, 1:p0*p1), p1,p0), view(theta, p0*p1+1:length(theta)))
theta = randn(p0*p1 + p1)
layer(theta)
ForwardDiff.jacobian(layer, theta)

kron(x',I(4))