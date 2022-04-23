using StaticArrays
using SparseArrays

Base.@kwdef struct DoubleIntegrator{D} <: RD.ContinuousDynamics 
    mass::Float64 = 1.0
    gravity::SVector{D,Float64} = push((@SVector zeros(D-1)), -9.81)
end

RD.state_dim(::DoubleIntegrator{D}) where D = 2D
RD.control_dim(::DoubleIntegrator{D}) where D = D

function RD.dynamics!(model::DoubleIntegrator{D}, xdot, x, u) where D
    for i = 1:D
        xdot[i] = x[i+D]
        xdot[i+D] = u[i] / model.mass + model.gravity[i]
    end
    nothing
end

function BilinearControl.getA(model::DoubleIntegrator{D}) where D
    A = spzeros(2D, 2D)
    for i = 1:D
        A[i,i+D] = 1.0
    end
    A
end

function BilinearControl.getB(model::DoubleIntegrator{D}) where D
    B = spzeros(2D, D)
    for i = 1:D
        B[i+D,i] = 1.0
    end
    B
end

function BilinearControl.getC(model::DoubleIntegrator{D}) where D
    [spzeros(2D,2D) for _ in 1:D]
end

function BilinearControl.getD(model::DoubleIntegrator{D}) where D
    spzeros(2D)
end

