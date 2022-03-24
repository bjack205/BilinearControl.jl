using SparseArrays
using LinearAlgebra
include("pendulum_bilinear.jl")

struct BilinearPendulum
    A::SparseMatrixCSC{Float64,Int}
    B::SparseMatrixCSC{Float64,Int}
    C::Vector{SparseMatrixCSC{Float64,Int}}
    D::SparseMatrixCSC{Float64,Int}
    x0::Vector{Float64}
end
function BilinearPendulum(x0)
    A,B,C,D = getsparsearrays()
    updateA!(A, x0)
    updateB!(B, x0)
    updateC!(C, x0)
    updateD!(D, x0)
    BilinearPendulum(A, B, C, D, x0)
end

"""
Build the bilinear constraint for the entire trajectory optimization problem, with a uniform
timestep `h` and `N` knot points.

Uses implicit midpoint to integrate the dynamics, maintaining the bilinear structure 
of the dynamics constraints.
"""
function buildbilinearconstraint(model::BilinearPendulum, h, N)
    n,m = size(model.B)
    Nx = N*n
    Nu = (N-1)*m
    Abar = spzeros(Nx,Nx)
    Bbar = spzeros(Nx,Nu)
    Cbar = [spzeros(Nx,Nx) for i = 1:Nu]
    Dbar = spzeros(Nx)
    ic = 1:n
    ix1 = 1:n
    ix2 = ix1 .+ n 
    iu1 = 1:m

    # Initial condition
    Abar[ic,ix1] .= -I(n)
    ic = ic .+ n

    # Dynamics
    A,B,C,D = model.A, model.B, model.C, model.D
    for k = 1:N-1
        Abar[ic, ix1] .= h/2*A + I
        Abar[ic, ix2] .= h/2*A - I
        Bbar[ic, iu1] .= h*B
        for (i,j) in enumerate(iu1)
            Cbar[j][ic,ix1] .= h/2 * C[i]
            Cbar[j][ic,ix2] .= h/2 * C[i]
        end
        Dbar[ic] .= h*D
        ic = ic .+ n
        ix1 = ix1 .+ n
        ix2 = ix2 .+ n
        iu1 = iu1 .+ m 
    end
    return Abar, Bbar, Cbar, Dbar
end

model = BilinearPendulum(zeros(2))
h = 0.1
model.D*h
A,B,C,D = buildbilinearconstraint(model, h, 11)
model.D
n = size(model.A,1)
A[(1:n) .+ n, 1:n] ≈ model.A * h/2 + I
A[(1:n) .+ n, (1:n) .+ n] ≈ model.A * h/2 - I
A[(1:n) .+ 2n, (1:n) .+ n] ≈ model.A * h/2 + I
A[(1:n) .+ 2n, (1:n) .+ 2n] ≈ model.A * h/2 - I