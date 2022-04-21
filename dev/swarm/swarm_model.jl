using BilinearControl.Problems
using LinearAlgebra
using SparseArrays
using StaticArrays

struct SwarmSE2{P} <: RD.ContinuousDynamics end

RD.state_dim(::SwarmSE2{P}) where P = 4P
RD.control_dim(::SwarmSE2{P}) where P = 2P
RD.default_diffmethod(::SwarmSE2) = RD.UserDefined()
RD.default_signature(::SwarmSE2) = RD.InPlace()

function getstate(::SwarmSE2{P}, x, i) where P
    offset = (i-1)*4 
    SA[x[offset+1], x[offset+2], x[offset+3], x[offset+4]]
end

function getcontrol(::SwarmSE2{P}, u, i) where P
    offset = (i-1)*2
    SA[u[offset+1], u[offset+2]]
end

function Base.rand(::SwarmSE2{P}) where P
    x = randn(4P)
    X = reshape(x, 4, P)
    for i = 1:P
        normalize!(view(X, 3:4, i))
    end
    u = randn(2P)
    return vec(x), u
end

function RD.dynamics!(model::SwarmSE2{P}, xdot, x, u) where P
    for i = 1:P
        px,py,α,β = getstate(model, x, i)
        ν,ω = getcontrol(model, u, i)
        xdot[1 + (i-1)*4] = ν * α
        xdot[2 + (i-1)*4] = ν * β 
        xdot[3 + (i-1)*4] = -β * ω 
        xdot[4 + (i-1)*4] = +α * ω 
    end
    return nothing
end

function RD.jacobian!(model::SwarmSE2{P}, J, xdot, x, u) where P
    for i = 1:P
        px,py,α,β = getstate(model, x, i)
        ν,ω = getcontrol(model, u, i)
        J[1 + (i-1)*4, 3 + (i-1)*4] = ν
        J[1 + (i-1)*4, 1 + 4P + (i-1)*2] = α 
        J[2 + (i-1)*4, 4 + (i-1)*4] = ν
        J[2 + (i-1)*4, 1 + 4P + (i-1)*2] = β 
        J[3 + (i-1)*4, 4 + (i-1)*4] = -ω 
        J[3 + (i-1)*4, 2 + 4P + (i-1)*2] = -β 
        J[4 + (i-1)*4, 3 + (i-1)*4] = +ω
        J[4 + (i-1)*4, 2 + 4P + (i-1)*2] = +α 
    end
    return nothing
end

BilinearControl.getA(::SwarmSE2{P}) where P = zeros(4P, 4P)
BilinearControl.getB(::SwarmSE2{P}) where P = zeros(4P, 2P)

function BilinearControl.getC(::SwarmSE2{P}) where P
    C = [spzeros(4P, 4P) for i = 1:2P]
    for i = 1:P
        C[1 + (i-1)*2][1 + (i-1)*4, 3 + (i-1)*4] = 1    # ν * α
        C[1 + (i-1)*2][2 + (i-1)*4, 4 + (i-1)*4] = 1    # ν * β
        C[2 + (i-1)*2][3 + (i-1)*4, 4 + (i-1)*4] = -1   # -ω * β
        C[2 + (i-1)*2][4 + (i-1)*4, 3 + (i-1)*4] = +1   # +ω * α
    end
    return C
end

BilinearControl.getD(::SwarmSE2{P}) where P = zeros(4P)

function buildformationconstraint(::SwarmSE2{P}, cons)
    ncons = length(cons)
    F = zeros(4ncons, 4P)
    for k = 1:ncons
        i,j = cons[k].i, cons[k].j
        x,y = cons[k].x, cons[k].y
        α,β = cons[k].α, cons[k].β
        Fi = view(F, (k-1)*4 .+ (1:4), (i-1)*4 .+ (1:4))
        Fi[1,1] = 1
        Fi[2,2] = 1
        Fi[1,3] = x
        Fi[2,4] = x
        Fi[1,4] = -y
        Fi[2,3] = y

        Gi = view(F, (k-1)*4 .+ (1:4), (j-1)*4 .+ (1:4))
        Gi[1,1] = -1
        Gi[2,2] = -1
        C[1 + (i-1)*2]
    end
    F
end

function BilinearControl.buildbilinearconstraintmatrices(model::SwarmSE2{P}, x0, Af, bf, h, N; 
        relcons=[]
    )
    n,m = RD.dims(model)
    ncons = length(relcons)
    Nx = N*n
    Nu = (N-1)*m
    Nc = length(x0) + size(Af,1) + (N-1) * n + (N-2)*2*ncons

    ic = 1:n
    ix1 = 1:n
    ix2 = ix1 .+ n
    iu1 = 1:m

    # Build matrices
    Abar = spzeros(Nc,Nx)
    Bbar = spzeros(Nc,Nu)
    Cbar = [spzeros(Nc,Nx) for i = 1:Nu]
    Dbar = spzeros(Nc)

    # Initial condition
    Abar[ic, ix1] .= -I(n)
    Dbar[ic] .= x0
    ic = ic[end] .+ (1:n)

    # Dynamics and Formation
    A,B,C,D = getA(model), getB(model), getC(model), getD(model)
    F = buildformationconstraint(model, relcons)
    for k = 1:N-1
        # Dynamics
        Abar[ic, ix1] .= h/2 * A + I
        Abar[ic, ix2] .= h/2 * A - I
        Bbar[ic, iu1] .= h*B
        for (i,j) in enumerate(iu1)
            Cbar[j][ic,ix1] .= h/2 * C[i]
            Cbar[j][ic,ix2] .= h/2 * C[i]
        end
        Dbar[ic] .= h*D

        if k > 1
            # Formation constraint
            ic2 = ic[end] .+ (1:2ncons)
            Abar[ic2, ix1] .= F
        else
            ic2 = ic[end]+1:ic[end]
        end

        @show ic
        ic = ic2.stop .+ (1:n)
        ix1 = ix1 .+ n
        ix2 = ix2 .+ n
        iu1 = iu1 .+ m 
    end

    # Terminal constraint
    @show ic
    ic = ic[1] - 1 .+ (1:size(Af,1))
    Abar[ic, ix1] .= Af
    Dbar[ic] .= bf

    Abar, Bbar, Cbar, Dbar
end