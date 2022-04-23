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

function buildformationconstraint(::SwarmSE2{P}, cons) where P
    ncons = length(cons)
    p = 2
    F = zeros(p*ncons, 4P)
    for k = 1:ncons
        i,j = cons[k].i, cons[k].j
        x,y = cons[k].x, cons[k].y
        α,β = cons[k].α, cons[k].β
        Fi = view(F, (k-1)*p .+ (1:p), (i-1)*4 .+ (1:4))
        Fi[1,1] = 1
        Fi[2,2] = 1
        Fi[1,3] = x
        Fi[2,4] = x
        Fi[1,4] = -y
        Fi[2,3] = y

        # Fi[3,3] = α
        # Fi[4,4] = α
        # Fi[3,4] = -β
        # Fi[4,3] = +β

        Gi = view(F, (k-1)*p .+ (1:p), (j-1)*4 .+ (1:4))
        Gi[1,1] = -1
        Gi[2,2] = -1
        # Gi[3,3] = -1
        # Gi[4,4] = -1
    end
    F
end

function BilinearControl.buildbilinearconstraintmatrices(model::SwarmSE2{P}, x0, Af, bf, h, N; 
        relcons=[]
    ) where P
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

        ic = ic2.stop .+ (1:n)
        ix1 = ix1 .+ n
        ix2 = ix2 .+ n
        iu1 = iu1 .+ m 
    end

    # Terminal constraint
    ic = ic[1] - 1 .+ (1:size(Af,1))
    Abar[ic, ix1] .= Af
    Dbar[ic] .= bf

    Abar, Bbar, Cbar, Dbar
end


struct Swarm{P,L} <: RD.ContinuousDynamics
    model::L
    Swarm{P}(model::L) where {P,L <: RD.ContinuousDynamics} = new{P,L}(model)
end

RD.state_dim(model::Swarm{P}) where P = P * RD.state_dim(model.model)
RD.control_dim(model::Swarm{P}) where P = P * RD.control_dim(model.model)
RD.default_diffmethod(model::Swarm) = RD.default_diffmethod(model.model)
RD.default_signature(model::Swarm) = RD.default_signature(model.model)

function Base.rand(model::Swarm{P})
    x = vcat([rand(model.model)[1] for _ in 1:P]...)
    u = vcat([rand(model.model)[2] for _ in 1:P]...)
    x,u
end

function RD.dynamics!(model::Swarm{P}, xdot, x, u) where P
    Xdot = reshape(xdot, :, P)
    X = reshape(x, :, P)
    U = reshape(u, :, P)

    for i = 1:P
        ẋk = @view Xdot[:,i] 
        xk = @view X[:,i] 
        uk = @view U[:,i] 
        RD.dynamics!(model.model, ẋk, xk, uk)
    end
    nothing
end

function BilinearControl.getA(model::Swarm{P}) where P
    A0 = BilinearControl.getA(model.model)
    blockdiag([sparse(A0) for _ = 1:P]...)
end

function BilinearControl.getB(model::Swarm{P}) where P
    B0 = BilinearControl.getB(model.model)
    blockdiag([sparse(B0) for _ = 1:P]...)
end

function BilinearControl.getC(model::Swarm{P}) where P
    n0 = RD.state_dim(model.model)
    m0 = RD.control_dim(model.model)
    p,q = 1:n0 * P , 1:n0 * P
    p = circshift(p, n0)
    q = circshift(q, n0)

    C0 = sparse.(BilinearControl.getC(model.model))
    C0 = map(C0) do C
        sparse(findnz(C)..., n0*P, n0*P)
    end
    C = [spzeros(n0*P, n0*P) for i = 1:m0*P]
    for i = 1:P
        for j = 1:m0
            C[j + (i-1)*m0] .= C0[j]
            permute!(C0[j], p, q)
        end
    end
    C
end

function BilinearControl.getD(model::Swarm{P}) where P
    D0 = BilinearControl.getD(model.model)
    repeat(D0, P)
end