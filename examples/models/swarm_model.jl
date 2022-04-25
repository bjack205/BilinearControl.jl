
struct Swarm{P,L} <: RD.ContinuousDynamics
    model::L
    Swarm{P}(model::L) where {P,L <: RD.ContinuousDynamics} = new{P,L}(model)
end

RD.state_dim(model::Swarm{P}) where P = P * RD.state_dim(model.model)
RD.control_dim(model::Swarm{P}) where P = P * RD.control_dim(model.model)
RD.default_diffmethod(model::Swarm) = RD.default_diffmethod(model.model)
RD.default_signature(model::Swarm) = RD.default_signature(model.model)

function Base.rand(model::Swarm{P}) where P
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