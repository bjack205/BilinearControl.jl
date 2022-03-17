export BiLinearDynamics, state_dim, control_dim


struct BiLinearDynamics  # TODO: generalize this to take sparse arrays
    A::Matrix{Float64}
    B::Matrix{Float64}
    C::Vector{Matrix{Float64}}
    function BiLinearDynamics(A, B, C)
        n,m = size(B)
        @assert size(A) == (n,n)
        @assert length(C) == m
        @assert all(x->size(x) == (n,n), C)
        new(A,B,C)
    end
end

state_dim(model::BiLinearDynamics) = size(model.A, 2)
control_dim(model::BiLinearDynamics) = size(model.B, 2)

function discrete_dynamics!(model::BiLinearDynamics, xn, x, u)
    A, B, C = model.A, model.B, model.C
    xn .= A * x .+ B * u
    for i = 1:length(u)
        xn .+= u[i] * C[i] * x
    end
end

function getAhat(model::BiLinearDynamics, u)
    Ahat = copy(model.A)
    for i = 1:length(model.C)
        Ahat .+= model.C[i] * u[i]
    end
    return Ahat
end

function getBhat(model::BiLinearDynamics, x)
    Bhat = copy(model.B)
    for i = 1:length(model.C)
        Bhat[:,i] .+= model.C[i] * x
    end
    return Bhat
end