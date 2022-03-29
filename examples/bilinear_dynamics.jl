using RobotDynamics
const RD = RobotDynamics

"""

Dynamics of the form:
```math 
\\dot{x} = A x + B u + \\sum_{i = 1}^{m} u_i C_i x 
```

"""
struct BilinearDynamics{T} <: RobotDynamics.ContinuousDynamics
    A::Matrix{T}
    B::Matrix{T}
    C::Vector{Matrix{T}}
end

function BilinearDynamics(A::AbstractMatrix, B::AbstractVector{<:AbstractMatrix}, 
                                 C::AbstractMatrix) where {Nx,Nu}
    BilinearDynamics(A, B, C)
end
RobotDynamics.state_dim(model::BilinearDynamics) = size(model.A,1) 
RobotDynamics.control_dim(model::BilinearDynamics) = length(model.C) 
RD.default_diffmethod(::BilinearDynamics) = RD.UserDefined()

function RobotDynamics.dynamics(model::BilinearDynamics, x, u)
    A, B, C = model.A, model.B, model.C
    xdot = A * x + B * u
    for i = 1:length(u)
        xdot += u[i] * C[i] * x
    end
    return xdot
end

function RD.dynamics!(model::BilinearDynamics, xdot, x, u)
    A, B, C = model.A, model.B, model.C
    mul!(xdot, A, x)
    mul!(xdot, B, u, 1.0, 1.0)
    for i = 1:length(u)
        mul!(xdot, C[i], x, u[i], 1.0)
    end
    return nothing
end

function RD.jacobian!(model::BilinearDynamics, J, xdot, x, u)
    n,m = RD.dims(model)
    A, B, C = model.A, model.B, model.C
    ix = 1:n
    iu = n .+ (1:m)

    # State derivative
    Jx = view(J, :, ix)
    Jx .= A
    for i = 1:length(u)
        Jx .+= u[i] .* C[i]
    end

    # Control derivative
    Ju = view(J, :, iu)
    Ju .= B
    for i = 1:length(u)
        Ji = view(J, :, n + i) 
        mul!(Ji, C[i], x, 1.0, 1.0)
    end
    return nothing
end

struct ControlIntegralDerivative{L <: RobotDynamics.ContinuousDynamics} <: RobotDynamics.ContinuousDynamics
    model::L 
end

RD.state_dim(model::ControlIntegralDerivative) = 
    RD.state_dim(model.model) + 3 * RD.control_dim(model.model)

RD.control_dim(model::ControlIntegralDerivative) = RD.control_dim(model.model)

function RD.dynamics!(model::ControlIntegralDerivative, xdot, x, u)
    n, m = RD.dims(model.model)
    xdot0 = view(xdot, 1:n)
    x0 = view(x, 1:n)
    u0 = view(x, (n+m) .+ (1:m))
    RD.dynamics!(model.model, xdot0, x0, u0)
    for i = 1:m
        ∫a = x[n + 0m + i] 
        a = x[n + 1m + i]
        da = x[n + 2m + i]
        dda = u[i]

        xdot[n + 0m + i] = a
        xdot[n + 1m + i] = da
        xdot[n + 2m + i] = dda
    end
    return nothing 
end

function RD.jacobian!(model::ControlIntegralDerivative, J, xdot, x, u)
    n, m = RD.dims(model.model)
    J0 = view(J, 1:n, 1:n + m)
    xdot0 = view(xdot, 1:n)
    x0 = view(x, 1:n)
    u0 = view(x, (n + m) .+ (1:m))
    RD.jacobian!(model.model, J0, xdot0, x0, u0)

    # Move controls to next set of columns
    B0 = view(J, 1:n, n .+ (1:m)) 
    J[1:n, (n + m) .+ (1:m)] .= B0 
    B0 .= 0

    # Set identity matrix
    for i = 1:3m
        J[n + i, n + m + i] = 1
    end
    return nothing
end