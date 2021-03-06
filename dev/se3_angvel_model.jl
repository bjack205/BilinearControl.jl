include("se3_angvel_dynamics.jl")

#############################################
# Rigid Body with Angular velocity control
#############################################
RD.@autodiff struct SE3AngVelDynamics <: RD.ContinuousDynamics
    mass::Float64
end

RD.state_dim(::SE3AngVelDynamics) = 10
RD.control_dim(::SE3AngVelDynamics) = 6
RD.default_diffmethod(::SE3AngVelDynamics) = RD.ForwardAD()
RD.default_signature(::SE3AngVelDynamics) = RD.InPlace()

function RD.dynamics(model::SE3AngVelDynamics, x, u)
    q = SA[x[4],x[5],x[6],x[7]]
    v = SA[x[8],x[9],x[10]]
    F = SA[u[1], u[2], u[3]]
    ω = SA[u[4],u[5],u[6]]
    ωhat = pushfirst(ω, 0)
    rdot = qrot(q)*v
    qdot = 0.5*lmult(q)*ωhat
    vdot = F / model.mass - (ω × v)
    return [rdot; qdot; vdot]
end

function RD.dynamics!(model::SE3AngVelDynamics, xdot, x, u)
    xs = SVector{10}(x)
    us = SVector{6}(u)
    xdot .= RD.dynamics(model, xs, us)
    return nothing
end


#############################################
# Bilinear dynamics with angular velocity control
#############################################
struct SE3AngVelBilinearDynamics <: RD.ContinuousDynamics
    mass::Float64
    A::SparseMatrixCSC{Float64,Int}
    B::SparseMatrixCSC{Float64,Int}
    C::Vector{SparseMatrixCSC{Float64,Int}}
    D::SparseVector{Float64,Int}
    function SE3AngVelBilinearDynamics(mass)
        constants = [mass]
        A,B,C,D = se3_angvel_genarrays()
        se3_angvel_updateA!(A, constants)
        se3_angvel_updateB!(B, constants)
        se3_angvel_updateC!(C, constants)
        se3_angvel_updateD!(D, constants)
        new(mass, A, B, C, D)
    end
end

RD.state_dim(::SE3AngVelBilinearDynamics) = 50
RD.control_dim(::SE3AngVelBilinearDynamics) = 6
RD.default_diffmethod(::SE3AngVelBilinearDynamics) = RD.UserDefined()
RD.default_signature(::SE3AngVelBilinearDynamics) = RD.InPlace()
base_state_dim(::SE3AngVelBilinearDynamics) = 10

function Base.rand(::SE3AngVelBilinearDynamics)
    x0 = [randn(3); normalize(randn(4)); randn(3)]
    u = randn(6)
    x = zeros(50)
    se3_angvel_expand!(x, x0)
    return x, u
end

function RD.dynamics!(model::SE3AngVelBilinearDynamics, xdot, x, u)
    se3_angvel_dynamics!(xdot, x, u, SA[model.mass])
end

function RD.jacobian!(model::SE3AngVelBilinearDynamics, J, y, x, u)
    Jx = view(J, :, 1:50)
    Ju = view(J, :, 51:56)
    Jx .= model.A
    Ju .= model.B
    for i = 1:length(u)
        Jx .+= model.C[i] .* u[i]
        Ju[:,i] .+= model.C[i] * x
    end
    return nothing
end
