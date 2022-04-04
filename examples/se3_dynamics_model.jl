using RobotDynamics
using ForwardDiff
using FiniteDiff    
const RD = RobotDynamics

include("rotation_utils.jl")
include("se3_bilinear_dynamics.jl")

RD.@autodiff struct SE3Dynamics <: RD.ContinuousDynamics
    mass::Float64
    J::Diagonal{Float64, SVector{3, Float64}}
    function SE3Dynamics(mass::Real, J::AbstractMatrix)
        new(Float64(mass), Diagonal(SA_F64[J[1,1], J[2,2], J[3,3]]))
    end
end

RD.state_dim(::SE3Dynamics) = 13
RD.control_dim(::SE3Dynamics) = 6
RD.default_diffmethod(::SE3Dynamics) = RD.ForwardAD()
RD.default_signature(::SE3Dynamics) = RD.InPlace()

function Base.rand(::SE3Dynamics)
    r = @SVector randn(3)
    q = normalize(@SVector randn(4))
    v = @SVector randn(3)
    ω = @SVector randn(3)
    u = @SVector randn(6)
    return [r; q; v; ω], u
end

function RD.dynamics(model::SE3Dynamics, x, u)
    J = Diagonal(SA[model.J[1,1], model.J[2,2], model.J[3,3]])
    Jinv = Diagonal(SA[1/model.J[1,1], 1/model.J[2,2], 1/model.J[3,3]])

    q = SA[x[4],x[5],x[6],x[7]]
    v = SA[x[8],x[9],x[10]]
    ω = SA[x[11],x[12],x[13]]
    F = SA[u[1], u[2], u[3]]
    τ = SA[u[4], u[5], u[6]]
    ωhat = pushfirst(ω, 0)
    rdot = qrot(q)*v
    qdot = 0.5*lmult(q)*ωhat
    vdot = F / model.mass - (ω × v)
    ωdot = Jinv * (τ - ω × (J*ω))
    return [rdot; qdot; vdot; ωdot]
end

function RD.dynamics!(model::SE3Dynamics, xdot, x, u)
    xs = SVector{13}(x)
    us = SVector{6}(u)
    xdot .= RD.dynamics(model, xs, us)
    return nothing
end

struct SE3BilinearDynamics <: RD.ContinuousDynamics
    mass::Float64
    J::Diagonal{Float64, SVector{3, Float64}}
    A::SparseMatrixCSC{Float64,Int}
    B::SparseMatrixCSC{Float64,Int}
    C::Vector{SparseMatrixCSC{Float64,Int}}
    D::SparseVector{Float64,Int}
    function SE3BilinearDynamics(mass, J::AbstractMatrix)
        constants = [mass, J[1,1], J[2,2], J[3,3]]
        A,B,C,D = se3_genarrays()
        se3_updateA!(A, constants)
        se3_updateB!(B, constants)
        se3_updateC!(C, constants)
        se3_updateD!(D, constants)
        J_ = Diagonal(SA_F64[J[1,1], J[2,2], J[3,3]])
        new(Float64(mass), J_, A, B, C, D)
    end
end

RD.state_dim(::SE3BilinearDynamics) = 152
RD.control_dim(::SE3BilinearDynamics) = 6
RD.default_diffmethod(::SE3BilinearDynamics) = RD.ForwardAD()
RD.default_signature(::SE3BilinearDynamics) = RD.InPlace()

expand!(::SE3BilinearDynamics, y, x) = se3_expand!(y, x)
getconstants(model::SE3BilinearDynamics) = pushfirst(diag(model.J), model.mass)

function RD.dynamics!(model::SE3BilinearDynamics, xdot, x, u)
    c = getconstants(model)
    se3_dynamics!(xdot, x, u, c)
end

function RD.jacobian!(model::SE3BilinearDynamics, J, y, x, u)
    Jx = view(J, :, 1:152)
    Ju = view(J, :, 153:158)
    Jx .= model.A
    Ju .= model.B
    for i = 1:length(u)
        Jx .+= model.C[i] .* u[i]
        Ju[:,i] .+= model.C[i] * x
    end
    return nothing
end

