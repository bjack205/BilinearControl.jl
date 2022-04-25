using ForwardDiff
using FiniteDiff    
using SparseArrays

include("se3_bilinear_dynamics.jl")
include("se3_angvel_dynamics.jl")

#############################################
# Kinematics Model
#############################################
RD.@autodiff struct SE3Kinematics <: RD.ContinuousDynamics end

RD.state_dim(::SE3Kinematics) = 12
RD.control_dim(::SE3Kinematics) = 6
RD.default_diffmethod(::SE3Kinematics) = RD.UserDefined()
RD.default_signature(::SE3Kinematics) = RD.InPlace()
base_state_dim(::SE3Kinematics) = 12

function Base.rand(::SE3Kinematics)
    x = [(@SVector randn(3)); vec(qrot(normalize(@SVector randn(4))))]
    u = @SVector randn(6)
    x,u
end

orientation(::SE3Kinematics, x) = RotMatrix{3}(SMatrix{3,3}(x[4:12]))

buildstate(::SE3Kinematics, x::RBState) = [x.r; vec(x.q)]

getangularvelocity(::SE3Kinematics, u) = SA[u[4], u[5], u[6]]

function RD.dynamics(model::SE3Kinematics, x, u)
    v = SA[u[1], u[2], u[3]]
    ω = getangularvelocity(model, u)
    ωhat = skew(ω) 
    R = SA[
        x[4] x[7] x[10]
        x[5] x[8] x[11]
        x[6] x[9] x[12]
    ]
    rdot = R*v
    Rdot = R*ωhat
    return [rdot; vec(Rdot)]
end

function RD.dynamics!(model::SE3Kinematics, xdot, x, u)
    xdot .= RD.dynamics(model, x, u)
    nothing
end

function RD.jacobian!(model::SE3Kinematics, J, y, x, u)
    Nu = 3
    J .= 0
    v = SA[u[1], u[2], u[3]]
    ω = getangularvelocity(model, u)
    R = SA[
        x[4] x[7] x[10]
        x[5] x[8] x[11]
        x[6] x[9] x[12]
    ]
    for i = 1:3
        J[i+0,i+3] = v[1]
        J[i+0,i+6] = v[2]
        J[i+0,i+9] = v[3]

        J[i+3+3,i+6+3] = ω[1]
        J[i+6+3,i+3+3] = -ω[1]
        J[i+6+3,i+0+3] = ω[2]
        J[i+0+3,i+6+3] = -ω[2]

        J[i+0,13] = R[i,1]
        J[i+0,14] = R[i,2]
        J[i+0,15] = R[i,3]

        J[i+3+3,16] = R[i,3]
        J[i+6+3,16] = -R[i,2]
        J[i+6+3,17] = R[i,1]
        J[i+0+3,17] = -R[i,3]
        if Nu > 2
            J[i+0+3,i+3+3] = ω[3]
            J[i+3+3,i+0+3] = -ω[3]

            J[i+0+3,18] = R[i,2]
            J[i+3+3,18] = -R[i,1]
        end
    end
end

BilinearControl.getA(::SE3Kinematics) = spzeros(12,12)
BilinearControl.getB(::SE3Kinematics) = spzeros(12,6)

function BilinearControl.getC(::SE3Kinematics)
    C = [spzeros(12,12) for i = 1:6]
    for i = 1:3
        C[1][i+0,i+0+3] = 1
        C[2][i+0,i+3+3] = 1
        C[3][i+0,i+6+3] = 1

        C[4][i+6,i+6+3] = 1
        C[4][i+9,i+3+3] = -1
        C[5][i+3,i+6+3] = -1
        C[5][i+9,i+0+3] = 1
        C[6][i+3,i+3+3] = 1
        C[6][i+6,i+0+3] = -1
    end
    return C
end

BilinearControl.getD(::SE3Kinematics) = spzeros(12)

#############################################
#  Dynamics Model
#############################################
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

#############################################
# Bilinear Dynamics Model
#############################################
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
base_state_dim(::SE3BilinearDynamics) = 13

expand!(::SE3BilinearDynamics, y, x) = se3_expand!(y, x)
expand(model::SE3BilinearDynamics, x) = expand!(model, zeros(RD.state_dim(model)), x)
getconstants(model::SE3BilinearDynamics) = pushfirst(diag(model.J), model.mass)

function buildstate(model::SE3BilinearDynamics, x̄::RBState)
    x0 = Vector(x̄)
    expand(model, x0)
end

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

BilinearControl.getA(model::SE3BilinearDynamics) = model.A
BilinearControl.getB(model::SE3BilinearDynamics) = model.B
BilinearControl.getC(model::SE3BilinearDynamics) = model.C
BilinearControl.getD(model::SE3BilinearDynamics) = model.D


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

expand!(::SE3AngVelBilinearDynamics, y, x) = se3_angvel_expand!(y, x)
expand(model::SE3AngVelBilinearDynamics, x) = expand!(model, zeros(RD.state_dim(model)), x) 

function buildstate(model::SE3AngVelBilinearDynamics, x̄::RBState)
    x0 = [x̄.r; Rotations.params(x̄.q); x̄.v]
    expand(model, x0)
end

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

BilinearControl.getA(model::SE3AngVelBilinearDynamics) = model.A
BilinearControl.getB(model::SE3AngVelBilinearDynamics) = model.B
BilinearControl.getC(model::SE3AngVelBilinearDynamics) = model.C
BilinearControl.getD(model::SE3AngVelBilinearDynamics) = model.D