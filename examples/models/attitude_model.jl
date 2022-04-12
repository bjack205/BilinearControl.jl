using StaticArrays
using Rotations

struct AttitudeDynamics{Nu} <: RD.ContinuousDynamics end
AttitudeDynamics() = AttitudeDynamics{3}()
RD.state_dim(::AttitudeDynamics) = 4
RD.control_dim(::AttitudeDynamics{Nu}) where Nu = Nu
RD.default_diffmethod(::AttitudeDynamics) = RD.UserDefined()
RD.default_signature(::AttitudeDynamics) = RD.InPlace()
Base.rand(::AttitudeDynamics{Nu}) where Nu = normalize(@SVector randn(4)), @SVector randn(Nu)

translation(::AttitudeDynamics, x) = @SVector zeros(3)
orientation(::AttitudeDynamics, x) = UnitQuaternion(x)


function RD.dynamics(::AttitudeDynamics{3}, x, u)
    q = x
    ωhat = SA[0.0, u[1], u[2], u[3]] 
    0.5 * lmult(q) * ωhat 
end

function RD.dynamics(::AttitudeDynamics{2}, x, u)
    q = x
    ωhat = SA[0.0, u[1], u[2], 0] 
    0.5 * lmult(q) * ωhat 
end

function RD.dynamics!(model::AttitudeDynamics, xdot, x, u)
    xdot .= RD.dynamics(model, x, u)
    return nothing
end

function RD.jacobian!(::AttitudeDynamics{2}, J, y, x, u)
    q = x
    ωhat = SA[0.0, u[1], u[2], 0] 
    L = lmult(q)
    J[:,1:4] .= 0.5*rmult(ωhat)
    J[:,5:end] .= 0.5*L[:,2:3]
    return nothing
end

function RD.jacobian!(::AttitudeDynamics{3}, J, y, x, u)
    q = x
    ωhat = SA[0.0, u[1], u[2], u[3]] 
    L = lmult(q)
    J[:,1:4] .= 0.5*rmult(ωhat)
    J[:,5:end] .= 0.5*L[:,2:4]
    return nothing
end


BilinearControl.getA(::AttitudeDynamics) = @SMatrix zeros(4,4)
BilinearControl.getB(::AttitudeDynamics{Nu}) where Nu = @SMatrix zeros(4,Nu)

function BilinearControl.getC(::AttitudeDynamics{Nu}) where Nu
    C1 = 0.5 * SA[
        0 -1 0 0
        +1 0 0 0
        0 0 0 +1
        0 0 -1 0 
    ]
    C2 = 0.5 * SA[
        0 0 -1 0
        0 0 0 -1
        +1 0 0 0
        0 +1 0 0
    ]
    if Nu == 2
        return [C1,C2]
    end
    C3 = 0.5 * SA[
        0 0 0 -1
        0 0 +1 0
        0 -1 0 0
        +1 0 0 0
    ]
    return [C1,C2,C3]
end

BilinearControl.getD(::AttitudeDynamics) = @SVector zeros(4)

struct SO3Dynamics{Nu} <: RD.ContinuousDynamics end
SO3Dynamics() = SO3Dynamics{3}()
RD.state_dim(::SO3Dynamics) = 9
RD.control_dim(::SO3Dynamics{Nu}) where Nu = Nu
RD.default_diffmethod(::SO3Dynamics) = RD.UserDefined()
RD.default_signature(::SO3Dynamics) = RD.InPlace()
Base.rand(::SO3Dynamics{Nu}) where Nu = vec(rand(UnitQuaternion)), @SVector randn(Nu)

translation(::SO3Dynamics, x) = @SVector zeros(3)
orientation(::SO3Dynamics, x) = RotMatrix{3}(x...)

getangularvelocity(::SO3Dynamics{3}, u) = SA[u[1], u[2], u[3]]
getangularvelocity(::SO3Dynamics{2}, u) = SA[u[1], u[2], 0.0]

function RD.dynamics(model::SO3Dynamics, x, u)
    ω = getangularvelocity(model, u)
    ωhat = skew(ω) 
    R = SMatrix{3,3}(x)
    vec(R*ωhat)
end

function RD.dynamics!(model::SO3Dynamics, xdot, x, u)
    xdot .= RD.dynamics(model, x, u)
    return nothing
end

function RD.jacobian!(::SO3Dynamics{Nu}, J, y, x, u) where Nu
    J .= 0
    R = SMatrix{3,3}(x)
    ω = u
    for i = 1:3
        J[i+3,i+6] = ω[1]
        J[i+6,i+3] = -ω[1]
        J[i+6,i+0] = ω[2]
        J[i+0,i+6] = -ω[2]

        J[i+3,10] = R[i,3]
        J[i+6,10] = -R[i,2]
        J[i+0,11] = -R[i,3]
        J[i+6,11] = R[i,1]
        if Nu > 2
            J[i+0,i+3] = ω[3]
            J[i+3,i+0] = -ω[3]

            J[i+0,12] = R[i,2]
            J[i+3,12] = -R[i,1]
        end
    end
    return nothing
end

BilinearControl.getA(::SO3Dynamics) = @SMatrix zeros(9,9)
BilinearControl.getB(::SO3Dynamics{Nu}) where Nu = @SMatrix zeros(9,Nu)

function BilinearControl.getC(::SO3Dynamics{Nu}) where Nu
    C1 = SA[
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 +1 0 0
        0 0 0 0 0 0 0 +1 0
        0 0 0 0 0 0 0 0 +1
        0 0 0 -1 0 0 0 0 0
        0 0 0 0 -1 0 0 0 0
        0 0 0 0 0 -1 0 0 0
    ]
    C2 = SA[
        0 0 0 0 0 0 -1 0 0
        0 0 0 0 0 0 0 -1 0
        0 0 0 0 0 0 0 0 -1
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        +1 0 0 0 0 0 0 0 0 
        0 +1 0 0 0 0 0 0 0 
        0 0 +1 0 0 0 0 0 0 
    ]
    if Nu == 2
        C3 = @SMatrix zeros(9,9)
        return [C1,C2,C3]
    end
    C3 = SA[
        0 0 0 +1 0 0 0 0 0 
        0 0 0 0 +1 0 0 0 0 
        0 0 0 0 0 +1 0 0 0 
        -1 0 0 0 0 0 0 0 0 
        0 -1 0 0 0 0 0 0 0 
        0 0 -1 0 0 0 0 0 0 
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
    ]
    return [C1,C2,C3]
end

BilinearControl.getD(::SO3Dynamics) = @SVector zeros(9)