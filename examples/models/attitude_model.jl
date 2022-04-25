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

"""
Uses torque as an input. The angular velocity is also an extra input to keep the 
bilinear dynamics, but should be constrained to be equal to the angular velocity 
in the state.

```math
\\dot{R} = R \\hat{w}
\\dot{\\omega} = J^{-1}(\\tau - w \\times J \\omega)
```
where ``w`` are the angular rates in the control vector, and ``\\omega`` are the 
those in the state vector.

"""
struct FullAttitudeDynamics <: RD.ContinuousDynamics 
    J::Diagonal{Float64, SVector{3,Float64}}
end

RD.state_dim(::FullAttitudeDynamics) = 12
RD.control_dim(::FullAttitudeDynamics) = 6
RD.default_diffmethod(::FullAttitudeDynamics) = RD.UserDefined()

translation(::FullAttitudeDynamics, x) = @SVector zeros(3)
orientation(::FullAttitudeDynamics, x) = RotMatrix{3}(x[1:9]...)

function Base.rand(::FullAttitudeDynamics)
    x = [vec(qrot(normalize(@SVector randn(4)))); (@SVector randn(3))] 
    u = @SVector randn(6)
    x,u
end

function RD.dynamics(model::FullAttitudeDynamics, x, u)
    J = model.J
    R = SA[
        x[1] x[4] x[7]
        x[2] x[5] x[8]
        x[3] x[6] x[9]
    ]
    ω = SA[x[10], x[11], x[12]]
    τ = SA[u[1], u[2], u[3]]
    w = SA[u[4], u[5], u[6]]
    Rdot = R*skew(w)
    ωdot = J\(τ - w × (J*ω))
    [vec(Rdot); ωdot]
end

BilinearControl.getA(::FullAttitudeDynamics) = @SMatrix zeros(12,12)

function BilinearControl.getB(model::FullAttitudeDynamics)
    J1 = 1 / model.J[1,1]
    J2 = 1 / model.J[2,2]
    J3 = 1 / model.J[3,3]
    [
        @SMatrix zeros(9,6);
        SA[
            J1 0 0 0 0 0
            0 J2 0 0 0 0
            0 0 J3 0 0 0
        ]
    ]
end

function BilinearControl.getC(model::FullAttitudeDynamics)
    J1 = model.J[1,1]
    J2 = model.J[2,2]
    J3 = model.J[3,3]
    C = [zeros(12,12) for i = 1:6]
    for i = 1:3
        C[4][3+i, 6+i] = 1
        C[4][6+i, 3+i] = -1
        C[5][0+i, 6+i] = -1
        C[5][6+i, 0+i] = 1
        C[6][0+i, 3+i] = 1
        C[6][3+i, 0+i] = -1

        C[4][11,12] = J3/J2
        C[4][12,11] = -J2/J3
        C[5][10,12] = -J3/J1
        C[5][12,10] = J1/J3
        C[6][10,11] = J2/J1
        C[6][11,10] = -J1/J2
    end
    C
end

BilinearControl.getD(::FullAttitudeDynamics) = @SVector zeros(12)

struct ConsensusDynamics <: RD.DiscreteDynamics
    J::Diagonal{Float64, SVector{3,Float64}}
end

RD.state_dim(::ConsensusDynamics) = 12
RD.control_dim(::ConsensusDynamics) = 6
RD.default_diffmethod(::ConsensusDynamics) = RD.UserDefined()
RD.output_dim(::ConsensusDynamics) = 15

function Base.rand(::ConsensusDynamics)
    x = [vec(qrot(normalize(@SVector randn(4)))); (@SVector randn(3))] 
    u = @SVector randn(6)
    x,u
end

function RD.dynamics_error(model::ConsensusDynamics, z2::RD.AbstractKnotPoint, z1::RD.AbstractKnotPoint)
    x1,u1 = RD.state(z1), RD.control(z1)
    h = RD.timestep(z1)
    x2 = RD.state(z2)

    J = model.J
    R1 = SA[
        x1[1] x1[4] x1[7]
        x1[2] x1[5] x1[8]
        x1[3] x1[6] x1[9]
    ]
    ω1 = SA[x1[10], x1[11], x1[12]]
    R2 = SA[
        x2[1] x2[4] x2[7]
        x2[2] x2[5] x2[8]
        x2[3] x2[6] x2[9]
    ]
    ω2 = SA[x2[10], x2[11], x2[12]]

    τ = SA[u1[1], u1[2], u1[3]]
    w = SA[u1[4], u1[5], u1[6]]

    R = (R1 + R2)/2
    ω = (ω1 + ω2)/2
    Rdot = R*skew(w)
    ωdot = J\(τ - w × (J*ω))
    [h*[vec(Rdot); ωdot] + x1 - x2; ω1 - w]
end

function BilinearControl.getA(::ConsensusDynamics, h)
    n = 12
    A = zeros(15,2n)
    for i = 1:3
        A[12+i,9+i] = 1
    end
    for i = 1:n
        A[i,i] = 1
        A[i,i+n] = -1
    end
    A
end

function BilinearControl.getB(model::ConsensusDynamics, h)
    J1 = h / model.J[1,1]
    J2 = h / model.J[2,2]
    J3 = h / model.J[3,3]
    [
        zeros(9,12);
        [
            J1 0 0 0 0 0
            0 J2 0 0 0 0
            0 0 J3 0 0 0
            0 0 0 -1 0 0
            0 0 0 0 -1 0
            0 0 0 0 0 -1
        ] zeros(6,6)
    ]
end

function BilinearControl.getC(model::ConsensusDynamics, h)
    J1 = model.J[1,1]
    J2 = model.J[2,2]
    J3 = model.J[3,3]
    C = [zeros(15,24) for i = 1:12]
    n = 12
    for i = 1:3
        for j in (0,n)
            C[4][3+i, 6+i+j] = 1 
            C[4][6+i, 3+i+j] = -1
            C[5][0+i, 6+i+j] = -1
            C[5][6+i, 0+i+j] = 1
            C[6][0+i, 3+i+j] = 1
            C[6][3+i, 0+i+j] = -1

            C[4][11,12+j] = J3/J2
            C[4][12,11+j] = -J2/J3
            C[5][10,12+j] = -J3/J1
            C[5][12,10+j] = J1/J3
            C[6][10,11+j] = J2/J1
            C[6][11,10+j] = -J1/J2
        end
    end
    C * h / 2
end

BilinearControl.getD(::ConsensusDynamics, h) = @SVector zeros(15)