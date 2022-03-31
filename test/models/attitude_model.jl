using StaticArrays

struct AttitudeDynamics{Nu} <: RD.ContinuousDynamics end
AttitudeDynamics() = AttitudeDynamics{3}()
RD.state_dim(::AttitudeDynamics) = 4
RD.control_dim(::AttitudeDynamics{Nu}) where Nu = Nu
RD.default_diffmethod(::AttitudeDynamics) = RD.UserDefined()
RD.default_signature(::AttitudeDynamics) = RD.InPlace()
Base.rand(::AttitudeDynamics{Nu}) where Nu = normalize(@SVector randn(4)), @SVector randn(Nu)

function lmult(q)
    w, x, y, z = q
    SA[
        w -x -y -z;
        x  w -z  y;
        y  z  w -x;
        z -y  x  w;
    ]
end

function rmult(q)
    w, x, y, z = q
    SA[
        w -x -y -z;
        x  w  z -y;
        y -z  w  x;
        z  y -x  w;
    ]
end

function RD.dynamics(::AttitudeDynamics{3}, x, u)
    q = x
    ωhat = SA[0.0, u[1], u[2], u[3]] 
    0.5 * lmult(q) * ωhat 
end

function RD.jacobian!(::AttitudeDynamics{3}, J, y, x, u)
    q = x
    ωhat = SA[0.0, u[1], u[2], u[3]] 
    L = lmult(q)
    J[:,1:4] .= 0.5*rmult(ωhat)
    J[:,5:end] .= 0.5*L[:,2:4]
    return nothing
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


