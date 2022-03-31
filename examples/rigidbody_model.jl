
struct RigidBodyDynamics{Nu} <: RD.ContinuousDynamics end
RD.state_dim(::RigidBodyDynamics) = 10
RD.control_dim(::RigidBodyDynamics{Nu}) where Nu = Nu
RD.default_diffmethod(::RigidBodyDynamics) = RD.UserDefined()
RD.default_signature(::RigidBodyDynamics) = RD.InPlace()

function Base.rand(::AttitudeDynamics{Nu}) where Nu
    q = normalize(@SVector rand(4))
    r = @SVector randn(3)
    v = @SVector randn(3)
    return [r; q; v]
end

getforce(::AttitudeDynamics)
getomega(::AttitudeDynamics{3}, u) = SA[u[1], u[2], x[3]]
getomega(::AttitudeDynamics{2}, u) = SA[u[1], u[2], 0.0]

function RD.dynamics(model::AttitudeDynamics{3}, x, u)
    r = SA[x[1], x[2], x[3]]
    q = SA[x[4], x[5], x[6], x[7]]
    v = SA[x[8], x[9], x[10]]

    ω = getomega(model, u)
    ωhat = pushfirst(ω, 0.0)
    rdot = v   # assumes velocity is in the world frame
    qdot = lmult(q) * ωhat
    vdot = 
end