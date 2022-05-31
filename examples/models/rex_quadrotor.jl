using Rotations
using ForwardDiff
using LinearAlgebra
# using ControlSystems
# using JSON
using StaticArrays
using Printf
using RobotDynamics: ContinuousDynamics, RigidBody, LieGroupModel, @autodiff
using RobotDynamics: dynamics!, dynamics, jacobian!, forces, moments, wrenches, inertia, inertia_inv, orientation
using RobotDynamics: state_dim, control_dim

const quad_mass = 2.0
const quad_inertia = SMatrix{3,3,Float64,9}([0.01566089 0.00000318037 0; 0.00000318037 0.01562078 0; 0 0 0.02226868])
const quad_motor_kf = 0.0244101
const quad_motor_bf = -30.48576
const quad_motor_km = 0.00029958
const quad_motor_bm = -0.367697
const quad_arm_len = 0.28
const quad_min_throttle = 1148.0
const quad_max_throttle = 1832.0

@autodiff struct RexQuadrotor{R} <: RigidBody{R}
    mass::Float64
    J::SMatrix{3,3,Float64,9}
    Jinv::SMatrix{3,3,Float64,9}
    gravity::SVector{3,Float64}
    motor_dist::Float64
    kf::Float64
    km::Float64
    bf::Float64
    bm::Float64
end

RobotDynamics.state_dim(::RexQuadrotor) = 6
RobotDynamics.control_dim(::RexQuadrotor) = 4

function RexQuadrotor{R}(;
        mass=quad_mass,
        J=quad_inertia,
        gravity=SVector(0,0,-9.81),
        motor_dist=quad_arm_len,
        kf=quad_motor_kf,
        km=quad_motor_km,
        bf=quad_motor_bf,
        bm=quad_motor_bm
    ) where R
    RexQuadrotor(mass,J,inv(J),gravity,motor_dist,kf,km,bf,bm)
end

(::Type{RexQuadrotor})(;kwargs...) = RexQuadrotor{MRP{Float64}}(;kwargs...)

# %%
function trim_controls(model::RexQuadrotor)
    kf, bf = model.kf, model.bf
    g = 9.81
    m = model.mass

    thrust = (g * m) / 4.0
    pwm = (thrust - bf) / kf

    return [pwm for _ in 1:4]
end

HOVER_STATE = let
    r0 = [-0.02, 0.17, 1.70]
    q0 = [1., 0, 0, 0]
    v0 = [0.0; 0.0; 0.0]
    ω0 = [0.0; 0.0; 0.0]
    [r0; q0; v0; ω0]
    end

  HOVER_INPUT = trim_controls(RexQuadrotor())

"""
* `x` - Quadrotor state
* `u` - Motor PWM commands
"""

function forces(model::RexQuadrotor, x, u)
    q = Rotations.MRP(x[4:6])
    kf = model.kf
    bf = model.bf

    Kf = [0.0  0   0   0;
          0.0  0   0   0;
          kf   kf  kf  kf];
    Bf = [0; 0; 4*bf];
    g = [0; 0; -9.81 * quad_mass]

    force = Kf * u + Bf + q' * g
    return force
end

function moments(model::RexQuadrotor, x, u)
    km = model.km
    bm = model.bm
    kf = model.kf
    bf = model.bf

    Km = [0.0 0 0 0; 0 0 0 0; km -km km -km];
    # Bm = [0; 0; 4*bm];
    # torque = Km * u + Bm
    torque = Km * u

    ss = normalize.([[1.;1;0], [1.;-1;0], [-1.;-1;0], [-1.;1;0]])
    for (si, ui) in zip(ss, u)
        torque += cross(quad_arm_len * si, [0; 0; kf * ui + bf])
    end

    return torque
end

function cont_dynamics(model::RexQuadrotor, x, u)
    p = x[1:3]
    q = Rotations.MRP(x[4:6])
    v = x[8:10]
    ω = x[11:13]
    m = model.mass
    J = model.J

    dp = q * v
    dq = Rotations.kinematics(q, ω)
    dv = 1/m * forces(model, x, u) - cross(ω, v)
    dω = J \ (moments(model, x, u) - cross(ω, J * ω))

    return [dp; dq; dv; dω]
end

function dynamics_rk4(model::RexQuadrotor, x, u, dt)
    k1 = cont_dynamics(model, x, u)
    k2 = cont_dynamics(model, x + 0.5 * dt * k1, u)
    k3 = cont_dynamics(model, x + 0.5 * dt * k2, u)
    k4 = cont_dynamics(model, x + dt * k3, u)
    tmp = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    tmp[4:6] .= Rotations.params(Rotations.MRP(tmp[4:6]))

    return tmp
end

# function state_error(x2, x1)
#     p1 = x1[1:3]
#     p2 = x2[1:3]
#     q1 = Rotations.MRP(x1[4:6])
#     q2 = Rotations.MRP(x2[4:6])
#     all1 = x1[7:end]
#     all2 = x2[7:end]

#     ori_er = Rotations.rotation_error(q2, q1, Rotations.CayleyMap())

#     dx =[p2-p1; ori_er; all2-all1]
#     return dx
# end

# function error_state_jacobian(x)
#     # Get various compoents
#     q = Rotations.MRP(x[4:6])
#     # Build error state to state jacobian
#     J = zeros(13, 12)
#     J[1:3, 1:3] .= I(3)
#     J[4:7, 4:6] .= Rotations.∇differential(q)
#     J[8:end, 7:end] .= I(6)

#     return J
# end