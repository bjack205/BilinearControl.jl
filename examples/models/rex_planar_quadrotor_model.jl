using ForwardDiff
using LinearAlgebra
using StaticArrays
using Printf
using RobotDynamics

const RD = RobotDynamics

const quad_mass = 2.0
const quad_inertia = SMatrix{3,3,Float64,9}([0.01566089 0.00000318037 0; 0.00000318037 0.01562078 0; 0 0 0.02226868])
const quad_motor_kf = 0.0244101
const quad_motor_bf = -30.48576
const quad_motor_km = 0.00029958
const quad_motor_bm = -0.367697
const quad_arm_len = 0.28
const quad_min_throttle = 1148.0
const quad_max_throttle = 1832.0

RD.@autodiff struct RexPlanarQuadrotor <: RD.ContinuousDynamics
    mass::Float64  # mass (kg)
    g::Float64     # gravity (m/s²)
    ℓ::Float64     # tip to tip distance (m)
    J::Float64
    kf::Float64
    km::Float64
    bf::Float64
    bm::Float64
end

function RexPlanarQuadrotor(; axis_rot=[1/sqrt(2), 1/sqrt(2), 0])
    mass=quad_mass
    g=9.81
    ℓ=2*quad_arm_len
    J=axis_rot'*quad_inertia*axis_rot
    kf=quad_motor_kf
    km=quad_motor_km
    bf=quad_motor_bf
    bm=quad_motor_bm

    RexPlanarQuadrotor(mass,g,ℓ,J,kf,km,bf,bm)
end

RD.state_dim(::RexPlanarQuadrotor) = 6
RD.control_dim(::RexPlanarQuadrotor) = 2

function trim_controls(model::RexPlanarQuadrotor)
    kf, bf = model.kf, model.bf
    g = 9.81
    m = model.mass

    thrust = (g * m) / 4.0
    pwm = (thrust - bf) / kf

    return [2.0.*pwm for _ in 1:2]
end

function RD.dynamics(model::RexPlanarQuadrotor, x, u)
    
    mass,g,ℓ,J = model.mass, model.g, model.ℓ, model.J
    kf, bf = model.kf, model.bf

    thrust_1 = u[1]*kf + 2.0*bf
    thrust_2 = u[2]*kf + 2.0*bf

    θ = x[3]
    s,c = sincos(θ)
    ẍ = (1/mass)*(thrust_1 + thrust_2)*s
    ÿ = (1/mass)*(thrust_1 + thrust_2)*c - g
    θddot = (1/J)*(ℓ/2)*(thrust_2 - thrust_1)

    return SA[x[4], x[5], x[6], ẍ, ÿ, θddot]

end

function RD.dynamics!(model::RexPlanarQuadrotor, xdot, x, u)
    
    mass,g,ℓ,J = model.mass, model.g, model.ℓ, model.J
    kf, bf = model.kf, model.bf

    thrust_1 = u[1]*kf + 2.0*bf
    thrust_2 = u[2]*kf + 2.0*bf

    θ = x[3]
    s,c = sincos(θ)
    ẍ = (1/mass)*(thrust_1 + thrust_2)*s
    ÿ = (1/mass)*(thrust_1 + thrust_2)*c - g
    θddot = (1/J)*(ℓ/2)*(thrust_2 - thrust_1)

    xdot[1:3] .= @view x[4:6]
    xdot[4] = ẍ
    xdot[5] = ÿ
    xdot[6] = θddot
    return nothing

end