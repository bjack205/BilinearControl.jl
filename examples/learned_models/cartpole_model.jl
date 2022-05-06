using ForwardDiff, FiniteDiff
using RobotDynamics

RD.@autodiff struct Cartpole2{T} <: RD.ContinuousDynamics 
    mc::T
    mp::T
    l::T
    g::T
    b::T
    function Cartpole2(mc, mp, l, g, b)
        T = eltype(promote(mc, mp, l, g, b))
        new{T}(mc, mp, l, g, b)
    end
end

Cartpole2(; mc=1.0, mp=0.2, l=0.5, g=9.81, b=0.01) = Cartpole2(mc, mp, l, g, b)

function RD.dynamics(model::Cartpole2, x, u)
    mc = model.mc  # mass of the cart in kg (10)
    mp = model.mp   # mass of the pole (point mass at the end) in kg
    l = model.l   # length of the pole in m
    g = model.g  # gravity m/s^2

    q = x[ @SVector [1,2] ]
    qd = x[ @SVector [3,4] ]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp*g*l*s]
    B = @SVector [1, 0]

    qdd = -H\(C*qd + G + model.b .* qd - B*u[1])
    return [qd; qdd]
end

function RD.dynamics!(model::Cartpole2, xdot, x, u)
    xdot .= RD.dynamics(model, x, u)
end

RD.state_dim(::Cartpole2) = 4
RD.control_dim(::Cartpole2) = 1