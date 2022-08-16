# struct BilinearCartpole <: RD.DiscreteDynamics
#     A::Matrix{Float64}
#     C::Matrix{Float64}
#     g::Matrix{Float64}  # mapping from extended to original states
#     dt::Float64
#     kf::Function
#     function BilinearCartpole()
#         datadir = joinpath(@__DIR__, "..", "..", "data")
#         data = load(joinpath(datadir, "cartpole_eDMD_data.jld2"))
#         A = Matrix(data["F"])
#         C = Matrix(data["C"])
#         g = Matrix(data["g"])
#         T_ref = data["T_ref"]
#         dt = T_ref[2] - T_ref[1]
#         eigfuns = data["eigfuns"]
#         eigorders = data["eigorders"]
#         kf(x) = BilinearControl.EDMD.koopman_transform(Vector(x), eigfuns, eigorders)
#         new(A, C, g, dt, kf)
#     end
# end

# Base.copy(model::BilinearCartpole) = BilinearCartpole()

# RD.output_dim(model::BilinearCartpole) = size(model.A,1)
# RD.state_dim(model::BilinearCartpole) = size(model.A,2)
# RD.control_dim(model::BilinearCartpole) = 1
# RD.default_diffmethod(::BilinearCartpole) = RD.UserDefined()
# RD.default_signature(::BilinearCartpole) = RD.InPlace()

# function RD.discrete_dynamics(model::BilinearCartpole, x, u, t, h)
#     @assert h ≈ model.dt "Timestep must be $(model.dt)."
#     return model.A*x .+ model.C*x .* u[1]
# end

# function RD.discrete_dynamics!(model::BilinearCartpole, xn, x, u, t, h)
#     @assert h ≈ model.dt "Timestep must be $(model.dt)."
#     mul!(xn, model.A, x)
#     mul!(xn, model.C, x, u[1], true)
#     nothing
# end

# function RD.jacobian!(model::BilinearCartpole, J, xn, x, u, t, h)
#     @assert h ≈ model.dt "Timestep must be $(model.dt)."
#     n,m = RD.dims(model)
#     J[:,1:n] .= model.A .+ model.C .* u[1]
#     Ju = view(J, :, n+1)
#     mul!(Ju, model.C, x)
#     nothing
# end

# expandstate(model::BilinearCartpole, x) = model.kf(x)


RD.@autodiff struct Cartpole2{T} <: RD.ContinuousDynamics 
    mc::T
    mp::T
    l::T
    g::T
    b::T         # viscous friction coefficient
    σ::T         # scaling property for tanh friction
    μ::T         # friction coefficient for the cart
    deadband::T  # control deadband
    u_max::T     # maximum control value
    function Cartpole2(mc, mp, l, g, b, σ, μ, deadband, u_max)
        T = eltype(promote(mc, mp, l, g, b))
        new{T}(mc, mp, l, g, b, σ, μ, deadband, u_max)
    end
end

Cartpole2(; mc=1.0, mp=0.2, l=0.5, g=9.81, b=0.01, σ=5, μ=0.0, deadband=0.0, u_max=Inf) = 
    Cartpole2(mc, mp, l, g, b, σ, μ, deadband, u_max)

NominalCartpole(;μ=0.0) = Cartpole2(b=0.0, deadband=0.0, u_max=Inf; μ)
SimulatedCartpole(;μ=0.1) = Cartpole2(mc=1.2, mp=0.25, b=0.07, deadband=0.05; μ)

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

    # Friction
    Fn = (mc + mp) * g  # normal force (approximate)
    σ = model.σ 
    μ = model.μ 
    cart_friction = tanh(σ * qd[1]) * μ * Fn  # approximate coloumb friction
    viscous_friction = model.b .* qd          # viscous friction
    friction = SA[cart_friction + viscous_friction[1], viscous_friction[2]]

    # Control 
    deadband = model.deadband
    u_max = model.u_max 
    u_true = map(u) do uk
        if abs(uk) < deadband
            return 0.0
        elseif abs(uk) > u_max
            return u_max * sign(uk)
        end
        uk
    end

    qdd = -H\(C*qd + G + friction - B*u_true[1])
    return [qd; qdd]
end

function RD.dynamics!(model::Cartpole2, xdot, x, u)
    xdot .= RD.dynamics(model, x, u)
end

RD.state_dim(::Cartpole2) = 4
RD.control_dim(::Cartpole2) = 1