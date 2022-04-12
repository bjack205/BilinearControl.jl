
abstract type AbstractTwoQubit <: RobotDynamics.ContinuousDynamics end

nqubits(::AbstractTwoQubit) = 4
qubitdim(::AbstractTwoQubit) = 8
RobotDynamics.default_signature(::AbstractTwoQubit) = RD.InPlace()

function getdrifthamiltonian(model::AbstractTwoQubit)
    ω1 = model.ω0
    ω2 = model.ω1

    I2 = I(2) 
    σz = paulimat(:z)
    σz_1 = kron(σz, I2)
    σz_2 = kron(I2, σz)
    H = σz_1 * ω1 / 2 + σz_2 * ω2 / 2
    return H
end

function getdrivehamiltonian(::AbstractTwoQubit)
    I2 = I(2) 
    σx = paulimat(:x)
    σx_1 = kron(σx, I2)
    σx_2 = kron(I2, σx)
    Hdrive = σx_1 * σx_2  / 2
    return Hdrive
end

function getqstateinds(model::AbstractTwoQubit, i::Int)
    nqstates = 4
    ir = (i-1)*2nqstates
    return ir .+ (1:8)
end

function getqstate(model::AbstractTwoQubit, x, i::Int)
    nqstates = 4
    # ir = (i-1)*nqstates
    # ic = ir + nqstates^2
    ir = (i-1)*2nqstates
    ic = ir + nqstates
    [
        x[ir+1] + x[ic+1]*1im,
        x[ir+2] + x[ic+2]*1im,
        x[ir+3] + x[ic+3]*1im,
        x[ir+4] + x[ic+4]*1im,
    ] 
end

function setqstate!(model::AbstractTwoQubit, x, ψ, i::Int)
    @assert length(ψ) == 4
    nqstates = 4
    # ir = (1:4) .+ (i-1)*nqstates
    # ic = ir .+ nqstates^2
    ir = (1:4) .+ (i-1)*2nqstates
    ic = ir .+ nqstates
    x[ir] .= real(ψ)
    x[ic] .= imag(ψ)
    return x
end

RD.@autodiff struct TwoQubit <: AbstractTwoQubit 
    ω0::Float64
    ω1::Float64
end
TwoQubit() = TwoQubit(1.0, 1.0)
RobotDynamics.control_dim(::TwoQubit) = 1
RobotDynamics.state_dim(::TwoQubit) = 8 * 4 + 3
# nqubits(::TwoQubit) = 4
# qubitdim(::TwoQubit) = 8
# RobotDynamics.default_signature(::TwoQubit) = RD.InPlace()

# function getdrifthamiltonian(model::TwoQubit)
#     ω1 = model.ω0
#     ω2 = model.ω1

#     I2 = I(2) 
#     σz = paulimat(:z)
#     σz_1 = kron(σz, I2)
#     σz_2 = kron(I2, σz)
#     H = σz_1 * ω1 / 2 + σz_2 * ω2 / 2
#     return H
# end

# function getdrivehamiltonian(::TwoQubit)
#     I2 = I(2) 
#     σx = paulimat(:x)
#     σx_1 = kron(σx, I2)
#     σx_2 = kron(I2, σx)
#     Hdrive = σx_1 * σx_2  / 2
#     return Hdrive
# end

# function getqstateinds(model::TwoQubit, i::Int)
#     nqstates = 4
#     ir = (i-1)*2nqstates
#     return ir .+ (1:8)
# end

# function getqstate(model::TwoQubit, x, i::Int)
#     nqstates = 4
#     # ir = (i-1)*nqstates
#     # ic = ir + nqstates^2
#     ir = (i-1)*2nqstates
#     ic = ir + nqstates
#     [
#         x[ir+1] + x[ic+1]*1im,
#         x[ir+2] + x[ic+2]*1im,
#         x[ir+3] + x[ic+3]*1im,
#         x[ir+4] + x[ic+4]*1im,
#     ] 
# end

# function setqstate!(model::TwoQubit, x, ψ, i::Int)
#     @assert length(ψ) == 4
#     nqstates = 4
#     # ir = (1:4) .+ (i-1)*nqstates
#     # ic = ir .+ nqstates^2
#     ir = (1:4) .+ (i-1)*2nqstates
#     ic = ir .+ nqstates
#     x[ir] .= real(ψ)
#     x[ic] .= imag(ψ)
#     return x
# end

function RobotDynamics.dynamics!(model::TwoQubit, xdot, x, u)
    ω0, ω1 = model.ω0, model.ω1
    ħ = 1.0  # Plancks Constant

    # Extract controls
    nqstates = 4
    nstates_quantum = nqstates^2 * 2  # number of state elements describing quantum bits
    ∫a = x[nstates_quantum + 1]
    a = x[nstates_quantum + 2]
    da = x[nstates_quantum + 3]
    dda = u[1]

    # Form the Hamiltonian
    H_drift = getdrifthamiltonian(model)
    H_drive = getdrivehamiltonian(model)
    H = (H_drift + H_drive * a) * (-im)


    # Compute the dynamics
    nqstates = 4  # number of quantum states
    for i = 1:nqstates
        ψi = getqstate(model, x, i)
        ψdot = H*ψi
        setqstate!(model, xdot, ψdot, i)
    end
    xdot[nstates_quantum + 1] = a
    xdot[nstates_quantum + 2] = da
    xdot[nstates_quantum + 3] = dda
    return nothing
end

function RobotDynamics.jacobian!(model::TwoQubit, J, xdot, x, u)
    J .= 0
    ω0, ω1 = model.ω0, model.ω1

    # Extract controls
    nqstates = 4
    nstates_quantum = nqstates^2 * 2  # number of state elements describing quantum bits
    ∫a = x[nstates_quantum + 1]
    a = x[nstates_quantum + 2]
    da = x[nstates_quantum + 3]
    dda = u[1]

    # Form the Hamiltonian
    H_drift = getdrifthamiltonian(model)
    H_drive = getdrivehamiltonian(model)
    H = -im * (H_drift + H_drive * a)
    Hiso = complex2real(H)
    for i = 1:nqstates
        ir = (i-1)*2nqstates
        iψ =  ir .+ (1:8)
        ψi = getqstate(model, x, i)
        J[iψ, iψ] .= Hiso
        J[iψ, nstates_quantum + 2] = complex2real(-im * H_drive * ψi)
    end
    for i = 1:3
        J[nstates_quantum + i, nstates_quantum + 1 + i] = 1
    end
    return nothing
end

RD.@autodiff struct TwoQubitBase <: AbstractTwoQubit 
    ω0::Float64
    ω1::Float64
end
TwoQubitBase() = TwoQubitBase(1.0, 1.0)
RobotDynamics.control_dim(::TwoQubitBase) = 1
RobotDynamics.state_dim(::TwoQubitBase) = 8 * 4

function RobotDynamics.dynamics!(model::TwoQubitBase, xdot, x, u)
    ω0, ω1 = model.ω0, model.ω1
    ħ = 1.0  # Plancks Constant

    # Extract controls
    nqstates = 4
    nstates_quantum = nqstates^2 * 2  # number of state elements describing quantum bits
    a = u[1]

    # Form the Hamiltonian
    H_drift = getdrifthamiltonian(model)
    H_drive = getdrivehamiltonian(model)
    H = (H_drift + H_drive * a) * (-im)

    # Compute the dynamics
    nqstates = 4  # number of quantum states
    for i = 1:nqstates
        ψi = getqstate(model, x, i)
        ψdot = H*ψi
        setqstate!(model, xdot, ψdot, i)
    end
    return nothing
end

function RobotDynamics.jacobian!(model::TwoQubitBase, J, xdot, x, u)
    J .= 0
    ω0, ω1 = model.ω0, model.ω1

    # Extract controls
    nqstates = 4
    nstates_quantum = nqstates^2 * 2  # number of state elements describing quantum bits
    a = u[1]

    # Form the Hamiltonian
    H_drift = getdrifthamiltonian(model)
    H_drive = getdrivehamiltonian(model)
    H = -im * (H_drift + H_drive * a)
    Hiso = complex2real(H)
    for i = 1:nqstates
        ir = (i-1)*2nqstates
        iψ =  ir .+ (1:8)
        ψi = getqstate(model, x, i)
        J[iψ, iψ] .= Hiso
        J[iψ, nstates_quantum + 1] = complex2real(-im * H_drive * ψi)
    end
    return nothing
end

RD.@autodiff struct DiscreteTwoQubit <: RobotDynamics.DiscreteDynamics
    continuous_model::TwoQubit
end
DiscreteTwoQubit(args...) = DiscreteTwoQubit(TwoQubit(args...))
@inline RD.state_dim(model::DiscreteTwoQubit) = RD.state_dim(model.continuous_model)
@inline RD.control_dim(model::DiscreteTwoQubit) = RD.control_dim(model.continuous_model)

function RD.discrete_dynamics!(model::DiscreteTwoQubit, xn, x, u, t, dt)
    cmodel = model.continuous_model

    # Extract controls
    nqstates = 4
    nstates_quantum = nqstates^2 * 2  # number of state elements describing quantum bits
    ∫a = x[nstates_quantum + 1]
    a = x[nstates_quantum + 2]
    da = x[nstates_quantum + 3]
    dda = u[1]

    Hdrift = getdrifthamiltonian(cmodel)
    Hdrive = getdrivehamiltonian(cmodel)
    H = (Hdrift + Hdrive * a) * (-im)
    Hiso = complex2real(H)
    Hprop = real2complex(exp(Hiso * dt))
    # display(complex2real(Hprop))

    # Calculate the dynamics
    # ψ1 = getqstate(cmodel, x, 1)
    # x1 = complex2real(ψ1)
    # display(complex2real(Hprop) * x1)
    for i = 1:nqstates
        ψi = getqstate(cmodel, x, i)
        ψn = Hprop * ψi 
        # @show ψn
        setqstate!(cmodel, xn, ψn, i)
    end
    # Use Euler for controls
    xn[nstates_quantum + 1] = ∫a + a * dt
    xn[nstates_quantum + 2] = a  + da * dt
    xn[nstates_quantum + 3] = da + dda * dt
    return nothing
end



struct ControlIntegralDerivative{L <: RobotDynamics.ContinuousDynamics} <: RobotDynamics.ContinuousDynamics
    model::L 
end

RD.state_dim(model::ControlIntegralDerivative) = 
    RD.state_dim(model.model) + 3 * RD.control_dim(model.model)

RD.control_dim(model::ControlIntegralDerivative) = RD.control_dim(model.model)

function RD.dynamics!(model::ControlIntegralDerivative, xdot, x, u)
    n, m = RD.dims(model.model)
    xdot0 = view(xdot, 1:n)
    x0 = view(x, 1:n)
    u0 = view(x, (n+m) .+ (1:m))
    RD.dynamics!(model.model, xdot0, x0, u0)
    for i = 1:m
        ∫a = x[n + 0m + i] 
        a = x[n + 1m + i]
        da = x[n + 2m + i]
        dda = u[i]

        xdot[n + 0m + i] = a
        xdot[n + 1m + i] = da
        xdot[n + 2m + i] = dda
    end
    return nothing 
end

function RD.jacobian!(model::ControlIntegralDerivative, J, xdot, x, u)
    n, m = RD.dims(model.model)
    J0 = view(J, 1:n, 1:n + m)
    xdot0 = view(xdot, 1:n)
    x0 = view(x, 1:n)
    u0 = view(x, (n + m) .+ (1:m))
    RD.jacobian!(model.model, J0, xdot0, x0, u0)

    # Move controls to next set of columns
    B0 = view(J, 1:n, n .+ (1:m)) 
    J[1:n, (n + m) .+ (1:m)] .= B0 
    B0 .= 0

    # Set identity matrix
    for i = 1:3m
        J[n + i, n + m + i] = 1
    end
    return nothing
end

struct ControlDerivative{L<:RD.DiscreteDynamics} <: RD.DiscreteDynamics
    model::L
    J0::Matrix{Float64}
    z0::Vector{Float64}
    function ControlDerivative(model::RD.DiscreteDynamics)
        n,m = RD.dims(model)
        J0 = zeros(n,n+m)
        z0 = zeros(n+m)
        new{typeof(model)}(model, J0, z0)
    end
end

RD.state_dim(model::ControlDerivative) = RD.state_dim(model.model) + 2 * RD.control_dim(model.model)
RD.control_dim(model::ControlDerivative) = RD.control_dim(model.model)

Base.copy(model::ControlDerivative) = ControlDerivative(model.model)
RD.default_signature(::ControlDerivative) = RD.InPlace()

function RD.discrete_dynamics!(model::ControlDerivative, xn, x, u, t, dt)
    n0 = RD.state_dim(model.model)
    m = RD.control_dim(model.model)
    xn0 = view(xn, 1:n0)
    for i = 1:n0
        model.z0[i] = x[i]
    end
    for i = 1:m
        model.z0[n0+i] = u[i]
    end

    # Call original dynamics 
    z0 = RD.StaticKnotPoint{Any,Any}(n0, m, model.z0, t, dt)
    RD.discrete_dynamics!(model.model, xn0, z0)

    # Add in extra control information
    for i = 1:m
        xn[n0 + i] = u[i]                   # store previous control
        uprev = x[n0 + i]
        xn[n0 + m + i] = (u[i] - uprev) / dt   # calculate control derivative
    end
    return nothing
end

function RD.jacobian!(sig::RD.FunctionSignature, diffmethod::RD.DiffMethod, 
                      model::ControlDerivative, J, y, z)
    n = length(y) 
    n0 = RD.state_dim(model.model)
    m = RD.control_dim(model.model)
    x = RD.state(z)
    u = RD.control(z)
    for i = 1:n0
        model.z0[i] = x[i]
    end
    for i = 1:m
        model.z0[n0+i] = u[i]
    end

    # Call original Jacobian and copy to new Jacobian
    z0 = RD.StaticKnotPoint{Any,Any}(n0, m, model.z0, z.t, z.dt)
    y0 = view(y, 1:n0)
    J0 = model.J0
    RD.jacobian!(sig, diffmethod, model.model, J0, y0, z0)
    J[1:n0,1:n0] .= view(J0, :, 1:n0)
    J[1:n0,n+1:n+m] .= view(J0, :, n0+1:n0+m)

    # Fill out new Jacobian
    dt = z.dt
    for i = 1:m
        J[n0 + i, n+i] = 1.0
        J[n0 + m + i, n + i] = 1/dt
        J[n0 + m + i, n0 + i] = - 1/dt
    end
    return nothing
end