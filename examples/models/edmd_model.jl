
struct EDMDModel <: RD.DiscreteDynamics
    A::Matrix{Float64}
    B::Matrix{Float64}
    C::Vector{Matrix{Float64}}
    g::Matrix{Float64}  # mapping from extended to original states
    kf::Function
    dt::Float64
    name::String
    function EDMDModel(A::AbstractMatrix, B::AbstractMatrix, C::Vector{<:AbstractMatrix}, 
                       g::AbstractMatrix, kf::Function, dt::AbstractFloat, 
                       name::AbstractString)

        p,n = size(A)
        m = size(B,2)
        m == length(C) || throw(DimensionMismatch("Length of C must be m. Expected $m, got $(length(C))"))
        p == size(B,1) || throw(DimensionMismatch("B should have the same number of rows as A."))
        all(c->size(c) == (p,n), C) || throw(DimensionMismatch("All C matrices should be the same size as A."))
        new(A,B,C,g,kf,dt,name)
    end
end

function EDMDModel(A::AbstractMatrix, B::AbstractMatrix, C::Vector{<:AbstractMatrix}, g::AbstractMatrix, 
    kf::Function, dt::AbstractFloat, name::AbstractString)
n = size(A,2)
m = size(C)
EDMDModel(A,B,C,g,kf,dt,name)
end

function EDMDModel(A::AbstractMatrix, C::Vector{<:AbstractMatrix}, g::AbstractMatrix, 
                    kf::Function, dt::AbstractFloat, name::AbstractString)
    n = size(A,2)
    m = size(C,1)
    B = zeros(n,m)
    EDMDModel(A,B,C,g,kf,dt,name)
end

function EDMDModel(datafile::String; name=splitext(datafile)[1])
    data = load(datafile)
    A = Matrix(data["A"])
    B = Matrix(data["B"])
    C = Matrix(data["C"])
    g = Matrix(data["g"])
    dt = data["dt"]
    eigfuns = data["eigfuns"]
    eigorders = data["eigorders"]
    kf(x) = BilinearControl.EDMD.koopman_transform(Vector(x), eigfuns, eigorders)
    EDMDModel(A, B, C, g, kf, dt, name)
end

Base.copy(model::EDMDModel) = EDMDModel(copy(model.A), copy(model.B), copy(model.C), copy(model.g), 
                                        model.kf, model.dt, model.name)

RD.output_dim(model::EDMDModel) = size(model.A,1)
RD.state_dim(model::EDMDModel) = size(model.A,2)
RD.control_dim(model::EDMDModel) = size(model.B,2)
RD.default_diffmethod(::EDMDModel) = RD.UserDefined()
RD.default_signature(::EDMDModel) = RD.InPlace()

function RD.discrete_dynamics(model::EDMDModel, x, u, t, h)
    @assert h ≈ model.dt "Timestep must be $(model.dt)."
    return model.A*x .+ model.B*u .+ sum(model.C[i]*x .* u[i] for i = 1:length(u))
end

function RD.discrete_dynamics!(model::EDMDModel, xn, x, u, t, h)
    @assert h ≈ model.dt "Timestep must be $(model.dt)."
    mul!(xn, model.A, x)
    mul!(xn, model.B, u, true, true)
    for i = 1:length(u)
        mul!(xn, model.C[i], x, u[i], true)
    end
    nothing
end

function RD.jacobian!(model::EDMDModel, J, xn, x, u, t, h)
    @assert h ≈ model.dt "Timestep must be $(model.dt)."
    n,m = RD.dims(model)
    J[:,1:n] .= model.A
    J[:,n .+ (1:m)] .= model.B
    for i = 1:m
       J[:,1:n] .+= model.C[i] .* u[i]
       Ju = view(J, :, n+i)
       mul!(Ju, model.C[i], x, true, true)
    end
    nothing
end

expandstate(model::EDMDModel, x) = model.kf(x)
originalstate(model::EDMDModel, z) = model.g*z
originalstatedim(model::EDMDModel) = size(model.g, 1)

originalstate(::RD.DiscreteDynamics, x) = x
originalstatedim(model::RD.DiscreteDynamics) =  RD.state_dim(model)
expandstate(::RD.DiscreteDynamics, x) = x

## Saved models
const DATADIR = joinpath(dirname(pathof(BilinearControl)), "..", "data")
BilinearPendulum() = EDMDModel(joinpath(DATADIR, "pendulum_eDMD_data.jld2"), name="pendulum")
BilinearCartpole() = EDMDModel(joinpath(DATADIR, "cartpole_eDMD_data.jld2"), name="cartpole")

struct EDMDErrorModel{L} <: RD.DiscreteDynamics
    nominal::L
    err::EDMDModel
end
RD.state_dim(model::EDMDErrorModel) = RD.state_dim(model.nominal)
RD.control_dim(model::EDMDErrorModel) = RD.control_dim(model.nominal)
RD.default_diffmethod(model::EDMDErrorModel) = RD.default_diffmethod(model.nominal)
RD.default_signature(model::EDMDErrorModel) = RD.default_signature(model.nominal)

function RD.discrete_dynamics(model::EDMDErrorModel, x, u, t, dt)
    y = expandstate(model.err, x)
    xn = RD.discrete_dynamics(model.nominal, x, u, t, dt)
    xn += originalstate(model.err, RD.discrete_dynamics(model.err, y, u, t, dt))
    xn
end

function RD.discrete_dynamics!(model::EDMDErrorModel, xn, x, u, t, dt)
    y = expandstate(model.err, x)
    yn = zero(y)
    RD.discrete_dynamics(model.nominal, xn, x, u, t, dt)
    RD.discrete_dynamics!(model.err, xn, y, u, t, dt)
    xn_err = originalstate(model.err, yn) 
    xn .+= xn_err
    nothing
end