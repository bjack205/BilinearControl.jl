
struct EDMDModel <: RD.DiscreteDynamics
    A::Matrix{Float64}
    C::Matrix{Float64}
    g::Matrix{Float64}  # mapping from extended to original states
    kf::Function
    dt::Float64
    name::String
    function EDMDModel(A::AbstractMatrix, C::AbstractMatrix, g::AbstractMatrix, 
                       kf::Function, dt::AbstractFloat, name::AbstractString)
        new(A,C,g,kf,dt,name)
    end
end

function EDMDModel(datafile::String; name=splitext(datafile)[1])
    data = load(datafile)
    A = Matrix(data["A"])
    C = Matrix(data["C"])
    g = Matrix(data["g"])
    dt = data["dt"]
    eigfuns = data["eigfuns"]
    eigorders = data["eigorders"]
    kf(x) = BilinearControl.EDMD.koopman_transform(Vector(x), eigfuns, eigorders)
    EDMDModel(A, C, g, kf, dt, name)
end

Base.copy(model::EDMDModel) = EDMDModel(copy(model.A), copy(model.C), copy(model.g), 
                                        model.kf, model.dt, model.name)

RD.output_dim(model::EDMDModel) = size(model.A,1)
RD.state_dim(model::EDMDModel) = size(model.A,2)
RD.control_dim(model::EDMDModel) = 1
RD.default_diffmethod(::EDMDModel) = RD.UserDefined()
RD.default_signature(::EDMDModel) = RD.InPlace()

function RD.discrete_dynamics(model::EDMDModel, x, u, t, h)
    @assert h ≈ model.dt "Timestep must be $(model.dt)."
    return model.A*x .+ model.C*x .* u[1]
end

function RD.discrete_dynamics!(model::EDMDModel, xn, x, u, t, h)
    @assert h ≈ model.dt "Timestep must be $(model.dt)."
    mul!(xn, model.A, x)
    mul!(xn, model.C, x, u[1], true)
    nothing
end

function RD.jacobian!(model::EDMDModel, J, xn, x, u, t, h)
    @assert h ≈ model.dt "Timestep must be $(model.dt)."
    n,m = RD.dims(model)
    J[:,1:n] .= model.A .+ model.C .* u[1]
    Ju = view(J, :, n+1)
    mul!(Ju, model.C, x)
    nothing
end

expandstate(model::EDMDModel, x) = model.kf(x)

## Saved models
const DATADIR = joinpath(dirname(pathof(BilinearControl)), "..", "data")
BilinearPendulum() = EDMDModel(joinpath(DATADIR, "pendulum_eDMD_data.jld2"), name="pendulum")