
struct EDMDModel <: RD.DiscreteDynamics
    A::Matrix{Float64}
    C::Matrix{Float64}
    g::Matrix{Float64}  # mapping from extended to original states
    dt::Float64
    kf::Function
    datafile::String
    name::String
    function EDMDModel(datafile::String; name=splitext(datafile)[1])
        data = load(datafile)
        A = Matrix(data["F"])
        C = Matrix(data["C"])
        g = Matrix(data["g"])
        T_ref = data["T_ref"]
        dt = T_ref[2] - T_ref[1]
        eigfuns = data["eigfuns"]
        eigorders = data["eigorders"]
        kf(x) = BilinearControl.EDMD.koopman_transform(Vector(x), eigfuns, eigorders)
        new(A, C, g, dt, kf, datafile)
    end
end

Base.copy(model::EDMDModel) = EDMDModel(model.datafile)

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