struct BilinearCartpole <: RD.DiscreteDynamics
    A::Matrix{Float64}
    C::Matrix{Float64}
    g::Matrix{Float64}  # mapping from extended to original states
    dt::Float64
    kf::Function
    function BilinearCartpole()
        datadir = joinpath(@__DIR__, "..", "..", "data")
        data = load(joinpath(datadir, "cartpole_eDMD_data.jld2"))
        A = Matrix(data["F"])
        C = Matrix(data["C"])
        g = Matrix(data["g"])
        T_ref = data["T_ref"]
        dt = T_ref[2] - T_ref[1]
        eigfuns = data["eigfuns"]
        eigorders = data["eigorders"]
        kf(x) = BilinearControl.EDMD.koopman_transform(Vector(x), eigfuns, eigorders)
        new(A, C, g, dt, kf)
    end
end

Base.copy(model::BilinearCartpole) = BilinearCartpole()

RD.output_dim(model::BilinearCartpole) = size(model.A,1)
RD.state_dim(model::BilinearCartpole) = size(model.A,2)
RD.control_dim(model::BilinearCartpole) = 1
RD.default_diffmethod(::BilinearCartpole) = RD.UserDefined()
RD.default_signature(::BilinearCartpole) = RD.InPlace()

function RD.discrete_dynamics(model::BilinearCartpole, x, u, t, h)
    @assert h ≈ model.dt "Timestep must be $(model.dt)."
    return model.A*x + model.C*x * u[1]
end

function RD.discrete_dynamics!(model::BilinearCartpole, xn, x, u, t, h)
    @assert h ≈ model.dt "Timestep must be $(model.dt)."
    mul!(xn, model.A, x)
    mul!(xn, model.C, x, u[1], true)
    nothing
end

function RD.jacobian!(model::BilinearCartpole, J, xn, x, u, t, h)
    @assert h ≈ model.dt "Timestep must be $(model.dt)."
    n,m = RD.dims(model)
    J[:,1:n] .= model.A .+ model.C .* u[1]
    Ju = view(J, :, n+1)
    mul!(Ju, model.C, x)
    nothing
end

expandstate(model::BilinearCartpole, x) = model.kf(x)
