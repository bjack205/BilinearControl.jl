using OSQP

## Define some useful controllers
## Simulation functions
function simulatewithcontroller(model::RD.DiscreteDynamics, ctrl::AbstractController, x0, 
                                tf, dt)
    simulatewithcontroller(RD.default_signature(model), model, ctrl, x0, tf, dt)
end

function simulatewithcontroller(sig::RD.FunctionSignature, 
                                model::RD.DiscreteDynamics, ctrl::AbstractController, x0, 
                                tf, dt)
    times = range(0, tf, step=dt)
    m = RD.control_dim(model)
    N = length(times)
    X = [copy(x0) for k = 1:N]
    U = [zeros(m) for k = 1:N-1]
    for k = 1:N-1 
        t = times[k]
        dt = times[k+1] - times[k]
        u = getcontrol(ctrl, X[k], t)
        U[k] = u
        RD.discrete_dynamics!(sig, model, X[k+1], X[k], u, times[k], dt)
    end
    X,U
end

function bilinearerror(model::EDMDModel, X, U)
    dt = model.dt
    map(CartesianIndices(U)) do cind
        k = cind[1]
        i = cind[2]

        uk = U[k,i]
        zk = expandstate(model, X[k,i]) 
        zn = zero(zk)
        t = (k-1)*dt
        RD.discrete_dynamics!(model, zn, zk, uk, t, dt)
        xn = originalstate(model, zn) 
        xn - X[k+1]
    end
end


function simulate(model::RD.DiscreteDynamics, U, x0, tf, dt)
    times = range(0, tf, step=dt)
    N = length(times)
    @assert length(U) in [N,N-1]
    X = [copy(x0) for k = 1:N]
    sig = RD.default_signature(model)
    for k = 1:N-1 
        dt = times[k+1] - times[k]
        RD.discrete_dynamics!(sig, model, X[k+1], X[k], U[k], times[k], dt)
    end
    X
end

function compare_models(sig::RD.FunctionSignature, model::EDMDModel,
                        model0::RD.DiscreteDynamics, x0, tf, U; 
                        doplot=false, inds=1:RD.state_dim(model0))
    N = length(U) + 1
    times = range(0, tf, length=N)
    dt = times[2]
    @show dt
    m = RD.control_dim(model)
    @assert m == RD.control_dim(model0)
    z0 = expandstate(model, x0) 
    Z = [copy(z0) for k = 1:N]
    X = [copy(x0) for k = 1:N]
    for k = 1:N-1
        RD.discrete_dynamics!(sig, model, Z[k+1], Z[k], U[k], times[k], dt)
        RD.discrete_dynamics!(sig, model0, X[k+1], X[k], U[k], times[k], dt)
    end
    X_bl = map(z->model.g * z, Z)
    X_bl, X
    if doplot
        X_mat = reduce(hcat, X_bl)
        X0_mat = reduce(hcat, X)
        p = plot(times, X0_mat[inds,:]', label="original", c=[1 2])
        plot!(p, times, X_mat[inds,:]', label="bilinear", c=[1 2], s=:dash)
        display(p)
    end
    sse = norm(X_bl - X)^2
    println("SSE: $sse")
    X_bl, X
end

function create_data(model::RD.DiscreteDynamics, ctrl::AbstractController, 
                              initial_conditions, tf, dt; sig=RD.InPlace())
    num_traj = length(initial_conditions)
    N = round(Int, tf/dt) + 1
    X_sim = Matrix{Vector{Float64}}(undef, N, num_traj)
    U_sim = Matrix{Vector{Float64}}(undef, N-1, num_traj)
    for i = 1:num_traj
        resetcontroller!(ctrl, initial_conditions[i])
        X,U = simulatewithcontroller(sig, model, ctrl, initial_conditions[i], tf, dt)
        X_sim[:,i] = X
        U_sim[:,i] = U
    end
    X_sim, U_sim
end

function calc_error(model::RD.DiscreteDynamics, X, U, dt)
    map(CartesianIndices(U)) do cind
        k = cind[1]  # time index
        j = cind[2]  # trajectory index
        xn_true = X[k+1,j]
        xn_nominal = RD.discrete_dynamics(model, X[k,j], U[k,j], (k-1)*dt, dt)
        Vector(xn_true - xn_nominal)
    end
end