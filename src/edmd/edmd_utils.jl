using OSQP

function create_data(model::RD.DiscreteDynamics, ctrl::AbstractController, 
                     x0_sampler, num_traj, xe, tf, dt; 
                     sig=RD.InPlace(), thresh=0.1, max_samples=3*num_traj
    )
    N = round(Int, tf/dt) + 1
    X_sim = Matrix{Vector{Float64}}(undef, N, num_traj)
    U_sim = Matrix{Vector{Float64}}(undef, N-1, num_traj)
    j = 0
    for i = 1:max_samples
        x0 = rand(x0_sampler)
        resetcontroller!(ctrl, x0)
        X,U = simulatewithcontroller(sig, model, ctrl, x0, tf, dt)
        did_stabilize = norm(X[end] - xe) < thresh
        if did_stabilize
            j += 1
            X_sim[:,j] = X
            U_sim[:,j] = U
        end
        if j == num_traj
            break
        end
    end
    X_sim[:,1:j], U_sim[:,1:j]
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