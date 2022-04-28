function linear_regression(Y::Vector{Float64}, X::Matrix{Float64}; gamma::Float64=0.0, 
                           lambda::Float64=0.0)
    
    (T, K) = (size(X, 1), size(X, 2))

    Q = X'X / T
    c = X'Y / T                   #c'b = Y'X*b

    b = Variable(K)               #define variables to optimize over
    L1 = quadform(b, Q)           #b'Q*b
    L2 = dot(c, b)                #c'b
    L3 = norm(b, 1)               #sum(|b|)
    L4 = sumsquares(b)            #sum(b^2)

    if gamma==0 && lambda==0
        return X \ Y
    end

    if lambda > 0
        # perform elastic net or ridge
        Sol = minimize(L1 - 2 * L2 + gamma * L3 + lambda * L4)
    else
        # perform lasso
        Sol = minimize(L1 - 2 * L2 + gamma * L3)
    end

    solve!(Sol, COSMO.Optimizer; silent_solver = true)
    Sol.status == Convex.MOI.OPTIMAL ? b = vec(evaluate(b)) : b = X \ Y
    
    return b

end

function LLS_fit(x::Matrix{Float64}, b::Matrix{Float64}, regression_type::String, 
                 weights::Vector{Float64})

    #= this function performs LLS to best fit A for system
    Ax = b where x and b are also matrices
    =#

    if regression_type == "lasso"
        gamma = weights[1]
        lambda = 0.0
    elseif regression_type == "ridge"
        gamma = 0.0
        lambda = weights[1]
    elseif regression_type == "elastic net"
        gamma = weights[1]
        lambda = weights[2]
    else
        gamma = 0.0
        lambda = 0.0
    end

    vec_b_T = Vector{Float64}(vec(b'))
    x_T_mat = Matrix(kron(I(size(b')[2]), x'))

    A_T_vec = linear_regression(vec_b_T, x_T_mat; gamma, lambda)
    
    A = reshape(A_T_vec, size(x')[2], size(b')[2])'

    return A

end

function learn_bilinear_model(X::VecOrMat{<:AbstractVector}, Z::VecOrMat{<:AbstractVector}, 
                              Zu::VecOrMat{<:AbstractVector},
                              regression_types::Vector{String}; 
                              edmd_weights::Vector{Float64}=[0.0, 0.0], 
                              mapping_weights::Vector{Float64}=[0.0, 0.0])

    X_mat = reduce(hcat, X[1:end-1,:])
    Z_mat = reduce(hcat, Z[1:end-1,:])
    Zu_mat = reduce(hcat, Zu)
    Zn_mat = reduce(hcat, Z[2:end,:])
    # X_mat = Matrix(mapreduce(permutedims, vcat, X)')
    # Z_mat = Matrix(mapreduce(permutedims, vcat, Z)')
    # Zu = Matrix(mapreduce(permutedims, vcat, Zu)')

    # extract data matrices
    # X = X_mat[:, 1:end-1]
    # Z = Z_mat[:, 1:end-1]
    # Z_prime = Z_mat[:, 2:end]
        
    dynamics_jacobians = LLS_fit(Zu_mat, Zn_mat, regression_types[1], edmd_weights)
    g = LLS_fit(Z_mat, X_mat, regression_types[2], mapping_weights)

    A = dynamics_jacobians[:, 1:size(dynamics_jacobians)[1]]
    C = dynamics_jacobians[:, (size(dynamics_jacobians)[1] + 1):end]

    return A, C, Matrix(g)

end