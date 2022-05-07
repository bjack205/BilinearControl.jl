
function linear_regression(Y::AbstractVector{<:AbstractFloat}, 
                           X::AbstractMatrix{<:AbstractFloat}; 
                           gamma::Float64=0.0, lambda::Float64=0.0,
                           algorithm=:qr)
    
    (T, K) = (size(X, 1), size(X, 2))
    @show T,K
    @show issparse(X)
    @show lambda

    λ = lambda
    if algorithm == :qdldl
        P = 2*(X'X + 2λ * I) / T
        gamma == zero(gamma) || error("Cannot use QDLDL with L1 regularization.")
        F = QDLDL.qdldl(P)
        b = QDLDL.solve(F, Y)
        return b
    elseif algorithm == :cholesky
        P = Symmetric(Matrix(X'X + 2λ * I))
        @show cond(P)
        F = cholesky!(P)
        b = F \ (X'Y)
        return b
    elseif algorithm == :qr
        F = qr([X; sqrt(λ)*I])
        b = F \ [Y; zeros(K)]
        return b
    elseif algorithm == :convex
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
    else
        error("Algorithm $algorithm not recognized.")
    end

end

function LLS_fit(x::Matrix{Float64}, b::Matrix{Float64}, regression_type::String, 
                 weights::Vector{Float64}; kwargs...)

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
    n = size(b, 1)
    x_T_mat = kron(sparse(I,n,n), sparse(x'))

    A_T_vec = linear_regression(vec_b_T, x_T_mat; gamma, lambda, kwargs...)
    
    A = reshape(A_T_vec, size(x')[2], size(b')[2])'

    return A

end

function learn_bilinear_model(X::VecOrMat{<:AbstractVector}, Z::VecOrMat{<:AbstractVector}, 
                              Zu::VecOrMat{<:AbstractVector},
                              regression_types::Vector{String}; 
                              edmd_weights::Vector{Float64}=[0.0, 0.0], 
                              mapping_weights::Vector{Float64}=[0.0, 0.0],
                              kwargs...)

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
        
    # dynamics_jacobians = LLS_fit(Zu_mat, Zn_mat, regression_types[1], edmd_weights; kwargs...)
    # g = LLS_fit(Z_mat, X_mat, regression_types[2], mapping_weights; kwargs...)
    dynamics_jacobians = fitA(Zu_mat, Zn_mat; rho=edmd_weights[1], kwargs...)
    g = fitA(Z_mat, X_mat; rho=mapping_weights[1], kwargs...)

    A = dynamics_jacobians[:, 1:size(dynamics_jacobians)[1]]
    C = dynamics_jacobians[:, (size(dynamics_jacobians)[1] + 1):end]

    return A, C, Matrix(g)

end

function build_edmd_data(X,U, A,B,F,G; verbose=true)
    if size(A) != size(B) != size(F)
        throw(DimensionMismatch("A,B, and F must all have the same dimension."))
    end
    n0 = size(A[1],1)
    n = length(X[1])
    m = length(U[1])
    p = n+m + n*m     # number of features
    N,T = size(X)
    P = (N-1)*T
    verbose && println("Concatentating data")
    Uj = reduce(hcat, U)
    Xj = reduce(hcat, X[1:end-1,:])
    Xn = reduce(hcat, X[2:end,:])
    Amat = reduce(hcat, A)
    Bmat = reduce(hcat, B)
    @assert size(Xn,2) == size(Xj,2) == P
    verbose && println("Creating feature matrix")
    Z = mapreduce(hcat, 1:P) do j
        x = Xj[:,j]
        u = Uj[:,j]
        [x; u; vec(x*u')]
    end
    @assert size(Z) == (p,P)
    verbose && println("Creating state Jacobian matrix")
    Ahat = mapreduce(hcat, 1:P) do j
        u = U[j] 
        In = sparse(I,n,n)
        vcat(sparse(I,n,n), spzeros(m,n), reduce(vcat, In*ui for ui in u)) * F[j] 
    end
    @assert size(Ahat) == (p,P*n0)
    verbose && println("Creating control Jacobian matrix")
    Bhat = mapreduce(hcat, 1:P) do j
        xB = spzeros(n,m)
        xB[:,1] .= X[j]
        vcat(spzeros(n,m), sparse(I,m,m), reduce(vcat, circshift(xB, (0,i)) for i = 1:m))
    end
    @assert size(Bhat) == (p,P*m)

    verbose && println("Creating least-squares data")
    W = ApplyArray(vcat,
        ApplyArray(kron, Xj', sparse(I,n,n)),
        ApplyArray(kron, Ahat', G),
        ApplyArray(kron, Bhat', G),
    ) 
    s = vcat(vec(Xn), vec(Amat), vec(Bmat))
    W,s
end

function fiterror(A,C,g,kf, X,U)
    P = size(X,2)
    norm(map(CartesianIndices(U)) do cind 
        k = cind[1]
        j = cind[2]
        x = X[k,j]
        u = U[k,j]
        y = kf(x)
        xn = X[k+1,j]
        yn = A*y + u[1]*C*y
        norm(g*yn - xn)
    end) / P
end

"""
Finds `A` that minimizes 

```math 
|| Ax - b||_2^2 + \\rho ||A||_F
```
where `x` and `b` are vector or matrices and ``||A||_F`` is the Frobenius norm of ``A``.
"""
function fitA(x,b; rho=0.0, kwargs...)
    n,p = size(x)
    m = size(b,1)
    if size(b,2) != p
        throw(DimensionMismatch("x and b must have same number of columns."))
    end
    Â = kron(sparse(x'), sparse(I,m,m))
    b̂ = vec(b)
    x̂ = linear_regression(b̂, Â, lambda=rho; kwargs...)
    return reshape(x̂,m,n)
end