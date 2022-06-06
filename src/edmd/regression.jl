function rls_qr(b::AbstractVector{<:AbstractFloat}, A::SparseMatrixCSC{Float64, Int64};
    batchsize=Int(floor(size(A)[1]/10)), Q=0.0, verbose=false)

    m, n = size(A)

    rem_m = m - 3*n
    
    if batchsize == 0 || batchsize >= rem_m
        m_batch = 1
        batch_size_1 = Int(3*n)
        batches = m
    else
        m_batch = batchsize 
        batches = Int(floor(rem_m/m_batch))
        batch_size_1 = Int(3*n) + Int(mod(rem_m, m_batch))
    end

    verbose && println("# Batches: $batches")
    verbose && println("")
    verbose && println("Batch 1")
    
    # QR factorization on 1st batch
    
    A_i = A[1:batch_size_1, :]
    b_i = b[1:batch_size_1]
    
    if Q == 0.0
        U, _, Pcol = qrr(A_i)
    else
        U, _, Pcol = qrr([A_i; sqrt(Q)*I])
    end

    rhs = A_i'*b_i

    # now at each batch we are going to
    for i = 0:batches-1

        verbose && print("\u1b[1F")
        verbose && println("Batch $(i+2)")
        verbose && print("\u1b[0K")
        
        # determine next batch
        batch_start = batch_size_1 + i*m_batch + 1
        batch_end = batch_start+m_batch-1

        A_i = A[batch_start:batch_end, :]
        b_i = b[batch_start:batch_end]

        # update our cholesky factor U with the new Ai
        U, _, Pcol = qrr([U*Pcol'; A_i])
        
        # add to the right hand side
        rhs += A_i'*b_i

    end

    U = U*Pcol'

    verbose && println("Solving LLS")
    verbose && @show norm(U'*U - A'*A)
    verbose && @show norm(rhs - A'*b)
    verbose && println("")

    x_rls = U\(U'\rhs)

    return x_rls

end

function rls_qr(b::AbstractVector{<:AbstractFloat}, A::AbstractMatrix{<:AbstractFloat};
    batchsize=Int(floor(size(A)[1]/10)), Q=0.0, verbose=false)

    m, _ = size(A)
    
    if batchsize == 0 || batchsize >= m
        m_batch = 1
        batch_size_1 = 1
        batches = m
    else
        m_batch = batchsize 
        batches = Int(floor(m/m_batch))
        batch_size_1 = m_batch + Int(mod(m, m_batch))
    end

    verbose && println("# Batches: $batches")
    verbose && println("")
    verbose && println("Batch 1")
    
    # QR factorization on 1st batch
    
    A_i = A[1:batch_size_1, :]
    b_i = b[1:batch_size_1]
    
    if Q == 0.0
        U = qr(A_i).R
    else
        U = qr([A_i; sqrt(Q)*I]).R
    end
    rhs = A_i'*b_i

    # now at each batch we are going to
    for i = 0:batches-2

        verbose && print("\u1b[1F")
        verbose && println("Batch $(i+2)")
        verbose && print("\u1b[0K")
        
        # determine next batch
        batch_start = batch_size_1 + i*m_batch + 1
        A_i = A[batch_start:batch_start+m_batch-1, :]
        b_i = b[batch_start:batch_start+m_batch-1]

        # update our cholesky factor U with the new Ai
        U = qr([U; A_i]).R
        
        # add to the right hand side
        rhs += A_i'*b_i

    end

    verbose && println("Solving LLS")
    verbose && @show norm(U'*U - A'*A)
    verbose && @show norm(rhs - A'*b)
    verbose && println("")

    x_rls = U\(U'\rhs)

    return x_rls

end

function rls_chol(Y::AbstractVector{<:AbstractFloat}, 
    X::AbstractMatrix{<:AbstractFloat}; verbose=false)

    m, n = size(X)

    verbose && println("# Iterations: $m")

    P = X[1, :]*X[1, :]'
    q = Y[1]*X[1, :]
    
    P_chol = cholesky(P, check = false)

    factorizable = issuccess(P_chol)
    ~factorizable || error("Cholesky Factorization Failed")

    verbose && println("")
    verbose && println("Iteration 1")

    for i in 2:m

        verbose && print("\u1b[1F")
        verbose && println("Iteration $i")
        verbose && print("\u1b[0K")

        lowrankupdate!(P_chol, X[i, :])
        q += Y[i]*X[i, :]

    end

    verbose && println("Solving LLS")
    verbose && println("")
    
    x = P_chol \ q

    return x

end

function qrr(A::AbstractMatrix{<:AbstractFloat})

    m, n = size(A)

    F = qr(A)

    R = F.R
    prow_vec = F.prow
    pcol_vec = F.pcol

    # build permutation matrices
    prow = sparse(I, m, m)[prow_vec, :]
    pcol = sparse(I, n, n)[:, pcol_vec]

    return R, prow, pcol

end

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

    num_X = size(X_mat, 1)
    num_Z = size(Z_mat,1)
    num_U = mod(size(Zu_mat,1), num_Z)

    dynamics_jacobians = fitA(Zu_mat, Zn_mat; rho=edmd_weights[1], kwargs...)

    if issubset(X_mat[:, 1], Z_mat[:, 1])
        g = spzeros(num_X, num_Z)
        g[:, 2:1+num_X] .= I(num_X)
    else
        g = Matrix(fitA(Z_mat, X_mat; rho=mapping_weights[1], kwargs...))
    end

    A = dynamics_jacobians[:, 1:size(dynamics_jacobians, 1)]
    B = dynamics_jacobians[:, (size(dynamics_jacobians, 1)+1):(size(dynamics_jacobians, 1)+num_U)]
    C = dynamics_jacobians[:, (size(dynamics_jacobians, 1)+num_U+1):end]
    
    C_list = Matrix{Float64}[]
    
    for i in 1:num_U
        C_i = C[:, (i-1)*num_Z+1:i*num_Z]
        push!(C_list, C_i)
    end

    return A, B, C_list, g
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
        ApplyArray(kron, Z', sparse(I,n,n)),
        ApplyArray(kron, Ahat', G),
        ApplyArray(kron, Bhat', G),
    ) 
    s = vcat(vec(Xn), vec(Amat), vec(Bmat))
    W,s
end

function fiterror(A,B,C,g,kf, X,U)
    P = size(X,2)
    norm(map(CartesianIndices(U)) do cind 
        k = cind[1]
        j = cind[2]
        x = X[k,j]
        u = U[k,j]
        y = kf(x)
        xn = X[k+1,j]
        # yn = A*y + u[1]*C*y
        yn = A*y + B*u + sum(C[i]*y .* u[i] for i = 1:length(u))
        norm(g*yn - xn)
    end) / P
end

function fiterror(A,B,C,g,kf, X,U)
    P = size(X,2)
    norm(map(CartesianIndices(U)) do cind 
        k = cind[1]
        j = cind[2]
        x = X[k,j]
        u = U[k,j]
        y = kf(x)
        xn = X[k+1,j]
        # yn = A*y + B*u + u[1]*C*y
        yn = A*y + B*u + sum(C[i]*y .* u[i] for i = 1:length(u))
        norm(g*yn - xn)
    end) / P
end

function fiterror(A,B,C,g,kf, X,U,Xn)
    P = size(X,2)
    norm(map(CartesianIndices(U)) do cind 
        k = cind[1]
        j = cind[2]
        x = X[k,j]
        u = U[k,j]
        y = kf(x)
        xn = Xn[k,j]
        yn = A*y + B*u + u[1]*C*y
        norm(g*yn - xn)
    end) / P
end

function fiterror(model::RD.DiscreteDynamics, dt, X, U)
    P = size(X,2)
    norm(map(CartesianIndices(U)) do cind 
        k = cind[1]
        j = cind[2]
        x = X[k,j]
        u = U[k,j]
        xn = X[k+1,j]
        xn_ = RD.discrete_dynamics(model, x, u, (k-1)*dt, dt)
        norm(xn - xn_)
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