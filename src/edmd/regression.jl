
function rls_qr(b::AbstractVector{<:AbstractFloat}, A::SparseMatrixCSC{Float64, Int64};
    batchsize=Int(floor(size(A)[1]/10)), Q=0.0, verbose=false, showprog=false)

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
    prog = Progress(batches, enabled=showprog)
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
        next!(prog)

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
                           algorithm=:qr, showprog=false)
    
    (T, K) = (size(X, 1), size(X, 2))
    # @show T,K
    # @show issparse(X)
    # @show lambda

    λ = lambda
    if algorithm == :qdldl
        P = 2*(X'X + 2λ * I) / T
        gamma == zero(gamma) || error("Cannot use QDLDL with L1 regularization.")
        F = QDLDL.qdldl(P)
        b = QDLDL.solve(F, Y)
        return b
    elseif algorithm == :cholesky
        P = Symmetric(Matrix(X'X + 2λ * I))
        # @show cond(P)
        F = cholesky!(P)
        b = F \ (X'Y)
        return b
    elseif algorithm == :qr
        F = qr([X; sqrt(λ)*I])
        b = F \ [Y; zeros(K)]
        return b
    elseif algorithm == :qr_rls
        b = rls_qr(Y, X; Q=λ, showprog)
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

function build_edmd_data(X,U, A,B,F,G; cinds_jac=CartesianIndices(U), α=0.5, verbose=true, learnB=true)
    if size(A) != size(B) != size(F)
        throw(DimensionMismatch("A,B, and F must all have the same dimension."))
    end
    n0 = size(A[1],1)
    n = length(X[1])
    m = length(U[1])
    mB = m * learnB
    p = n+mB + n*m     # number of features
    N,T = size(X)
    P = (N-1)*T        # number of dynamics samples
    Pj = length(A)     # number of Jacobian samples
    Uj = reduce(hcat, U)
    Xj = reduce(hcat, X[1:end-1,:])
    Xn = reduce(hcat, X[2:end,:])
    Amat = reduce(hcat, A)
    Bmat = reduce(hcat, B)
    @assert size(Xn,2) == size(Xj,2) == P
    Z = mapreduce(hcat, 1:P) do j
        x = Xj[:,j]
        u = Uj[:,j]
        if learnB
            [x; u; vec(x*u')]
        else
            [x; vec(x*u')]
        end
    end
    @assert size(Z) == (p,P)
    Ahat = mapreduce(hcat, CartesianIndices(cinds_jac)) do cind0
        cind = cinds_jac[cind0]
        u = U[cind] 
        In = sparse(I,n,n)
        vcat(sparse(I,n,n), spzeros(mB,n), reduce(vcat, In*ui for ui in u)) * F[cind0] 
    end
    @assert size(Ahat) == (p,Pj*n0)
    Bhat = mapreduce(hcat, cinds_jac) do cind 
        xB = spzeros(n,m)
        xB[:,1] .= X[cind]
        vcat(spzeros(n,m), sparse(I,mB,m), reduce(vcat, circshift(xB, (0,i)) for i = 1:m))
    end
    @assert size(Bhat) == (p,Pj*m)

    # W = ApplyArray(vcat,
    #     (1-α) * ApplyArray(kron, Z', sparse(I,n,n)),
    #     α * ApplyArray(kron, Ahat', G),
    #     α * ApplyArray(kron, Bhat', G),
    # ) 
    verbose && println("Forming sparse coefficient matrix...")
    W = vcat(
        (1-α) * kron(sparse(Z'), sparse(I,n,n)),
        α * kron(vcat(Ahat', Bhat'), G)
    )
    verbose && println("Matrix formed!")
    s = vcat((1-α) * vec(Xn), α * vec(Amat), α * vec(Bmat))
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

"""
    run_eDMD

Run the eDMD algorithm on the training data. Returns an EDMDModel.
"""
function run_eDMD(X_train, U_train, dt, function_list, order_list; reg=1e-6, name="edmd_model",
        alg=:qr, kwargs...
    )
    Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, function_list, order_list);

    A, B, C, g = learn_bilinear_model(X_train, Z_train, Zu_train,
        ["na", "na"]; 
        edmd_weights=[reg], 
        mapping_weights=[0.0],
        algorithm=alg,
        kwargs...
    )
    EDMDModel(A, B, C, g, kf, dt, name)
end

function subsamplerows(x::AbstractArray, α::Real)
    α === one(α) && return x
    @assert 0 <= α <= 1
    n = size(x,1)
    m = size(x,2)
    inds = map(x->round(Int,x), range(1,n,length=round(Int, α*n)))
    if x isa Vector
        x[inds]
    else
        x[inds,:]
    end
end

"""
    run_jDMD

Run the jDMD algorithm on the training data, using the provided model to regularize the 
Jacobians of the learned model.
"""
function run_jDMD(X_train, U_train, dt, function_list, order_list, model::RD.DiscreteDynamics; 
        reg=1e-6, name="jdmd_model", α=0.5, learnB=true, β=1.0, showprog=false, verbose=false
    )
    n0 = length(X_train[1])
    m = length(U_train[1])
    T_train = range(0,step=dt,length=size(X_train,1))

    # Generate transform
    Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, function_list, order_list);

    ## Generate Jacobians
    xn = zeros(n0)
    n = length(kf(xn))  # new state dimension
    cinds_jac = subsamplerows(CartesianIndices(U_train), β)
    jacobians = map(cinds_jac) do cind
        k = cind[1]
        x = X_train[cind]
        u = U_train[cind]
        z = RD.KnotPoint{n0,m}(x,u,T_train[k],dt)
        J = zeros(n0,n0+m)
        RD.jacobian!(
            RD.InPlace(), RD.ForwardAD(), model, J, xn, z 
        )
        J
    end
    A_train = map(J->J[:,1:n0], jacobians)
    B_train = map(J->J[:,n0+1:end], jacobians)

    ## Convert states to lifted Koopman states
    Y_train = map(kf, X_train)

    ## Calculate Jacobian of Koopman transform
    F_train = map(cinds_jac) do cind 
        x = X_train[cind]
        sparse(ForwardDiff.jacobian(kf, x))
    end

    ## Create a sparse version of the G Jacobian
    G = spdiagm(n0,n,1=>ones(n0)) 
    if function_list isa AbstractVector
        @assert function_list[1] == "state"
    end

    ## Build Least Squares Problem
    verbose && println("Generating least squares data")
    W,s = BilinearControl.EDMD.build_edmd_data(
        Y_train, U_train, A_train, B_train, F_train, G; cinds_jac, α, learnB, verbose)

    n = length(Z_train[1])

    ## Create sparse LLS matrix
    #   TODO: avoid forming this matrix explicitly (i.e. use LazyArrays)
    # verbose && println("Forming sparse matrix...")
    # Wsparse = sparse(W)

    ## Solve with RLS
    verbose && println("Solving least-squares problem")
    x_rls = BilinearControl.EDMD.rls_qr(s, W; Q=reg, showprog)
    E = reshape(x_rls,n,:)

    ## Extract out bilinear dynamics
    mB = m * learnB
    A = E[:,1:n]
    B = E[:,n .+ (1:mB)]
    C = E[:,n+mB .+ (1:n*m)]

    C_list = Matrix{Float64}[]
        
    for i in 1:m
        C_i = C[:, (i-1)*n+1:i*n]
        push!(C_list, C_i)
    end

    C = C_list

    EDMDModel(A,B,C,G,kf,dt,name)
end

"""
    open_loop_error

Calculate the error between the given trajectories when simulating the EDMD model open-loop.
"""
function open_loop_error(model::EDMDModel, X_test, U_test)
    num_test = size(X_test, 2)
    dt = model.dt
    tf = (size(X_test,1) - 1) * dt
    openloop_errors = map(1:num_test) do i
        Y, = simulate(model, U_test[:,i], expandstate(model, X_test[1,i]), tf, dt)
        X = map(x->originalstate(model, x), Y)
        norm(X - X_test[:,i])
    end
    return mean(openloop_errors)
end