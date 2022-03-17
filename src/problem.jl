export BiLinearProblem, getmodel, 
    cost, statecost, controlcost, statecost_grad!, controlcost_grad!,
    constraints!, constraint_state_jacobian!, constraint_control_jacobian!

struct BiLinearProblem
    model::BiLinearDynamics
    Q::Diagonal{Float64, Vector{Float64}}
    R::Diagonal{Float64, Vector{Float64}}
    Qf::Diagonal{Float64, Vector{Float64}}
    x0::Vector{Float64}
    xf::Vector{Float64}
    N::Int
    xinds::Vector{UnitRange{Int}}
    uinds::Vector{UnitRange{Int}}
    yinds::Vector{UnitRange{Int}}
end

function BiLinearProblem(model, Q, R, Qf, x0, xf, N)
    n = state_dim(model)
    m = control_dim(model)
    @assert size(Q) == (n,n)
    @assert size(Qf) == (n,n)
    @assert size(R) == (m,m)
    @assert length(x0) == n
    @assert length(xf) == n
    Np = N * n + (N - 1) * m  # number of primals
    xinds = [(1:n) .+ (k-1)*n for k = 1:N]
    uinds = [(1:m) .+ (k-1)*m .+ N*n for k = 1:N-1]
    yinds = [(1:n) .+ (k-1)*n .+ Np for k = 1:N]
    BiLinearProblem(model, Q, R, Qf, x0, xf, N, xinds, uinds, yinds)
end

dims(prob::BiLinearProblem) = (state_dim(prob.model), control_dim(prob.model), prob.N)
getmodel(prob::BiLinearProblem) = prob.model

function num_primals(prob::BiLinearProblem)
    n,m,N = dims(prob)
    return N*n + (N-1) * m
end

function num_duals(prob::BiLinearProblem)
    n,_,N = dims(prob)
    return n*N
end

function unpackZ(prob::BiLinearProblem, Z)
    n,m,N = dims(prob)
    iX = 1:N*n
    iU = N*n .+ (1:(N-1)*m)
    X = @view Z[iX]
    U = @view Z[iU]
    return X, U
end

function cost(prob, Z)
    X, U = unpackZ(prob, Z)
    return statecost(prob, X) + controlcost(prob, U)
end

function statecost(prob, X)
    n,_,N = dims(prob)
    J = zero(eltype(X))
    for k = 1:N
        ix = (k-1)*n .+ (1:n)
        Q = k == N ? prob.Qf : prob.Q
        x = @view X[ix]
        dx = x - prob.xf

        J += dot(dx, Q, dx) / 2
    end
    return J
end

function controlcost(prob, U)
    _,m,N = dims(prob)
    R = prob.R
    J = zero(eltype(U))
    for k = 1:N-1
        iu = (k-1)*m .+ (1:m)
        u = @view U[iu]

        J += dot(u, R, u) / 2
    end
    return J
end

function statecost_grad!(prob, gx, X)
    n,_,N = dims(prob)
    for k = 1:N
        ix = (k-1)*n .+ (1:n)
        Q = k == N ? prob.Qf : prob.Q
        x = @view X[ix]
        dx = x - prob.xf
        gx[ix] .= Q * dx
    end
    return gx
end

function controlcost_grad!(prob, gu, U)
    _,m,N = dims(prob)
    R = prob.R
    for k = 1:N-1
        iu = (k-1)*m .+ (1:m)
        u = @view U[iu]
        gu[iu] .= R * u
    end
    return gu
end

function constraints!(prob, c, X, U)
    n,m,N = dims(prob)
    
    # Initial condition
    ic = 1:n
    ix = 1:n
    x = @view X[ix]
    c[ic] .= prob.x0 - x
    ic = ic .+ n

    # Dynamics constraints
    model = prob.model
    for k = 1:N-1
        ix1 = (k-1)*n .+ (1:n)
        iu1 = (k-1)*m .+ (1:m)
        ix2 = ix1 .+ n 
        x1 = @view X[ix1]
        u1 = @view U[iu1]
        x2 = @view X[ix2]
        y = zero(x)
        discrete_dynamics!(model, y, x1, u1)
        c[ic] .= y .- x2
        ic = ic .+ n
    end
    return c
end

function constraint_state_jacobian!(prob, jx, X, U)
    n,m,N = dims(prob)
    jx .= 0
    
    # Initial condition
    ic = 1:n
    ix = 1:n
    jx[ic,ix] .= -I(n) 
    ic = ic .+ n

    # Dynamics constraints
    model = prob.model
    for k = 1:N-1
        ix1 = (k-1)*n .+ (1:n)
        iu1 = (k-1)*m .+ (1:m)
        ix2 = ix1 .+ n 
        u1 = @view U[iu1]
        Ahat = getAhat(model, u1)
        jx[ic, ix1] .= Ahat
        jx[ic, ix2] .= -I(n)
        ic = ic .+ n
    end
    return jx
end

function constraint_control_jacobian!(prob, ju, X, U)
    n,m,N = dims(prob)
    ju .= 0
    
    # Initial condition
    ic = n+1:2n

    # Dynamics constraints
    model = prob.model
    for k = 1:N-1
        ix1 = (k-1)*n .+ (1:n)
        iu1 = (k-1)*m .+ (1:m)
        x1 = @view X[ix1]
        Bhat = getBhat(model, x1)
        ju[ic, iu1] .= Bhat
        ic = ic .+ n
    end
    return ju
end