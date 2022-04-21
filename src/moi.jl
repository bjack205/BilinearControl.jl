struct BilinearMOI <: MOI.AbstractNLPEvaluator
    A::SparseMatrixCSC{Float64,Int}
    B::SparseMatrixCSC{Float64,Int}
    C::Vector{SparseMatrixCSC{Float64,Int}}
    D::SparseVector{Float64,Int}

    Q::SparseMatrixCSC{Float64,Int}
    q::Vector{Float64}
    R::SparseMatrixCSC{Float64,Int}
    r::Vector{Float64}
    c::Vector{Float64}

    Ahat::SparseMatrixCSC{Float64}
    Bhat::SparseMatrixCSC{Float64}
    nzindsA::Vector{Vector{Int}}
    nzindsB::Vector{Vector{Int}}

    Nx::Int
    Nu::Int
end

function BilinearMOI(A,B,C,d, Q,q,R,r,c=0.0)
    n = size(A,2)
    m = size(B,2)
    p = length(d)
    x = zeros(n)
    u = zeros(m)

    # Build Ahat and Bhat
    Ahat = A + sum(c*rand() for c in C)
    Bhat = copy(B)
    x_ = randn(n)
    for i = 1:m
        Bhat[:,i] += C[i] * x_ * rand()
    end

    # Precompute index caches for sparse matrices
    nzindsA = map(eachindex(C)) do i
        getnzindsA(Ahat, C[i])
    end
    pushfirst!(nzindsA, getnzindsA(Ahat, A))
    nzindsB = map(eachindex(C)) do i
        getnzindsB(Bhat, C[i], i)
    end
    pushfirst!(nzindsB, getnzindsA(Bhat, B))

    Nx = size(Ahat,2)
    Nu = size(Bhat,2)

    BilinearMOI(A,B,C,d, Q,q,R,r,[c], Ahat,Bhat,nzindsA,nzindsB, Nx,Nu)
end

num_primals(nlp::BilinearMOI) = nlp.Nx + nlp.Nu
num_duals(nlp::BilinearMOI) = length(nlp.D)

MOI.initialize(::BilinearMOI, requested_features) = nothing
MOI.features_available(::BilinearMOI) = [:Grad, :Jac]

getstatevec(nlp::BilinearMOI, z) = view(z, 1:nlp.Nx)
getcontrolvec(nlp::BilinearMOI, z) = view(z, nlp.Nx+1:nlp.Nx + nlp.Nu)

function MOI.eval_objective(nlp::BilinearMOI, z)
    x = getstatevec(nlp, z)
    u = getcontrolvec(nlp, z)
    0.5 * dot(x, nlp.Q, x) + dot(x, nlp.q) + 
        0.5 * dot(u, nlp.R, u) + dot(u, nlp.r) + sum(nlp.c)
end

function MOI.eval_constraint(nlp::BilinearMOI, g, z)
    x = getstatevec(nlp, z)
    u = getcontrolvec(nlp, z)
    A, B, C = nlp.A, nlp.B, nlp.C
    g .= nlp.D
    mul!(g, A, x, 1.0, 1.0)
    mul!(g, B, u, 1.0, 1.0)
    for i in eachindex(u)
        mul!(g, C[i], x, u[i], 1.0)
    end
    g
end

function MOI.eval_objective_gradient(nlp::BilinearMOI, df, z)
    Q,q = nlp.Q, nlp.q
    R,r = nlp.R, nlp.r
    x = getstatevec(nlp, z)
    u = getcontrolvec(nlp, z)
    dx = getstatevec(nlp, df)
    du = getcontrolvec(nlp, df)
    mul!(dx, Q, x)
    mul!(du, R, u)
    dx .+= nlp.q
    du .+= nlp.r
    df
end

function updateAhat!(nlp::BilinearMOI, u)
    Ahat = nlp.Ahat
    Ahat .= 0
    nzinds = nlp.nzindsA
    for (nzind0,nzind) in enumerate(nzinds[1])
        nonzeros(Ahat)[nzind] = nonzeros(nlp.A)[nzind0]
    end
    for i in eachindex(u)
        for (nzind0, nzind) in enumerate(nzinds[i+1])
            nonzeros(Ahat)[nzind] += nonzeros(nlp.C[i])[nzind0] * u[i]
        end
    end
    Ahat
end

function updateBhat!(solver::BilinearMOI, x)
    nzinds = solver.nzindsB

    # Copy B to Bhat
    Bhat = solver.Bhat
    Bhat .= 0
    for (nzind0, nzind) in enumerate(nzinds[1])
        nonzeros(Bhat)[nzind] = nonzeros(solver.B)[nzind0]
    end

    # Copy C[i]*x to B[:,i]
    C = solver.C
    for i in axes(Bhat, 2)
        for c in axes(C[i],2)
            for j in nzrange(C[i], c)
                nzindB = nzinds[i+1][j]
                nonzeros(Bhat)[nzindB] += nonzeros(C[i])[j] * x[c]
            end
        end
    end
    Bhat
end

function MOI.eval_constraint_jacobian(nlp::BilinearMOI, J, z)
    x = getstatevec(nlp, z)
    u = getcontrolvec(nlp, z)
    updateAhat!(nlp, u)
    updateBhat!(nlp, x)
    nnzA = nnz(nlp.Ahat)
    nnzB = nnz(nlp.Bhat)
    J[1:nnzA] .= nonzeros(nlp.Ahat) 
    J[nnzA+1:nnzA+nnzB] .= nonzeros(nlp.Bhat)
end

function MOI.jacobian_structure(nlp::BilinearMOI)
    Ahat = nlp.Ahat
    Bhat = nlp.Bhat

    rA,cA = findnz(Ahat)
    rB,cB = findnz(Bhat)
    cB .+= nlp.Nx  # shift columns of B
    rcA = map(zip(rA,cA)) do (r,c)
        (r,c)
    end
    rcB = map(zip(rB,cB)) do (r,c)
        (r,c)
    end
    append!(rcA, rcB)
end

function solve(nlp::BilinearMOI, x0;
        max_iter=1000,
        tol=1e-6,
        c_tol=1e-6,
        verbose=0
    )
    n_nlp = num_primals(nlp)
    m_nlp = num_duals(nlp)

    x_l = fill(-Inf, n_nlp)
    x_u = fill(+Inf, n_nlp)
    c_l = fill(0.0, m_nlp)
    c_u = fill(0.0, m_nlp)
    nlp_bounds = MOI.NLPBoundsPair.(c_l, c_u)
    has_objective = true
    block_data = MOI.NLPBlockData(nlp_bounds, nlp, has_objective)

    solver = Ipopt.Optimizer()
    x = MOI.add_variables(solver, n_nlp)
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol
    solver.options["print_level"] = verbose 

    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    res = MOI.get(solver, MOI.VariablePrimal(), x)
    res, solver
end
