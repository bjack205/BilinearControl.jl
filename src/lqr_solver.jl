struct RiccatiSolver{n,m,T}
    data::TOQP{n,m,T}
    K::Vector{SizedMatrix{m,n,T,2,Matrix{T}}}
    d::Vector{SVector{m,T}}
    P::Vector{SizedMatrix{n,n,T,2,Matrix{T}}}
    p::Vector{SVector{n,T}}
    X::Vector{SVector{n,T}}
    U::Vector{SVector{m,T}}
    λ::Vector{SVector{n,T}}
end

function RiccatiSolver(data::TOQP{n,m}) where {n,m}
    N = length(data.Q)
    K = [SizedMatrix{m,n}(zeros(m,n)) for k = 1:N-1]
    d = [@SVector zeros(m) for k = 1:N-1]
    P = [SizedMatrix{n,n}(zeros(n,n)) for k = 1:N]
    p = [@SVector zeros(n) for k = 1:N]
    X = [@SVector zeros(n) for k = 1:N]
    U = [@SVector zeros(m) for k = 1:N-1]
    λ = [@SVector zeros(n) for k = 1:N]
    RiccatiSolver(data, K, d, P, p, X, U, λ)
end

function backwardpass!(solver::RiccatiSolver{n,m}) where {n,m}
    A,B,f = solver.data.A, solver.data.B, solver.data.d
    Q,q = solver.data.Q, solver.data.q
    R,r = solver.data.R, solver.data.r
    # K,d = solver.K, solver.d
    P,p = solver.P, solver.p
    N = length(Q) 

    P[N] .= Q[N]
    p[N] = q[N]

    for k = reverse(1:N-1)
        P′ = SMatrix(P[k+1])
        Ak = SMatrix{n,n}(A[k])
        Bk = SMatrix{n,m}(B[k])

        Qx = q[k] + Ak'*(P′*f[k] + p[k+1]) 
        Qu = r[k] + Bk'*(P′*f[k] + p[k+1])
        Qxx = Q[k] + Ak'P′*Ak
        Quu = R[k] + Bk'P′*Bk
        Qux = Bk'P′*Ak

        cholQ = cholesky(Symmetric(Quu))
        K = -(cholQ \ Qux)
        d = -(cholQ \ Qu)

        P[k] .= Qxx .+ K'Quu*K .+ K'Qux .+ Qux'K
        p[k] = Qx .+ K'Quu*d .+ K'Qu .+ Qux'd
        solver.K[k] .= K
        solver.d[k] = d
    end
end

function forwardpass!(solver::RiccatiSolver{n,m}) where {n,m}
    A,B,f = solver.data.A, solver.data.B, solver.data.d
    X,U,λ = solver.X, solver.U, solver.λ
    K,d = solver.K, solver.d
    X[1] = solver.data.x0
    N = length(X)
    for k = 1:N-1
        λ[k] = SMatrix(solver.P[k])*X[k] .+ solver.p[k]
        U[k] = SMatrix(K[k])*X[k] + d[k]
        X[k+1] = SMatrix{n,n}(A[k])*X[k] .+ SMatrix{n,m}(B[k])*U[k] .+ f[k]
    end
    λ[N] = SMatrix(solver.P[N])*X[N] .+ solver.p[N]
end

function solve!(solver::RiccatiSolver)
    backwardpass!(solver)
    forwardpass!(solver)
    return nothing
end