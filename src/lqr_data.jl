struct TOQP{n,m,T}
    Q::Vector{Diagonal{T,SVector{n,T}}}
    R::Vector{Diagonal{T,SVector{m,T}}}
    q::Vector{SVector{n,T}}
    r::Vector{SVector{m,T}}
    c::Vector{T}
    A::Vector{SizedMatrix{n,n,T,2,Matrix{T}}}
    B::Vector{SizedMatrix{n,m,T,2,Matrix{T}}}
    C::Vector{SizedMatrix{n,n,T,2,Matrix{T}}}
    D::Vector{SizedMatrix{n,m,T,2,Matrix{T}}}
    d::Vector{SVector{n,T}}
    x0::SVector{n,T}
end

function Base.rand(::Type{<:TOQP{n,m}}, N::Integer; cond=1.0, implicit=false) where {n,m}
    Nx,Nu = n,m
    Q = [Diagonal(@SVector rand(Nx)) * 10^cond for k = 1:N]
    R = [Diagonal(@SVector rand(Nu)) for k = 1:N-1]
    q = [@SVector randn(Nx) for k = 1:N]
    r = [@SVector randn(Nu) for k = 1:N-1]
    A = SizedMatrix.([@SMatrix zeros(Nx,Nx) for k = 1:N-1])
    B = SizedMatrix.([@SMatrix zeros(Nx,Nu) for k = 1:N-1])
    C = SizedMatrix.([@SMatrix zeros(Nx,Nx) for k = 1:N-1])
    D = SizedMatrix.([@SMatrix zeros(Nx,Nu) for k = 1:N-1])
    for k = 1:N-1
        Ak, Bk = RandomLinearModels.gencontrollable(n, m)
        A[k] .= Ak
        B[k] .= Bk
        if implicit
            Ck,Dk = RandomLinearModels.gencontrollable(n ,m)
            C[k] .= Ck
            D[k] .= Dk
        else
            C[k] .= -I(n)
        end
    end
    d = [@SVector randn(Nx) for k = 1:N-1]
    x0 = @SVector randn(Nx)
    c = randn(N) * 10
    data = TOQP(Q,R,q,r,c,A,B,C,D,d,x0)
end
Base.size(data::TOQP{n,m}) where {n,m} = (n,m, length(data.Q))
nhorizon(data::TOQP) = length(data.Q)
RD.state_dim(data::TOQP) = length(data.X[1])
RD.control_dim(data::TOQP) = length(data.U[1])

function Base.copy(data::TOQP)
    TOQP(
        deepcopy(data.Q),
        deepcopy(data.R),
        deepcopy(data.q),
        deepcopy(data.r),
        deepcopy(data.c),
        deepcopy(data.A),
        deepcopy(data.B),
        deepcopy(data.C),
        deepcopy(data.D),
        deepcopy(data.d),
        deepcopy(data.x0)
    )
end

function unpackY(data::TOQP, Y)
    n,m,N = size(data)
    yinds = [(k-1)*(2n+m) .+ (1:n) for k = 1:N]
    xinds = [(k-1)*(2n+m) + n .+ (1:n) for k = 1:N]
    uinds = [(k-1)*(2n+m) + 2n .+ (1:m) for k = 1:N-1]
    λ = [Y[yi] for yi in yinds]
    X = [Y[xi] for xi in xinds]
    U = [Y[ui] for ui in uinds]
    X,U,λ
end

function primal_residual(data, X, U)
    r = norm(data.x0 - X[1], Inf) 
    N = nhorizon(data)
    for k = 1:N-1
        r = max(r, norm(
            data.A[k] * X[k] + 
            data.B[k]*U[k] + 
            data.d[k] + 
            data.C[k]*X[k+1]
        ))
    end
    r
end

function dual_residual(data, X, U, λ)
    r = norm(data.Q[1]*X[1] + data.q[1] + data.A[1]'λ[2] - λ[1], Inf)
    for k = 1:nhorizon(data)-1
        rx = norm(data.Q[k]*X[k] + data.q[k] + data.A[k]'λ[k+1] + data.C[k]'λ[k], Inf)
        ru = norm(data.R[k]*U[k] + data.r[k] + data.B[k]'λ[k+1], Inf)
        r = max(r, rx, ru)
    end
    r
end

#############################################
# Methods to convert LQR problem to 
#   Linear System of Equations
#############################################
function build_block_diagonal(blocks)
    n = 0
    m = 0
    for block in blocks
        n += size(block, 1)
        m += size(block, 2)
    end
    A = spzeros(n, m)
    off1 = 0
    off2 = 0
    for block in blocks
        inds1 = off1 .+ (1:size(block, 1))
        inds2 = off2 .+ (1:size(block, 2))
        A[inds1, inds2] .= block
        off1 += size(block, 1)
        off2 += size(block, 2)
    end
    return A
end

function stack_vectors(vectors)
    n = 0
    for vec in vectors 
        n += size(vec, 1)
    end
    b = spzeros(n)
    off = 0
    for vec in vectors 
        inds = off .+ (1:size(vec, 1))
        b[inds] .= vec 
        off += size(vec, 1)
    end
    return b
end

function build_Ab(data::TOQP{n,m}; remove_x1::Bool=false, reg=0.0) where {n,m}
    N = length(data.Q)
    Q,R,q,r = data.Q, data.R, data.q, data.r
    A,B,d   = data.A, data.B, data.d
    C,D     = data.C, data.D
   
    Ds = [[
            Q[k] zeros(n,m) A[k]';
            zeros(m,n) R[k] B[k]';
            A[k] B[k] -I(n)*reg 
        ] for k = 1:N-1
    ]
    push!(Ds, Q[N])

    Is = [
        [
            zeros(n,n) C[k] D[k];
            C[k]' zeros(n,n) zeros(n,m);
            D[k]' zeros(m,n) zeros(m,m);
        ] for k = 1:N-1
    ]

    b = map(1:N) do k
        dk = k == 1 ? data.x0 : d[k-1]
        if k == N
            [dk; q[k]]
        else
            [dk; q[k]; r[k]]
        end
    end

    if remove_x1
        Is[1] = zeros(m,m)
        Ds[1] = Ds[1][n+1:end, n+1:end]
        b[1] = r[1]
        b[2] = [A[1]*data.x0 + d[1]; q[2]; r[2]]
    else
        pushfirst!(Ds, -I(n)*reg)
    end
    push!(Is, Is[end][1:2n,1:2n])

    Ds = build_block_diagonal(Ds)
    Is = build_block_diagonal(Is)
    b = Vector(stack_vectors(b))
    A = Ds + Is
    return A,-b
end
