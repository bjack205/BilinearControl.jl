
struct TrajectoryTracker{T} <: AbstractController
    # Reference trajectory
    Xref::Vector{Vector{T}}
    Uref::Vector{Vector{T}}
    Tref::Vector{T}

    # Dynamics 
    A::Vector{Matrix{T}}
    B::Vector{Matrix{T}}
    f::Vector{Vector{T}}

    # Cost
    Q::Vector{Diagonal{T,Vector{T}}}
    R::Vector{Diagonal{T,Vector{T}}}
    q::Vector{Vector{T}}
    r::Vector{Vector{T}}

    Nt::Int
    K::Vector{Matrix{T}}  # TVLQR gains
    xmax::Vector{T}
    xmin::Vector{T}
    umax::Vector{T}
    umin::Vector{T}
end

function TrajectoryTracker(model, Xref, Uref, Tref, Qk, Rk, Qf; Nt=length(Xref),
        xmax=fill(+Inf,length(Xref[1])),
        xmin=fill(-Inf,length(Xref[1])),
        umax=fill(+Inf,length(Uref[1])),
        umin=fill(-Inf,length(Uref[1])),
    )
    N = length(Xref)
    n = length(Xref[1])
    m = length(Uref[1])
    A,B = linearize(model, Xref, Uref, Tref)
    Q = [copy(Qk) for k = 1:Nt-1]
    R = [copy(Rk) for k = 1:Nt-1]
    q = [zeros(n) for k = 1:Nt]
    r = [zeros(m) for k = 1:Nt-1]
    push!(Q,Qf)
    f = map(1:N) do k
        dt = k < N ? Tref[k+1] - Tref[k] : Tref[k] - Tref[k-1] 
        xn = k < N ? copy(Xref[k+1]) : copy(Xref[k])
        Vector(RD.discrete_dynamics(model, Xref[k], Uref[k], Tref[k], dt) - xn)
    end
    tvlqr_inds = N - Nt + 1:N
    K, = tvlqr(A[tvlqr_inds], B[tvlqr_inds], Q, R)
    TrajectoryTracker(Xref,Uref,Tref, A,B,f, Q,R,q,r, Nt,K, xmax,xmin, umax,umin)
end

gettime(ctrl::TrajectoryTracker) = ctrl.Tref

function getcontrol(mpc::TrajectoryTracker, x, t)
    N = length(mpc.Xref)
    Nt = mpc.Nt
    k = get_k(mpc, t)
    if k + Nt > N
        i = k - (N - Nt)
        @show i
        dx = x - mpc.Xref[k]
        return mpc.K[i]*dx + mpc.Uref[k]
    else
        n,m = size(mpc.B[1])
        mpc_inds = k-1 .+ (1:Nt-1)
        A = mpc.A[mpc_inds]
        B = mpc.B[mpc_inds]
        f = mpc.f[mpc_inds]
        x0 = x - mpc.Xref[k]
        X,U,y = solve_lqr_osqp(mpc.Q, mpc.R, mpc.q, mpc.r, A, B, f, x0)
        return U[1] + mpc.Uref[k]
    end
end

function solve_lqr_osqp(Q,R,q,r,A,B,f,x0;
        xmax=fill(+Inf,length(q[1])),
        xmin=fill(-Inf,length(q[1])),
        umax=fill(+Inf,length(r[1])),
        umin=fill(-Inf,length(r[1])),
    )
    Nt = length(q)
    n,m = size(B[1])
    Np = Nt*n + (Nt-1) * m
    Nd = Nt*n
    println("hi")
    @assert length(Q) == Nt
    @assert length(R) == Nt-1
    @assert length(A) == Nt-1
    @assert length(B) == Nt-1
    @assert length(f) == Nt-1
    @assert length(q) == Nt
    @assert length(r) == Nt-1
    P_qp = spdiagm(vcat(mapreduce(diag, vcat, Q), mapreduce(diag, vcat, R)))
    q_qp = vcat(reduce(vcat, q), reduce(vcat, r))
    b = [-x0; reduce(vcat, f)]
    D = spzeros(Nd,Np)
    D[1:n,1:n] .= -I(n)
    for k = 1:Nt-1
        ic = k*n .+ (1:n)
        ix = (k-1) * n .+ (1:n)
        iu = Nt*n + (k-1) * m .+ (1:m)
        D[ic,ix] .= A[k]
        D[ic,iu] .= B[k]
        D[ic,ix .+ n] .= -I(n)
    end
    C = sparse(I,Np,Np)
    lp = [repeat(xmin, Nt); repeat(umin, Nt-1)]
    up = [repeat(xmax, Nt); repeat(umax, Nt-1)]
    osqp = OSQP.Model()
    OSQP.setup!(osqp, P=P_qp, q=q_qp, A=[D; C], l=[b;lp], u=[b;up], verbose=false)
    res = OSQP.solve!(osqp)
    X = [x for x in eachcol(reshape(res.x[1:n*Nt], n,:))]
    U = [u for u in eachcol(reshape(res.x[n*Nt .+ (1:(Nt-1)*m)], m,:))]
    λ = [y for y in eachcol(reshape(res.y[1:n*Nt], n, :))]
    X,U,λ
end
