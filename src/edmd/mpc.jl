
struct LinearMPC{T} <: AbstractController
    # Reference trajectory
    Xref::Vector{Vector{T}}
    Uref::Vector{Vector{T}}
    Tref::Vector{T}

    # Dynamics 
    Aref::Vector{Matrix{T}}
    Bref::Vector{Matrix{T}}
    fref::Vector{Vector{T}}

    # Cost
    Q::Vector{Diagonal{T,Vector{T}}}
    R::Vector{Diagonal{T,Vector{T}}}
    q::Vector{Vector{T}}
    r::Vector{Vector{T}}

    Nt::Int
    xmax::Vector{T}
    xmin::Vector{T}
    umax::Vector{T}
    umin::Vector{T}

    # Trajectory storage
    X::Vector{Vector{T}}
    U::Vector{Vector{T}}
    num_fails::Vector{Int}
end

function LinearMPC(model, Xref, Uref, Tref, Qk, Rk, Qf; Nt=length(Xref),
        xmax=fill(+Inf,length(Xref[1])),
        xmin=fill(-Inf,length(Xref[1])),
        umax=fill(+Inf,length(Uref[1])),
        umin=fill(-Inf,length(Uref[1])),
    )
    N = length(Xref)
    n = length(Xref[1])
    m = length(Uref[1])
    A,B = linearize(model, Xref, Uref, Tref)
    f = [zeros(n) for k = 1:Nt-1]

    Q = [copy(Qk) for k = 1:Nt]
    Q[end] .= Qf
    R = [copy(Rk) for k = 1:Nt-1]
    q = [zeros(n) for k = 1:Nt]
    r = [zeros(m) for k = 1:Nt-1]
    push!(Q,Qf)
    f = map(1:N) do k
        dt = k < N ? Tref[k+1] - Tref[k] : Tref[k] - Tref[k-1] 
        xn = k < N ? copy(Xref[k+1]) : copy(Xref[k])
        Vector(RD.discrete_dynamics(model, Xref[k], Uref[k], Tref[k], dt) - xn)
    end
    X = [zeros(n) for k = 1:Nt]
    U = [zeros(m) for k = 1:Nt]
    LinearMPC(Xref,Uref,Tref, A,B,f, Q,R,q,r, Nt, xmax,xmin, umax,umin, X,U, [0])
end

gettime(ctrl::LinearMPC) = ctrl.Tref

function getcontrol(mpc::LinearMPC, x, t)
    N_ref = length(mpc.Xref)
    Nt = mpc.Nt
    k = get_k(mpc, t)

    Nh = min(Nt, N_ref - k)     # actual MPC length (shrinks at the end of the horizon)
    # if there's only one step left, use control from the previous step
    if Nh == 1
        return mpc.U[1] + mpc.Uref[k]
    end
    mpc_inds = k-1 .+ (1:Nh)
    A = mpc.Aref[mpc_inds[1:end-1]]
    B = mpc.Bref[mpc_inds[1:end-1]]
    f = mpc.fref[(Nt-Nh) .+ (1:Nh-1)]
    Q = mpc.Q[(Nt-Nh) .+ (1:Nh)]
    R = mpc.R[(Nt-Nh) .+ (1:Nh-1)]
    q = mpc.q[(Nt-Nh) .+ (1:Nh)]
    r = mpc.r[(Nt-Nh) .+ (1:Nh-1)]
    xmax,xmin = mpc.xmax, mpc.xmin
    umax,umin = mpc.umax, mpc.umin

    dx = x - mpc.Xref[k]
    dX,dU,_,solved = EDMD.solve_lqr_osqp(Q,R,q,r,A,B,f,dx; xmin, xmax, umin, umax)
    if !solved
        @warn "OSQP solve failed"
        mpc.num_fails[1] += 1
        i = mpc.num_fails[1] + 1  # 0-to-1 based index shift
        return mpc.U[i] + mpc.Uref[k]
    else
        mpc.num_fails[1]
        mpc.X[1:Nh] .= dX
        mpc.U[1:Nh-1] .= dU
        return dU[1] + mpc.Uref[k]
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
    success = res.info.status == :Solved
    X = [x for x in eachcol(reshape(res.x[1:n*Nt], n,:))]
    U = [u for u in eachcol(reshape(res.x[n*Nt .+ (1:(Nt-1)*m)], m,:))]
    λ = [y for y in eachcol(reshape(res.y[1:n*Nt], n, :))]
    X,U,λ,success
end
