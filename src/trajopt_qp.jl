struct TOQP{n,m,T}
    # Objective
    Q::Vector{Diagonal{T,Vector{T}}}
    R::Vector{Diagonal{T,Vector{T}}}
    q::Vector{Vector{T}}
    r::Vector{Vector{T}}
    c::Vector{T}

    # Dynamics constraints
    A::Vector{Matrix{T}}
    B::Vector{Matrix{T}}
    C::Vector{Matrix{T}}
    D::Vector{Matrix{T}}
    d::Vector{Vector{T}}
    x0::Vector{T}

    # Equality constraints
    Hx::Vector{Matrix{T}}
    hx::Vector{Vector{T}}
    Hu::Vector{Matrix{T}}
    hu::Vector{Vector{T}}

    # Inequality constraints 
    Gx::Vector{Matrix{T}}
    gx::Vector{Vector{T}}
    Gu::Vector{Matrix{T}}
    gu::Vector{Vector{T}}
    # TODO: Add conic constraints 
end

function TOQP(model::DiscreteLinearModel, obj::TO.Objective, x0;
        Hx=[zeros(0, state_dim(model)) for k = 1:length(obj)],
        hx=[zeros(0) for k = 1:length(obj)],
        Hu=[zeros(0, control_dim(model)) for k = 1:length(obj)],
        hu=[zeros(0) for k = 1:length(obj)],
        Gx=[zeros(0, state_dim(model)) for k = 1:length(obj)],
        gx=[zeros(0) for k = 1:length(obj)],
        Gu=[zeros(0, control_dim(model)) for k = 1:length(obj)],
        gu=[zeros(0) for k = 1:length(obj)],
    )
    N = length(obj)
    n,m = RD.dims(model)
    Q = map(c->c.Q, obj.cost)
    R = map(c->c.R, obj.cost)
    q = map(c->c.q, obj.cost)
    r = map(c->c.r, obj.cost)
    c = map(c->c.c, obj.cost)
    D = [zeros(n,m) for k = 1:N-1]
    TOQP{n,m,Float64}(Q, R, q, r, c, A, B, C, D, d, x0, Hx,hx, Hu,hu, Gx,gx, Gu,gu)
end

function Base.rand(::Type{<:TOQP{n,m}}, N::Integer; cond=1.0, implicit=false) where {n,m}
    Nx,Nu = n,m
    Q = [Diagonal(rand(Nx)) * 10^cond for k = 1:N]
    R = [Diagonal(rand(Nu)) for k = 1:N-1]
    q = [randn(Nx) for k = 1:N]
    r = [randn(Nu) for k = 1:N-1]
    A = [zeros(Nx,Nx) for k = 1:N-1]
    B = [zeros(Nx,Nu) for k = 1:N-1]
    C = [zeros(Nx,Nx) for k = 1:N-1]
    D = [zeros(Nx,Nu) for k = 1:N-1]
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
    d = [randn(Nx) for k = 1:N-1]
    x0 = randn(Nx)
    c = randn(N) * 10
    Hx=[zeros(0, n) for k = 1:N]
    hx=[zeros(0) for k = 1:N]
    Hu=[zeros(0, m) for k = 1:N]
    hu=[zeros(0) for k = 1:N]
    Gx=[zeros(0, n) for k = 1:N]
    gx=[zeros(0) for k = 1:N]
    Gu=[zeros(0, m) for k = 1:N]
    gu=[zeros(0) for k = 1:N]

    data = TOQP{n,m,Float64}(Q,R,q,r,c,A,B,C,D,d,x0, Hx,hu, Hu,hu, Gx,gx, Gu,gu)
end
Base.size(data::TOQP{n,m}) where {n,m} = (n,m, length(data.Q))
nhorizon(data::TOQP) = length(data.Q)
RD.state_dim(data::TOQP) = length(data.q[1])
RD.control_dim(data::TOQP) = length(data.r[1])
num_primals(qp::TOQP) = sum(length, qp.q) + sum(length, qp.r)

function num_duals(qp::TOQP)
    nduals = state_dim(qp) * nhorizon(qp)              # dynamics and initial condition
    nduals += num_equality(qp) + num_inequality(qp)
    nduals
end

num_equality(qp::TOQP) = num_state_equality(qp) + num_control_equality(qp) 
num_inequality(qp::TOQP) = num_state_inequality(qp) + num_control_inequality(qp) 

num_state_equality(qp::TOQP) = sum(length(qp.hx))
num_state_inequality(qp::TOQP) = sum(length(qp.gx))

num_control_equality(qp::TOQP) = sum(length(qp.hu))
num_control_inequality(qp::TOQP) = sum(length(qp.gu))

num_states(qp::TOQP) = sum(length, qp.q)
num_controls(qp::TOQP) = sum(length, qp.r)

function getxind(qp::TOQP, k)
    n = state_dim(qp)
    (k-1)*n .+ (1:n)
end

function getuind(qp::TOQP, k)
    n,m = state_dim(qp), control_dim(qp) 
    N*n + (k-1)*m .+ (1:m)
end

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
        deepcopy(data.x0),
        deepcopy(data.Gx),
        deepcopy(data.gx),
        deepcopy(data.Gu),
        deepcopy(data.gu),
        deepcopy(data.Hx),
        deepcopy(data.hx),
        deepcopy(data.Hu),
        deepcopy(data.hu),
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
            data.C[k]*X[k+1],
            Inf
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

function build_objective(qp::TOQP)
    Np = num_primals(qp)
    P = spzeros(Np, Np)
    q = zeros(Np)

    for k in eachindex(qp.Q) 
        ix = getxind(qp, k)
        iu = getuind(qp, k)
        P[ix,ix] .= qp.Q[k]
        q[ix] .= qp.q
    end
    for k in eachindex(qp.R)
        P[iu,iu] .= qp.R[k]
        q[iu] .= qp.r[k]
    end

    P, q, sum(qp.c)
end

function build_dynamics(qp::TOQP)
    n = state_dim(qp)
    Nc = nhorizon(qp) * n 
    Nx = num_states(qp) 
    Nu = num_controls(qp) 
    A = spzeros(Nc,Nx)
    B = spzeros(Nc,Nu)
    C = spzeros(Nc,Nx)
    d = spzeros(Nc)

    # Initial condition
    ix1 = getxind(qp, 1)
    ic = 1:n
    A[ic,ix1] .= -I(n)
    d[ic] .= qp.x0
    ic = ic .+ n

    # Dynamics
    for k in eachindex(qp.A)
        ix1 = getxind(qp, k)
        iu1 = getuind(qp, k) .- Nx
        ix2 = getxind(qp, k+1)

        A[ic,ix1] .= qp.A[k]
        B[ic,iu1] .= qp.B[k]
        C[ic,ix2] .= qp.C[k]
        d[ic] .= qp.d[k]
    end
    A,B,C,d
end

function build_state_equalities(qp::TOQP)
    Nc = num_state_equality(qp)
    Nx = num_states(qp) 
    Nu = num_controls(qp) 
    Hx = spzeros(Nc,Nx)
    hx = zeros(Nc)
    off = 1
    for k in eachindex(qp.Hx)
        ic = off .+ eachindex(qp.hx[k])
        ix = getxind(qp, k)
        Hx[ic,ix] .= qp.Hx[k]
        hx[ic] .= qp.hx[k]
    end
    Hx, hx
end

function build_state_inequalities(qp::TOQP)
    Nc = num_state_inequality(qp)
    Nx = num_states(qp) 
    Nu = num_controls(qp) 
    Gx = spzeros(Nc,Nx)
    gx = zeros(Nc)
    off = 1
    for k in eachindex(qp.Gx)
        ic = off .+ eachindex(qp.gx[k])
        ix = getxind(qp, k)
        Gx[ic,ix] .= qp.Gx[k]
        gx[ic] .= qp.gx[k]
    end
    Gx, gx
end

function build_control_equalities(qp::TOQP)
    Nc = num_control_equality(qp)
    Nx = num_states(qp) 
    Nu = num_controls(qp) 
    Hu = spzeros(Nc,Nu)
    hu = zeros(Nc)
    off = 1
    for k in eachindex(qp.Hu)
        ic = off .+ eachindex(qp.hu[k])
        iu = getuind(qp, k) .- Nx
        Hu[ic,iu] .= qp.Hu[k]
        hu[ic] .= qp.hu[k]
    end
    Hu, hu
end

function build_control_inequalities(qp::TOQP)
    Nc = num_control_inequality(qp)
    Nx = num_states(qp) 
    Nu = num_controls(qp) 
    Gx = spzeros(Nc,Nu)
    gx = zeros(Nc)
    off = 1
    for k in eachindex(qp.Gu)
        ic = off .+ eachindex(qp.gu[k])
        ix = getxind(qp, k)
        Gu[ic,ix] .= qp.Gu[k]
        gu[ic] .= qp.gu[k]
    end
    Gx, gx
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

function setup_osqp!(qp::TOQP; kwargs...)
    Nx = num_states(qp)
    Nu = num_controls(qp)

    P,q,c = build_objective(qp)
    A,B,C,d = build_dynamics(qp)
    Hx,hx = build_state_equalities(qp) 
    Hu,hu = build_control_equalities(qp) 
    Gx,gx = build_state_inequalities(qp) 
    Gu,gu = build_control_iequalities(qp) 
    Ahat = [
        [A+C B];  # dynamics
        [Hx spzeros(size(Hx,Nu))];  # state equalities
        [spzeros(size(Hu,Nx)) Hu];  # control equalities
        [Gx spzeros(size(Gx,Nu))];  # state inequalities
        [spzeros(size(Gu,Nx)) Gu];  # control inequalities
    ]
    lb = [
        -d; 
        -hx; 
        -hu;
        fill(-Inf, length(gx) + length(gu));
    ]
    ub = [
        -d; 
        -hx; 
        -hu;
        -gx;
        -gu;
    ]
    model = OSQP.Model()
    OSQP.setup!(model, P=P, q=q, A=A, l=lb, u=ub; kwargs...)
    model
end

function setup_convex!(qp::TOQP; kwargs...)
    Nx = num_states(qp)
    Nu = num_controls(qp)

    A,B,C,d = build_dynamics(qp)
    Hx,hx = build_state_equalities(qp) 
    Hu,hu = build_control_equalities(qp) 
    Gx,gx = build_state_inequalities(qp) 
    Gu,gu = build_control_iequalities(qp) 

    Q = build_block_diagonal(qp.Q)
    R = build_block_diagonal(qp.Q)
    q = stack_vectors(qp.q)
    r = stack_vectors(qp.r)

    x = Convex.Variable(Nx)
    u = Convex.Variable(Nu)

    problem = minimize(Convex.quadform(x, Q) + dot(x,q) + Convex.quadform(u, R) + dot(u, r))
    problem.constraints += Hx*x + hx == 0
    problem.constraints += Hu*u + hu == 0
    problem.constraints += Gx*x + gx <= 0
    problem.constraints += Gu*u + bu <= 0
    problem
end