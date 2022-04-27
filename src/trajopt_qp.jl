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

num_state_equality(qp::TOQP) = sum(length,qp.hx)
num_state_inequality(qp::TOQP) = sum(length,qp.gx)

num_control_equality(qp::TOQP) = sum(length,qp.hu)
num_control_inequality(qp::TOQP) = sum(length, qp.gu)

num_states(qp::TOQP) = sum(length, qp.q)
num_controls(qp::TOQP) = sum(length, qp.r)

function getxind(qp::TOQP, k)
    n = state_dim(qp)
    (k-1)*n .+ (1:n)
end

function getuind(qp::TOQP, k)
    n,m = state_dim(qp), control_dim(qp) 
    N = nhorizon(qp)
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

function primal_feasibility(data, X, U)
    r = norm(data.x0 - X[1], Inf) 
    N = nhorizon(data)
    pos(x) = max(zero(x), x)
    for k = 1:N
        x = X[k]
        u = U[k]
        r = max(r, norm(
            data.Hx[k]*x + data.hx[k],
            Inf
        ))
        r = max(r, norm(
            pos.(data.Gx[k]*x + data.gx[k]), 
            Inf
        ))
        if k < N
            xn = X[k+1]
            r = max(r, norm(
                data.A[k]*x + 
                data.B[k]*u + 
                data.d[k] + 
                data.C[k]*xn,
                Inf
            ))
            r = max(r, norm(
                data.Hu[k]*u + data.hu[k],
                Inf
            ))
            r = max(r, norm(
                pos.(data.Gu[k]*u + data.gu[k]), 
                Inf
            ))
        end
    end
    r
end

function stationarity(data, X, U, λ, μ, ν)
    r = norm(data.Q[1]*X[1] + data.q[1] + data.A[1]'λ[2] - λ[1], Inf)
    for k = 1:nhorizon(data)-1
        rx = norm(data.Q[k]*X[k] + data.q[k] + data.A[k]'λ[k+1] + data.C[k]'λ[k] + data.Hx[k]'μ[k,1] + data.Gx[k]'ν[k,1], Inf)
        ru = norm(data.R[k]*U[k] + data.r[k] + data.B[k]'λ[k+1] + data.Hu[k]'μ[k,2] + data.Gu[k]'ν[k,2], Inf)
        r = max(r, rx, ru)
    end
    r
end

function dual_feasibility(data, λ, μ, ν)
    neg(x) = min(zero(x), x)
    r = 0.0
    for nu in ν 
        if !isempty(nu)
            r = max(r, abs(minimum(neg.(nu))))
        end
    end
    r
end

function complementary_slackness(data, X, U, λ, μ, ν)
    r = 0.0
    for k in eachindex(X)
        gx = data.Gx[k] * X[k]  + data.gx[k]
        r = max(r, abs(gx'ν[k,1]))
    end
    for k in eachindex(data.Gu)
        gu = data.Gu[k] * U[k]  + data.gu[k]
        r = max(r, abs(gu'ν[k,2]))
    end
    r
end

function unpackprimals(qp::TOQP, x)
    n = state_dim(qp)
    m = control_dim(qp)
    Nx = num_states(qp)
    X = tovecs(x[1:Nx], n) 
    U = tovecs(x[Nx+1:end], m) 
    X, U
end

function unpackduals(qp::TOQP, y)
    n = state_dim(qp)
    m = control_dim(qp)
    N = nhorizon(qp)

    Nx = num_states(qp)
    λ = tovecs(view(y, 1:Nx), n)
    off = Nx
    μ = [zeros(0) for k = 1:N, i = 1:2]
    ν = [zeros(0) for k = 1:N, i = 1:2]

    function getduals!(λ,c,col)
        for k in eachindex(c)
            p = length(c[k])
            if p > 0
                λ[k,col] = view(y, off .+ (1:p)) 
                off += p
            end
        end
    end

    getduals!(μ, qp.hx, 1)
    getduals!(μ, qp.hu, 2)
    getduals!(ν, qp.gx, 1)
    getduals!(ν, qp.gu, 2)
    @assert off == num_duals(qp) == length(y)
    λ, μ, ν
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
        P[ix,ix] .= qp.Q[k]
        q[ix] .= qp.q[k]
    end
    for k in eachindex(qp.R)
        iu = getuind(qp, k)
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
        ic = ic .+ n
    end
    A,B,C,d
end

function build_state_equalities(qp::TOQP)
    Nc = num_state_equality(qp)
    Nx = num_states(qp) 
    Nu = num_controls(qp) 
    Hx = spzeros(Nc,Nx)
    hx = zeros(Nc)
    off = 0
    for k in eachindex(qp.Hx)
        ic = off .+ eachindex(qp.hx[k])
        ix = getxind(qp, k)
        Hx[ic,ix] .= qp.Hx[k]
        hx[ic] .= qp.hx[k]
        off += length(qp.hx[k])
    end
    Hx, hx
end

function build_state_inequalities(qp::TOQP)
    Nc = num_state_inequality(qp)
    Nx = num_states(qp) 
    Nu = num_controls(qp) 
    Gx = spzeros(Nc,Nx)
    gx = zeros(Nc)
    off = 0
    for k in eachindex(qp.Gx)
        ic = off .+ eachindex(qp.gx[k])
        ix = getxind(qp, k)
        Gx[ic,ix] .= qp.Gx[k]
        gx[ic] .= qp.gx[k]
        off += length(qp.gx[k])
    end
    Gx, gx
end

function build_control_equalities(qp::TOQP)
    Nc = num_control_equality(qp)
    Nx = num_states(qp) 
    Nu = num_controls(qp) 
    Hu = spzeros(Nc,Nu)
    hu = zeros(Nc)
    off = 0
    for k in eachindex(qp.Hu)
        ic = off .+ eachindex(qp.hu[k])
        iu = getuind(qp, k) .- Nx
        Hu[ic,iu] .= qp.Hu[k]
        hu[ic] .= qp.hu[k]
        off += length(qp.hu[k])
    end
    Hu, hu
end

function build_control_inequalities(qp::TOQP)
    Nc = num_control_inequality(qp)
    Nx = num_states(qp) 
    Nu = num_controls(qp) 
    Gu = spzeros(Nc,Nu)
    gu = zeros(Nc)
    off = 0
    for k in eachindex(qp.Gu)
        ic = off .+ eachindex(qp.gu[k])
        iu = getuind(qp, k) .- Nx
        Gu[ic,iu] .= qp.Gu[k]
        gu[ic] .= qp.gu[k]
        off += length(qp.gu[k])
    end
    Gu, gu
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

function setup_osqp(qp::TOQP; kwargs...)
    Nx = num_states(qp)
    Nu = num_controls(qp)

    P,q,c = build_objective(qp)
    A,B,C,d = build_dynamics(qp)
    Hx,hx = build_state_equalities(qp) 
    Hu,hu = build_control_equalities(qp) 
    Gx,gx = build_state_inequalities(qp) 
    Gu,gu = build_control_inequalities(qp) 
    Ahat = [
        [A+C B];  # dynamics
        [Hx spzeros(size(Hx,1), Nu)];  # state equalities
        [spzeros(size(Hu,1), Nx) Hu];  # control equalities
        [Gx spzeros(size(Gx,1), Nu)];  # state inequalities
        [spzeros(size(Gu,1), Nx) Gu];  # control inequalities
    ]
    lb = Vector([
        -d; 
        -hx; 
        -hu;
        fill(-Inf, num_inequality(qp));
    ])
    ub = Vector([
        -d; 
        -hx; 
        -hu;
        -gx;
        -gu;
    ])
    model = OSQP.Model()
    OSQP.setup!(model, P=P, q=q, A=Ahat, l=lb, u=ub; kwargs...)
    model
end

function setup_cosmo(qp::TOQP; kwargs...)
    Nx = num_states(qp)
    Nu = num_controls(qp)

    P,q,c = BilinearControl.build_objective(qp)
    A,B,C,d = build_dynamics(qp)
    Hx,hx = build_state_equalities(qp) 
    Hu,hu = build_control_equalities(qp) 
    Gx,gx = build_state_inequalities(qp) 
    Gu,gu = build_control_inequalities(qp) 

    Hx = [Hx spzeros(size(Hx,1), Nu)]
    Hu = [spzeros(size(Hu,1), Nx) Hu]
    Gx = [Gx spzeros(size(Gx,1), Nu)]
    Gu = [spzeros(size(Gu,1), Nx) Gu]

    A,B,C,d = BilinearControl.build_dynamics(qp)
    cons = [
        COSMO.Constraint([(A+C) B], d, COSMO.ZeroSet),
        COSMO.Constraint(Hx, hx, COSMO.ZeroSet),
        COSMO.Constraint(Hu, hu, COSMO.ZeroSet),
        COSMO.Constraint(-Gx, -gx, COSMO.Nonnegatives),
        COSMO.Constraint(-Gu, -gu, COSMO.Nonnegatives),
    ]
    model = COSMO.Model() 
    settings = COSMO.Settings(;kwargs...)
    COSMO.assemble!(model, P, q, cons; settings)
    model
end

function stack_linearized_constraints(constraints::TO.ConstraintList, con_inds, Z, input)

    # Evaluate the constraint Jacobian and value for each constraint
    jacvals = map(con_inds) do i
        con = constraints[i]
        inds = TO.constraintindices(constraints, i)
        sig = TO.functionsignature(constraints, i)
        diffmethod = TO.diffmethod(constraints, i)
        J = TO.gen_jacobian(con)
        c = zeros(RD.output_dim(con))
        vals = map(inds) do k
            TO.evaluate_constraint!(sig, con, c, Z[k]) 
            c
        end
        jacs = map(inds) do k
            TO.constraint_jacobian!(sig, diffmethod, con, J, c, Z[k]) 
            J
        end
        vals, jacs
    end

    # Separate Jacobian and constraint values
    vals = getindex.(jacvals, 1)
    jacs = getindex.(jacvals, 2)

    # Stack the values by time step
    N = length(Z) 
    h = map(1:N) do k
        con_inds_k = findall(con_inds) do i
            k in TO.constraintindices(constraints, i)
        end
        con_inds_k = con_inds[con_inds_k]
        inds = map(con_inds_k) do i
            inds = TO.constraintindices(constraints, i)
            searchsortedfirst(inds, k)
        end
        if !isempty(con_inds_k)
            return reduce(vcat, vals[i][j] for (i,j) in zip(eachindex(con_inds_k),inds))
        else
            return zeros(0)
        end
    end
    H = map(1:N) do k
        con_inds_k = findall(con_inds) do i
            k in TO.constraintindices(constraints, i)
        end
        con_inds_k = con_inds[con_inds_k]
        inds = map(con_inds_k) do i
            inds = TO.constraintindices(constraints, i)
            searchsortedfirst(inds, k)
        end
        if !isempty(con_inds_k)
            return reduce(vcat, jacs[i][j] for (i,j) in zip(eachindex(con_inds_k),inds))
        else
            if input == :state
                w = RD.state_dim(constraints, k)
            elseif input == :control
                w = RD.control_dim(constraints, k)
            else
                w = length(Z[k])
            end
            return zeros(0, w)
        end
    end
    H, h
end

function TOQP(prob::TO.Problem, Z=TO.get_trajectory(prob))
    N = TO.horizonlength(prob)
    Nx = sum(k->state_dim(prob, k), N)
    Nu = sum(k->control_dim(prob, k), N-1)

    # Get objective
    Q = map(c->Diagonal(Vector(diag(c.Q))), prob.obj.cost)
    R = map(c->Diagonal(Vector(diag(c.R))), prob.obj.cost)
    q = map(c->Vector(c.q), prob.obj.cost)
    r = map(c->Vector(c.r), prob.obj.cost)
    c = map(c->c.c, prob.obj.cost)

    # Get dynamics
    ABCd = map(1:N-1) do k
        model2 = TO.get_model(prob, k+1)
        model1 = TO.get_model(prob, k)
        n2,m2,p2 = RD.dims(model2)
        n1,m1,p1 = RD.dims(model1)
        J2 = zeros(p2, n2+m2)
        J1 = zeros(p1, n1+m1)
        y2 = zeros(p2)
        y1 = zeros(p1)
        RD.dynamics_error_jacobian!(RD.default_signature(model1), RD.default_diffmethod(model1),
            model1, J2, J1, y2, y1, Z[k+1], Z[k]
        )
        RD.dynamics_error!(RD.default_signature(model1), model1, y2, y1, Z[k+1], Z[k])
        
        A = J1[:,1:n1]
        B = J1[:,n1+1:n1+m1]
        C = J2[:,1:n2]
        d = y2
        @assert norm(J2[:,n2+1:n2+m2]) ≈ 0 "Dynamics cannot be a function of the next control."
        A,B,C,d
    end
    A = getindex.(ABCd, 1)
    B = getindex.(ABCd, 2)
    C = getindex.(ABCd, 3)
    d = getindex.(ABCd, 4)

    nx,nu = RD.dims(prob)
    D = [zeros(nx[k], nu[k]) for k = 1:N-1]

    ## Linearize constraints
    constraints = TO.get_constraints(prob)
    if !all(con->con isa Union{TO.StateConstraint,TO.ControlConstraint}, constraints)
        @warn "Problem contains at least one generic StageConstraint.\n" * 
              "Constraints coupling states and controls are not supported."
    end

    # Get equality constraints
    xeq_inds = findall(constraints) do con
        (TO.sense(con) isa TO.Equality) && (con isa TO.StateConstraint)
    end
    ueq_inds = findall(constraints) do con
        (TO.sense(con) isa TO.Equality) && (con isa TO.ControlConstraint)
    end

    Hx,hx = stack_linearized_constraints(constraints, xeq_inds, Z, :state)
    Hu,hu = stack_linearized_constraints(constraints, ueq_inds, Z, :control)

    # Get inequality constraints
    xineq_inds = findall(constraints) do con
        (TO.sense(con) isa TO.Inequality) && (con isa TO.StateConstraint)
    end
    uineq_inds = findall(constraints) do con
        (TO.sense(con) isa TO.Inequality) && (con isa TO.ControlConstraint)
    end

    Gx,gx = stack_linearized_constraints(constraints, xineq_inds, Z, :state)
    Gu,gu = stack_linearized_constraints(constraints, uineq_inds, Z, :control)

    # Trim off last control
    pop!(Hu)
    pop!(hu)
    pop!(Gu)
    pop!(gu)

    # Build QP
    n = state_dim(prob, 1)
    m = control_dim(prob, 1)
    T = eltype(c)
    TOQP{n,m,T}(Q, R, q, r, c, A, B, C, D, d, Vector(prob.x0), Hx, hx, Hu, hu, Gx, gx, Gu, gu)
end