using OSQP

## Define some useful controllers
abstract type AbstractController end

resetcontroller!(::AbstractController, x0) = nothing
function get_k(ctrl::AbstractController, t)
    times = gettime(ctrl)
    inds = searchsorted(times, t)
    t1 = times[inds.stop]
    t2 = times[min(inds.start, length(times))]
    if abs(t-t1) < abs(t-t2)
        return inds.stop
    else
        return min(inds.start, length(times))
    end
end

struct RandomController{D} <: AbstractController
    m::Int
    distribution::D
    function RandomController(model::RD.AbstractModel, 
            distribution::D=Normal()) where D 
        new{D}(RD.control_dim(model), distribution)
    end
end

function getcontrol(ctrl::RandomController, x, t)
    u = rand(ctrl.distribution, ctrl.m)
    return u
end

"""
Picks a random control value each time the controller is reset, 
and applies it as a constant.
"""
struct RandConstController{D} <: AbstractController
    distribution::D
    u::Vector{Float64}
    function RandConstController(distribution::D) where D
        u = rand(distribution)
        new{D}(distribution, u)
    end
end

resetcontroller!(ctrl::RandConstController, x0) = rand!(ctrl.distribution, ctrl.u)

getcontrol(ctrl::RandConstController, x, t) = ctrl.u

struct ZeroController <: AbstractController 
    m::Int
    ZeroController(m::Int) = new(m)
    ZeroController(model::RD.AbstractModel) = new(RD.control_dim(model))
end
getcontrol(ctrl::ZeroController, x, t) = zeros(ctrl.m)

struct LQRController <: AbstractController
    K::Matrix{Float64}
    xeq::Vector{Float64}
    ueq::Vector{Float64}
    state_error::Function
end

function LQRController(A,B,Q,R, xeq, ueq; state_error=(x,x0)->x-x0, kwargs...)
    K, = dlqr(A,B,Q,R; kwargs...)
    LQRController(Matrix(K), Vector(xeq), Vector(ueq), state_error)
end

function LQRController(model::RD.DiscreteDynamics, Q, R, xeq, ueq, dt; kwargs...)
    n,m = RD.dims(model)
    n != length(xeq) && throw(DimensionMismatch("Expected a state vector of length $n, got $(length(xeq))."))
    m != length(ueq) && throw(DimensionMismatch("Expected a control vector of length $m, got $(length(ueq))."))
    zeq = RD.KnotPoint{n,m}(xeq, ueq, 0.0, dt)
    J = zeros(n, n+m)
    xn = zeros(n)
    RD.jacobian!(RD.default_signature(model), RD.default_diffmethod(model), model, J, xn, zeq)
    A = J[:,1:n]
    B = J[:,n+1:n+m]
    LQRController(A,B,Q,R,xeq,ueq; kwargs...)
end

function getcontrol(ctrl::LQRController, x, t)
    # dx = x - ctrl.xeq
    dx = ctrl.state_error(x, ctrl.xeq)
    ctrl.ueq - ctrl.K*dx
end

function dlqr(A,B,Q,R; max_iters=1000, tol=1e-6, verbose=false)
    P = Matrix(copy(Q))
    n,m = size(B)
    K = zeros(m,n)
    K_prev = copy(K)
    
    for k = 1:max_iters
        K .= (R + B'P*B) \ (B'P*A)
        P .= Q + A'P*A - A'P*B*K
        if norm(K-K_prev,Inf) < tol
            verbose && println("Converged in $k iterations")
            return K,P
        end
        K_prev .= K
    end
    @warn "dlqr didn't converge in the given number of iterations ($max_iters)"
    return K,P
end

function linearize(model::RD.DiscreteDynamics, X, U, times)
    linearize(RD.default_signature(model), RD.default_diffmethod(model), model, X, U, times)
end

function linearize(sig, diffmethod, model::RD.DiscreteDynamics, X, U, times)
    N = length(X)
    n,m = RD.dims(model)
    J = zeros(n,n+m)
    xn = zeros(n)
    uN = length(U)
    A = [zeros(n,n) for k = 1:uN]
    B = [zeros(n,m) for k = 1:uN]
    for k = 1:uN
        dt = k < N ? times[k+1] - times[k] : times[k] - times[k-1]
        z = RD.KnotPoint(X[k], U[k], times[k], dt)
        # RD.jacobian!(sig, diffmethod, model, J, xn, X[k], U[k], times[k], dt)
        RD.jacobian!(sig, diffmethod, model, J, xn, z)
        A[k] = J[:,1:n]
        B[k] = J[:,n+1:end]
    end
    A,B
end

function tvlqr(A,B,Q,R)
    # initialize the output
    n,m = size(B[1])
    N = length(Q)
    P = [zeros(n,n) for k = 1:N]
    K = [zeros(m,n) for k = 1:N]
    
    Kf, Pf = dlqr(A[end], B[end], Q[end], R[end])
    K[end] .= Kf
    P[end] .= Pf
    for k = reverse(1:N-1) 
        K[k] .= (R[k] + B[k]'P[k+1]*B[k])\(B[k]'P[k+1]*A[k])
        P[k] .= Q[k] + A[k]'P[k+1]*A[k] - A[k]'P[k+1]*B[k]*K[k]
    end
    
    # return the feedback gains and ctg matrices
    return K,P
end

struct TVLQRController <: AbstractController
    K::Vector{Matrix{Float64}}
    xref::Vector{Vector{Float64}}
    uref::Vector{Vector{Float64}}
    time::Vector{Float64}
end

function TVLQRController(model::RD.DiscreteDynamics, Q, R, Xref, Uref, times)
    A, B = linearize(model, Xref, Uref, times)
    TVLQRController(A, B, Q, R, Xref, Uref, times)
end

function TVLQRController(model::RD.DiscreteDynamics, Qk::Diagonal, Rk::Diagonal, Qf::Diagonal, xref, uref, time)
    N = length(time)
    Q = [copy(Qk) for k = 1:N-1]
    push!(Q,Qf)
    R = [copy(Rk) for k = 1:N]
    println("hi")
    TVLQRController(model,Q,R,xref,uref,time)
end
function TVLQRController(A, B, Q, R, xref, uref, time)
    K, _ = tvlqr(A,B,Q,R)
    TVLQRController(K, xref, uref, time)
end
gettime(ctrl::TVLQRController) = ctrl.time

function getcontrol(ctrl::TVLQRController, x, t)
    k = min(get_k(ctrl, t), length(ctrl.K))
    # @show k,t
    K = ctrl.K[k]
    dx = x - ctrl.xref[k]
    ctrl.uref[k] + K*dx
end

# struct BilinearController{C} <: AbstractController
#     model::EDMDModel
#     ctrl::C  # controller defined on bilinear model
# end

# resetcontroller!(crl::BilinearController, x) = resetcontroller!(ctrl.ctrl, expandstate(ctrl.model, x))

# function getcontrol(ctrl::BilinearController, x, t)
#     z = expandstate(ctrl.model, x)
#     u = getcontrol(ctrl.ctrl, z, t)
#     return u
# end

@doc raw"""
    TrackingMPC

Solves the tracking linear MPC problem:

```math
\begin{align*} 
\underset{\delta x_{1:N}, \delta u_{1:N}}{\text{minimize}} &&& \frac{1}{2} \sum_{k=0}^N \delta x_k^T Q_k \delta x_k + \delta u_k^T R_k \delta u_k \\
\text{subject to} &&& A_k \delta x_k + B_k \delta u_k + f_k = \delta x_{k+1} \\
&&& \delta x_1 = x_\text{init}
\end{align*}
```
where ``f_k = f(\bar{x}_k, \bar{u}_k) - \bar{x}_{k+1}``.

Once the mpc horizon `Nt` goes beyond the end of the reference trajectory, it uses the last 
point of the reference trajectory.
"""
struct TrackingMPC{T,L} <: AbstractController
    # Reference trajectory
    Xref::Vector{Vector{T}}
    Uref::Vector{Vector{T}}
    Tref::Vector{T}

    # Dynamics 
    model::L
    A::Vector{Matrix{T}}
    B::Vector{Matrix{T}}
    f::Vector{Vector{T}}

    # Cost
    Q::Vector{Diagonal{T,Vector{T}}}
    R::Vector{Diagonal{T,Vector{T}}}
    q::Vector{Vector{T}}
    r::Vector{Vector{T}}

    # Storage
    K::Vector{Matrix{T}}
    d::Vector{Vector{T}}
    P::Vector{Matrix{T}}
    p::Vector{Vector{T}}
    X::Vector{Vector{T}}
    U::Vector{Vector{T}}
    λ::Vector{Vector{T}}
    Pqp::SparseMatrixCSC{T,Int}
    Aqp::SparseMatrixCSC{T,Int}
    lqp::Vector{T}
    uqp::Vector{T}
    osqp::OSQP.Model
    Nt::Vector{Int}  # horizon length
    state_error::Function
    opts::Dict{String,Any}
end

mpchorizon(mpc::TrackingMPC) = mpc.Nt[1]
setmpchorizon(mpc::TrackingMPC, N) = mpc.Nt[1] = N

function TrackingMPC(model::L, Xref, Uref, Tref, Qk, Rk, Qf; Nt=length(Xref),
        state_error=(x,x0)->(x-x0)
    ) where {L<:RD.DiscreteDynamics}
    @assert size(Rk,1) == length(Uref[1])
    if length(Uref) != length(Xref)
        error("State and control reference trajectories must be the same length (must provide a terminal control)")
    end
    N = length(Xref)
    n = length(Xref[1])
    m = length(Uref[1])
    A,B = linearize(model, Xref, Uref, Tref)
    f = map(1:N) do k
        dt = k < N ? Tref[k+1] - Tref[k] : Tref[k] - Tref[k-1] 
        xn = k < N ? copy(Xref[k+1]) : copy(Xref[k])
        Vector(RD.discrete_dynamics(model, Xref[k], Uref[k], Tref[k], dt) - xn)
    end

    Q = [copy(Qk) for k in 1:Nt-1]
    push!(Q, Qf)
    R = [copy(Rk) for k = 1:Nt] 
    q = [zeros(n) for k in 1:Nt]
    r = [zeros(m) for k in 1:Nt]

    K = [zeros(m, n) for k in 1:(Nt - 1)]
    d = [zeros(m) for k in 1:(Nt - 1)]
    P = [zeros(n, n) for k in 1:Nt]
    p = [zeros(n) for k in 1:Nt]
    X = [zeros(n) for k in 1:Nt]
    U = [zeros(m) for k in 1:Nt]
    λ = [zeros(n) for k in 1:Nt]

    Np = N*n + (N-1)*m
    Nd = N*n
    Pqp = spzeros(Np,Np)
    Aqp = spzeros(Nd,Np)
    lqp = zeros(Nd)
    uqp = zeros(Nd)
    osqp = OSQP.Model()
    opts = Dict{String,Any}()

    TrackingMPC(
        Xref, Uref, Tref, model, A, B, f, Q, R, q, r, K, d, P, p, X, U, λ, 
        Pqp,Aqp,lqp,uqp,osqp,
        [Nt], state_error,opts
    )
end

function backwardpass!(mpc::TrackingMPC, i)
    A, B = mpc.A, mpc.B
    Q, q = mpc.Q, mpc.q
    R, r = mpc.R, mpc.r
    P, p = mpc.P, mpc.p
    f = mpc.f
    N = length(mpc.Xref)
    Nt = mpchorizon(mpc)

    P[Nt] .= Q[Nt]
    p[Nt] .= q[Nt]

    for j in reverse(1:(Nt - 1))
        k = min(N, j + i - 1)
        P′ = P[j + 1]
        Ak = A[k]
        Bk = B[k]
        fk = f[k]

        Qx = q[j] + Ak' * (P′*fk + p[j + 1])
        Qu = r[j] + Bk' * (P′*fk + p[j + 1])
        Qxx = Q[j] + Ak'P′ * Ak
        Quu = R[j] + Bk'P′ * Bk
        Qux = Bk'P′ * Ak

        cholQ = cholesky(Symmetric(Quu))
        K = -(cholQ \ Qux)
        d = -(cholQ \ Qu)

        P[j] .= Qxx .+ K'Quu * K .+ K'Qux .+ Qux'K
        p[j] = Qx .+ K'Quu * d .+ K'Qu .+ Qux'd
        mpc.K[j] .= K
        mpc.d[j] .= d
    end
end

function forwardpass!(mpc::TrackingMPC, x0, i)
    A,B,f = mpc.A, mpc.B, mpc.f
    X,U,λ = mpc.X, mpc.U, mpc.λ
    K,d = mpc.K, mpc.d
    # X[1] = x0  - mpc.Xref[i]
    X[1] = mpc.state_error(x0, mpc.Xref[i])
    Nt = mpchorizon(mpc)
    for j = 1:Nt-1
        λ[j] = mpc.P[j]*X[j] .+ mpc.p[j]
        U[j] = K[j]*X[j] + d[j]
        X[j+1] = A[j]*X[j] .+ B[j]*U[j] .+ f[j]
    end
    λ[Nt] = mpc.P[Nt]*X[Nt] .+ mpc.p[Nt]
end

function solve!(mpc::TrackingMPC, x0, i=1)
    backwardpass!(mpc, i)
    forwardpass!(mpc, x0, i)
    return nothing
end

function cost(mpc::TrackingMPC)
    Nt = mpchorizon(mpc)
    mapreduce(+, 1:Nt) do k
        Jx = 0.5 * mpc.X[k]'mpc.Q[k]*mpc.X[k]
        Ju = 0.5 * mpc.U[k]'mpc.R[k]*mpc.U[k]
        Jx + Ju
    end
end

gettime(mpc::TrackingMPC) = mpc.Tref

function solve_lqr(Q,R,q,r,A,B,f,x0)
    n,m = size(B[1])
    Nt = length(q)
    K = [zeros(m,n) for k = 1:Nt-1]
    d = [zeros(m) for k = 1:Nt-1] 
    P = [zeros(n,n) for k = 1:Nt]
    p = [zeros(n) for k = 1:Nt]
    X = [zeros(n) for k = 1:Nt]
    U = [zeros(m) for k = 1:Nt-1] 
    λ = [zeros(n) for k = 1:Nt]

    P[Nt] .= Q[Nt]
    p[Nt] .= q[Nt]
    for j in reverse(1:(Nt - 1))
        P′ = P[j + 1]
        Ak = A[j]
        Bk = B[j]
        fk = f[j]

        Qx = q[j] + Ak' * (P′*fk + p[j + 1])
        Qu = r[j] + Bk' * (P′*fk + p[j + 1])
        Qxx = Q[j] + Ak'P′ * Ak
        Quu = R[j] + Bk'P′ * Bk
        Qux = Bk'P′ * Ak

        cholQ = cholesky(Symmetric(Quu))
        K[j] = -(cholQ \ Qux)
        d[j] = -(cholQ \ Qu)

        P[j] .= Qxx .+ K[j]'Quu * K[j] .+ K[j]'Qux .+ Qux'K[j]
        p[j] = Qx .+ K[j]'Quu * d[j] .+ K[j]'Qu .+ Qux'd[j]
    end
    X[1] .= x0
    for j = 1:Nt-1
        λ[j] = P[j]*X[j] .+ p[j]
        U[j] = K[j]*X[j] + d[j]
        X[j+1] = A[j]*X[j] .+ B[j]*U[j] .+ f[j]
    end
    λ[Nt] = P[Nt]*X[Nt] .+ p[Nt]
    return X,U,λ
end

function getcontrol(mpc::TrackingMPC, x, t)
    k = get_k(mpc, t) 
    Nt = mpchorizon(mpc)
    inds = k-1 .+ (1:Nt)
    Q = 
    X,U,λ = solve_lqr(Qr, Rr, qr, rr, Ar, Br, fr, x0)
    mpc.X .= X
    # mpc.U .= U
    mpc.λ .= λ
    return U[1] + Ur[1]
    # solve!(mpc, x, k)
    # return mpc.U[1] + mpc.Uref[k]
end

function gettrajectory(mpc::TrackingMPC, t)
    Nt = mpchorizon(mpc) 
    N = length(mpc.Xref)
    k = get_k(mpc, t) 
    X = map(1:Nt) do i
        j = min(k + i - 1, N)
        mpc.Xref[j] + mpc.X[i]
    end
    U = map(1:Nt) do i
        j = min(k + i - 1, N)
        mpc.Uref[j] + mpc.U[i]
    end
    X,U
end

struct BilinearMPC{L} <: AbstractController
    model::L
    Q::Diagonal{Float64,Vector{Float64}}
    R::Diagonal{Float64,Vector{Float64}}
    P::SparseMatrixCSC{Float64,Int}
    q::Vector{Float64}
    c::Vector{Float64}
    A::SparseMatrixCSC{Float64,Int}
    l::Vector{Float64}
    u::Vector{Float64}
    Xref::Vector{Vector{Float64}}
    Uref::Vector{Vector{Float64}}
    Yref::Vector{Vector{Float64}}
    times::Vector{Float64}
    Ā::Vector{Matrix{Float64}}
    B̄::Vector{Matrix{Float64}}
    d̄::Vector{Vector{Float64}}
    Nmpc::Int
end
function BilinearMPC(model::RD.DiscreteDynamics, Nmpc, x0, Q, R, Xref, Uref, times;
        x_min=fill(-Inf,length(x0)),
        x_max=fill(+Inf,length(x0)),
        u_min=fill(-Inf,length(Uref[1])),
        u_max=fill(+Inf,length(Uref[1])),
    )
    @assert length(Xref) == length(Uref) "State and control trajectory must have equal lengths. Must include terminal control."
    n,m = RD.dims(model)
    n0 = originalstatedim(model)
    N = length(Xref)
    Nx = Nmpc*n0
    Nu = (Nmpc-1)*m
    Ny = Nmpc*n
    Nc = Nmpc*n #+ (Nmpc-1)*n0 + Nu

    # QP Data
    if model isa EDMDModel
        G = model.g
    else
        G = I(n)
    end
    P = blockdiag(kron(sparse(I,Nmpc,Nmpc),G'Q*G), kron(sparse(I,Nmpc-1,Nmpc-1), R))
    q = zeros(Ny+Nu)
    c = zeros(Nmpc)
    A = sparse(-1.0*I, Nc, Ny+Nu)
    l = zeros(Nc)
    u = zeros(Nc)

    # Initial conditions
    A[1:n,1:n] .= -I(n)

    # Reference trajectory data
    Yref = map(x->expandstate(model, x), Xref)
    Ā,B̄ = linearize(model, Yref, Uref, times)
    d̄ = map(1:N-1) do k
        dt = times[k+1] - times[k]
        d = RD.discrete_dynamics(model, Yref[k], Uref[k], times[k], dt) - (Ā[k]*Yref[k] + B̄[k]*Uref[k])
        Vector(d)
    end

    # Bound constraints
    xlo = repeat(x_min, Nmpc - 1)
    xhi = repeat(x_max, Nmpc - 1)
    ulo = repeat(u_min, Nmpc - 1)
    uhi = repeat(u_max, Nmpc - 1)
    X = [kron(spdiagm(Nmpc-1,Nmpc,1=>ones(Nmpc-1)), G) spzeros(Nx-n0, Nu)]
    U = [spzeros(Nu,Ny) sparse(I,Nu,Nu)] 
    A = [A; X; U]
    l = [l; xlo; ulo]
    u = [u; xhi; uhi]
    
    ctrl = BilinearMPC(model, Q, R, P, q, c, A, l, u, Xref, Uref, Yref, collect(times), Ā, B̄, d̄, Nmpc)
    build_qp!(ctrl, Xref[1], 1)
end

gettime(ctrl::BilinearMPC) = ctrl.times

function build_qp!(ctrl::BilinearMPC, x0, kstart::Integer)
    model = ctrl.model
    n = RD.state_dim(model)

    # Indices from reference trajectory
    kinds = kstart - 1 .+ (1:ctrl.Nmpc)

    # Extract out info from controller
    Xref = ctrl.Xref
    Uref = ctrl.Uref
    Yref = ctrl.Yref
    Ā,B̄,d̄ = ctrl.Ā, ctrl.B̄, ctrl.d̄

    if model isa EDMDModel
        G = model.g
    else
        G = I(n)
    end
    Q,R = ctrl.Q, ctrl.R

    # Some useful sizes
    n,m = RD.dims(model)
    Nmpc = ctrl.Nmpc
    N = length(Xref)
    Nu = (Nmpc-1) * m
    Ny = Nmpc * n 

    # Build Cost
    q = reshape(view(ctrl.q,1:Ny), n, :)
    r = reshape(view(ctrl.q,Ny+1:Ny+Nu), m, :)
    for i = 1:Nmpc
        k = min(kinds[i], N)
        q[:,i] .= -G'Q*Xref[k]
        ctrl.c[i] = 0.5 * dot(Xref[k], Q, Xref[k])
        if i < Nmpc
            r[:,i] .= -R*Uref[k]
            ctrl.c[i] += 0.5 * dot(Uref[k], R, Uref[k])
        end
    end

    # Build dynamics
    A,l,u = ctrl.A, ctrl.l, ctrl.u
    ic = 1:n
    # D[ic,1:n] .= -I(n)
    y0 = expandstate(model, x0)
    l[ic] .= .-y0
    u[ic] .= .-y0
    ic = ic .+ n
    for i = 1:Nmpc-1
        k = min(kinds[i], N-1)
        ix1 = (i-1)*n .+ (1:n)
        iu1 = (i-1)*m + Ny .+ (1:m)
        ix2 = ix1 .+ n
        A[ic,ix1] .= Ā[k]
        A[ic,iu1] .= B̄[k]
        # A[ic,ix2] .= -I(n)
        l[ic] .= -d̄[k]
        u[ic] .= -d̄[k]

        ic = ic .+ n
    end

    ctrl
end

function solveqp!(ctrl::BilinearMPC, x, t; x0=nothing)
    k = get_k(ctrl, t)
    build_qp!(ctrl, x, k)
    model = OSQP.Model()
    n = RD.state_dim(ctrl.model)
    if norm(ctrl.l[1:n]) > 1e8
        error("Large state detected")
    end
    OSQP.setup!(model, P=ctrl.P, q=ctrl.q, A=ctrl.A, l=ctrl.l, u=ctrl.u, verbose=false)
    if !isnothing(x0)
        OSQP.warm_start_x!(model, x0)
    end
    res = OSQP.solve!(model)  # TODO: implement warm-starting
    if res.info.status != :Solved
        @warn "OSQP didn't solve! Got status $(res.info.status)"
    end
    res.x
end

function getcontrol(ctrl::BilinearMPC, x, t)
    n,m = RD.dims(ctrl.model)
    Ny = ctrl.Nmpc * n 
    uind = Ny .+ (1:m)  # indices of first control
    z = solveqp!(ctrl, x, t)
    return z[uind]
end

function updatereference!(ctrl::BilinearMPC, Xref, Uref, tref)
    @assert length(Xref) == length(ctrl.Xref) "Changing length of reference trajectory not supported yet."
    Yref = map(x->expandstate(model, x), Xref)
    Ā,B̄ = linearize(model, Yref, Uref, times)
    d̄ = map(1:N-1) do k
        dt = times[k+1] - times[k]
        RD.discrete_dynamics(model, Yref[k], Uref[k], times[k], dt) - (Ā[k]*Yref[k] + B̄[k]*Uref[k])
    end
    for k = 1:N
        ctrl.Ā[k] .= Ā[k]
        ctrl.B̄[k] .= B̄[k]
        ctrl.d̄[k] .= d̄[k]
        ctrl.Xref[k] .= Xref[k]
        ctrl.Uref[k] .= uref[k]
    end
    copyto!(ctrl.tref, tref)
    ctrl
end

## Simulation functions
function simulatewithcontroller(model::RD.DiscreteDynamics, ctrl::AbstractController, x0, 
                                tf, dt; kwargs...)
    simulatewithcontroller(RD.default_signature(model), model, ctrl, x0, tf, dt; kwargs...)
end

function simulatewithcontroller(sig::RD.FunctionSignature, 
                                model::RD.DiscreteDynamics, ctrl::AbstractController, x0, 
                                tf, dt; printrate=false, umod=identity)
    times = range(0, tf, step=dt)
    m = RD.control_dim(model)
    N = length(times)
    X = [copy(x0)*NaN for k = 1:N]
    U = [zeros(m)*NaN for k = 1:N-1]
    X[1] = x0
    tstart = time_ns()
    for k = 1:N-1 
        t = times[k]
        # dt = times[k+1] - times[k]
        try
            u = getcontrol(ctrl, X[k], t)
            U[k] .= umod(u)
            RD.discrete_dynamics!(sig, model, X[k+1], X[k], U[k], times[k], dt)
        catch e
            break
        end
    end
    t_total_s = (time_ns() - tstart) / 1e9
    rate = (N-1) / t_total_s
    if printrate
        println("Average controller rate = $rate Hz")
    end
    X,U,times
end

# function bilinearerror(model::EDMDModel, X, U)
#     dt = model.dt
#     map(CartesianIndices(U)) do cind
#         k = cind[1]
#         i = cind[2]

#         uk = U[k,i]
#         zk = expandstate(model, X[k,i]) 
#         zn = zero(zk)
#         t = (k-1)*dt
#         RD.discrete_dynamics!(model, zn, zk, uk, t, dt)
#         xn = originalstate(model, zn) 
#         xn - X[k+1]
#     end
# end


function simulate(model::RD.DiscreteDynamics, U, x0, tf, dt)
    N = round(Int, tf / dt) + 1
    times = range(0, tf, length=N)
    @assert length(U) in [N,N-1]
    X = [copy(x0) for k = 1:N]
    sig = RD.default_signature(model)
    for k = 1:N-1 
        dt = times[k+1] - times[k]
        RD.discrete_dynamics!(sig, model, X[k+1], X[k], U[k], times[k], dt)
    end
    X,U,times
end

# function compare_models(sig::RD.FunctionSignature, model::EDMDModel,
#                         model0::RD.DiscreteDynamics, x0, tf, U; 
#                         doplot=false, inds=1:RD.state_dim(model0))
#     N = length(U) + 1
#     times = range(0, tf, length=N)
#     dt = times[2]
#     @show dt
#     m = RD.control_dim(model)
#     @assert m == RD.control_dim(model0)
#     z0 = expandstate(model, x0) 
#     Z = [copy(z0) for k = 1:N]
#     X = [copy(x0) for k = 1:N]
#     for k = 1:N-1
#         RD.discrete_dynamics!(sig, model, Z[k+1], Z[k], U[k], times[k], dt)
#         RD.discrete_dynamics!(sig, model0, X[k+1], X[k], U[k], times[k], dt)
#     end
#     X_bl = map(z->model.g * z, Z)
#     X_bl, X
#     if doplot
#         X_mat = reduce(hcat, X_bl)
#         X0_mat = reduce(hcat, X)
#         p = plot(times, X0_mat[inds,:]', label="original", c=[1 2])
#         plot!(p, times, X_mat[inds,:]', label="bilinear", c=[1 2], s=:dash)
#         display(p)
#     end
#     sse = norm(X_bl - X)^2
#     println("SSE: $sse")
#     X_bl, X
# end

function create_data(model::RD.DiscreteDynamics, ctrl::AbstractController, 
                     x0_sampler, num_traj, xe, tf, dt; 
                     sig=RD.InPlace(), thresh=0.1, max_samples=3*num_traj
    )
    N = round(Int, tf/dt) + 1
    X_sim = Matrix{Vector{Float64}}(undef, N, num_traj)
    U_sim = Matrix{Vector{Float64}}(undef, N-1, num_traj)
    j = 0
    for i = 1:max_samples
        x0 = rand(x0_sampler)
        resetcontroller!(ctrl, x0)
        X,U = simulatewithcontroller(sig, model, ctrl, x0, tf, dt)
        did_stabilize = norm(X[end] - xe) < thresh
        if did_stabilize
            j += 1
            X_sim[:,j] = X
            U_sim[:,j] = U
        end
        if j == num_traj
            break
        end
    end
    X_sim[:,1:j], U_sim[:,1:j]
end

function create_data(model::RD.DiscreteDynamics, ctrl::AbstractController, 
                              initial_conditions, tf, dt; sig=RD.InPlace())
    num_traj = length(initial_conditions)
    N = round(Int, tf/dt) + 1
    X_sim = Matrix{Vector{Float64}}(undef, N, num_traj)
    U_sim = Matrix{Vector{Float64}}(undef, N-1, num_traj)
    for i = 1:num_traj
        resetcontroller!(ctrl, initial_conditions[i])
        X,U = simulatewithcontroller(sig, model, ctrl, initial_conditions[i], tf, dt)
        X_sim[:,i] = X
        U_sim[:,i] = U
    end
    X_sim, U_sim
end

function calc_error(model::RD.DiscreteDynamics, X, U, dt)
    map(CartesianIndices(U)) do cind
        k = cind[1]  # time index
        j = cind[2]  # trajectory index
        xn_true = X[k+1,j]
        xn_nominal = RD.discrete_dynamics(model, X[k,j], U[k,j], (k-1)*dt, dt)
        Vector(xn_true - xn_nominal)
    end
end