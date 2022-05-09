
abstract type AbstractController end

resetcontroller!(::AbstractController, x0) = nothing
get_k(ctrl::AbstractController, t) = searchsortedfirst(gettime(ctrl), t)

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
end

function LQRController(A,B,Q,R, xeq, ueq)
    K = dlqr(A,B,Q,R)
    LQRController(Matrix(K), Vector(xeq), Vector(ueq))
end

function LQRController(model::RD.DiscreteDynamics, Q, R, xeq, ueq, dt)
    n,m = RD.dims(model)
    n != length(xeq) && throw(DimensionMismatch("Expected a state vector of length $n, got $(length(xeq))."))
    m != length(ueq) && throw(DimensionMismatch("Expected a control vector of length $m, got $(length(ueq))."))
    zeq = RD.KnotPoint{n,m}(xeq, ueq, 0.0, dt)
    J = zeros(n, n+m)
    xn = zeros(n)
    RD.jacobian!(RD.default_signature(model), RD.default_diffmethod(model), model, J, xn, zeq)
    A = J[:,1:n]
    B = J[:,n+1:n+m]
    LQRController(A,B,Q,R,xeq,ueq)
end

function getcontrol(ctrl::LQRController, x, t)
    dx = x - ctrl.xeq
    ctrl.ueq - ctrl.K*dx
end

function dlqr(A,B,Q,R; max_iters=200, tol=1e-6, verbose=false)
    P = Matrix(copy(Q))
    n,m = size(B)
    K = zeros(m,n)
    K_prev = copy(K)
    
    for k = 1:max_iters
        K .= (R + B'P*B) \ (B'P*A)
        P .= Q + A'P*A - A'P*B*K
        if norm(K-K_prev,Inf) < tol
            verbose && println("Converged in $k iterations")
            return K
        end
        K_prev .= K
    end
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
    for k = 1:N-1
        dt = times[k+1] - times[k]
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
    N = length(A) + 1
    P = [zeros(n,n) for k = 1:N]
    K = [zeros(m,n) for k = 1:N-1]
    
    P[end] .= dlqr(A[end], B[end], Q[end], R[end])[2]
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

function TVLQRController(model::RD.DiscreteDynamics, Q::Diagonal, R::Diagonal, Xref, Uref, times)
    Qs = [Q for t in times]
    Rs = [R for t in times]
    A,B = linearize(model, Xref, Uref, times)
    K,_ = tvlqr(A,B,Qs,Rs)
    TVLQRController(K, Xref, Uref, times)
end

gettime(ctrl::TVLQRController) = ctrl.time

function getcontrol(ctrl::TVLQRController, x, t)
    k = min(get_k(ctrl, t), length(ctrl.K))
    K = ctrl.K[k]
    dx = x - ctrl.xref[k]
    ctrl.uref[k] + K*dx
end

struct BilinearController{C} <: AbstractController
    model::EDMDModel
    ctrl::C  # controller defined on bilinear model
end

resetcontroller!(crl::BilinearController, x) = resetcontroller!(ctrl.ctrl, expandstate(ctrl.model, x))

function getcontrol(ctrl::BilinearController, x, t)
    z = expandstate(ctrl.model, x)
    u = getcontrol(ctrl.ctrl, z, t)
    return u
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

mutable struct StraightLineController{L,D} <: AbstractController
    model::L
    Q::Diagonal{Float64,Vector{Float64}}
    R::Diagonal{Float64,Vector{Float64}}
    tf::Float64
    dt::Float64
    dx::D
    Uref::Vector{Vector{Float64}}
    tvlqr::TVLQRController
    function StraightLineController(model::L, Q::Diagonal, R::Diagonal, 
                                    tf, dt, dx::D, u0::Vector{<:Real}) where {L<:RD.DiscreteDynamics,D}
        times = range(0,tf,step=dt)
        n = RD.state_dim(model)
        Xref = [zeros(n) for t in times]
        Uref = [u0 for t in times]
        tvlqr = TVLQRController(model, Q, R, Xref, Uref, times)
        new{L,D}(model, Q, R, tf, dt, dx, Uref, tvlqr)
    end
end

function resetcontroller!(ctrl::StraightLineController, x0, t)
    # Select a random new final position
    dx = rand(ctrl.dx) 
    Nsample = length(ctrl.Uref)
    Xref = map(range(0,1,length=Nsample)) do θ
        x0 .+ θ .* dx  # Assumes Euclidean states
    end
    times = t .+ range(0,ctrl.tf,step=ctrl.dt)
    ctrl.tvlqr = TVLQRController(ctrl.model, ctrl.Q, ctrl.R, Xref, ctrl.Uref, times)
    ctrl
end

getcontrol(ctrl::StraightLineController, x, t) = getcontrol(ctrl.tvlqr, x, t)