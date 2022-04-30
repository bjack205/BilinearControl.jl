
## Define some useful controllers
abstract type AbstractController end

resetcontroller!(::AbstractController, x0) = nothing

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

function dlqr(A,B,Q,R; max_iters=200, tol=1e-6)
    P = Matrix(copy(Q))
    n,m = size(B)
    K = zeros(m,n)
    K_prev = copy(K)
    
    for k = 1:max_iters
        K .= (R + B'P*B) \ (B'P*A)
        P .= Q + A'P*A - A'P*B*K
        if norm(K-K_prev,Inf) < tol
            println("Converged in $k iterations")
            return K
        end
        K_prev .= K
    end
    return K * NaN
end

function getcontrol(ctrl::LQRController, x, t)
    dx = x - ctrl.xeq
    ctrl.ueq - ctrl.K*dx
end

struct TVLQRController <: AbstractController
    K::Vector{Matrix{Float64}}
    xref::Vector{Vector{Float64}}
    uref::Vector{Vector{Float64}}
    time::Vector{Float64}
end

get_k(ctrl::TVLQRController, t) = searchsortedfirst(ctrl.time, t)

function getcontrol(ctrl::TVLQRController, x, t)
    k = get_k(ctrl, t)
    K = ctrl.K[k]
    dx = x - ctrl.xref[k]
    ctrl.uref[k] + K*dx
end

mutable struct ALTROController{D} <: AbstractController
    genprob::Function
    distribution::D
    tvlqr::TVLQRController
    prob::TO.Problem{Float64}
    opts::Altro.SolverOptions{Float64}
    function ALTROController(genprob::Function, distribution::D; opts=Altro.SolverOptions()) where D
        params = rand(distribution)
        prob = genprob(params...)
        n = RD.state_dim(prob,1)
        m = RD.control_dim(prob,1)
        N = TO.horizonlength(prob)
        K = [zeros(m,n) for k = 1:N-1]
        Xref = RD.states(prob)
        Uref = RD.controls(prob)
        time = RD.gettimes(prob)
        tvlqr = TVLQRController(K, Xref, Uref, time)
        new{D}(genprob, distribution, tvlqr, prob, opts)
    end
end

function resetcontroller!(ctrl::ALTROController, x0)
    params = rand(ctrl.distribution)
    prob = ctrl.genprob(params...)
    TO.set_initial_state!(prob, x0)
    solver = Altro.ALTROSolver(prob, ctrl.opts)
    solve!(solver)
    status = Altro.status(solver)
    if status != Altro.SOLVE_SUCCEEDED
        @warn "ALTRO solve failed."
    end
    X = RD.states(solver)
    U = RD.controls(solver)
    t = RD.gettimes(prob)
    K = Altro.get_ilqr(solver).K
    N = TO.horizonlength(prob)
    ctrl.prob = prob
    resize!(ctrl.tvlqr.K, N-1)
    resize!(ctrl.tvlqr.xref, N)
    resize!(ctrl.tvlqr.uref, N-1)
    resize!(ctrl.tvlqr.time, N)
    copyto!(ctrl.tvlqr.K, K)
    copyto!(ctrl.tvlqr.xref, X)
    copyto!(ctrl.tvlqr.uref, U)
    copyto!(ctrl.tvlqr.time, t)
    ctrl
end

function getcontrol(ctrl::ALTROController, x, t)
    getcontrol(ctrl.tvlqr, x, t)
end

## Simulation functions
function simulatewithcontroller(model::RD.DiscreteDynamics, ctrl::AbstractController, x0, 
                                tf, dt)
    simulatewithcontroller(RD.default_signature(model), model, ctrl, x0, tf, dt)
end

function simulatewithcontroller(sig::RD.FunctionSignature, 
                                model::RD.DiscreteDynamics, ctrl::AbstractController, x0, 
                                tf, dt)
    times = range(0, tf, step=dt)
    m = RD.control_dim(model)
    N = length(times)
    X = [copy(x0) for k = 1:N]
    U = [zeros(m) for k = 1:N-1]
    for k = 1:N-1 
        t = times[k]
        dt = times[k+1] - times[k]
        u = getcontrol(ctrl, X[k], t)
        U[k] = u
        RD.discrete_dynamics!(sig, model, X[k+1], X[k], u, times[k], dt)
    end
    X,U
end

function compare_models(sig::RD.FunctionSignature, model::EDMDModel,
                        model0::RD.DiscreteDynamics, x0, tf, U; 
                        doplot=false, inds=1:RD.state_dim(model0))
    N = length(U) + 1
    times = range(0, tf, length=N)
    dt = times[2]
    @show dt
    m = RD.control_dim(model)
    @assert m == RD.control_dim(model0)
    z0 = expandstate(model, x0) 
    Z = [copy(z0) for k = 1:N]
    X = [copy(x0) for k = 1:N]
    for k = 1:N-1
        RD.discrete_dynamics!(sig, model, Z[k+1], Z[k], U[k], times[k], dt)
        RD.discrete_dynamics!(sig, model0, X[k+1], X[k], U[k], times[k], dt)
    end
    X_bl = map(z->model.g * z, Z)
    X_bl, X
    if doplot
        X_mat = reduce(hcat, X_bl)
        X0_mat = reduce(hcat, X)
        p = plot(times, X0_mat[inds,:]', label="original", c=[1 2])
        plot!(p, times, X_mat[inds,:]', label="bilinear", c=[1 2], s=:dash)
        display(p)
    end
    sse = norm(X_bl - X)^2
    println("SSE: $sse")
    X_bl, X
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