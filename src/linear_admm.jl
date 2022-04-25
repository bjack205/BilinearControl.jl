struct TrajOptADMM{Nx,Nu,T,A}
    data::TOQP{Nx,Nu,T}
    x::Vector{Vector{T}}
    u::Vector{Vector{T}}
    y::Vector{Vector{T}}
    rho::Vector{T}

    constraint_vector::Vector{T}
    dual_residual::Vector{T}

    # Acceleration
    aa::A
    uy::Vector{Vector{Float64}}   # storage of stacked u,y pair

    opts::ADMMOptions
    stats::ADMMStats
end

function TrajOptADMM(data::TOQP{<:Any,<:Any,T}; 
        acceleration::Type{AA} = COSMOAccelerators.EmptyAccelerator, 
        opts...
    ) where {T,AA}
    n = state_dim(data)
    m = control_dim(data)
    N = nhorizon(data)
    Nx = N*n       # total number of states
    Nu = (N-1)*m   # total number of controls
    Nc = N*n       # total number of constraints (duals)

    x = [zeros(n) for k = 1:N]
    u = [zeros(m) for k = 1:N-1]
    y = [zeros(n) for k = 1:N]
    rho = ones(T,1) 
    c = zeros(Nc)
    s = zeros(Nx)
    opts = ADMMOptions(;opts...)
    stats = ADMMStats()

    aa = AA(Nu + Nc)
    uy = [zeros(Nu + Nc) for _ = 1:2]
    
    TrajOptADMM(data, x, u, y, rho, c, s, aa, uy, opts, stats)
end

getpenalty(solver::TrajOptADMM) = solver.rho[1]
setpenalty!(solver::TrajOptADMM, rho) = solver.rho[1] = rho
getstates(solver::TrajOptADMM) = solver.x
getcontrols(solver::TrajOptADMM) = solver.u
# getduals(solver::TrajOptADMM) = solver.λ
getscaledduals(solver::TrajOptADMM) = solver.y
iterations(solver::TrajOptADMM) = length(solver.stats.cost)

num_primals(solver::TrajOptADMM) = sum(length, solver.x) + sum(length, solver.u)
num_duals(solver::TrajOptADMM) = sum(length, solver.y)

for method in (:nhorizon, :state_dim, :control_dim)
    @eval $method(solver::TrajOptADMM) = $method(solver.data)
end

function eval_f(solver::TrajOptADMM{Nx,Nu,T}, x=getstates(solver)) where {Nx,Nu,T}
    J = zero(T)
    Q = solver.data.Q
    q = solver.data.q
    for k = 1:nhorizon(solver)
        J += dot(x[k], Q[k], x[k]) / 2 + dot(q[k], x[k])
    end
    J
end

function eval_g(solver::TrajOptADMM{Nx,Nu,T}, u=getcontrols(solver)) where {Nx,Nu,T}
    J = zero(T)
    R = solver.data.R
    r = solver.data.r
    for k = 1:nhorizon(solver)-1
        J += dot(u[k], R[k], u[k]) / 2 + dot(r[k], u[k])
    end
    J
end

function eval_c!(solver::TrajOptADMM, c, x=getstates(solver), u=getcontrols(solver))
    n = state_dim(solver)
    m = control_dim(solver)
    N = nhorizon(solver)
    
    Nc = N*n
    Nx = N*n
    Nu = (N-1)*m

    Â = spzeros(Nc, Nx)
    B̂ = spzeros(Nc, Nu)
    ĉ = zeros(Nc)

    # Extract data
    A = solver.data.A
    B = solver.data.B
    C = solver.data.C
    f = solver.data.d

    # Initial condition
    c[1:n] .= solver.data.x0 .- x[1]

    # Dynamics constraints
    ic = (1:n) .+ n
    for k = 1:N-1
        c[ic] .= A[k]*x[k] + B[k]*u[k] + f[k] + C[k]*x[k+1]
        ic = ic .+ n
    end
    c
end

function cost(solver::TrajOptADMM, x=getstates(solver), u=getcontrols(solver))
    eval_f(solver, x) + eval_g(solver, u) + sum(solver.data.c)
end

function auglag(solver::TrajOptADMM, x=getstates(solver), u=getcontrols(solver), 
                y=getscaledduals(solver), ρ=getpenalty(solver))
    J = cost(solver, x, u)
    J += ρ * normsquared(solver.data.x0 - x[1] + y[1]) / 2 - ρ * normsquared(y[1]) / 2

    A = solver.data.A
    B = solver.data.B
    C = solver.data.C
    f = solver.data.d
    for k = 1:nhorizon(solver)-1
        J += ρ * normsquared(A[k]*x[k] + B[k]*u[k] + f[k] + C[k]*x[k+1] + y[k+1]) / 2
        J -= ρ * normsquared(y[k+1]) / 2
    end
    J
end

function primal_residual(solver::TrajOptADMM, x=getstates(solver), u=getcontrols(solver); p=Inf)
    c = zeros(num_duals(solver))  # TODO: put this in the solver
    eval_c!(solver, c, x, u)
    norm(c, p)
end

function dual_residual(solver::TrajOptADMM, u, uprev, ρ=getpenalty(solver); p=Inf)
    n = state_dim(solver)
    N = nhorizon(solver)
    s = zeros(sum(length, solver.x))  # TODO: put this in the solver

    ix = 1:n
    A = solver.data.A
    B = solver.data.B
    C = solver.data.C
    for k = 1:N
        s[ix] .= 0
        if k < N
            s[ix] .+= A[k]'B[k]*(u[k] - uprev[k])
        end
        if k > 1
            s[ix] .+= C[k-1]'B[k-1]*(u[k-1] - uprev[k-1])
        end
        ix = ix .+ n
    end
    s .*= ρ
    norm(s, p)
end

function buildstatesystem(solver::TrajOptADMM, u=getcontrols(solver), 
                          y=getscaledduals(solver), ρ=getpenalty(solver))
    n = state_dim(solver)
    N = nhorizon(solver)

    Nx = N*n
    H = spzeros(Nx, Nx)
    g = zeros(Nx)
    ix1 = 1:n
    ix2 = ix1 .+ n

    Q = solver.data.Q
    q = solver.data.q
    A = solver.data.A
    B = solver.data.B
    C = solver.data.C
    f = solver.data.d
    ρ = getpenalty(solver)

    H[ix1,ix1] .= Q[1] + ρ * A[1]'A[1] + ρ*I(n)
    g[ix1] .= q[1] + ρ * A[1]'*(B[1]*u[1] + f[1] + y[2]) - ρ*(solver.data.x0 + y[1])
    for k = 2:N-1
        H[ix1, ix2] .= ρ * A[k-1]'C[k-1]
        H[ix2, ix1] .= ρ * C[k-1]'A[k-1]
        H[ix2,ix2] .= Q[k] + ρ * A[k]'A[k] + ρ * C[k-1]'C[k-1]
        g[ix2] .= q[k] + ρ * A[k]'*(B[k] * u[k] + f[k] + y[k+1]) + ρ * C[k-1]'*(B[k-1]*u[k-1] + f[k-1] + y[k])
        ix1 = ix1 .+ n
        ix2 = ix2 .+ n
    end
    let k = N
        H[ix1, ix2] .= ρ * A[k-1]'C[k-1]
        H[ix2, ix1] .= ρ * C[k-1]'A[k-1]
        H[ix2,ix2] .= Q[k] + ρ * C[k-1]'C[k-1]
        g[ix2] .= q[k] + ρ * C[k-1]'*(B[k-1]*u[k-1] + f[k-1] + y[k])
    end
    H,g
end

function buildcontrolsystem(solver::TrajOptADMM, x=getstates(solver), 
                          y=getscaledduals(solver), ρ=getpenalty(solver))
    m = control_dim(solver)
    N = nhorizon(solver)

    Nu = (N-1)*m
    H = spzeros(Nu, Nu)
    g = zeros(Nu)
    iu = 1:m

    R = solver.data.R
    r = solver.data.r
    A = solver.data.A
    B = solver.data.B
    C = solver.data.C
    f = solver.data.d
    ρ = getpenalty(solver)

    for k = 1:N-1
        H[iu,iu] .= R[k] + ρ * B[k]'B[k]
        g[iu] .= r[k] + ρ * B[k]'*(A[k] * x[k] + f[k] + C[k]*x[k+1] + y[k+1])
        iu = iu .+ m
    end
    H,g
end

function solvex!(solver::TrajOptADMM, x=getstates(solver), u=getcontrols(solver), 
                 y=getscaledduals(solver), ρ=getpenalty(solver))
    n = state_dim(solver)
    N = nhorizon(solver)
    H,g = buildstatesystem(solver, u, y, ρ)
    g .*= -1
    F = cholesky(Symmetric(H))
    Xn = F\g

    xn = reshape(Xn, n, N)
    for k = 1:N
        x[k] .= @view xn[:,k]
    end
    x
end

function solveu!(solver::TrajOptADMM, x=getstates(solver), u=getcontrols(solver), 
                y=getscaledduals(solver), ρ=getpenalty(solver))
    m = control_dim(solver)
    N = nhorizon(solver)
    H,g = buildcontrolsystem(solver, x, y, ρ)
    g .*= -1
    F = cholesky(H)
    Un = F\g

    un = reshape(Un, m, N-1)
    for k = 1:N-1
        u[k] .= @view un[:,k]
    end
    u
end

function updateduals!(solver::TrajOptADMM, x=getstates(solver), u=getcontrols(solver), 
                      y=getscaledduals(solver))
    y[1] .+= solver.data.x0 - x[1]

    A = solver.data.A
    B = solver.data.B
    C = solver.data.C
    f = solver.data.d
    for k = 1:nhorizon(solver)-1
        y[k+1] .+= A[k]*x[k] + B[k]*u[k] + f[k] + C[k]*x[k+1]
    end
    y
end

function buildadmmconstraint(solver::TrajOptADMM)
    n = state_dim(solver)
    m = control_dim(solver)
    N = nhorizon(solver)
    
    Nc = N*n
    Nx = N*n
    Nu = (N-1)*m

    Â = spzeros(Nc, Nx)
    B̂ = spzeros(Nc, Nu)
    ĉ = zeros(Nc)

    # Extract data
    A = solver.data.A
    B = solver.data.B
    C = solver.data.C
    f = solver.data.d

    # Initial condition
    Â[1:n,1:n] .= -I(n)
    ĉ[1:n] .= .-solver.data.x0

    # Dynamics constraints
    ic = (1:n) .+ n
    ix1 = 1:n
    ix2 = ix1 .+ n 
    iu = 1:m
    for k = 1:N-1
        Â[ic,ix1] .= A[k]
        Â[ic,ix2] .= C[k]
        B̂[ic,iu] .= B[k]
        ĉ[ic] .= .-f[k]

        ic = ic .+ n
        ix1 = ix1 .+ n
        ix2 = ix2 .+ n
        iu = iu .+ m
    end
    Â,B̂,ĉ
end

function solve(solver::TrajOptADMM{<:Any,<:Any,<:Any,AA}, x0=getstates(solver), u0=getcontrols(solver), 
        y0=getscaledduals(solver); 
        max_iters=100, 
        verbose=true,
    ) where {AA}
    n = state_dim(solver)
    m = control_dim(solver)
    N = nhorizon(solver)
    Nu = (N-1)*m
    Nc = N*n

    x = solver.x
    u = solver.u
    y = solver.y
    uprev = deepcopy(u)  # TODO: store this in the solver
    for k = 1:N
        x[k] .= x0[k]
        y[k] .= y0[k]
        if k < N
            u[k] .= u0[k]
        end
    end

    reset!(solver.stats)

    eps_primal = solver.opts.ϵ_abs_primal
    eps_dual = solver.opts.ϵ_abs_dual

    verbose = verbose > 0
    verbose && @printf("%8s %10s %10s %10s %10s\n", 
                       "iter", "cost", "||r||", "||s||", "ρ")
    r = primal_residual(solver, x, u)
    s = Inf 
    J = cost(solver, x, u)
    ρ = getpenalty(solver)
    iters = 0
    verbose && @printf("%8d %10.2g %10.2g %10.2g %10.2g\n",
                        iters, J, r, s, ρ)

    tstart = time_ns()
    for iter = 1:max_iters
        solvex!(solver, x, u, y, ρ)    # overwrites X
        solveu!(solver, x, u, y, ρ)    # overwrites u
        updateduals!(solver, x, u, y)  # overwrites y

        # Calculate residuals
        r = primal_residual(solver, x, u)
        s = dual_residual(solver, u, uprev, ρ)
        J = cost(solver, x, u)

        # Printing
        verbose && @printf("%8d %10.2g %10.2g %10.2g %10.2g\n",
                           iter, J, r, s, ρ)

        # Record stats
        push!(solver.stats.cost, J)
        push!(solver.stats.primal_residual, r)
        push!(solver.stats.dual_residual, s)

        # Check convergence
        if r < eps_primal && s < eps_dual
            iters = iter
            break
        end

        # Acceleration
        # TODO: add safeguarding
        if !(AA <: COSMOAccelerators.EmptyAccelerator)
            solver.uy[1] .= [vcat(u...); vcat(y...)]  # TODO: avoid this terrible thing...
            COSMOAccelerators.update!(solver.aa, solver.uy[1], solver.uy[2], iter)
            COSMOAccelerators.accelerate!(solver.uy[1], solver.uy[2], solver.aa, iter)
            uacc = reshape(view(solver.uy[1], 1:Nu), m, N-1)
            yacc = reshape(view(solver.uy[1], Nu+1:Nu+Nc), n, N)
            for k = 1:N
                y[k] .= @view yacc[:,k]
                if k < N
                    u[k] .= @view uacc[:,k]
                end
            end
            solver.uy[2] .= solver.uy[1]
        end

        # Update penalty
        penaltyupdate!(solver, r, s)

        # Update previous control
        for k = 1:N-1
            uprev[k] .= u[k]
        end
    end
    tsolve = (time_ns() - tstart) / 1e9
    verbose && println("Solve took $iters iterations $tsolve seconds.")
    x,u,y
end