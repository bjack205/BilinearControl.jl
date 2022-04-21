using Printf

Base.@kwdef mutable struct ADMMOptions
    ϵ_abs_primal::Float64 = 1e-3
    ϵ_rel_primal::Float64 = 1e-3
    ϵ_abs_dual::Float64 = 1e-3
    ϵ_rel_dual::Float64 = 1e-3
    relaxation_paramter::Float64 = 1.0  # no relaxation
    τ_incr::Float64 = 2.0
    τ_decr::Float64 = 2.0
    penalty_threshold::Float64 = 10.0
    x_solver::Symbol = :ldl
    z_solver::Symbol = :cholesky
    calc_x_residual::Bool = false
    calc_z_residual::Bool = false
end

Base.@kwdef mutable struct ADMMStats
    iterations::Int = 0
    cost::Vector{Float64} = sizehint!(Float64[], 1000)
    primal_residual::Vector{Float64} = sizehint!(Float64[], 1000)
    dual_residual::Vector{Float64} = sizehint!(Float64[], 1000)
    x_solve_residual::Vector{Float64} = sizehint!(Float64[], 1000)
    z_solve_residual::Vector{Float64} = sizehint!(Float64[], 1000)
    x_solve_iters::Vector{Int} = sizehint!(Int[], 1000)
    z_solve_iters::Vector{Int} = sizehint!(Int[], 1000)
end

function reset!(stats::ADMMStats)
    empty!(stats.cost)
    empty!(stats.x_solve_residual)
    empty!(stats.z_solve_residual)
    empty!(stats.x_solve_iters)
    empty!(stats.z_solve_iters)
    stats
end

struct BilinearADMM{M,A}
    # Objective
    Q::SparseMatrixCSC{Float64,Int}
    q::Vector{Float64}
    R::SparseMatrixCSC{Float64,Int} 
    r::Vector{Float64}
    c::Ref{Float64}

    # Bilinear Constraints
    A::M
    B::M
    C::Vector{M}
    d::M

    # Bound constraints
    xlo::Vector{Float64}  # lower state bound
    xhi::Vector{Float64}  # upper state bound
    ulo::Vector{Float64}  # lower control bound
    uhi::Vector{Float64}  # upper control bound

    # Parameters
    ρ::Ref{Float64}

    # Storage
    Ahat::M
    Bhat::M
    nzindsA::Vector{Vector{Int}}
    nzindsB::Vector{Vector{Int}}
    x::Vector{Float64}
    z::Vector{Float64}
    w::Vector{Float64}  # scaled duals

    x_prev::Vector{Float64}
    z_prev::Vector{Float64}
    w_prev::Vector{Float64}

    # Conic Constraints (using COSMO)
    constraints::Vector{COSMO.Constraint{Float64}}

    # Acceleration
    aa::A
    zw::Vector{Vector{Float64}}  # storage for acceleration

    opts::ADMMOptions
    stats::ADMMStats
end

boundvector(v::Real, n) = fill(v, n)
function boundvector(v::Vector, n)
    n % length(v) == 0 || throw(ArgumentError("Length of v ($(length(v))) must divide evenly into n ($n)."))
    repeat(v, n ÷ length(v))
end

function BilinearADMM(A,B,C,d, Q,q,R,r,c=0.0; ρ = 10.0, 
        xmin=-Inf, xmax=Inf,
        umin=-Inf, umax=Inf,
        acceleration::Type{AA} = COSMOAccelerators.EmptyAccelerator,
        constraints::Vector{COSMO.Constraint{Float64}} = COSMO.Constraint{Float64}[]
    ) where {AA<:COSMOAccelerators.AbstractAccelerator}
    n = size(A,2)
    m = size(B,2)
    p = length(d)
    x = zeros(n)
    z = zeros(m)
    w = zeros(p)
    x_prev = zero(x)
    z_prev = zero(z)
    w_prev = zero(w)
    ρref = Ref(ρ)
    opts = ADMMOptions() 
    M = typeof(A)

    # Build Ahat and Bhat
    Ahat = A + sum(c*rand() for c in C)
    Bhat = copy(B)
    x_ = randn(n)
    for i = 1:m
        Bhat[:,i] += C[i] * x_ * rand()
    end

    # Set bounds
    xlo = boundvector(xmin, n)
    xhi = boundvector(xmax, n)
    ulo = boundvector(umin, m)
    uhi = boundvector(umax, m)

    # Precompute index caches for sparse matrices
    nzindsA = map(eachindex(C)) do i
        getnzindsA(Ahat, C[i])
    end
    pushfirst!(nzindsA, getnzindsA(Ahat, A))
    nzindsB = map(eachindex(C)) do i
        getnzindsB(Bhat, C[i], i)
    end
    pushfirst!(nzindsB, getnzindsA(Bhat, B))

    # Acceleration
    aa = AA(m+p)
    zw = [zeros(m+p) for _ = 1:2]

    BilinearADMM{M,AA}(
        Q, q, R, r, Ref(c), A, B, C, d, xlo, xhi, ulo, uhi, 
        ρref, Ahat, Bhat, nzindsA, nzindsB, x, z, w, x_prev, z_prev, w_prev, 
        constraints,
        aa, zw,
        opts, ADMMStats()
    )
end

setpenalty!(solver::BilinearADMM, rho) = solver.ρ[] = rho
getpenalty(solver::BilinearADMM) = solver.ρ[]

eval_f(solver::BilinearADMM, x) = 0.5 * dot(x, solver.Q, x) + dot(solver.q, x)
eval_g(solver::BilinearADMM, z) = 0.5 * dot(z, solver.R, z) + dot(solver.r, z)
cost(solver::BilinearADMM, x, z) = eval_f(solver, x) + eval_g(solver, z) + solver.c[]

getA(solver::BilinearADMM) = solver.A
getB(solver::BilinearADMM) = solver.B
getC(solver::BilinearADMM) = solver.C
getD(solver::BilinearADMM) = solver.d

hasstateconstraints(solver::BilinearADMM) = any(isfinite, solver.xlo) || any(isfinite, solver.xhi) || !isempty(solver.constraints)
hascontrolconstraints(solver::BilinearADMM) = any(isfinite, solver.ulo) || any(isfinite, solver.uhi)

function eval_c(solver::BilinearADMM, x, z)
    A, B, C = solver.A, solver.B, solver.C
    # A*x + B*z + sum(z[i] * C[i]*x for i in eachindex(z)) + solver.d
    c = zeros(length(solver.d)) 
    c .= solver.d
    mul!(c, A, x, 1.0, 1.0)
    mul!(c, B, z, 1.0, 1.0)
    for i in eachindex(z)
        mul!(c, C[i], x, z[i], 1.0)
    end
    c
end

function getAhat(solver::BilinearADMM, z)
    Ahat = copy(solver.A)
    for i in eachindex(z)
        Ahat .+= solver.C[i] * z[i]
    end
    return Ahat
end

function updateAhat!(solver::BilinearADMM, Ahat::SparseMatrixCSC, z)
    Ahat .= 0
    nzinds = solver.nzindsA
    for (nzind0,nzind) in enumerate(nzinds[1])
        nonzeros(Ahat)[nzind] = nonzeros(solver.A)[nzind0]
    end
    for i in eachindex(z)
        for (nzind0, nzind) in enumerate(nzinds[i+1])
            nonzeros(Ahat)[nzind] += nonzeros(solver.C[i])[nzind0] * z[i]
        end
    end
    Ahat
end

function updateAhat!(solver::BilinearADMM, Ahat, z)
    Ahat .= solver.A
    for i in eachindex(z)
        axpy!(z[i], solver.C[i], Ahat)
    end
    return Ahat
end

function getBhat(solver::BilinearADMM, x)
    Bhat = copy(solver.B)
    for i in eachindex(solver.C)
        Bhat[:,i] .+= solver.C[i] * x
    end
    return Bhat
end

function updateBhat!(solver::BilinearADMM, Bhat::SparseMatrixCSC, x)
    nzinds = solver.nzindsB

    # Copy B to Bhat
    Bhat .= 0
    for (nzind0, nzind) in enumerate(nzinds[1])
        nonzeros(Bhat)[nzind] = nonzeros(solver.B)[nzind0]
    end

    # Copy C[i]*x to B[:,i]
    C = solver.C
    for i in axes(Bhat, 2)
        for c in axes(C[i],2)
            for j in nzrange(C[i], c)
                nzindB = nzinds[i+1][j]
                nonzeros(Bhat)[nzindB] += nonzeros(C[i])[j] * x[c]
            end
        end
    end
    Bhat
end

function updateBhat!(solver::BilinearADMM, Bhat, x)
    Bhat .= solver.B
    for i in eachindex(solver.C)
        Bi = view(Bhat, :, i)
        mul!(Bi, solver.C[i], x, 1.0, 1.0)
    end
    Bhat
end

geta(solver::BilinearADMM, z) = solver.B*z + solver.d
getb(solver::BilinearADMM, x) = solver.A*x + solver.d

primal_residual(solver::BilinearADMM, x, z) = eval_c(solver, x, z)

function dual_residual(solver::BilinearADMM, x, z; updatemats=true)
    ρ = getpenalty(solver)
    Ahat = solver.Ahat
    Bhat = solver.Bhat
    if updatemats
        updateAhat!(solver, Ahat, z)
        updateBhat!(solver, Bhat, x)
        # Ahat = getAhat(solver, solver.z_prev)
        # Bhat = getBhat(solver, x)
    end
    s = ρ * Ahat'*(Bhat*(z - solver.z_prev))
    return s
end

# function dual_residual2(solver::BilinearADMM, x, z)
#     ρ = getpenalty(solver)
#     Ahat = getAhat(solver, solver.z_prev)
#     Bhat = getBhat(solver, x)
#     ρ*Ahat'Bhat*(z - solver.z_prev)
# end

function solvex(solver::BilinearADMM, z, w)
    ρ = getpenalty(solver)
    p = length(w)
    Ahat = solver.Ahat
    # updateA && updateAhat!(solver, solver.Ahat, z)
    # Ahat = getAhat(solver, z)
    a = geta(solver, z)
    n = size(Ahat,2) 

    method = solver.opts.x_solver
    docalcres = solver.opts.calc_x_residual
    res = solver.stats.x_solve_residual
    iters = solver.stats.x_solve_iters
    local x
    # Primal methods
    if hasstateconstraints(solver) && method ∉ (:osqp, :cosmo)
        method_prev = method
        if !isempty(solver.constraints)
            method = :cosmo
        else
            method = :osqp
        end
        @warn "Can't use $method_prev method with state constraints.\n" * 
              "Switching to using $(uppercase(string(method)))."
        solver.opts.x_solver = method 
    end
    if method == :cosmo && isempty(solver.constraints)
        method = :osqp
        @warn "Can't use :cosmo method without specifying state constraints.\n" *
              "Switching to $(uppercase(string(method)))."
        solver.opts.x_solver = method
    end
    if method ∈ (:cholesky, :osqp, :cg, :cosmo)
        P̂ = solver.Q + ρ * Ahat'Ahat
        q̂ = solver.q + ρ * Ahat'*(vec(a) + w)

        if method == :cholesky
            F = cholesky(Symmetric(P̂))
            x = F\(-q̂)
            docalcres && push!(res, norm(P̂*x + q̂))
            push!(iters, 1)
        elseif method == :cg
            # x,ch = IterativeSolvers.cg!(solver.x, P̂, -q̂, log=true)
            x,ch = IterativeSolvers.cg(P̂, -q̂, log=true)
            push!(iters, IterativeSolvers.niters(ch))
            push!(res, ch[:resnorm][end])
        elseif method == :osqp
            model = OSQP.Model()
            P̂ = solver.Q + ρ * Ahat'Ahat
            q̂ = solver.q + ρ * Ahat'*(vec(a) + w)
            OSQP.setup!(model, P=P̂, q=q̂, A=sparse(I,n,n), l=solver.xlo, u=solver.xhi, verbose=false)
            OSQP.warm_start_x!(model, solver.x)
            res_osqp = OSQP.solve!(model)
            x = res_osqp.x
            docalcres && push!(res, norm(P̂*x + q̂))
            push!(iters, res_osqp.info.iter)
        elseif method == :cosmo
            model = COSMO.Model()
            opts = COSMO.Settings(eps_prim_inf=1e-8)
            COSMO.assemble!(model, P̂, q̂, solver.constraints, settings=opts)
            COSMO.warm_start_primal!(model, solver.x)
            res_cosmo = COSMO.optimize!(model)
            x = res_cosmo.x
            push!(iters, res_cosmo.iter)
            push!(res, res_cosmo.info.r_prim)
        end
    # Primal-dual methods
    else
        H = [solver.Q Ahat'; Ahat -I(p)*inv(ρ)]
        g = [solver.q; a + w]

        # TODO: add option to use QDLDL
        if method == :ldl
            δ = -(H\g)
            docalcres && push!(res, norm(H*δ + g))
            push!(iters, 1)
        elseif method == :minres
            δ,ch = IterativeSolvers.minres(H, -g, log=true)
            push!(solver.stats.x_solve_iters, IterativeSolvers.niters(ch))
            push!(res, ch[:resnorm][end])
        else
            error("Got unexpected method for x solve: $method.")
        end
        x = δ[1:n]
    end

    return x
end

function solvez(solver::BilinearADMM, x, w)
    R = solver.R
    ρ = getpenalty(solver)
    # Bhat = updateBhat!(solver, solver.Bhat, x)
    # Bhat = getBhat(solver, x)
    Bhat = solver.Bhat
    b = getb(solver, x)
    H = R + ρ * Bhat'Bhat
    g = solver.r + ρ * Bhat'*(b + w)

    method = solver.opts.z_solver
    docalcres = solver.opts.calc_z_residual
    res = solver.stats.z_solve_residual
    iters = solver.stats.z_solve_iters
    if hascontrolconstraints(solver) && method != :osqp
        @warn "Cannot solve with control bounds with $method method.\n" * 
              "Switching to OSQP."
        solver.opts.z_solver = :osqp
        method = :osqp
    end
    local z
    if method == :cholesky
        F = cholesky(Symmetric(H))
        z = F \ (-g)
        docalcres && push!(res, norm(H*z + g))
        push!(iters, 1)
    elseif method == :cg
        z,ch = IterativeSolvers.cg!(solver.z, H, -g, log=true)
        push!(iters, IterativeSolvers.niters(ch))
        push!(res, ch[:resnorm][end])
    elseif method == :osqp
        # Solve with OSQP
        m = length(g)
        model = OSQP.Model()
        OSQP.setup!(model, P=H, q=vec(g), A=sparse(I,m,m), l=solver.ulo, u=solver.uhi, verbose=0)
        OSQP.warm_start_x!(model, solver.z)
        res_osqp = OSQP.solve!(model)
        z = res_osqp.x
        docalcres && push!(res, norm(H*z + g))
        push!(iters, res_osqp.info.iter)
    end
    return z
end

function updatew(solver::BilinearADMM, x, z, w)
    return w + eval_c(solver, x, z)
end

function penaltyupdate!(solver::BilinearADMM, r, s)
    τ_incr = solver.opts.τ_incr
    τ_decr = solver.opts.τ_decr
    μ = solver.opts.penalty_threshold
    ρ = getpenalty(solver) 
    nr = norm(r)
    ns = norm(s)
    if nr > μ * ns  # primal residual too large
        ρ_new = ρ * τ_incr
    elseif ns > μ * nr  # dual residual too large
        ρ_new = ρ / τ_decr
    else
        ρ_new = ρ
    end
    setpenalty!(solver, ρ_new)
end

function get_primal_tolerance(solver::BilinearADMM, x=solver.x, z=solver.z, w=solver.w)
    ϵ_abs = solver.opts.ϵ_abs_primal
    ϵ_rel = solver.opts.ϵ_rel_primal
    p = length(w)
    # Ahat = getAhat(solver, z)
    # Bhat = getBhat(solver, x)
    Ahat = solver.Ahat
    Bhat = solver.Bhat
    √p*ϵ_abs + ϵ_rel * max(norm(Ahat * x), norm(Bhat * z), norm(solver.d))
end

function get_dual_tolerance(solver::BilinearADMM, x=solver.x, z=solver.z, w=solver.w)
    ρ = getpenalty(solver)
    ϵ_abs = solver.opts.ϵ_abs_dual
    ϵ_rel = solver.opts.ϵ_rel_dual
    Ahat = getAhat(solver, z)
    n = length(x) 
    √n*ϵ_abs + ϵ_rel * norm(ρ*Ahat'w)
end

function solve(solver::BilinearADMM{<:Any,AA}, x0=solver.x, z0=solver.z, w0=zero(solver.w); 
        max_iters=100,
        verbose::Bool=false
    ) where AA

    xn, zn, wn = solver.x, solver.z, solver.w
    x, z, w = solver.x_prev, solver.z_prev, solver.w_prev
    x .= x0
    z .= z0
    w .= w0
    n = length(x)
    m = length(z)
    p = length(w)

    reset!(solver.stats)

    verbose && @printf("%8s %10s %10s %10s, %10s %10s\n", "iter", "cost", "||r||", "||s||", "ρ", "dz")
    solver.z_prev .= z 
    tstart = time_ns()
    updateAhat!(solver, solver.Ahat, z)
    updateBhat!(solver, solver.Bhat, x)
    for iter = 1:max_iters
        xn .= solvex(solver, z, w)
        updateBhat!(solver, solver.Bhat, xn)
        zn .= solvez(solver, xn, w)
        updateAhat!(solver, solver.Ahat, zn)
        wn .= updatew(solver, xn, zn, w)

        # updateAhat!(solver, solver.Ahat, z)  # updates Ahat
        r = norm(primal_residual(solver, xn, zn))
        s = norm(dual_residual(solver, xn, zn, updatemats=false))
        # J = eval_f(solver, xn) + eval_g(solver, zn) + solver.c
        J = cost(solver, x, z)
        dz = norm(zn - z)
        ϵ_primal = get_primal_tolerance(solver, xn, zn, wn)
        ϵ_dual = get_dual_tolerance(solver, xn, zn, wn)
        if iter > 1
            penaltyupdate!(solver, r, s)
        else
            s = NaN
        end
        ρ = getpenalty(solver)

        # Record stats
        push!(solver.stats.cost, J)
        push!(solver.stats.primal_residual, r)
        push!(solver.stats.dual_residual, s)

        verbose && @printf("%8d %10.2g %10.2g %10.2g %10.2g %10.2g\n", iter, J, r, s, ρ, norm(dz))
        if r < ϵ_primal && s < ϵ_dual
            break
        end
        solver.z_prev .= z

        # Acceleration
        solver.zw[1] .= [zn; wn]
        COSMOAccelerators.update!(solver.aa, solver.zw[1], solver.zw[2], iter)
        COSMOAccelerators.accelerate!(solver.zw[1], solver.zw[2], solver.aa, iter)
        # TODO: add safeguarding on acceleration
        zn .= solver.zw[1][1:m]
        wn .= solver.zw[1][m+1:end]

        # Set variables for next iteration
        x .= xn
        z .= zn
        w .= wn
        solver.zw[2] .= solver.zw[1]

    end
    solver.stats.iterations = length(solver.stats.cost)
    tsolve = (time_ns() - tstart) / 1e9
    verbose && println("Solve took $tsolve seconds.")
    return xn,zn,wn
end
