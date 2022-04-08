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
    ϵ_cor::Float64 = 0.2   # safeguarding threshold for AADMM
    Tf::Int = 2            # AADMM update rate
    penalty_update::Symbol = :threshold # options (:aadmm, :threshold)
end

Base.@kwdef mutable struct ADMMStats
    iterations::Int = 0
    ϵ_primal::Float64 = 0.0  # actual primal tolerance
    ϵ_dual::Float64 = 0.0    # actual dual tolerance
    αhat::Float64 = 0.0
    βhat::Float64 = 0.0
    α_cor::Float64 = 0.0
    β_cor::Float64 = 0.0
end

struct BilinearADMM{M}
    # Objective
    Q::Diagonal{Float64, Vector{Float64}}
    q::Vector{Float64}
    R::Diagonal{Float64, Vector{Float64}}
    r::Vector{Float64}
    c::Float64

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

    # AADMM storage
    y::Vector{Vector{Float64}}
    ŷ::Vector{Vector{Float64}}
    Δy::Vector{Float64}
    Δŷ::Vector{Float64}
    ΔH::Vector{Float64}             # A (x₊ - x)
    ΔG::Vector{Float64}             # B (z₊ - z)

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
    )
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
    Ahat = A + sum(C)
    Bhat = copy(B)
    x_ = randn(n)
    for i = 1:m
        Bhat[:,i] += C[i] * x_
    end

    # Set bounds
    xlo = boundvector(xmin, n)
    xhi = boundvector(xmax, n)
    ulo = boundvector(umin, m)
    uhi = boundvector(umax, m)

    # Caches for AADMM
    y = [zeros(p) for i = 1:2]
    ŷ = [zeros(p) for i = 1:2]
    Δy = zeros(p)
    Δŷ = zeros(p)
    ΔH = zeros(p)
    ΔG = zeros(p)

    # Precompute index caches for sparse matrices
    nzindsA = map(eachindex(C)) do i
        getnzindsA(Ahat, C[i])
    end
    pushfirst!(nzindsA, getnzindsA(Ahat, A))
    nzindsB = map(eachindex(C)) do i
        getnzindsB(Bhat, C[i], i)
    end
    pushfirst!(nzindsB, getnzindsA(Bhat, B))

    BilinearADMM{M}(
        Q, q, R, r, c, A, B, C, d, xlo, xhi, ulo, uhi, 
        ρref, Ahat, Bhat, nzindsA, nzindsB, x, z, w, x_prev, z_prev, w_prev, 
        y, ŷ, Δy, Δŷ, ΔH, ΔG,
        opts, ADMMStats() 
    )
end

setpenalty!(solver::BilinearADMM, rho) = solver.ρ[] = rho
getpenalty(solver::BilinearADMM) = solver.ρ[]

eval_f(solver::BilinearADMM, x) = 0.5 * dot(x, solver.Q, x) + dot(solver.q, x)
eval_g(solver::BilinearADMM, z) = 0.5 * dot(z, solver.R, z) + dot(solver.r, z)

getA(solver::BilinearADMM) = solver.A
getB(solver::BilinearADMM) = solver.B
getC(solver::BilinearADMM) = solver.C
getD(solver::BilinearADMM) = solver.d

hasstateconstraints(solver::BilinearADMM) = any(isfinite, solver.xlo) || any(isfinite, solver.xhi)
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

function solvex(solver::BilinearADMM, z, w; updateA=true)
    ρ = getpenalty(solver)
    p = length(w)
    Ahat = solver.Ahat
    # updateA && updateAhat!(solver, solver.Ahat, z)
    # Ahat = getAhat(solver, z)
    a = geta(solver, z)
    n = size(Ahat,2) 

    if hasstateconstraints(solver)
        model = OSQP.Model()
        P̂ = solver.Q + ρ * Ahat'Ahat
        q̂ = solver.q + ρ * Ahat'*(a + w)
        OSQP.setup!(model, P=P̂, q=q̂, A=sparse(I,n,n), l=solver.xlo, u=solver.xhi, verbose=false)
        res = OSQP.solve!(model)
        return res.x
    else
        H = [solver.Q Ahat'; Ahat -I(p)*inv(ρ)]
        g = [solver.q; a + w]
        δ = -(H\g)
        x = δ[1:n]
    end
    return x
end

function solvez(solver::BilinearADMM{M}, x, w) where M
    R = solver.R
    ρ = getpenalty(solver)
    # Bhat = updateBhat!(solver, solver.Bhat, x)
    # Bhat = getBhat(solver, x)
    Bhat = solver.Bhat
    b = getb(solver, x)
    H = R + ρ * Bhat'Bhat
    g = solver.r + ρ * Bhat'*(b + w)

    # Solve with OSQP
    m = length(g)
    if M <: AbstractSparseMatrix
        model = OSQP.Model()
        OSQP.setup!(model, P=H, q=vec(g), A=sparse(I,m,m), l=solver.ulo, u=solver.uhi, verbose=0)
        res = OSQP.solve!(model)
        z = res.x
    else
        @assert !hascontrolconstraints(solver) "Control constraints not supported for dense systems."
        z = -(H\g)
    end
    return z
end

function updatew(solver::BilinearADMM, x, z, w)
    return w + eval_c(solver, x, z)
end

function penaltyupdate!(solver::BilinearADMM, r, s)
    ρ = getpenalty(solver) 
    if solver.opts.penalty_update == :threshold
        τ_incr = solver.opts.τ_incr
        τ_decr = solver.opts.τ_decr
        μ = solver.opts.penalty_threshold
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
    elseif solver.opts.penalty_update == :aadmm
        x,z,w = solver.x, solver.z, solver.w
        # w = y / ρ
        x_prev = solver.x_prev
        z_prev = solver.z_prev
        w_prev = solver.w_prev
        ĉ = eval_c(solver, x, z_prev)
        # c = eval_c(solver, x, z)
        solver.y[1] .= w .* ρ              # y_k+1
        solver.ŷ[1] .= (w_prev .+ ĉ) .* ρ  # ŷ_k+1
        Ahat = solver.Ahat
        Bhat = solver.Bhat
        Δy = solver.Δy
        Δŷ = solver.Δŷ
        ΔH = solver.ΔH
        ΔG = solver.ΔG
        Δy .= solver.y[1] .- solver.y[2]
        Δŷ .= solver.ŷ[1] .- solver.ŷ[2]
        ΔH .= Ahat * (x .- x_prev)
        ΔG .= Bhat * (z .- z_prev)
        α_sd = dot(Δŷ, Δŷ) / dot(ΔH, Δŷ)
        α_mg = dot(ΔH, Δŷ) / dot(ΔH, ΔH)
        β_sd = dot(Δy, Δy) / dot(ΔG, Δy)
        β_mg = dot(ΔG, Δy) / dot(ΔG, ΔG)
        if 2α_mg > α_sd
            α = α_mg
        else
            α = α_sd - α_mg / 2
        end
        if 2β_mg > β_sd
            β = β_mg
        else
            β = β_sd - β_mg / 2
        end
        ϵ_cor = solver.opts.ϵ_cor
        α_cor = dot(ΔH,Δŷ) / (norm(ΔH) * norm(Δŷ))
        β_cor = dot(ΔG,Δy) / (norm(ΔG) * norm(Δy))
        if α_cor > ϵ_cor && β_cor > ϵ_cor
            ρ = sqrt(α*β)
        elseif α_cor > ϵ_cor && β_cor <= ϵ_cor
            ρ = α
        elseif α_cor <= ϵ_cor && β_cor > ϵ_cor
            ρ = β
        end
        setpenalty!(solver, ρ)

        solver.ŷ[2] .= solver.ŷ[1]
        solver.y[2] .= solver.y[1]
    end
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

function solve(solver::BilinearADMM, x0=solver.x, z0=solver.z, w0=zero(solver.w); 
        max_iters=100,
        verbose::Bool=false
    )
    x, z, w = solver.x, solver.z, solver.w
    x .= x0
    z .= z0
    w .= w0
    @printf("%8s %10s %10s %10s, %10s %10s\n", "iter", "cost", "||r||", "||s||", "ρ", "dz")
    solver.z_prev .= z 
    tstart = time_ns()
    updateBhat!(solver, solver.Bhat, x)
    s = NaN
    for iter = 1:max_iters
        updateAhat!(solver, solver.Ahat, z)
        x .= solvex(solver, z, w)                 # Bhat out of sync
        updateBhat!(solver, solver.Bhat, x)
        z .= solvez(solver, x, w)                 # Ahat out of sync
        updateAhat!(solver, solver.Ahat, z)
        w .= updatew(solver, x, z, w)

        r = primal_residual(solver, x, z)
        s = dual_residual(solver, x, z, updatemats=false)
        J = eval_f(solver, x) + eval_g(solver, z) + solver.c
        dz = norm(z - solver.z_prev)
        ϵ_primal = get_primal_tolerance(solver, x, z, w)
        ϵ_dual = get_dual_tolerance(solver, x, z, w)
        ρ = getpenalty(solver)
        @printf("%8d %10.2g %10.2g %10.2g %10.2g %10.2g\n", iter, J, norm(r), norm(s), ρ, norm(dz))
        if norm(r) < ϵ_primal && norm(s) < ϵ_dual
            break
        end

        penaltyupdate!(solver, r, s)

        solver.x_prev .= x
        solver.z_prev .= z
        solver.w_prev .= w

    end
    tsolve = (time_ns() - tstart) / 1e9
    println("Solve took $tsolve seconds.")
    return x,z,w
end
