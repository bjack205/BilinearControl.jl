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

    # Parameters
    ρ::Ref{Float64}

    # Storage
    x::Vector{Float64}
    z::Vector{Float64}
    w::Vector{Float64}  # scaled duals

    x_prev::Vector{Float64}
    z_prev::Vector{Float64}
    w_prev::Vector{Float64}

    opts::ADMMOptions
end

function BilinearADMM(A,B,C,d, Q,q,R,r,c=0.0; ρ = 10.0)
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
    BilinearADMM{M}(Q, q, R, r, c, A, B, C, d, ρref, x, z, w, x_prev, z_prev, w_prev, opts)
end

setpenalty!(solver::BilinearADMM, rho) = solver.ρ[] = rho
getpenalty(solver::BilinearADMM) = solver.ρ[]

eval_f(solver::BilinearADMM, x) = 0.5 * dot(x, solver.Q, x) + dot(solver.q, x)
eval_g(solver::BilinearADMM, z) = 0.5 * dot(z, solver.R, z) + dot(solver.r, z)

getA(solver::BilinearADMM) = solver.A
getB(solver::BilinearADMM) = solver.B
getC(solver::BilinearADMM) = solver.C
getD(solver::BilinearADMM) = solver.d

function eval_c(solver::BilinearADMM, x, z)
    A, B, C = solver.A, solver.B, solver.C
    A*x + B*z + sum(z[i] * C[i]*x for i in eachindex(z)) + solver.d
end

function getAhat(solver::BilinearADMM, z)
    Ahat = copy(solver.A)
    for i in eachindex(z)
        Ahat .+= solver.C[i] * z[i]
    end
    return Ahat
end

function updateAhat!(solver::BilinearADMM, Ahat, z, nzinds)
    for (nzind0,nzind) in enumerate(nzinds[1])
        nonzeros(Ahat)[nzind] = nonzeros(solver.A)[nzind0]
    end
    for i in eachindex(z)
        for (nzind0, nzind) in enumerate(nzinds[i+1])
            nonzeros(Ahat)[nzind] = nonzeros(solver.C[i])[nzind0] * z[i]
        end
    end
end

function getBhat(solver::BilinearADMM, x)
    Bhat = copy(solver.B)
    for i in eachindex(solver.C)
        Bhat[:,i] .+= solver.C[i] * x
    end
    return Bhat
end

geta(solver::BilinearADMM, z) = solver.B*z + solver.d
getb(solver::BilinearADMM, x) = solver.A*x + solver.d

primal_residual(solver::BilinearADMM, x, z) = eval_c(solver, x, z)

function dual_residual(solver::BilinearADMM, x, z)
    ρ = getpenalty(solver)
    Ahat = getAhat(solver, solver.z_prev)
    Bhat = getBhat(solver, x)
    s = ρ * Ahat'*(Bhat*(z - solver.z_prev))
    return s
end

function dual_residual2(solver::BilinearADMM, x, z)
    ρ = getpenalty(solver)
    Ahat = getAhat(solver, solver.z_prev)
    Bhat = getBhat(solver, x)
    ρ*Ahat'Bhat*(z - solver.z_prev)
end

function solvex(solver::BilinearADMM, z, w)
    ρ = getpenalty(solver)
    p = length(w)
    Ahat = getAhat(solver, z)
    a = geta(solver, z)
    n = size(Ahat,2) 

    H = [solver.Q Ahat'; Ahat -I(p)*inv(ρ)]
    g = [solver.q; a + w]
    δ = -(H\g)
    x = δ[1:n]
    return x
end

function solvez(solver::BilinearADMM, x, w)
    R = solver.R
    ρ = getpenalty(solver)
    Bhat = getBhat(solver, x)
    b = getb(solver, x)
    H = R + ρ * Bhat'Bhat
    g = solver.r + ρ * Bhat'*(b + w)
    z = -(H\g)
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

function get_primal_tolerance(solver::BilinearADMM, x, z, w)
    ϵ_abs = solver.opts.ϵ_abs_primal
    ϵ_rel = solver.opts.ϵ_rel_primal
    p = length(w)
    Ahat = getAhat(solver, z)
    Bhat = getBhat(solver, x)
    √p*ϵ_abs + ϵ_rel * max(norm(Ahat * x), norm(Bhat * z), norm(solver.d))
end

function get_dual_tolerance(solver::BilinearADMM, x, z, w)
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
    for iter = 1:max_iters
        r = primal_residual(solver, x, z)
        s = dual_residual(solver, x, z)
        J = eval_f(solver, x) + eval_g(solver, z) + solver.c
        dz = norm(z - solver.z_prev)
        ϵ_primal = get_primal_tolerance(solver, x, z, w)
        ϵ_dual = get_primal_tolerance(solver, x, z, w)
        if iter > 1
            penaltyupdate!(solver, r, s)
        else
            s = NaN
        end
        ρ = getpenalty(solver)
        @printf("%8d %10.2g %10.2g %10.2g %10.2g %10.2g\n", iter, J, norm(r), norm(s), ρ, norm(dz))
        if norm(r) < ϵ_primal && norm(s) < ϵ_dual
            break
        end
        solver.z_prev .= z

        x .= solvex(solver, z, w)
        z .= solvez(solver, x, w)
        w .= updatew(solver, x, z, w)

    end
    tsolve = (time_ns() - tstart) / 1e9
    println("Solve took $tsolve seconds.")
    return x,z,w
end
