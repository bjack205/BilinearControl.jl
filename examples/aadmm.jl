import Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra
using ForwardDiff
using Printf
using Random

struct ADMMProblem{T}
    Q::Diagonal{T,Vector{T}}
    R::Diagonal{T,Vector{T}}
    q::Vector{T}
    r::Vector{T}
    A0::Matrix{T}
    B0::Matrix{T}
    C::Vector{Matrix{T}}
    b::Vector{T}
    A::Matrix{T}
    B::Matrix{T}
end
function ADMMProblem(Q,R,q,r, A,B,C,b)
    Ahat = zero(A)
    Bhat = zero(B)
    ADMMProblem(Q,R,q,r,A,B,C,b,Ahat,Bhat)
end

eval_f(prob::ADMMProblem, x) = 0.5 * dot(x, prob.Q, x) + dot(prob.q, x)
eval_g(prob::ADMMProblem, z) = 0.5 * dot(z, prob.R, z) + dot(prob.r, z)
eval_c(prob::ADMMProblem, x, z) = prob.A0*x + prob.B0*z + sum(z[i]*prob.C[i]*x for i in eachindex(z)) - prob.b

function auglag(prob::ADMMProblem, x, z, λ, ρ)
    r = prob.b - prob.A*x - prob.B*z + λ / ρ
    eval_f(prob, x) + eval_g(prob, z) + ρ / 2 * r'r
end

function solvex(prob::ADMMProblem, x, z, λ, ρ)
    H = prob.Q + ρ * prob.A'prob.A
    b = prob.q + ρ * prob.A'*(prob.B0*z - prob.b + λ/ρ)
    return -(H\b)
end

function solvez(prob::ADMMProblem, x, z, λ, ρ)
    H = prob.R + ρ * prob.B'prob.B
    b = prob.r + ρ * prob.B'*(prob.A0*x - prob.b + λ/ρ)
    return -(H\b)
end

function dualupdate(prob::ADMMProblem, x, z, λ, ρ)
    λ + ρ * eval_c(prob, x, z) 
end

function updateA!(prob::ADMMProblem, z)
    prob.A .= prob.A0 .+ sum(prob.C[i] * z[i] for i in eachindex(z))
end

function updateB!(prob::ADMMProblem, x)
    prob.B .= prob.B0
    for i = 1:size(prob.B, 2)
        prob.B[:,i] .+= prob.C[i] * x
    end
end

function calcstep(H, y)
    α_sd = dot(y, y) / dot(H, y)
    α_mg = dot(H, y) / dot(H, H)
    if 2α_mg > α_sd
        α =  α_mg
    else
        α =  α_sd - α_mg / 2
    end
    α_cor = dot(H, y) / (norm(H) * norm(y))
    return α, α_cor
end

function solve(prob, x0, z0, λ0, ρ; 
        max_iters=100, 
        eps_primal=1e-3, 
        eps_dual=1e-2,
        Tf = 2,
        ϵ_cor = 0.2 
    )
    x = copy(x0)
    z = copy(z0)
    λ = copy(λ0)
    xn = copy(x)
    zn = copy(z)
    λn = copy(λ)
    
    x0 = copy(x)
    z0 = copy(z)
    λ0 = copy(λ)
    λhat0 = copy(λ)

    @printf("%8s %10s %10s %10s, %10s\n", "iter", "cost", "||r||", "||s||", "ρ")

    A,B,b = prob.A, prob.B, prob.b
    for iter = 1:max_iters
        updateA!(prob, z)
        xn .= solvex(prob, x, z, λ, ρ)
        updateB!(prob, xn)
        zn .= solvez(prob, xn, z, λ, ρ)
        updateA!(prob, zn)
        λn .= dualupdate(prob, xn, zn, λ, ρ) 

        r = eval_c(prob, xn, zn) 
        s = ρ * A'B*(zn - z)
        J = eval_f(prob, x) + eval_g(prob, z)
        @printf(
            "%8d %10.2g %10.2g %10.2g %10.2g\n", 
            iter, J, norm(r), norm(s), ρ
        )
        if norm(r) < eps_primal && norm(s) < eps_dual
            break
        end

        # Penalty update
        if iter % Tf == 0
            λhat = λ + ρ * eval_c(prob, xn, z) 
            Δλhat = λhat - λhat0
            ΔH = A*(xn - x0)
            α, α_cor = calcstep(ΔH, Δλhat)

            Δλ = λn - λ0
            ΔG = B*(zn - z0)
            β, β_cor = calcstep(ΔG, Δλ)

            if α_cor > ϵ_cor && β_cor > ϵ_cor
                ρ = sqrt(α*β)
            elseif α_cor > ϵ_cor && β_cor <= ϵ_cor
                ρ = α
            elseif α_cor <= ϵ_cor && β_cor > ϵ_cor
                ρ = β
            end
            x0 .= xn
            z0 .= zn
            λhat0 .= λn
        end

        # Set variables for next iteration
        x .= xn
        z .= zn
        λ .= λn
    end
    return x,z,λ
end


##
Random.seed!(2)
n,m,p = 10,5,5
prob = ADMMProblem(
    Diagonal(fill(1.0, n)), 
    Diagonal(fill(0.1, m)), 
    randn(n), 
    randn(m), 
    randn(p,n),
    randn(p,m),
    [randn(p,n)*1 for i = 1:m],
    randn(p)
)
x = randn(n)
z = randn(m)
λ = zeros(p)
ρ = 1.0
# solve(prob, x, z, λ, ρ, max_iters=4000, Tf=typemax(Int))
solve(prob, x, z, λ, ρ, max_iters=4000, ϵ_cor=0.2, Tf=5)