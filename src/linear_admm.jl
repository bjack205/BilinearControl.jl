struct TrajOptADMM{Nx,Nu,T}
    data::TOQP{Nx,Nu,T}
    x::Vector{Vector{T}}
    u::Vector{Vector{T}}
    y::Vector{Vector{T}}
    rho::Vector{T}
end

function TrajOptADMM(data::TOQP{<:Any,<:Any,T}) where T
    n = state_dim(data)
    m = control_dim(data)
    N = nhorizon(data)
    x = [zeros(n) for k = 1:N]
    u = [zeros(m) for k = 1:N-1]
    y = [zeros(n) for k = 1:N]
    rho = ones(T,1) 
    TrajOptADMM(data, x, u, y, rho)
end

getpenalty(solver::TrajOptADMM) = solver.rho[1]
setpenalty!(solver::TrajOptADMM, rho) = solver.rho[1] = rho
getstates(solver::TrajOptADMM) = solver.X
getcontrols(solver::TrajOptADMM) = solver.U
# getduals(solver::TrajOptADMM) = solver.λ
getscaledduals(solver::TrajOptADMM) = solver.y

for method in (:nhorizon, :state_dim, :control_dim)
    @eval $method(solver::TrajOptADMM) = $method(solver.data)
end

function eval_f(solver::TrajOptADMM{Nx,Nu,T}, x=solver.data.X) where {Nx,Nu,T}
    J = zero(T)
    Q = solver.data.Q
    q = solver.data.q
    for k = 1:nhorizon(solver)
        J += dot(x[k], Q[k], x[k]) / 2 + dot(q[k], x[k])
    end
    J
end

function eval_g(solver::TrajOptADMM{Nx,Nu,T}, u=solver.data.U) where {Nx,Nu,T}
    J = zero(T)
    R = solver.data.R
    r = solver.data.r
    for k = 1:nhorizon(solver)-1
        J += dot(u[k], R[k], u[k]) / 2 + dot(r[k], u[k])
    end
    J
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

function buildstatesystem(solver::TrajOptADMM, u=getcontrols(solver), 
                          y=getscaledduals(solver), ρ=getpenalty(solver))
    n = state_dim(solver)
    N = nhorizon(solver)

    Nx = N*n
    Nc = N*n
    P = spzeros(Nc, Nx)
    p = zeros(Nc)
    ix1 = 1:n
    ix2 = ix1 .+ n

    Q = solver.data.Q
    q = solver.data.q
    A = solver.data.A
    B = solver.data.B
    C = solver.data.C
    f = solver.data.d
    ρ = getpenalty(solver)

    P[ix1,ix1] .= Q[1] + ρ * A[1]'A[1] + ρ*I(n)
    p[ix1] .= q[1] + ρ * A[1]'*(B[1]*u[1] + f[1] + y[2]) - ρ*(solver.data.x0 + y[1])
    for k = 2:N-1
        P[ix1, ix2] .= ρ * A[k-1]'C[k-1]
        P[ix2, ix1] .= ρ * C[k-1]'A[k-1]
        P[ix2,ix2] .= Q[k] + ρ * A[k]'A[k] + ρ * C[k-1]'C[k-1]
        p[ix2] .= q[k] + ρ * A[k]'*(B[k] * u[k] + f[k] + y[k+1]) + ρ * C[k-1]'*(B[k-1]*u[k-1] + f[k-1] + y[k])
        ix1 = ix1 .+ n
        ix2 = ix2 .+ n
    end
    let k = N
        P[ix1, ix2] .= ρ * A[k-1]'C[k-1]
        P[ix2, ix1] .= ρ * C[k-1]'A[k-1]
        P[ix2,ix2] .= Q[k] + ρ * C[k-1]'C[k-1]
        p[ix2] .= q[k] + ρ * C[k-1]'*(B[k-1]*u[k-1] + f[k-1] + y[k])
    end
    P,p
end

function solvex(solver::TrajOptADMM, U=getcontrols(solver), y=getscaledduals(solver))
end