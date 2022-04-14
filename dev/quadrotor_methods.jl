import Pkg; Pkg.activate(@__DIR__)

using BilinearControl
using BilinearControl.Problems
using COSMO
import RobotDynamics as RD
import TrajectoryOptimization as TO
using LinearAlgebra
using StaticArrays
using Statistics
using Rotations
using BilinearControl.Problems: qrot, skew
using SparseArrays
using Test
using BilinearControl: getA, getB, getC, getD
using BenchmarkTools

model = QuadrotorRateLimited()
n,m = RD.dims(model)
h = 0.02

function Ax!(y,x1,x2,h)
    for i = 1:3
        y[i] = (x1[12+i] + x2[12+i]) * h /2 + x1[i] - x2[i]
        y[15+i] = x2[15+i] * h 
    end
    for i = 4:15
        y[i] = x1[i] - x2[i]
    end
    return y
end

function Bu!(y,u1,u2,h)
    for i = 1:15
        y[i] = 0.0
    end
    for i = 1:3 
        y[15+i] = u1[i] - u2[i]
    end
    y
end

function Cxu!(y,x1,u1,x2,u2,h,mass)
    for i = 1:3
        y[i] = 0.0
        y[3+i] = h*(x1[6+i] + x2[6+i]) * u1[3] / 2 - h*(x1[9+i] + x2[9+i]) * u1[2] / 2
        y[6+i] = h*(x1[9+i] + x2[9+i]) * u1[1] / 2 - h*(x1[3+i] + x2[3+i]) * u1[3] / 2
        y[9+i] = h*(x1[3+i] + x2[3+i]) * u1[2] / 2 - h*(x1[6+i] + x2[6+i]) * u1[1] / 2
        y[12+i] = h*(x1[9+i] + x2[9+i]) * u1[4] / 2mass
        y[15+i] = 0.0
    end
    y
end

function Aty!(x, y, h)
    n = 18
    for i = 1:15
        x[i] = y[i]
        x[n+i] = -y[i]
    end
    for i = 1:3
        x[15+i] = 0.0
        x[n+15+i] = 0.0
        x[12+i] += y[i] * h/2
        x[n+12+i] += y[i] * h/2
        x[n+15+i] += y[15+i] * h
    end
    x
end

function Bty!(u, y, h)
    m = 4
    for i = 1:3
        u[i] = y[15+i]
        u[m+i] = -y[15+i]
    end
    u[4] = 0.0
    u[m+4] = 0.0
    u
end

function Ctuy!(x, u, y, h, mass)
    for i = 1:3
        x[i] = 0.0
        x[i+n] = 0.0
        
        x[3+i] = y[9+i] * h / 2 * u[2] - y[6+i] * h / 2 * u[3]
        x[6+i] = y[3+i] * h / 2 * u[3] - y[9+i] * h / 2 * u[1]
        x[9+i] = y[6+i] * h / 2 * u[1] - y[3+i] * h / 2 * u[2] +  y[12+i] * h / 2mass * u[4]

        x[3+i+n] = y[9+i] * h / 2 * u[2] - y[6+i] * h / 2 * u[3]
        x[6+i+n] = y[3+i] * h / 2 * u[3] - y[9+i] * h / 2 * u[1]
        x[9+i+n] = y[6+i] * h / 2 * u[1] - y[3+i] * h / 2 * u[2] +  y[12+i] * h / 2mass * u[4]

        x[12+i] = 0.0
        x[12+i+n] = 0.0

        x[15+i] = 0.0
        x[15+i+n] = 0.0
    end
    x
end

function Ctxy!(u, x, y, h, mass)
    u .= 0
    n = 18
    for i = 1:3
        u[1] += h/2 * y[6+i] * x[9+i] + h/2 * y[6+i] * x[9+i+n]
        u[1] += -h/2 * y[9+i] * (x[6+i] + x[6+i+n])
        u[2] += -h/2 * y[3+i] * (x[9+i] + x[9+i+n])
        u[2] += h/2 * y[9+i] * (x[3+i] + x[3+i+n])
        u[3] += h/2 * y[3+i] * (x[6+i] + x[6+i+n])
        u[3] += -h/2 * y[6+i] * (x[3+i] + x[3+i+n])
        u[4] += h/2mass * y[12+i] * (x[9+i] + x[9+i+n])
    end
    u
end

mass = model.mass
A,B,C,D = getA(model,h), getB(model,h), getC(model,h), getD(model,h)
x1,u1 = Vector.(rand(model))
x2,u2 = Vector.(rand(model))
x12 = [x1; x2]
u12 = [u1; u2]
y = zeros(n)
Ax!(y, x1, x2, h) ≈ A*x12 
Bu!(y, u1, u2, h) ≈ B*u12
Cxu!(y, x1, u1, x2, u2, h, mass) ≈ sum(u12[i] * C[i] * x12 for i = 1:2m)
y = randn(n)
Aty!(x12, y, h) ≈ A'y
Bty!(u12, y, h) ≈ B'y
y = randn(n)
x12 = [x1; x2]
Ctuy!(x12, u12, y, h, mass) ≈ sum(u12[i] * C[i]'y for i = 1:2m)   # Ahat
u12 = [u1; u2]
Ctxy!(u12, x12, y, h, mass) ≈ hcat([C[i]*x12 for i = 1:2m]...)'y  # Bhat

@btime Ax!($y, $x1, $x2, $h)
@btime mul!($y, $A, $x12)
@btime Bu!($y, $u1, $u2, $h)
@btime mul!($y, $B, $u12)
@btime Cxu!($y, $x1, $u1,$x2,$u2, $h, $mass)
@btime sum($u12[i] * $C[i] * x12 for i = 1:(2*$m)) 
@btime Aty!($x12, $y, $h)
@btime mul!($x12, $(A'), $y)

@btime Bty!($u12, $y, $h)
@btime mul!($u12, $(B'), $y)
@btime Ctxy!($u12, $x12, $y, $h, $mass)
Cbar_x = hcat([C[i] * x12 for i = 1:2m]...)
@btime mul!($u12, $(Cbar_x'), $y)


## Bilinear Constraint functions
function Abarx!(y, x, h, n, N)
    for i = 1:n 
        y[i] = -x[i]
    end
    for k = 1:N-1
        yk = view(y, k*n .+ (1:n))
        x1 = view(x, (k-1)*n .+ (1:n))
        x2 = view(x, k*n .+ (1:n))
        Ax!(yk, x1, x2, h)
    end
    for i = 1:n
        y[N*n + i] = -x[(N-1)*n + i]
    end
    y
end

function Abarty!(x, xtmp, y, h, n, N)
    x[1:n] .= 0.0
    for k = 1:N-1 
        yk = view(y, k*n .+ (1:n))
        x1 = view(x, (k-1)*n .+ (1:n))
        x2 = view(x, k*n .+ (1:n))
        Aty!(xtmp, yk, h)
        x1 .+= view(xtmp, 1:n)
        x2 .= view(xtmp, n+1:2n)
    end
    for i = 1:n
        x[i] += -y[i]
        x[(N-1)*n + i] += -y[N*n + i]
    end
    x
end

function Bbaru!(y, u, h, n, m, N)
    for i = 1:n
        y[i] = 0.0
    end
    for k = 1:N-1
        yk = view(y, k*n .+ (1:n))
        u1 = view(u, (k-1)*m .+ (1:m))
        u2 = view(u, k*m .+ (1:m))
        Bu!(yk, u1, u2, h)
    end
    for i = 1:n
        y[N*n + i] = 0.0
    end
    y
end

function Bbarty!(u, utmp, y, h, n, m, N)
    u[1:n] .= 0
    for k = 1:N-1
        yk = view(y, k*n .+ (1:n))
        u1 = view(u, (k-1)*m .+ (1:m))
        u2 = view(u, k*m .+ (1:m))
        Bty!(utmp, yk, h)  # fine since uk isn't used
        u1 .+= view(utmp, 1:m)
        u2 .= view(utmp, m+1:2m)
    end
    u
end

function Cbarxu!(y, x, u, h, n, m, N, mass)
    for i = 1:n
        y[i] = 0.0
    end
    for k = 1:N-1
        yk = view(y, k*n .+ (1:n))
        x1 = view(x, (k-1)*n .+ (1:n))
        x2 = view(x, k*n .+ (1:n))
        u1 = view(u, (k-1)*m .+ (1:m))
        u2 = view(u, k*m .+ (1:m))
        Cxu!(yk, x1, u1, x2, u2, h, mass)
    end
    for i = 1:n
        y[N*n + i] = 0.0
    end
    y
end

function Cbartxy!(u, x, y, h, n, m, N, mass)
    for k = 1:N-1
        uk = view(u, (k-1)*m .+ (1:2m))
        xk = view(x, (k-1)*n .+ (1:2n)) 
        yk = view(y, k*n .+ (1:n))
        Ctxy!(uk, xk, yk, h, mass)
    end
    u
end

function Cbartuy!(x, xtmp, u, y, h, n, m, N, mass)
    x[1:n] .= 0
    for k = 1:N-1
        x1 = view(x, (k-1)*n .+ (1:n)) 
        x2 = view(x, k*n .+ (1:n)) 
        uk = view(u, (k-1)*m .+ (1:2m))
        yk = view(y, k*n .+ (1:n))
        Ctuy!(xtmp, uk, yk, h, mass)
        x1 .+= view(xtmp, 1:n)
        x2 .= view(xtmp, n+1:2n)
        # xk[n+1:2n] .= view(xtmp, n+1:2n)
        # xk .+= xtmp
    end
    x
end


function Ahatx!(y, ytmp, x, u, h, n, m, N, mass)
    Abarx!(y, x, h, n, N)
    Cbarxu!(ytmp, x, u, h, n, m, N, mass)
    y .+= ytmp
    y
end

function Bhatu!(y, ytmp, x, u, h, n, m, N, mass)
    Bbaru!(y, u, h, n, m, N)
    Cbarxu!(ytmp, x, u, h, n, m, N, mass)
    y .+= ytmp
    y
end

function Ahatty!(x, xtmp1, xtmp2, u, y, h, n, m, N, mass)
    Abarty!(x, xtmp2, y, h, n, N)
    Cbartuy!(xtmp1, xtmp2, u, y, h, n, m, N, mass)
    x .+= xtmp1
    x
end

function Bhatty!(u, utmp1, utmp2, x, y, h, n, m, N, mass)
    Bbarty!(u, utmp2, y, h, n, m, N)
    Cbartxy!(utmp1, x, y, h, n, m, N, mass)
    u .+= utmp1
    u
end

function AtAx!(xout, xtmp1, xtmp2, y, ytmp, x, u, h, n, m, N, mass)
    Ahatx!(y, ytmp, x, u, h, n, m, N, mass)
    Ahatty!(xout, xtmp1, xtmp2, u, y, h, n, m, N, mass)
end

function BtBu!(uout, utmp1, utmp2, y, ytmp, x, u, h, n, m, N, mass)
    Bhatu!(y, ytmp, x, u, h, n, m, N, mass)
    Bhatty!(uout, utmp1, utmp2, x, y, h, n, m, N, mass)
end

function Qhatmul!(Q, ρ, xout, xtmp1, xtmp2, y, ytmp, x, u, h, n, m, N, mass)
    AtAx!(xout, xtmp1, xtmp2, y, ytmp, x, u, h, n, m, N, mass)
    mul!(xtmp1, Q, x)
    for i = 1:length(x)
        xout[i] = xout[i] * ρ + xtmp1[i]
    end
    xout
end

function Rhatmul!(R, ρ, uout, utmp1, utmp2, y, ytmp, x, u, h, n, m, N, mass)
    BtBu!(uout, utmp1, utmp2, y, ytmp, x, u, h, n, m, N, mass)
    mul!(utmp1, R, u)
    for i = 1:length(u)
        uout[i] = uout[i] * ρ + utmp1[i]
    end
    uout
end


N = 101
tf = 3.0
h = tf / (N-1)
model = QuadrotorRateLimited()
mass = model.mass
n,m = RD.dims(model)
admm = Problems.QuadrotorRateLimitedSolver(N=N, tf=tf)
Abar = admm.A
Bbar = admm.B
Cbar = admm.C
Dbar = admm.d

BilinearControl.updateBhat!(admm, admm.Bhat, X)

X = vcat([Vector(RD.rand(model)[1]) for k = 1:N]...)
U = vcat([Vector(RD.rand(model)[2]) for k = 1:N]...)

y = zeros(size(Abar,1))
ytmp = zero(y)

Abarx!(y, X, h, n, N) ≈ Abar*X
Bbaru!(y, U, h, n, m, N) ≈ Bbar*U
Cbarxu!(y, X, U, h, n, m, N, mass) ≈ sum(U[i] * Cbar[i] *X for i = 1:length(U))

xtmp = zeros(2n)
utmp = zeros(2m)
y = randn(size(Abar,1))
Abarty!(X, xtmp, y, h, n, N) ≈ Abar'y
Bbarty!(U, utmp, y, h, n, m, N) ≈ Bbar'y

Cbartuy!(X, xtmp,  U, y, h, n, m, N, mass) ≈ 
    sum(U[i] * Cbar[i] for i = 1:length(U))'y

Cbartxy!(U, X, y, h, n, m, N, mass) ≈ 
    hcat([Cbar[i] * X for i = 1:length(U)]...)'y

Ahat = admm.Ahat
BilinearControl.updateAhat!(admm, Ahat, U)
Ahatx!(y, ytmp, X, U, h, n, m, N, mass) ≈ Ahat*X

Bhat = admm.Bhat
BilinearControl.updateBhat!(admm, Bhat, X)
Bhatu!(y, ytmp, X, U, h, n, m, N, mass) ≈ Bhat*U

xtmp1 = zero(X)
xtmp2 = zeros(2n)
BilinearControl.updateAhat!(admm, Ahat, U)
Ahatty!(X, xtmp1, xtmp2, U, y, h, n, m, N, mass) ≈ Ahat'y

utmp1 = zero(U)
utmp2 = zeros(2m)
BilinearControl.updateBhat!(admm, Bhat, X)
Bhatty!(U, utmp1, utmp2, X, y, h, n, m, N, mass) ≈ Bhat'y

xout = zero(X)
X = vcat([Vector(RD.rand(model)[1]) for k = 1:N]...)
X0 = copy(X)
U = vcat([Vector(RD.rand(model)[2]) for k = 1:N]...)
BilinearControl.updateAhat!(admm, Ahat, U)
AtAx!(xout, xtmp1, xtmp2, y, ytmp, X, U, h, n, m, N, mass) ≈ Ahat'Ahat*X

uout = zero(U)
X = vcat([Vector(RD.rand(model)[1]) for k = 1:N]...)
U = vcat([Vector(RD.rand(model)[2]) for k = 1:N]...)
BilinearControl.updateBhat!(admm, Bhat, X)
BtBu!(uout, utmp1, utmp2, y, ytmp, X, U, h, n, m, N, mass) ≈ Bhat'Bhat*U

Q = Diagonal(fill(1.0, length(X)))
ρ = 10.0
BilinearControl.updateAhat!(admm, Ahat, U)
Qhatmul!(Q, ρ, xout, xtmp1, xtmp2, y, ytmp, X, U, h, n, m, N, mass) ≈ 
    (Q + ρ*Ahat'Ahat)*X

R = Diagonal(fill(1.0, length(U)))
ρ = 10.0
BilinearControl.updateBhat!(admm, Bhat, X)
Rhatmul!(R, ρ, uout, utmp1, utmp2, y, ytmp, X, U, h, n, m, N, mass) ≈ 
    (R + ρ*Bhat'Bhat)*U


@btime Abarx!($y, $X, $h, $n, $N)
@btime mul!($y, $Abar, $X)
@btime Cbarxu!($y, $X, $U, $h, $n, $m, $N, $mass)
@btime sum($U[i] * $Cbar[i] * $X for i = 1:length($U))
@btime let x = $X, z = $U
    for i in eachindex(z)
        mul!($y, $Cbar[i], x, z[i], 1.0)
    end
end
@btime Ahatx!($y, $ytmp, $X, $U, $h, $n, $m, $N, $mass)
@btime begin
    BilinearControl.updateAhat!($admm, $Ahat, $U) 
    mul!($y,$Ahat,$X)
end
@btime Bhatu!($y, $ytmp, $X, $U, $h, $n, $m, $N, $mass)
@btime begin
    BilinearControl.updateBhat!($admm, $Bhat, $X) 
    mul!($y,$Bhat,$U)
end

@btime Cbartxy!($U, $X, $y, $h, $n, $m, $N, $mass)
@btime Cbartuy!($X, $xtmp, $U, $y, $h, $n, $m, $N, $mass)

@btime Ahatty!($X, $xtmp1, $xtmp2, $U, $y, $h, $n, $m, $N, $mass)
@btime begin
    BilinearControl.updateAhat!($admm, $Ahat, $U)
    mul!($X, $(Ahat'), $y)
end

@btime Bhatty!($U, $utmp1, $utmp2, $X, $y, $h, $n, $m, $N, $mass)
@btime begin
    BilinearControl.updateBhat!($admm, $Bhat, $X)
    mul!($U, $(Bhat'), $y)
end