using Pkg; Pkg.activate(joinpath(@__DIR__, ".."));
Pkg.instantiate();
using BilinearControl
using BilinearControl.Problems
using BilinearControl.EDMD
import BilinearControl.Problems
import RobotDynamics as RD
using JLD2
using Plots
using LinearAlgebra
using Rotations
using StaticArrays
using SparseArrays
using OSQP

function solve_lqr_osqp(Q,R,q,r,A,B,f,x0)
    Nt = length(q)
    n,m = size(B[1])
    Np = Nt*n + (Nt-1) * m
    Nd = Nt*n
    P_qp = spdiagm(vcat(mapreduce(diag, vcat, Q), mapreduce(diag, vcat, R)))
    q_qp = vcat(reduce(vcat, q), reduce(vcat, r))
    b = [-x0; reduce(vcat, f)]
    Aqp = spzeros(Nd,Np)
    Aqp[1:n,1:n] .= -I(n)
    for k = 1:Nt-1
        ic = k*n .+ (1:n)
        ix = (k-1) * n .+ (1:n)
        iu = Nt*n + (k-1) * m .+ (1:m)
        Aqp[ic,ix] .= A[k]
        Aqp[ic,iu] .= B[k]
        Aqp[ic,ix .+ n] .= -I(n)
    end
    osqp = OSQP.Model()
    OSQP.setup!(osqp; P_qp, q_qp, A=Aqp, l=b, u=b, verbose=false)
    res = OSQP.solve!(osqp)
    res.x - Z
    res.x[1:n] - x0
    X = [x for x in eachcol(reshape(res.x[1:n*Nt], n,:))]
    U = [u for u in eachcol(reshape(res.x[n*Nt .+ (1:(Nt-1)*m)], m,:))]
    λ = [y for y in eachcol(reshape(res.y, n, :))]
    X,U,λ
end

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

## Random QP
n,m,N = 10,5,11
A = [zeros(n,n) for k=1:N-1]
B = [zeros(n,m) for k=1:N-1]
for k = 1:N-1
    A[k],B[k] = BilinearControl.RandomLinearModels.gencontrollable(n,m)
end
f = [zeros(n) for k=1:N-1]
Q = [Diagonal(rand(n)) for k = 1:N]
R = [Diagonal(rand(m)) for k = 1:N-1]
q = [randn(n) for k = 1:N]
r = [randn(m) for k = 1:N-1]
x0 = randn(n)

X,U,λ = solve_lqr(Q, R, q, r, A, B, f, x0)
Z = vcat(reduce(vcat, X), reduce(vcat, U))
norm(map(1:N-1) do k
    Q[k] * X[k] + q[k] - λ[k] + A[k]'λ[k+1]
end)
norm(map(1:N-1) do k
    R[k] * U[k] + r[k] + B[k]'λ[k+1]
end)
norm(map(1:N-1) do k
    A[k]*X[k] + B[k]*U[k] + f[k] - X[k+1]
end)

##
Nt = length(q)
n,m = size(B[1])
Np = Nt*n + (Nt-1) * m
Nd = Nt*n
P_qp = spdiagm(vcat(mapreduce(diag, vcat, Q), mapreduce(diag, vcat, R)))
q_qp = vcat(reduce(vcat, q), reduce(vcat, r))
b = [-x0; reduce(vcat, f)]
Aqp = spzeros(Nd,Np)
Aqp[1:n,1:n] .= -I(n)
for k = 1:Nt-1
    ic = k*n .+ (1:n)
    ix = (k-1) * n .+ (1:n)
    iu = Nt*n + (k-1) * m .+ (1:m)
    Aqp[ic,ix] .= A[k]
    Aqp[ic,iu] .= B[k]
    Aqp[ic,ix .+ n] .= -I(n)
end
osqp = OSQP.Model()
OSQP.setup!(osqp; P_qp, q_qp, A=Aqp, l=b, u=b, verbose=true, eps_abs=1e-6, eps_rel=1e-6)
res = OSQP.solve!(osqp)

X2 = [x for x in eachcol(reshape(res.x[1:n*Nt], n,:))]
U2 = [u for u in eachcol(reshape(res.x[n*Nt .+ (1:(Nt-1)*m)], m,:))]
λ2 = [y for y in eachcol(reshape(res.y, n, :))]

mapreduce(+,1:N) do k
    J = 0.5 * dot(X2[k], Q[k], X2[k]) + dot(q[k], X2[k])
    if k < N
        J += 0.5 * dot(U2[k], R[k], U2[k]) + dot(r[k], U2[k])
    end
    J
end
0.5 * dot(res.x, P_qp, res.x) + dot(q_qp, res.x)

norm(map(1:N-1) do k
    Q[k] * X2[k] + q[k] - λ2[k] + A[k]'λ2[k+1]
end)
norm(map(1:N-1) do k
    R[k] * U2[k] + r[k] + B[k]'λ2[k+1]
end)
norm(map(1:N-1) do k
    A[k]*X2[k] + B[k]*U2[k] + f[k] - X2[k+1]
end)

res.x[1:n] - x0
c = Aqp*res.x - b 
c[1:n] ≈  x0 - X[1] 
c[1n+1:2n] ≈ A[1] * X[1] + B[1] * U[1] + f[1] - X[2]
c[2n+1:3n] ≈ A[2] * X[2] + B[2] * U[2] + f[2] - X[3]

X,U,λ = solve_lqr_osqp(Q, R, q, r, A, B, f, x0)
norm(res.x - Z)

Q[1]
norm.(map(1:Nt-1) do k
    Q[k] * X2[:,k] + q[k] - λ2[:,k] + A[k]'λ2[:,k+1]
end)
norm(map(1:Nt-1) do k
    R[k] * U2[:,k] + r[k] + B[k]'λ2[:,k+1]
end)
norm(map(1:Nt-1) do k
    A[k] * X2[:,k] + B[k]*U2[:,k] + f[k] - X2[:,k+1]
end)
P_qp * res.x + q_qp - Aqp'res.y


# Load swingup trajectory to track
cartpole_data = load(joinpath(Problems.DATADIR, "cartpole_swingup_data.jld2"))
X_ref = cartpole_data["X_ref"][:,1]
U_ref = push!(cartpole_data["U_ref"][:,1], zeros(1))
t_ref = cartpole_data["tf"]
dt = cartpole_data["dt"]
T_ref = collect(range(0,t_ref,step=dt))

model = Problems.NominalCartpole()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
Qmpc = Diagonal(fill(1e-0,4))
Rmpc = Diagonal(fill(1e-3,1))
Qfmpc = Diagonal([1e4,1e2,1e1,1e1]) 
mpc = EDMD.TrackingMPC(dmodel, X_ref, U_ref, T_ref, Qmpc, Rmpc, Qfmpc, Nt=41)

dmodel_true = RD.DiscretizedDynamics{RD.RK4}(Problems.SimulatedCartpole())
t_sim = 1.5 * t_ref
X_sim,_,T_sim = simulatewithcontroller(dmodel_true, mpc, X_ref[1], t_sim, dt)
plotstates(T_ref, X_ref, inds=1:2, c=:black, label="ref")
plotstates!(T_sim, X_sim, inds=1:2, c=[1 2], label="mpc", s=:dash)

t = 0.0
x = copy(X_ref[1])
##
u = EDMD.getcontrol(mpc, x, t)
x = RD.discrete_dynamics(dmodel_true, x, u, t, dt)
t += dt
X_mpc, U_mpc = EDMD.gettrajectory(mpc, t)
T_mpc = range(0,step=dt,length=EDMD.mpchorizon(mpc))
plotstates(T_ref, X_ref, inds=1:2, c=:black, label="ref")
plotstates!(t .+ T_mpc, X_mpc, inds=1:2, c=[1 2], label="mpc", lw=2, legend=:bottomleft)

##
function getreference(A,inds)
    N = length(A)
    map(inds) do i
        k = min(i, N)
        A[k]
    end
end

t = 0.0
Nt = 41
kref = EDMD.get_k(mpc, t)
N_ref = length(X_ref)
inds = kref-1 .+ (1:Nt)

X = getreference(X_ref, inds)
U = getreference(U_ref, inds)[1:end-1]
A = getreference(mpc.A, inds)[1:end-1]
B = getreference(mpc.B, inds)[1:end-1]
f = getreference(mpc.f, inds)[1:end-1]
Q = getreference(mpc.Q, inds)
R = getreference(mpc.R, inds)[1:end-1]
q = getreference(mpc.q, inds)
r = getreference(mpc.r, inds)[1:end-1]