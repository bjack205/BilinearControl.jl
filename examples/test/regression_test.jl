using Pkg; Pkg.activate(joinpath(@__DIR__, ".."));
Pkg.instantiate();
using BilinearControl
using BilinearControl.Problems
using BilinearControl.EDMD
using Rotations
using StaticArrays
using Test
using LinearAlgebra 
using Altro
using RobotDynamics
using TrajectoryOptimization
const TO = TrajectoryOptimization
import RobotDynamics as RD
using BilinearControl: Problems
using JLD2
using Plots
using ForwardDiff
using SparseArrays
using BilinearControl: matdensity
using LazyArrays
using ProgressMeter

include("../airplane_constants.jl")

##
airplane_data = load(AIRPLANE_DATAFILE)
X_train = airplane_data["X_train"][:,1:10]
U_train = airplane_data["U_train"][:,1:10]
T_train = airplane_data["T_ref"]
dt = T_train[2]

dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(Problems.NominalAirplane())

## Generate Jacobians
n0 = length(X_train[1])
m = length(U_train[1])
xn = zeros(n0)
cinds_jac = CartesianIndices(U_train)
A,B,F,G = let model=dmodel_nom, kf=airplane_kf
    xn = zeros(n0)
    n = length(kf(xn))  # new state dimension
    cinds_jac = CartesianIndices(U_train)
    jacobians = map(cinds_jac) do cind
        k = cind[1]
        x = X_train[cind]
        u = U_train[cind]
        z = RD.KnotPoint{n0,m}(x,u,T_train[k],dt)
        J = zeros(n0,n0+m)
        RD.jacobian!(
            RD.InPlace(), RD.ForwardAD(), model, J, xn, z 
        )
        J
    end
    A_train = map(J->J[:,1:n0], jacobians)
    B_train = map(J->J[:,n0+1:end], jacobians)

    ## Convert states to lifted Koopman states
    # Y_train = map(kf, X_train)

    ## Calculate Jacobian of Koopman transform
    F_train = map(cinds_jac) do cind 
        x = X_train[cind]
        sparse(ForwardDiff.jacobian(kf, x))
    end

    ## Create a sparse version of the G Jacobian
    G = spdiagm(n0,n,1=>ones(n0)) 
    A_train, B_train, F_train, G
end

Y_train = map(airplane_kf, X_train)
X = Y_train
U = U_train

learnB = true
n0 = size(A[1],1)
n = length(X[1])
m = length(U[1])
mB = m * learnB
p = n+mB + n*m     # number of features
N,T = size(X)
P = (N-1)*T        # number of dynamics samples
Pj = length(A)     # number of Jacobian samples
Uj = reduce(hcat, U)
Xj = reduce(hcat, X[1:end-1,:])
Xn = reduce(hcat, X[2:end,:])
Amat = reduce(hcat, A)
Bmat = reduce(hcat, B)
@assert size(Xn,2) == size(Xj,2) == P
Z = mapreduce(hcat, 1:P) do j
    x = Xj[:,j]
    u = Uj[:,j]
    if learnB
        [x; u; vec(x*u')]
    else
        [x; vec(x*u')]
    end
end
@assert size(Z) == (p,P)

@time Ahat = mapreduce(hcat, CartesianIndices(cinds_jac)) do cind0
    cind = cinds_jac[cind0]
    u = U[cind] 
    In = sparse(I,n,n)
    vcat(sparse(I,n,n), spzeros(mB,n), reduce(vcat, In*ui for ui in u)) * F[cind0]
end
@assert size(Ahat) == (p,Pj*n0)

Bhat = mapreduce(hcat, cinds_jac) do cind 
    xB = spzeros(n,m)
    xB[:,1] .= X[cind]
    vcat(spzeros(n,m), sparse(I,mB,m), reduce(vcat, circshift(xB, (0,i)) for i = 1:m))
end
@assert size(Bhat) == (p,Pj*m)

α = 0.5
W = ApplyArray(vcat,
    (1-α) * ApplyArray(kron, Z', sparse(I,n,n)),
    α * ApplyArray(kron, Ahat', G),
    α * ApplyArray(kron, Bhat', G),
) 
s = vcat((1-α) * vec(Xn), α * vec(Amat), α * vec(Bmat))
@time Wsparse = sparse(W)
@time Ws = begin
    vcat(
        (1-α) * kron(Z', sparse(I,n,n)),
        α * kron([Ahat'; Bhat'], G),
    )
end

Z_t = map(cinds_jac) do cind 
    x = X[cind]
    u = U[cind]
    # x = Xj[:,j]
    # u = Uj[:,j]
    if learnB
        [x; u; vec(x*u')]
    else
        [x; vec(x*u')]
    end
end
@time Ahat_t = map(CartesianIndices(cinds_jac)) do cind0
    cind = cinds_jac[cind0]
    u = U[cind]
    In = sparse(I,n,n)
    F[cind0]'hcat(sparse(I,n,n), spzeros(n,mB), reduce(hcat, In*ui for ui in u))
end
@time Bhat_t = map(cinds_jac) do cind
    xB = spzeros(m,n)
    xB[1,:] .= X[cind]
    hcat(spzeros(m,n), sparse(I,m,mB), reduce(hcat, circshift(xB, (i,0)) for i = 1:m))
end
reduce(vcat, Ahat_t) ≈ Ahat'
reduce(vcat, Bhat_t) ≈ Bhat'

W0 = (1-α)*kron(sparse(Z'),sparse(I,n,n))
b0 = (1-α)*vec(Xn)
qrW = qr(W0)

blocks = map(1:length(Ahat_t)) do i
    kron(α * vcat(Ahat_t[i], Bhat_t[i]), G)
end
Amat

vecs = map(1:length(Ahat_t)) do i
    α * vcat(vec(A[i]), vec(B[i]))
end

using BilinearControl.EDMD: qrr
function rls_qr(A0,b0, A, b; reg=0.0, show_progress=false)

    if reg > zero(reg)
        U,_,Pcol = qrr([A0; sqrt(reg) * I])
    else
        U,_,Pcol = qrr(A0)
    end
    rhs = A0'b0

    prog = Progress(length(A), enabled=show_progress)
    for (Ai,bi) in zip(A,b)
        U,_,Pcol = qrr([U*Pcol'; Ai])
        rhs += Ai'bi
        next!(prog)
    end
    U = U*Pcol'
    return U,rhs
end
U,rhs = rls_qr(W0,b0,blocks,vecs, reg=1e-4, show_progress=true)
U