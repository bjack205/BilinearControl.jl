import Pkg; Pkg.activate(joinpath(@__DIR__)); Pkg.instantiate();
using BilinearControl
using BilinearControl.Problems
using BilinearControl.EDMD
import RobotDynamics as RD
using LinearAlgebra
using RobotZoo
using JLD2
using SparseArrays
using Plots
using Distributions
using Distributions: Normal
using Random
using FiniteDiff, ForwardDiff
using StaticArrays
using Test
import TrajectoryOptimization as TO
using Altro
import BilinearControl.Problems
using Test

const CARTPOLE_RESULTS_FILE = joinpath(Problems.DATADIR, "cartpole_results.jld2")
function generate_lsl_data(alg=:eDMD; num_lqr=0, num_swingup=10)
    #############################################
    ## Load training data
    #############################################
    cartpole_data = load(joinpath(Problems.DATADIR, "cartpole_swingup_data.jld2"))

    # Training data
    X_train_lqr = cartpole_data["X_train_lqr"][:,1:num_lqr]
    U_train_lqr = cartpole_data["U_train_lqr"][:,1:num_lqr]
    X_train_swingup = cartpole_data["X_train_swingup"][:,1:num_swingup]
    U_train_swingup = cartpole_data["U_train_swingup"][:,1:num_swingup]
    X_train = [X_train_lqr X_train_swingup]
    U_train = [U_train_lqr U_train_swingup]

    # Metadata
    tf = cartpole_data["tf"]
    t_sim = cartpole_data["t_sim"]
    dt = cartpole_data["dt"]

    #############################################
    ## Generate Least Squares data 
    #############################################
    # Define basis functions
    eigfuns = ["state", "sine", "cosine", "sine", "sine", "chebyshev"]
    eigorders = [[0],[1],[1],[2],[4],[2, 4]]

    if alg == :eDMD
        Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders);

        # learn_bilinear_model(X_trian, Z_train, Zu_train)
        let X = X_train, Z = Z_train, Zu = Zu_train
            X_mat = reduce(hcat, X[1:end-1,:])
            Z_mat = reduce(hcat, Z[1:end-1,:])
            Zu_mat = reduce(hcat, Zu)
            Zn_mat = reduce(hcat, Z[2:end,:])

            # Solve min || E Zn_mat - Zu_mat ||
            # fitA(Zu_mat, Zn_mat)
            let x = Zu_mat, b = Zn_mat
                n,p = size(x)
                m = size(b,1)
                Â = kron(sparse(x'), sparse(I,m,m))
                b̂ = vec(b)
                return Â, b̂
            end
        end
    elseif alg == :jDMD

        model = RD.DiscretizedDynamics{RD.RK4}(Problems.NominalCartpole())
        α = 0.5
        learnB = true
        n0 = length(X_train[1])
        m = length(U_train[1])
        T_train = range(0,step=dt,length=size(X_train,1))

        # Generate transform
        Z_train, Zu_train, kf = build_eigenfunctions(X_train, U_train, eigfuns, eigorders);

        # Generate Jacobians
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

        # Calculate Jacobian of Koopman transform
        F_train = map(cinds_jac) do cind 
            x = X_train[cind]
            sparse(ForwardDiff.jacobian(kf, x))
        end

        # Create a sparse version of the G Jacobian
        G = spdiagm(n0,n,1=>ones(n0)) 
        @assert eigfuns[1] == "state"

        # Build least squares problem
        W,s = BilinearControl.EDMD.build_edmd_data(
            Z_train, U_train, A_train, B_train, F_train, G; cinds_jac, α, learnB)
        return W,s
    end
end

function solve_neq(b,A; reg=zero(eltype(b)))
    (A'A + I*reg)\(A'b)
end

function solve_qr(b,A; reg=zero(eltype(b)))
    n = size(A,2)
    qr([A; I*sqrt(reg)]) \ [b; zeros(n)]
end
using BilinearControl.EDMD: rls_qr

A,b = generate_lsl_data(:jDMD, num_swingup=10, num_lqr=0)
A = sparse(A)
size(A)
size(b)

n = 100_000
m = 3_000
A = sprandn(n,m,0.2)
b = randn(n)
m,n = size(A)

reg = 1e-4
using StatProfilerHTML
@profilehtml for i = 1:50
    rls_qr(b, A; Q=reg, verbose=false)
end
@time x_rls = rls_qr(b, A; Q=reg, verbose=true)
@time x_neq = solve_neq(b,A;reg)
@time x_qr = solve_qr(b,A;reg) 

norm(x_rls - x_neq)
norm(x_rls - x_qr)

using BenchmarkTools
b_rls = @benchmark rls_qr($b,$A; Q=$reg)
b_neq = @benchmark solve_neq($b,$A;reg=$reg)
b_qr  = @benchmark solve_qr($b,$A;reg=$reg) 

function house(x)
    n = length(x)
    v = copy(x)
    x2 = view(x,2:n)
    sigma = dot(x2,x2)
    if sigma ≈ 0
        return 0.0, v
    end
    normx = sqrt(x[1]^2 + sigma)
    if x[1] > 0  # v = x + ||x||
        v[1] += normx 
    else         # v = x - ||x||
        v[1] -= normx 
    end
    vtv = v[1]^2 + sigma
    β = 2 / vtv 
    return β, v
end

function givens(a,b)
    if b == 0
        s = 1
        c = 1
    else
        if abs(b) > abs(a)  # closer to vertical
            tau = -a/b
            s = inv(sqrt(1 + tau*tau))
            c = s * tau
        else
            tau = -b/a
            c = inv(sqrt(1+tau*tau))
            s = c*tau
        end
    end
    return (c,s)
end

function qr_house(A)
    m,n = size(A)
    vA = zeros(n)
    istall = m > n
    kend = istall ? n : m - 1

    # Loop over columns
    for k = 1:kend
        beta, v = house(A[k:end,k])  # get reflection for column k

        # Compute vA = beta * v'A
        for j = k:n  # columns of A
            vA[j] = 0.0
            for i = k:m  # rows of A
                # compute inner product of v with the ith column of A
                vA[j] += v[i-k+1] * A[i,j]
            end
            vA[j] *= beta
        end

        # Compute A = A - v vA
        # only need to compute it for the lower right block
        for j = k:n      # loop over cols
            for i = k:m  # loop over rows
                A[i,j] -= v[i-k+1] * vA[j]
            end
        end
        # save v in lower triangular part of A
        A[k+1:end,k] .= v[2:end]
    end
    A
end

A = randn(15,10)
qrA = qr_house(copy(A))
UpperTriangular(qrA[1:10,1:10])
qr(A,NoPivot()).R