using SparseArrays

"""
Builds a bilinear constraint for the entire trajectory optimization problem, using 
    implicit midpoint integration to preserve the bilinear structure of the dynamics.
"""
function buildbilinearconstraintmatrices(model, x0, xf, h, N)
    A = getA(model)  # continuous dynamics A
    B = getB(model)  # continuous dynamics B
    C = getC(model)  # continuous dynamics C
    D = getD(model)  # continuous dynamics D

    usetermcon = all(isfinite, xf)

    # Get sizes
    n = size(A,2)
    p,m = size(B)
    Nc = N*p + usetermcon * n
    Nx = N*n
    Nu = (N-1)*m

    # Build matrices
    Abar = spzeros(Nc,Nx)
    Bbar = spzeros(Nc,Nu)
    Cbar = [spzeros(Nc,Nx) for i = 1:Nu]
    Dbar = spzeros(Nc)

    # Initialize some useful ranges
    ic = 1:p
    ix1 = 1:n
    ix2 = ix1 .+ n 
    iu1 = 1:m

    # Initial condition
    Abar[ix1, ix1] .= -I(n)
    Dbar[ix1] .= x0
    ic = ic .+ n

    # Dynamics
    for k = 1:N-1
        Abar[ic, ix1] .= h/2*A + I
        Abar[ic, ix2] .= h/2*A - I
        Bbar[ic, iu1] .= h*B
        for (i,j) in enumerate(iu1)
            Cbar[j][ic,ix1] .= h/2 * C[i]
            Cbar[j][ic,ix2] .= h/2 * C[i]
        end
        Dbar[ic] .= h*D
        ic = ic .+ n
        ix1 = ix1 .+ n
        ix2 = ix2 .+ n
        iu1 = iu1 .+ m 
    end

    # Goal constraint
    ic = ic[1]:ic[1]+n-1
    if usetermcon
        Abar[ic, ix1] .= -I(n)
        Dbar[ic] .= xf
    end

    return Abar, Bbar, Cbar, Dbar
end

"""
Builds the bilinear matrices for a discrete dynamics model.
Assumes that `getA`, `getB`, `getC`, and `getD` return the matrices for evaluating
the dynamics error between two consecutive time steps (so that implicit dynamics
are fully supported).
"""
function buildbilinearconstraintmatrices(model::RD.DiscreteDynamics, x0,xf,h, N)
    # get sizes
    n,m,p = RD.dims(model) 
    Nx = N*n 
    Nu = N*m
    Nc = (N-1)*p + 2n

    # Build matrices
    Abar = spzeros(Nc, Nx)
    Bbar = spzeros(Nc, Nu)
    Cbar = [spzeros(Nc, Nx) for i = 1:Nu]
    Dbar = spzeros(Nc)

    # Initialize some useful ranges
    ic = 1:n
    ix12 = 1:2n
    iu12 = 1:2m

    # Initial conditio
    Abar[ic, 1:n] .= -I(n)
    Dbar[ic] .= x0
    ic = (1:p) .+ n

    # Dynamics
    A,B,C,D = getA(model,h), getB(model,h), getC(model,h), getD(model,h)
    for k = 1:N-1
        Abar[ic, ix12] .+= A
        Bbar[ic, iu12] .+= B
        for (i,j) in enumerate(iu12)
            Cbar[j][ic,ix12] .= C[i]
        end
        Dbar[ic] .+= D

        ix12 = ix12 .+ n
        iu12 = iu12 .+ m
        ic = ic .+ p
    end

    # Terminal constraint
    ic = ic[1] - 1 .+ (1:n)
    Abar[ic, ix12[1:n]] .= -I(n)
    Dbar[ic] .= xf

    return Abar, Bbar, Cbar, Dbar
end

"""
Evaluates the bilinear constraint. Useful for verification of the bilinear matrices.
"""
function evaluatebilinearconstraint(model, x0, xf, h, N, Z)
    A = getA(model)  # continuous dynamics A
    B = getB(model)  # continuous dynamics B
    C = getC(model)  # continuous dynamics C
    D = getD(model)  # continuous dynamics D

    usetermcon = all(isfinite, xf)

    # Get sizes
    n,m = size(B)
    Nc = N*n + usetermcon * n
    Nx = N*n
    Nu = (N-1)*m

    # Initialize constraint vector
    c = zeros(Nc)

    # Initialize some useful ranges
    ic = 1:n
    ix1 = 1:n
    ix2 = ix1 .+ (n + m)
    iu1 = (1:m) .+ n

    # Initial condition
    c[ic] = x0 - Z[ix1]
    ic = ic .+ n

    # Dynamics
    for k = 1:N-1
        x1 = Z[ix1]
        u1 = Z[iu1]
        x2 = Z[ix2]
        xm = (x1 + x2) / 2
        xdot = A * xm + B * u1 + sum(u1[i] * C[i] * xm for i = 1:m) + D
        c[ic] .= h*xdot + x1 - x2
        ic = ic .+ n
        ix1 = ix1 .+ (n + m) 
        ix2 = ix2 .+ (n + m)
        iu1 = iu1 .+ (n + m) 
    end

    # Goal constraint
    if usetermcon
        c[ic] .= xf - Z[ix1]
    end

    return c
end

function buildcostmatrices(prob::TO.Problem)
    Q = Diagonal(vcat([Vector(diag(cst.Q)) for cst in prob.obj]...))
    R = Diagonal(vcat([Vector(diag(prob.obj[k].R)) for k = 1:prob.N-1]...))
    q = vcat([Vector(cst.q) for cst in prob.obj]...)
    r = vcat([Vector(prob.obj[k].r) for k = 1:prob.N-1]...)
    c = sum(cst.c for cst in prob.obj)
    Q, q, R, r, c
end

function buildcostmatrices(Q,R,Qf,xf,N; u0=zeros(size(R,1)), uN = N-1)
    Qbar = Diagonal(vcat([diag(Q) for i = 1:N-1]...))
    Qbar = Diagonal([diag(Qbar); diag(Qf)])
    Rbar = Diagonal(vcat([diag(R) for i = 1:uN]...))
    q = repeat(-Q*xf, N)
    r = repeat(-R*u0, uN)
    c = 0.5*sum(dot(xf,Q,xf) for k = 1:N-1) + 0.5*dot(xf,Qf,xf) + 
        0.5*sum(dot(u0,R,u0) for k = 1:N)
    Qbar,q,Rbar,r,c
end

function evaluatebilinearconstraint(prob::TO.Problem)
    model = prob.model[1]
    n,m = RD.dims(model)

    # Get sizes
    n,m = RD.dims(model)
    N = prob.N
    Nc = (N+1)*n
    Nx = N*n
    Nu = (N-1)*m

    # Initialize constraint vector
    c = zeros(Nc)

    # Initialize some useful ranges
    ic = 1:n

    # Initial condition
    c[ic] = prob.x0 - RD.state(prob.Z[1])
    ic = ic .+ n

    # Dynamics
    # y1 = zeros(n)
    for k = 1:N-1
        z1 = prob.Z[k]
        z2 = prob.Z[k+1]
        ci = view(c, ic)

        y2 = view(c, ic)
        y2 .= RD.dynamics_error(model, z2, z1)
        # RD.dynamics_error!(model, y2, y1, z2, z1)

        ic = ic .+ n
    end

    # Goal constraint
    c[ic] .= prob.xf - RD.state(prob.Z[end]) 

    return c
end

