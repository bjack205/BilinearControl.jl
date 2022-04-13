using BilinearControl.Problems: qrot, skew
using SparseArrays

Base.@kwdef struct QuadrotorRateLimited <: RD.DiscreteDynamics
    mass::Float64 = 2.0
    gravity::Float64 = 9.81
end

RD.state_dim(::QuadrotorRateLimited) = 18
RD.control_dim(::QuadrotorRateLimited) = 4

BilinearControl.Problems.translation(::QuadrotorRateLimited, x) = SVector{3}(x[1], x[2], x[3])
BilinearControl.Problems.orientation(::QuadrotorRateLimited, x) = RotMatrix{3}(x[4:12]...)

function Base.rand(::QuadrotorRateLimited)
    x = [
            @SVector randn(3);
            vec(qrot(normalize(@SVector randn(4))));
            @SVector randn(6)
    ]
    u = push((@SVector randn(3)), rand())
    x,u
end

# function RD.dynamics(model::QuadrotorRateLimited, x, u)
function RD.dynamics_error(model::QuadrotorRateLimited, z2::RD.KnotPoint, z1::RD.KnotPoint)
    x1 = RD.state(z1)
    x2 = RD.state(z2)
    u1 = RD.control(z1)
    u2 = RD.control(z2)

    xm = (x1 + x2) / 2
    h = RD.timestep(z1)
    xdot0 = let x = xm, u = u1
        mass = model.mass
        g = model.gravity 
        R = SA[
            x[4] x[7] x[10]
            x[5] x[8] x[11]
            x[6] x[9] x[12]
        ]
        v = SA[x[13], x[14], x[15]]
        ω = SA[u[1], u[2], u[3]]
        Fbody = [0, 0, u[4]]

        rdot = v;
        Rdot = R * Rotations.skew(ω)
        vdot = R*Fbody ./ mass - [0,0,g]
        [rdot; vec(Rdot); vdot]
    end
    dx0 = x1[1:15] - x2[1:15]
    α2 = SA[x2[16], x2[17], x2[18]]
    ω1 = SA[u1[1], u1[2], u1[3]]
    ω2 = SA[u2[1], u2[2], u2[3]]
    [h*xdot0 + dx0; h*α2 + ω1 - ω2]
end

function BilinearControl.getA(::QuadrotorRateLimited, h)
    n = 18 
    A = zeros(n, 2n)
    for i = 1:3
        A[i,12+i] = h/2
        A[i,n+12+i] = h/2
        A[15+i,n+15+i] = h 
    end
    for i = 1:15
        A[i,i] = 1.0
        A[i,n+i] = -1.0
    end
    A
end

function BilinearControl.getB(::QuadrotorRateLimited, h)
    n,m = 18,4
    B = zeros(n,2m)
    for i = 1:3
        B[15+i,i] = 1.0
        B[15+i,m+i] = -1.0
    end
    B
end

function BilinearControl.getC(model::QuadrotorRateLimited, h)
    n,m = 18,4
    C = [zeros(n,2n) for i = 1:2m]
    mass = model.mass
    for i = 1:3
        for j in (0,1)
            C[1][6+i,9+i+j*n] = +h*0.5
            C[1][9+i,6+i+j*n] = -h*0.5
            C[2][3+i,9+i+j*n] = -h*0.5
            C[2][9+i,3+i+j*n] = +h*0.5
            C[3][3+i,6+i+j*n] = +h*0.5
            C[3][6+i,3+i+j*n] = -h*0.5
            C[4][12+i,9+i+j*n] = h/2mass
        end
    end
    C
end

function BilinearControl.getD(model::QuadrotorRateLimited, h)
    g = model.gravity 
    d = zeros(18)
    d[15] = -g*h
    d
end

## Test dynamics
model = QuadrotorRateLimited()
n,m = RD.dims(model)
r1,r2 = randn(3), randn(3) 
R1,R2 = qrot(normalize(randn(4))), qrot(normalize(randn(4)))
v1,v2 = randn(3), randn(3) 
α1,α2 = randn(3), randn(3) 
ω1,ω2 = randn(3), randn(3) 
F1,F2 = rand(), rand()

x1 = [r1; vec(R1); v1; α1]
x2 = [r2; vec(R2); v2; α2]
u1 = [ω1; F1]
u2 = [ω2; F2]

h = 0.1
z1 = RD.KnotPoint{n,m}(n,m,[x1;u1],0.0,h)
z2 = RD.KnotPoint{n,m}(n,m,[x2;u2],h,h)
err = RD.dynamics_error(model, z2, z1)

err[1:3] ≈ h * (v1 + v2) / 2 + r1 - r2
err[4:12] ≈ vec(h * (R1 + R2) /2 * skew(ω1) + R1 - R2)
err[13:15] ≈ h*( (R1 + R2) /2 * [0,0,F1]) / model.mass - 
    h*[0,0,model.gravity] + v1 - v2
err[16:18] ≈ h*α2 - (ω2 - ω1)


using BilinearControl: getA, getB, getC, getD
# Test dynamics match bilinear dynamics
A,B,C,D = getA(model,h), getB(model,h), getC(model,h), getD(model,h)
x12 = [x1;x2]
u12 = [u1;u2]
err2 = A*x12 + B*u12 + sum(u12[i]*C[i]*x12 for i = 1:length(u12)) + D
err ≈ err2
A1 = A[:,1:n]
A2 = A[:,n+1:end]
A*x12 ≈ A1*x1 + A2*x2

function eval_dynamics_constraint(model, x0,xf,h,  X, U)
    n,m = RD.dims(model)
    Xs = reshape(X, n, :)
    Us = reshape(U, m, :)
    N = size(Xs,2)

    # Initialize some useful ranges
    ic = 1:n
    ix12 = 1:2n
    iu12 = 1:2m

    # Initialize
    c = zeros(n*(N+1))

    # Initial condition
    c[ic] = x0 - Xs[:,1] 
    ic = ic .+ n

    # Dynamics
    A,B,C,D = getA(model,h), getB(model,h), getC(model,h), getD(model,h)
    for k = 1:N-1
        x12 = X[ix12]
        u12 = U[iu12]
        c[ic] .= A*x12 + B*u12 + sum(u12[i] * C[i] * x12 for i = 1:2m) + D

        ix12 = ix12 .+ n
        iu12 = iu12 .+ m
        ic = ic .+ n
    end

    # Terminal constraint
    c[ic] .= xf .- Xs[:,end]

    c
end

function build_bilinear_matrices(model, x0,xf,h, N)
    # Get sizes
    n,m = RD.dims(model) 
    Nx = N*n 
    Nu = N*m
    Nc = N*n + n

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
    ic = ic .+ n

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
        ic = ic .+ n
    end

    # Terminal constraint
    Abar[ic, ix12[1:n]] .= -I(n)
    Dbar[ic] .= xf

    return Abar, Bbar, Cbar, Dbar
end

## Visualization 
using MeshCat
vis = Visualizer()
open(vis)
visdir = joinpath(@__DIR__, "../examples/visualization")
include(joinpath(visdir, "visualization.jl"))
setquadrotor!(vis)

tf = 3.0
N = 51
h = tf / (N-1)
x0 = [0; 0; 1.0; vec(I(3)); zeros(3); zeros(3)]
xf = [5; 0; 2.0; vec(RotZ(deg2rad(90))); zeros(3); zeros(3)]

A,B,C,D = getA(model,h), getB(model,h), getC(model,h), getD(model,h)
Xs = [rand(model)[1] for k = 1:N]
Us = [rand(model)[2] for k = 1:N]
X = vcat(Vector.(Xs)...)
U = vcat(Vector.(Us)...)
x0 = [zeros(3); vec(I(3)); zeros(6)]
c1 = eval_dynamics_constraint(model, x0,xf,h, X, U)
c1[1:n] ≈ x0 - Xs[1]
c1[n+1:2n] ≈ A*[Xs[1]; Xs[2]] + B* [Us[1]; Us[2]] + 
    sum([Us[1]; Us[2]][i] * C[i]*[Xs[1]; Xs[2]] for i = 1:2m) + D
c1[2n+1:3n] ≈ A*[Xs[2]; Xs[3]] + B* [Us[2]; Us[3]] + 
    sum([Us[2]; Us[3]][i] * C[i]*[Xs[2]; Xs[3]] for i = 1:2m) + D
c1[end-n+1:end] ≈ xf - Xs[end]


Abar,Bbar,Cbar,Dbar = build_bilinear_matrices(model, x0, xf, h, N)
c2 = Abar*X + Bbar*U + sum(U[i] * Cbar[i]*X for i = 1:length(U)) + Dbar
c1 ≈ c2


# Build cost
u0 = [0,0,0,model.mass*model.gravity]
Q = Diagonal([fill(1e-2, 3); fill(1e-2, 9); fill(1e-2, 3); fill(1e-1, 3)])
Qf = Q*(N-1)
R = Diagonal([fill(1e-2,3); 1e-2])
Qbar = Diagonal(vcat([diag(Q) for i = 1:N-1]...))
Qbar = Diagonal([diag(Qbar); diag(Qf)])
Rbar = Diagonal(vcat([diag(R) for i = 1:N]...))
q = repeat(-Q*xf, N)
r = repeat(-R*u0, N)
c = 0.5*sum(dot(xf,Q,xf) for k = 1:N-1) + 0.5*dot(xf,Qf,xf) + 0.5*sum(dot(u0,R,u0) for k = 1:N)

# Build Solver
admm = BilinearADMM(Abar,Bbar,Cbar,Dbar, Qbar,q,Rbar,r,c)
X = repeat(x0, N)
U = repeat(u0, N)
admm.opts.penalty_threshold = 1e2
BilinearControl.setpenalty!(admm, 1e4)
Xsol, Usol = BilinearControl.solve(admm, X, U, verbose=true, max_iters=200)

Xs = collect(eachcol(reshape(Xsol, RD.state_dim(model), :)))
visualize!(vis, model, TO.get_final_time(prob), Xs)

# Plot controls
using Plots
Us = collect(eachrow(reshape(Usol, RD.control_dim(model), :)))
times = range(0,tf, length=N)
p1 = plot(times, Us[1], label="ω₁", ylabel="angular rates (rad/s)", xlabel="time (s)")
plot!(times, Us[2], label="ω₂")
plot!(times, Us[3], label="ω₃")
# savefig(p1, "quadrotor_angular_rates.png")

p2 = plot(times, Us[4], label="", ylabel="Thrust", xlabel="times (s)")
# savefig(p2, "quadrotor_force.png")