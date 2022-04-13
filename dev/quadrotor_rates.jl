using BilinearControl.Problems: qrot, skew
using SparseArrays


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


function eval_dynamics_constraint(model, x0,xf,h,  x, u)
    n,m = rd.dims(model)
    xs = reshape(x, n, :)
    us = reshape(u, m, :)
    n = size(xs,2)

    # initialize some useful ranges
    ic = 1:n
    ix12 = 1:2n
    iu12 = 1:2m

    # initialize
    c = zeros(n*(n+1))

    # initial condition
    c[ic] = x0 - xs[:,1] 
    ic = ic .+ n

    # dynamics
    a,b,c,d = geta(model,h), getb(model,h), getc(model,h), getd(model,h)
    for k = 1:n-1
        x12 = x[ix12]
        u12 = u[iu12]
        c[ic] .= a*x12 + b*u12 + sum(u12[i] * c[i] * x12 for i = 1:2m) + d

        ix12 = ix12 .+ n
        iu12 = iu12 .+ m
        ic = ic .+ n
    end

    # terminal constraint
    c[ic] .= xf .- xs[:,end]

    c
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