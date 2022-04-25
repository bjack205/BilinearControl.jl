import Pkg; Pkg.activate(@__DIR__)
using SparseArrays
using LinearAlgebra
using BilinearControl
using Test
using RobotZoo 
import RobotDynamics as RD
using TrajectoryOptimization
const TO = TrajectoryOptimization
include("pendulum_bilinear.jl")

function build_original_problem()
    tf = 3.0
    N = 51
    h = tf / (N-1) 
    n,m = RD.dims(dpend)
    Q = Diagonal(fill(1e-3, n)) * h
    R = Diagonal(fill(1e-3, m)) * h 
    Qf = Diagonal(fill(1e-3, n))
    x0 = [0,0.]
    xf = [pi,0]
   
    obj = LQRObjective(Q,R,Qf,xf,N)
    goal = GoalConstraint(xf)
    cons = ConstraintList(n,m,N)
    add_constraint!(cons, goal, N)
    Problem(dpend, obj, x0, tf, xf=xf, constraints=cons)
end

function build_lifted_problem(prob0)
    
end

## Solve with ALTRO
using TrajectoryOptimization
using Altro
RD.default_diffmethod(::RobotZoo.Pendulum) = RD.ForwardAD()
pend = RobotZoo.Pendulum()
dpend = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(pend)

tf = 3.0
N = 51
h = tf / (N-1) 
n,m = RD.dims(dpend)
Q = Diagonal(fill(1e-3, n)) * h
R = Diagonal(fill(1e-3, m)) * h 
Qf = Diagonal(fill(1e-3, n))
x0 = [0,0.]
xf = [pi,0]

altro = ALTROSolver(prob)
altro.opts.dynamics_diffmethod = RD.ImplicitFunctionTheorem(RD.ForwardAD())
solve!(altro)

struct BilinearPendulum
    A::SparseMatrixCSC{Float64,Int}
    B::SparseMatrixCSC{Float64,Int}
    C::Vector{SparseMatrixCSC{Float64,Int}}
    D::SparseMatrixCSC{Float64,Int}
    x0::Vector{Float64}
end
function BilinearPendulum(x0)
    A,B,C,D = getsparsearrays()
    updateA!(A, x0)
    updateB!(B, x0)
    updateC!(C, x0)
    updateD!(D, x0)
    BilinearPendulum(A, B, C, D, x0)
end

"""
Build the bilinear constraint for the entire trajectory optimization problem, with a uniform
timestep `h` and `N` knot points.

Uses implicit midpoint to integrate the dynamics, maintaining the bilinear structure 
of the dynamics constraints.
"""
function buildbilinearconstraint(model::BilinearPendulum, x0, xf, h, N; T = Diagonal(ones(size(model.A,2))))
    n,m = size(model.B)
    Nc = (N+1)*n
    Nx = N*n
    Nu = (N-1)*m
    Abar = spzeros(Nc,Nx)
    Bbar = spzeros(Nc,Nu)
    Cbar = [spzeros(Nc,Nx) for i = 1:Nu]
    Dbar = spzeros(Nc)
    ic = 1:n
    ix1 = 1:n
    ix2 = ix1 .+ n 
    iu1 = 1:m
    Tinv = inv(T)

    # Initial condition
    y0 = zeros(n)
    expand!(y0, x0)
    Abar[ic,ix1] .= -I(n)
    Dbar[ic] .= Tinv * y0 
    ic = ic .+ n

    # Dynamics
    A = Tinv * model.A * T
    B = Tinv * model.B
    C = map(model.C) do Ci
        Tinv * Ci * T
    end
    D = Tinv * model.D
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
    yf = zeros(n)
    expand!(yf, xf)
    Abar[ic, ix1] .= -I(n)
    Dbar[ic] .= Tinv * yf
    return Abar, Bbar, Cbar, Dbar
end

xeq = [deg2rad(90), 0.0]
model = BilinearPendulum(xeq)

# Convert ALTRO trajectory to expanded state
Y = [zeros(ny) for k = 1:N]
U = [zeros(m) for k = 1:N-1]
Zsol = get_trajectory(altro)
Xsol = states(Zsol)
Usol = controls(Zsol)
for k = 1:N
    expand!(Y[k], Xsol[k])
    if k < N
        U[k] .= Usol[k]
    end
end

# Calculate scaling
T = Diagonal(sum(Y) ./ N)
Tinv = inv(T)
W = deepcopy(Y)
for k = 1:N
    W[k] = Tinv * Y[k]
end

# Build problem, using scaling
x0 = prob.x0
xf = prob.xf
A,B,C,D = buildbilinearconstraint(model, x0, xf, h, N, T=T)
ny,m = size(model.B)
Nx = size(A,2)
Nc,Nu = size(B)

# Test that the matrices match the expected
Â = Tinv * model.A * T
@test A[1:ny,1:ny] == -I(ny)
@test A[(1:ny) .+ 1ny,  1:ny] ≈ Â * h/2 + I
@test A[(1:ny) .+ 1ny, (1:ny) .+ 1ny] ≈ Â * h/2 - I
@test A[(1:ny) .+ 2ny, (1:ny) .+ 1ny] ≈ Â * h/2 + I
@test A[(1:ny) .+ 2ny, (1:ny) .+ 2ny] ≈ Â * h/2 - I
@test A[end-ny+1:end, end-ny+1:end] == -I(ny)

B̂ = Tinv * model.B
@test norm(B[1:ny,:]) == 0
@test B[(1:ny) .+ 1ny, 1:m] == B̂*h
@test B[(1:ny) .+ 2ny, (1:m) .+ m] == B̂*h

Ĉ = map(x->Tinv * x * T, model.C) 
@test norm(C[1][1:ny,:]) == 0
@test C[1][(1:ny) .+ 1ny, 1:ny] == h*Ĉ[1]/2
@test C[1][(1:ny) .+ 1ny, (1:ny) .+ 1ny] == h*Ĉ[1]/2
@test C[2][(1:ny) .+ 2ny, (1:ny) .+ 1ny] == h*Ĉ[1]/2
@test C[2][(1:ny) .+ 2ny, (1:ny) .+ 2ny] == h*Ĉ[1]/2

y0 = zeros(ny)
yf = zeros(ny)
expand!(y0, prob.x0)
expand!(yf, prob.xf)
@test D[1:ny] == Tinv * y0
@test D[(1:ny) .+ ny] ≈ Tinv*model.D*h
@test D[end-ny+1:end] ≈ Tinv*yf

# Test dynamics
Yvec = vcat(Y...)  # expanded states
Wvec = vcat(W...)  # scaled expanded states
Uvec = vcat(U...)
res0 = map(1:N-1) do k
    RD.dynamics_error(dpend, Zsol[k+1], Zsol[k])
end
res = A*Wvec + B*Uvec + sum(Uvec[i]*C[i]*Wvec for i = 1:Nu) + D

# Check residuals for initial and final conditions
@test norm(res[1:ny]) < 1e-8
@test norm(res[end-ny+1:end]) < 1e-8

# Check the difference in the continuous dynamics
Ydot = map(zip(Y,U)) do (y,u)
    ydot = model.A * y + model.B * u + u[1] * model.C[1]*y + vec(model.D)
    ydot[1:2]
end
Xdot = map(zip(Xsol,Usol)) do (x,u)
    RD.dynamics(pend, x, u)
end
norm(Xdot - Ydot, Inf)

# Build ADMM problem
using BilinearControl: BilinearADMM
A,B,C,D = buildbilinearconstraint(model, x0, xf, h, N, T=T)
Nx = size(A,2)
Nc,Nu = size(B)
Qbar = Diagonal(vcat([[diag(prob.obj[k].Q); fill(1e-4, ny-n)] for k = 1:N]...))
Rbar = Diagonal(vcat([Vector(diag(prob.obj[k].R)) for k = 1:N-1]...))
qbar = vcat([[prob.obj[k].q; fill(0, ny-n)] for k = 1:N]...)
rbar = vcat([Vector(prob.obj[k].r) for k = 1:N-1]...)
cbar = sum(obj.c for obj in prob.obj)

Tfull = Diagonal(repeat(diag(T), N))
Qbar_scaled = Tfull*Qbar*Tfull
qbar_scaled = Tfull*qbar

# Get initial trajectory
U0 = [fill(0.1, m) for k = 1:N-1] 
X0 = [zeros(n) for k = 1:N]
Y0 = [zeros(ny) for k = 1:N]
W0 = deepcopy(Y0) 
X0[1] .= prob.x0
expand!(Y0[1], X0[1])
W0[1] .= Tinv*Y0[1]
for k = 1:N-1
    U0[k] .= Usol[k]
    z = KnotPoint{2,1}(X0[k], U0[k], 0.0, h)
    X0[k+1] = RD.discrete_dynamics(dpend, z)
    expand!(Y0[k+1], X0[k+1])
    W0[k] .= Tinv * Y0[k+1]
end
Yvec = vcat(Y0...)
Wvec = vcat(W0...)
Uvec = vcat(U0...)

# Create the solver
solver = BilinearControl.BilinearADMM(A,B,C,D, Qbar,qbar, Rbar,rbar)
solver.opts.ϵ_rel_primal = 1e-3
solver.opts.ϵ_rel_dual = 1e-3
solver.opts.ϵ_abs_primal = 1e-3
solver.opts.ϵ_abs_dual = 1e-3
solver.opts.penalty_threshold = 4 
BilinearControl.setpenalty!(solver, 100.)
Wsol, Usol = BilinearControl.solve(solver, Wvec, Uvec, max_iters=100)

x,z,w = solver.x, solver.z, solver.w
BilinearControl.get_primal_tolerance(solver, x,z,w)
Ahat = BilinearControl.getAhat(solver, z)
Bhat = BilinearControl.getBhat(solver, x)
norm(Ahat*x)
norm(Bhat*z)
norm(z)
norm(A)
norm(C)
norm([norm(C*x) for C in solver.C], Inf)

res = A*Ysol + B*Usol + sum(Usol[i] * C[i] * Ysol for i = 1:Nu) + D
solver.A == A
norm(res)
norm(BilinearControl.primal_residual(solver, Ysol, Usol))