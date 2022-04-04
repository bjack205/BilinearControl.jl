import Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra
using SparseArrays
using Symbolics
using SymbolicUtils
using StaticArrays
using Rotations
using Symbolics: value
using IterTools
using Test

include("se3_dynamics.jl")
##

xdot_sym, x_sym, u_sym, c_sym, s0_sym = se3_symbolic_dynamics()

allfunctions = build_bilinear_dynamics_functions(
    "se3", xdot_sym, x_sym, u_sym, c_sym, s0_sym;
    filename=joinpath(@__DIR__,"se3_bilinear_dynamics.jl")
)
eval(allfunctions)

let nx = length(x_sym), nu = length(u_sym)
    # Create random state and control
    r = randn(3)
    q = normalize(randn(4))
    v = randn(3)
    ω = randn(3)
    F = randn(3)
    τ = randn(3)
    x = [r; q; v; ω]
    u = [F; τ] 

    # Constants
    m = 2.0  # mass
    J = Diagonal(fill(1.0, 3))
    c = [m, diag(J)...]

    # Create expanded state vector
    y = zeros(nx)
    se3_expand!(y, x)
    ydot = zero(y)

    # Test dynamics
    se3_dynamics!(ydot, y, u, c)
    xdot = [qrot(q) * v; 0.5*lmult(q)*[0; ω]; F/m - (ω × v); J\(τ - (ω × (J*ω)))]
    @test ydot[1:13] ≈ xdot

    # Test bilinear dynamics
    A,B,C,D = se3_genarrays()
    se3_updateA!(A, c)
    se3_updateB!(B, c)
    se3_updateC!(C, c)
    se3_updateD!(D, c)
    ydot2 = A*y + B*u + sum(u[i]*C[i]*y for i = 1:nu) + D
    @test ydot2 ≈ ydot
end

## Test generated functions
xdot_sym, x_sym, u_sym, c_sym, y0_sym = se3_angvel_symbolic_dynamics()
allfunctions = build_bilinear_dynamics_functions(
    "se3_angvel", xdot_sym, x_sym, u_sym, c_sym, y0_sym;
    filename=joinpath(@__DIR__,"se3_angvel_dynamics.jl")
)
eval(allfunctions)

let nx = length(x_sym), nu = length(u_sym)
    # Create random state and control
    r = randn(3)
    q = normalize(randn(4))
    v = randn(3)
    F = randn(3)
    ω = randn(3)
    x = [r; q; v]
    u = [F; ω] 
    m = 2.0  # mass
    constants = [m]

    # Test expand function
    y = zeros(nx)
    se3_angvel_expand!(y, x)
    @test y[1:10] == x
    @test y[11] == q[1]^2
    @test y[12] == q[1]*q[2]
    @test y[end] == q[4]^2 * v[3] 

    # Test dynamics
    ydot = zero(y)
    se3_angvel_dynamics!(ydot, y, u, constants)
    xdot = [qrot(q) * v; 0.5*lmult(q)*[0; ω]; F/m - (ω × v)]
    @test xdot ≈ ydot[1:10]

    # Build matrices
    A,B,C,D = se3_angvel_genarrays()
    se3_angvel_updateA!(A, constants)
    se3_angvel_updateB!(B, constants)
    se3_angvel_updateC!(C, constants)
    se3_angvel_updateD!(D, constants)

    # Test bilinear dynamics
    ydot2 = A*y + B*u + sum(u[i]*C[i]*y for i = 1:nu) + D
    @test ydot2 ≈ ydot
    @test ydot2[1:10] ≈ xdot
end

##
include("se3_dynamics_model.jl")
model0 = SE3Dynamics(2.0, I(3))
model = SE3BilinearDynamics(2.0, I(3))

n,m = RD.dims(model)
x0,u0 = rand(model0)
x = zeros(n)
expand!(model, x, x0)
xdot = zero(x)
RD.dynamics!(model, xdot, x, u0)

x0dot = RD.dynamics(model0, x0, u0)
@test xdot[1:13] ≈ x0dot

# Jacobian
J = zeros(n, n+m)
RD.jacobian!(model, J, xdot, x, u)

Jfd = zero(J)
FiniteDiff.finite_difference_jacobian!(
    Jfd, (y,z)->RD.dynamics!(model, y, z[1:152], z[153:end]), [x; u]
)
@test Jfd ≈ J rtol=1e-8

## Compare simulations
dmodel0 = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model0)
dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

times = range(0,1.0,step=1e-2)
length(times)
X0 = [copy(x0) for t in times]
X = [copy(x) for t in times]
for i = 1:length(times) - 1
    h = times[i+1] - times[i]
    z0 = KnotPoint(X0[i], u0, times[i], h)
    X0[i+1] = RD.discrete_dynamics(dmodel0, z0)

    RD.discrete_dynamics!(dmodel, X[i+1], X[i], u, times[i], h)
end
X0_ = [x[1:13] for x in X]
[norm(x[4:7]) for x in X]
norm(X0 - X0_, Inf)
norm(X0[end][1:3] - X0_[end][1:3], Inf)
norm(X0[end][8:10] - X0_[end][8:10], Inf)
norm(X0[end][11:13] - X0_[end][11:13], Inf)
x0f = X[end][1:13]
xf = zero(X[1])
expand!(model, xf, x0f)
norm(xf - X[end], Inf)