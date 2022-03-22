import Pkg; Pkg.activate(@__DIR__)
include("taylorseries.jl")
using Test
using Symbolics
using SparseArrays
using BenchmarkTools 

# getdifferential
vars = @variables x y z
@test expand_derivatives(getdifferential(value(x*y))(x^2 + 2x*y)) == 2
@test expand_derivatives(getdifferential(value(x^2))(x^2 + 2x*y)) == 1
e = x*(3y^2 + 4y) + y*x*(4 - 2y)
D = getdifferential(value(x*y^2))
@test expand_derivatives(D(Symbolics.expand(e))) == 1

@variables t θ(t)
e = θ^2 + 3 * θ
@test expand_derivatives(getdifferential(value(θ))(e)) - (3 + 2θ) == 0

θdot = Differential(t)(θ)
e = 2θ + θdot * θ + 4*Differential(t)(θ) + cos(θdot)
@test expand_derivatives(getdifferential(value(θdot))(e)) - (θ + 4 - sin(θdot)) == 0

# getconstant
vars = @variables x y z
vars0 = @variables x0 y0 z0
@test getconstant(value(x), vars) == 0
@test getconstant(value(x + y + 2), vars) == 2
@test getconstant(value(x0), vars) === value(x0)
@test iszero(getconstant(value(3 + 4*x0*sin(y0)), vars) - (3 + 4*x0*sin(y0)))

@variables t x(t) y(t)
xdot = Differential(t)(x)
ydot = Differential(t)(y)
vars = [x, y, xdot, ydot]
@test getconstant(value(x * xdot + xdot^2 + xdot + x), vars) == 0

# Taylor expansions
vars = @variables x y z
vars0 = @variables x0 y0 z0
e0 = sin(x0) + cos(x0)*(x - x0)
e1 = taylorexpand(sin(x), vars, vars0, 1)
@test iszero(e0 - e1)

e0 = sin(x0) + cos(x0)*(x - x0) - sin(x0)*(x - x0)^2/2
e1 = taylorexpand(sin(x), vars, vars0, 2)
@test iszero(e0 - e1)

e0 = sin(x0) + cos(x0)*(x - x0) - sin(x0)*(x - x0)^2/2 - cos(x0)*(x - x0)^3/6
e1 = taylorexpand(sin(x), vars, vars0, 3)
@test iszero(e0 - e1)

f = 2x - y
f0 = 2x0 - y0
e0 = sin(f0) + cos(f0)*(f - f0) - sin(f0)*(f - f0)^2/2
e1 = taylorexpand(sin(f), vars, vars0, 2)
@test iszero(e1 - e0)

f = 2x^2 - y
f0 = 2x0^2 - y0
e0 = sin(f0) + cos(f0)*(f - f0) - sin(f0)*(f - f0)^2/2
e1 = taylorexpand(sin(f), vars, vars0, 2)
@test iszero(e1 - e0)

e0 = x0/y0 + 1/y0*(x-x0) - x0/(y0^2)*(y-y0)
e1 = taylorexpand(x/y, vars, vars0, 1)
@test iszero(simplify(e0 - e1))

e0 = sin(exp(2x0^2)) + cos(exp(2x0^2))*(exp(2x0^2) * (2x^2 - 2x0^2) + exp(2x0^2) - exp(2x0^2))
e1 = taylorexpand(sin(exp(2x^2)), vars, vars0, 1)
@test iszero(simplify(expand(e0) - expand(e1)))

# Taylor expansions with dependent variables
@variables t x(t) y(t) x0 y0
vars = [x,y]
vars0 = [x0,y0]
@test taylorexpand(value(x), vars, vars0, 1) === value(x)
@test taylorexpand(x, vars, vars0, 1) === value(x)

@test taylorexpand(2x, vars, vars0, 1) - 2x == 0
@test taylorexpand(2x^2, vars, vars0, 1) - 2x^2 == 0
@test taylorexpand(2x^2, vars, vars0, 1) - 2x^2 == 0

e0 = sin(x0) + cos(x0)*(x - x0)
e1 = taylorexpand(sin(x), vars, vars0, 1)
@test iszero(e0 - e1)

# getpow
@variables t a(t) b(t) x y 
@test getpow(x) == 1
@test getpow(x^2) == 2
@test getpow(x^(3*2)) == 6
@test getpow(x * y) == 2
@test getpow(x * y^2) == 3
@test getpow(x * y * y) == 3
@test getpow(x + y) == 1
@test getpow(x + y^2) == 2
@test getpow(x + x * y^2) == 3
@test getpow((x - y)^2) == 2
@test getpow((x * x - x*y)^2) == 4

D = Differential(t)
@test getpow(a * b) == 2
@test getpow(a * b^2) == 3
@test getpow(a * D(b)^2) == 3
@test getpow(D(b^3)) == 1

# getcoeffs
@variables x y x0 y0
vars = [x,y]
nvars = [x y x^2 x*y y^2 x^3 x^2*y x*y^2 y^3]
exprs = [
    10x + y^3
    2*y^2
    cos(y0)*(y - y0) - sin(y0)*(y - y0)^2/2
    x*(x - 4y^2) - 4x
]
coeffs, rvals = getcoeffs(exprs, x, vars)
@test coeffs == [10,-4]
@test rvals == [1,4]

coeffs, rvals = getcoeffs(exprs, y, vars)
@test all(iszero, coeffs - [cos(y0) + sin(y0)*y0])
@test rvals == [3]

coeffs, rvals = getcoeffs(exprs, x^2, vars)
@test coeffs == [1]
@test rvals == [4]

coeffs, rvals = getcoeffs(exprs, x*y, vars)
@test isempty(coeffs)
@test isempty(rvals)

coeffs, rvals = getcoeffs(exprs, y^2, vars)
@test all(iszero, coeffs - [2, -sin(y0)/2])
@test rvals == [2,3] 

coeffs, rvals = getcoeffs(exprs, x^3, vars)
@test isempty(coeffs)
@test isempty(rvals)

coeffs, rvals = getcoeffs(exprs, x^2*y, vars)
@test isempty(coeffs)
@test isempty(rvals)

coeffs, rvals = getcoeffs(exprs, x*y^2, vars)
@test coeffs == [-4]
@test rvals == [4]

coeffs, rvals = getcoeffs(exprs, y^2*x, vars)
@test coeffs == [-4]
@test rvals == [4]

coeffs, rvals = getcoeffs(exprs, y^3, vars)
@test coeffs == [1] 
@test rvals == [1]

# Building state vector
@variables x
statevec = [x]
yvec = buildstatevector(statevec, 5)
@test all(iszero, yvec - [x, x^2, x^3, x^4, x^5])

@variables x y
statevec = [x,y]
yvec1 = buildstatevector(statevec, 4)
yvec0 = [x, y, x^2, x*y, y^2, x^3, x^2*y, x*y^2, y^3, x^4, x^3*y, x^2*y^2, x*y^3, y^4]
@test all(iszero, yvec1 - yvec0)

@variables x y z
statevec = [x,y,z]
yvec1 = buildstatevector(statevec, 2)
yvec0 = [x, y, z, x^2, x*y, x*z, y^2, y*z, z^2]
@test all(iszero, yvec1 - yvec0)

@variables t x(t)
ẋ = Differential(t)(x)
statevec = [x,ẋ]
yvec1 = buildstatevector(statevec, 3)
yvec0 = [x, ẋ, x^2, x*ẋ, ẋ^2, x^3, x^2*ẋ, x*ẋ^2, ẋ^3]
@test all(iszero, yvec1 - yvec0)

#############################################
## Pendulum
#############################################

@variables t x(t) x0 y(t)
Dt = Differential(t)
xdot = Dt(x)
ẋ = xdot
ẍ = (Dt^2)(x)
vars = [x, xdot]
order = 3

# Define the dynamics
function pendulum_dynamics(vars)
    x = vars[1]
    xdot = vars[2]
    a = -2.1
    b = 0.1
    xddot = a * sin(x)  + b * xdot 
    return [xdot, xddot]
end
a = -2.1
b = 0.1
xddot = a * sin(x)  + b * xdot 
statederivative = pendulum_dynamics(vars) 

# Get Taylor approximation of dynamics
@variables x0 ẋ0
vars0 = [x0, ẋ0]
order = 3
approx_dynamics = map(statederivative) do xdot
    Num(taylorexpand(xdot, vars, vars0, order))
end
xddot_approx = b*xdot + 
    a *(sin(x0) + cos(x0)*(x-x0) - sin(x0)*(x-x0)^2/2 - cos(x0)*(x-x0)^3/6)
@test approx_dynamics[2] - xddot_approx == 0
xddot_approx_const = a*(sin(x0) - cos(x0)*x0 - sin(x0)*x0^2/2 + cos(x0)*x0^3/6)

# Form the expanded vector
y = buildstatevector(statevec, order)
y0 = [x, ẋ, x^2, x*ẋ, ẋ^2, x^3, x^2*ẋ, x*ẋ^2, ẋ^3]
@test all(iszero, y - y0)

# Form the expanded state derivative
ydot = expand_derivatives.(Dt.(y))
ydot0 = [
    ẋ,               # x
    ẍ,               # xdot
    2x*ẋ,            # x^2
    ẋ^2 + x*ẍ,       # x*xdot
    2ẋ*ẍ,            # xdot^2
    3x^2*ẋ,          # x^3
    2x*ẋ^2 + x^2*ẍ,  # x^2*xdot
    ẋ^3 + x*2ẋ*ẍ,    # x*xdot^2
    3ẋ^2*ẍ,          # xdot^3
]
@test ydot0 - ydot == zeros(9)

# Substitute in the approximate dynamics
subs = Dict(Dt(statevec[i])=>approx_dynamics[i] for i = 1:length(statevec))
ydot_approx = map(ydot) do yi
    substitute(yi, subs)
end
@test iszero(ydot_approx[1] - xdot)
@test iszero(ydot_approx[2] - xddot_approx)
@test iszero(ydot_approx[3] - (2x*xdot))
@test iszero(ydot_approx[4] - (ẋ^2 + x*xddot_approx))
@test iszero(ydot_approx[5] - (2ẋ*xddot_approx))

# Get coeffs 
coeffs, rvals = getcoeffs(ydot_approx, y[1], vars)
r = coeffs[1] - a*(cos(x0) + sin(x0)*x0 - cos(x0)*(x0)^2 * 0.5)  # derivative of xddot wrt x
@test norm(filter(x->x isa Real, arguments(value(r)))) < 1e-12
@test coeffs[2] - xddot_approx_const == 0
@test rvals == [2, 4]

coeffs, rvals = getcoeffs(ydot_approx, y[2], vars)
@test coeffs - [1, b, 2*xddot_approx_const] == zeros(3)
@test rvals == [1, 2, 5]

coeffs, rvals = getcoeffs(ydot_approx, y[3], vars)
@test coeffs[3] - xddot_approx_const == 0
@test rvals == [2,4,7]

coeffs, rvals = getcoeffs(ydot_approx, x*ẋ^2, vars)
@test coeffs[1:2] == [2.0, 2*b]
@test rvals == 7:9

## Test build expanded vector function
pendulum_expand_expr = build_expanded_vector_function(y)
pendulum_expand! = eval(pendulum_expand_expr)
y_ = zeros(length(y))
x_ = [deg2rad(30), deg2rad(10)]
x0_ = zeros(2)
pendulum_expand!(y_, x_)
@test y_[1:2] == x_
@test y_[3] == x_[1]^2
@test y_[end] == x_[2]^3

# Test build A
Asym = getAsym(ydot_approx, y)
build_A_expr = build_bilinear_dynamics_functions(Asym, vars0)
pendulum_build_A! = eval(build_A_expr)
nterms = nnz(Asym)
n = length(y)
A = SparseMatrixCSC(n,n, copy(Asym.colptr), copy(Asym.rowval), zeros(nterms))
pendulum_build_A!(A, x0_, y_)

# Compare dynamics
xdot0 = pendulum_dynamics(x_)
xdot1 = (A*y_)[1:2]
@test norm(xdot0 - xdot1) < 1e-3

#############################################
## Forced Pendulum
#############################################

function pendulum_dynamics(states, controls)
    x = states[1]
    xdot = states[2]
    tau = controls[1]
    a = -2.1 # g / J⋅ℓ
    b = 0.1  # damping / J
    c = 0.5  # 1/J
    xddot = a * sin(x)  + b * xdot  + c*tau
    return [xdot, xddot]
end

# Set up analytical dynamics
@variables t x(t) τ 
n0,m = 2,1
a = -2.1
b = 0.1
c = 0.5
Dt = Differential(t)
xdot = Dt(x)

states = [x, xdot]
controls = [τ]
statederivative = pendulum_dynamics(states, controls)
@test statederivative[2] - (c*τ + b*xdot + a*sin(x)) == 0

# Get Taylor approximation of dynamics
@variables x0 ẋ0
vars0 = [x0, ẋ0]
order = 3
approx_dynamics = map(statederivative) do xdot
    Num(taylorexpand(xdot, states, vars0, order))
end
approx_dynamics
xddot_approx = c*τ + b*xdot + a *(
    sin(x0) + cos(x0)*(x-x0) - sin(x0)*(x-x0)^2/2 - cos(x0)*(x-x0)^3/6)
@test approx_dynamics[2] - xddot_approx == 0

# Form the expanded vector
y = buildstatevector(statevec, order)
n = length(y)  # new state dimension

# Form the expanded state derivative
ydot = expand_derivatives.(Dt.(y))

# Substitute in approximate dynamics
subs = Dict(Dt(statevec[i])=>approx_dynamics[i] for i = 1:length(statevec))
ydot_approx = map(ydot) do yi
    substitute(yi, subs)  # expand here?
end
ydot_approx

# Build symbolic matrices
Asym = getA(ydot_approx, y, controls)
@test Asym[1,2] == 1
@test Asym[4,5] == 1
Bsym = getB(ydot_approx, y, controls)
@test Bsym[2] == c
@test norm(Vector(Bsym[4:end])) == 0
@test norm(Matrix(Csym[1][1:3,:])) == 0
Csym = getC(ydot_approx, y, controls)
@test Csym[1][4,1] == c
@test Csym[1][5,2] == 2c
@test Csym[1][7,3] == c
@test Csym[1][8,4] == 2c
@test Csym[1][9,5] == 3c

# Test dynamics
A = similar(Asym, Float64)
B = similar(Bsym, Float64)
C = map(x->similar(x, Float64), Csym)
@test nnz(A) == nnz(Asym)
@test nnz(B) == nnz(B)
@test nnz(C[1]) == nnz(C[1])

updateA_expr, updateB_expr, updateC_expr = 
    build_bilinear_dynamics_functions(Asym, Bsym, Csym, vars0, controls)

pendulum_updateA! = eval(updateA_expr)
pendulum_updateB! = eval(updateB_expr)
pendulum_updateC! = eval(updateC_expr)

# Generate some inputs
x0_ = zeros(n0)
x_ = [deg2rad(30), deg2rad(10)]
y_ = zeros(n)
u_ = [0.5]
pendulum_expand!(y_, x_)
pendulum_updateA!(A, x0_, y_, u_)
pendulum_updateB!(B, x0_, y_, u_)
pendulum_updateC!(C, x0_, y_, u_)
ydot_ = A*y_ + B*u_ + u_[1]*C[1]*y_
xdot1 = ydot_[1:2]
xdot0 = pendulum_dynamics(x_, u_)
norm(xdot1 - xdot0)
@test norm(xdot0 - xdot1) < 1e-3