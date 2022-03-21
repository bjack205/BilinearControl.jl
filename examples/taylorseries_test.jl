using Test
using Symbolics

# getdifferential
vars = @variables x y z
@test expand_derivatives(getdifferential(value(x*y))(x^2 + 2x*y)) == 2
@test expand_derivatives(getdifferential(value(x^2))(x^2 + 2x*y)) == 1
e = x*(3y^2 + 4y) + y*x*(4 - 2y)
D = getdifferential(value(x*y^2))
@test expand_derivatives(D(Symbolics.expand(e))) == 1

# getconstant
vars0 = @variables x0 y0 z0
@test getconstant(value(x), vars) == 0
@test getconstant(value(x + y + 2), vars) == 2
@test getconstant(value(x0), vars) === value(x0)
@test iszero(getconstant(value(3 + 4*x0*sin(y0)), vars) - (3 + 4*x0*sin(y0)))

# Taylor expansions
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
@test terms == [10,-4]
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

# Define the dynamics
a = -2.1
b = 0.1
xddot = a * sin(x)  + b * xdot 
statederivative = [xdot, xddot]
e = statederivative[1]
typeof(value(x))
istree(value(x))
taylorexpand(statederivative[1], vars, vars0, order)

# Get Taylor approximation of dynamics
@variables x0 ẋ0
vars0 = [x0, ẋ0]
order = 3
approx_dynamics = map(statederivative) do xdot
    taylorexpand(xdot, vars, vars0, order)
end

# Form the expanded vector
y = buildstatevector(statevec, order)
y0 = [x, ẋ, x^2, x*ẋ, ẋ^2, x^3, x^2*ẋ, x*ẋ^2, ẋ^3]
@test all(iszero, y - y0)

# Form the expanded state derivative
ydot = expand_derivatives.(D.(y))
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

# Substitute in the dynamics
subs = Dict(Dt(statevec[i])=>statederivative[i] for i = 1:length(statevec))
for j = 1:length(ydot)
    ydot[j] = substitute(ydot[j], subs)
end
@test iszero(ydot[1] - xdot)
@test iszero(ydot[2] - xddot)
@test iszero(ydot[3] - (2x*xdot))
@test iszero(ydot[4] - (ẋ^2 + x*xddot))
@test iszero(ydot[5] - (ẋ^2 + x*xddot))