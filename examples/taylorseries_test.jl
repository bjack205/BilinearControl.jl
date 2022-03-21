using Test
using Symbolics

vars = @variables x y z
@test expand_derivatives(getdifferential(value(x*y))(x^2 + 2x*y)) == 2
@test expand_derivatives(getdifferential(value(x^2))(x^2 + 2x*y)) == 1
e = x*(3y^2 + 4y) + y*x*(4 - 2y)
D = getdifferential(value(x*y^2))
@test expand_derivatives(D(Symbolics.expand(e))) == 1

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