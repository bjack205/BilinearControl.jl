
# Test filtersubstitute 
@variables x y z a b[1:2]
constants = Set(value.([a, b...]))
iscoeff(x) = x isa Real || x in constants
iscoeff(value(a))
@test splitcoeff(iscoeff, value(x)) .- (x,1) == (0,0)
@test splitcoeff(iscoeff, value(2a)) .- (1,2a) == (0,0)
@test splitcoeff(iscoeff, value(2x*y)) .- (x*y,2) == (0,0)
@test splitcoeff(iscoeff, value(2x*y + z)) .- (2x*y + z, 1) == (0,0)
@test splitcoeff(iscoeff, value(a*x)) .- (x,a)  == (0,0)
@test splitcoeff(iscoeff, value(b[1]*x*y*2)) .- (x*y, 2b[1]) == (0,0)
@test splitcoeff(iscoeff, 2) == (1,2)
@test splitcoeff(iscoeff, value(x/m)) .- (x, 1//m) == (0,0)

@variables xy yz x2y
subs = Dict(value(x*y)=>value(xy), value(y*z)=>value(yz), value(x^2*y)=>value(x2y))
e = value(x*y)
@test filtersubstitute(iscoeff, expand(e), subs, fold=true) - xy == 0
e = value(x*y + z*y)
@test filtersubstitute(iscoeff, expand(e), subs) - (xy + yz) == 0
e = value(x*y + (x + y + z)*x)
@test filtersubstitute(iscoeff, expand(e), subs, fold=true) - (2xy + x^2 + x*z) == 0
e = value(x*y*z + (z + y + x)*y)
@test filtersubstitute(iscoeff, expand(e), subs, fold=true) - (x*y*z + yz + xy + y^2) == 0
e = x*y*x + x*y
e2 = filtersubstitute(iscoeff, e, subs)
@test e2 isa Num
@test e2 - (x2y + xy) == 0

# Test getcoeffs
e = 3q[2] + 2.1r[1] + m/3 + q[1]*ω[1] + 2ω[2]*m - 3.2
coeffs = getcoeffs(value(e), stateinds, controlinds, constants)
@test length(coeffs) == 6
@test (3,5,0) in coeffs
@test (2.1, 1, 0) in coeffs
@test (m/3, 0, 0) in coeffs
@test (1,4,4) in coeffs
@test (2m,0,5) in coeffs
@test (-3.2, 0, 0) in coeffs
