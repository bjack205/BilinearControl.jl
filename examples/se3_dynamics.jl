import Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra
using Symbolics
using SymbolicUtils
using StaticArrays
using Rotations
using Symbolics: value
using IterTools
using Test

function lmult(q)
    w, x, y, z = q
    SA[
        w -x -y -z;
        x  w -z  y;
        y  z  w -x;
        z -y  x  w;
    ]
end

function rmult(q)
    w, x, y, z = q
    SA[
        w -x -y -z;
        x  w  z -y;
        y -z  w  x;
        z  y -x  w;
    ]
end

function getcoeff(f, expr::SymbolicUtils.Mul)
    similarterm(
        expr,
        operation(expr),
        filter(f,  SymbolicUtils.unsorted_arguments(expr)),
        SymbolicUtils.symtype(expr);
        metadata=SymbolicUtils.metadata(expr)
    )
end
getcoeff(f, expr) = 1

function filtercoeff(f, expr::SymbolicUtils.Mul)
    istree(expr) || return expr
    similarterm(
        expr,
        operation(expr),
        filter(f |> !, SymbolicUtils.unsorted_arguments(expr)),
        SymbolicUtils.symtype(expr);
        metadata=SymbolicUtils.metadata(expr)
    )
end
filtercoeff(f, expr) = expr

function filtersubstitute(f, expr, dict; fold=true)
    coeff = getcoeff(f, expr)
    expr_nocoeff = filtercoeff(f, expr)
    haskey(dict, expr_nocoeff) && return coeff*dict[expr_nocoeff]
    if istree(expr)
        op = filtersubstitute(f, operation(expr), dict; fold=fold)
        if fold
            canfold = !(op isa SymbolicUtils.Symbolic)
            args = map(SymbolicUtils.unsorted_arguments(expr)) do x
                x_ = filtersubstitute(f, x, dict; fold=fold)
                canfold = canfold && !(x_ isa SymbolicUtils.Symbolic)
                x_
            end
            canfold && return op(args...)
            args
        else
            args = map(
                x->substitutemul(x, dict; fold=fold), 
                SymbolicUtils.unsorted_arguments(expr)
            )
        end
        similarterm(
            expr, 
            op, 
            args, 
            SymbolicUtils.symtype(expr); 
            metadata=SymbolicUtils.metadata(expr)
        )
    else
        expr
    end
end

filtersubstitute(f, expr::Num, dict; fold=true) = Num(filtersubstitute(f, value(expr), dict; fold=fold))

# Test substitutemul
@variables x y z a b[1:2]
constants = Set(value.([a, b...]))
iscoeff(x) = x isa Real || x in constants
@test getcoeff(iscoeff, value(x)) == 1
getcoeff(iscoeff, value(2x*y)) == 2
getcoeff(iscoeff, value(2x*y + z)) == 1
@test getcoeff(iscoeff, value(a*x)) - a == 0
@test getcoeff(iscoeff, value(b[1]*x*y*2)) - 2b[1] == 0

@variables xy yz x2y
subs = Dict(value(x*y)=>value(xy), value(y*z)=>value(yz), value(x^2*y)=>value(x2y))
e = value(x*y + (x + y + z)*x)
filtersubstitute(iscoeff, expand(e), subs, fold=true)
@test filtersubstitute(iscoeff, expand(e), subs, fold=true) - (2xy + x^2 + x*z) == 0
e = value(x*y*z + (z + y + x)*y)
@test filtersubstitute(iscoeff, expand(e), subs, fold=true) - (x*y*z + yz + xy + y^2) == 0
e = x*y*x + x*y
e2 = filtersubstitute(iscoeff, e, subs)
@test e2 isa Num
@test e2 - (x2y + xy) == 0

# Original variables
@variables r[1:3] q[1:4] v[1:3] ω[1:3] F[1:3]
r_ = SA[r...]
q_ = SA[q...]
v_ = SA[v...]
ω_ = SA[ω...]
F_ = SA[F...]

# Expanded variables
qq = Symbolics.variables(:qq, 1:4, 1:4)
qqv = Symbolics.variables(:qqv, 1:4, 1:4, 1:3)

# Create dictionary of substitutions
ij = NTuple{2,Int}[]
ijk = NTuple{3,Int}[]
for j = 1:4, i = j:4
    push!(ij, (i,j))
    for k = 1:3
        push!(ijk, (i,j,k))
    end
end
subs = Dict(value(q[i]*q[j])=>value(qq[i,j]) for (i,j) in ij)
subs2 = Dict(value(q[i]*q[j]*v[k])=>value(qqv[i,j,k]) for (i,j,k) in ijk)
merge!(subs, subs2)
@test length(subs) == 40

# Constants
@variables m J1 J2 J3
J = Diagonal(SA[J1, J2, J3])
Jinv = inv(J) 
constants = Set(value.([m, J1, J2, J3]))
controls = Set(value.([ω..., F...]))
iscoeff(x) = (x isa Number) || (x in constants)
isconstorcontrol(x) = iscoeff(x) || (x in controls)

# Dynamics of the original state
quat = UnitQuaternion(q, false)
A = SMatrix(quat)
rdot = A*v_
qdot = lmult(q)*SA[0, ω[1], ω[2], ω[3]] / 2
vdot = F_ / m - ω_ × v_

rdot = map(rdot) do expr
    filtersubstitute(iscoeff, expand(expr), subs)
end

qqdot = map(ij) do (i,j)
    filtersubstitute(isconstorcontrol, expand(q[i]*qdot[j] + qdot[i]*q[j]), subs)
end

qqvdot = map(ijk) do (i,j,k)
    expr = q[i]*q[j]*vdot[k] + q[i]*qdot[j]*v[k] + qdot[i]*q[j]*v[k]
    filtersubstitute(isconstorcontrol, expand(expr), subs)
end
qqvdot[1]