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

function splitcoeff(f, expr::SymbolicUtils.Mul)
    args = Symbolics.unsorted_arguments(expr)
    newargs = Any[]
    coeffs = Any[]
    for arg in args
        if f(arg)
            push!(coeffs, arg)
        else
            push!(newargs, arg)
        end
    end
    newterm = similarterm(
        expr,
        operation(expr),
        newargs,
        SymbolicUtils.symtype(expr);
        metadata=SymbolicUtils.metadata(expr)
    )
    coeff = similarterm(
        expr,
        operation(expr),
        coeffs,
        SymbolicUtils.symtype(expr);
        metadata=SymbolicUtils.metadata(expr)
    )
    return newterm, coeff
end
splitcoeff(f, expr) = f(expr) ? (1,expr) : (expr, 1)

function filtersubstitute(f, expr, dict; fold=true)
    expr_nocoeff, coeff = splitcoeff(f, expr)
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
iscoeff(value(a))
@test splitcoeff(iscoeff, value(x)) .- (x,1) == (0,0)
@test splitcoeff(iscoeff, value(2a)) .- (1,2a) == (0,0)
@test splitcoeff(iscoeff, value(2x*y)) .- (x*y,2) == (0,0)
@test splitcoeff(iscoeff, value(2x*y + z)) .- (2x*y + z, 1) == (0,0)
@test splitcoeff(iscoeff, value(a*x)) .- (x,a)  == (0,0)
@test splitcoeff(iscoeff, value(b[1]*x*y*2)) .- (x*y, 2b[1]) == (0,0)
@test splitcoeff(iscoeff, 2) == (1,2)

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
qq_vec = [qq[i,j] for (i,j) in ij]
qqv_vec = [qqv[i,j,k] for (i,j,k) in ijk]

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

# Create expanded state vector and control vector
x_vec = [r; q; v; qq_vec; qqv_vec]
u_vec = [F..., ω...]

# Store in a dictionary for fast look-ups
stateinds = Dict(value(x_vec[i])=>i for i in eachindex(x_vec))
controlinds = Dict(value(u_vec[i])=>i for i in eachindex(u_vec))

function getcoeffs(expr::SymbolicUtils.Symbolic, stateinds, controlinds, constants)
    iscoeff(x) = (x isa Real) || (x in constants)
    iscoefforcontrol(x) = iscoeff(x) || haskey(controlinds, x)

    # Extract off either a control or coefficient
    statevar, controlcoeff = splitcoeff(iscoefforcontrol, expr)

    # Split control from coefficient
    controlvar, coeff = splitcoeff(iscoeff, controlcoeff)

    if haskey(stateinds, statevar) 
        stateindex = stateinds[statevar]
        if haskey(controlinds, controlvar)
            # C matrix coefficient (bilinear)
            controlindex = controlinds[controlvar]
            return (coeff, stateindex, controlindex)
        else
            # A matrix coefficient (state only)
            @assert controlvar == 1 
            return (coeff, stateindex, 0)
        end
    elseif haskey(controlinds, controlvar)
        # B matrix coefficient (control only)
        @assert statevar == 1
        controlindex = controlinds[controlvar]
        return (coeff, 0, controlindex)
    else
        # D vector coefficient (constant)
        return (coeff, 0, 0)
    end
end

getcoeffs(expr::Real, args...) = (expr, 0, 0)

function getcoeffs(expr::SymbolicUtils.Add, args...)
    coeffs = map(SymbolicUtils.unsorted_arguments(expr)) do arg
        getcoeffs(arg, args...)
    end
    filter!(isnothing |> !, coeffs)
end

e = 3q[2] + 2.1r[1] + m/3 + q[1]*ω[1] + 2ω[2]*m - 3.2
coeffs = getcoeffs(value(e), stateinds, controlinds, constants)
@test length(coeffs) == 6
@test (3,5,0) in coeffs
@test (2.1, 1, 0) in coeffs
@test (m/3, 0, 0) in coeffs
@test (1,4,4) in coeffs
@test (2m,0,5) in coeffs
@test (-3.2, 0, 0) in coeffs