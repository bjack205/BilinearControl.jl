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

function splitcoeff(f, expr::SymbolicUtils.Div)
    @assert f(expr.den) "All denominators must be constant."
    expr_nocoeff, coeff = splitcoeff(f, expr.num)
    return (expr_nocoeff, coeff/expr.den)
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

# Original variables
@variables r[1:3] q[1:4] v[1:3] ω[1:3] F[1:3]
r_ = SA[r...]
q_ = SA[q...]
v_ = SA[v...]
ω_ = SA[ω...]
F_ = SA[F...]


# Create original state and control vectors
x0_vec = [r; q; v]
u_vec = [F..., ω...]
nx0 = length(x0_vec)
nu = length(u_vec)

# Expanded variables
qq = Symbolics.variables(:qq, 1:4, 1:4)
qqv = Symbolics.variables(:qqv, 1:4, 1:4, 1:3)

# Create extended states y
qq0_vec = [q[i]*q[j] for (i,j) in ij]
qqv0_vec = [q[i]*q[j]*v[k] for (i,j,k) in ijk]
qq_vec = [qq[i,j] for (i,j) in ij]
qqv_vec = [qqv[i,j,k] for (i,j,k) in ijk]
y0_vec = [qq0_vec; qqv0_vec]
y_vec = [qq_vec; qqv_vec]

# Create dictionary of substitutions, converting state to extended state
ij = NTuple{2,Int}[]
ijk = NTuple{3,Int}[]
for j = 1:4, i = j:4
    push!(ij, (i,j))
    for k = 1:3
        push!(ijk, (i,j,k))
    end
end
x2y_dict = Dict(value(x)=>value(y) for (x,y) in zip(y0_vec,y_vec))
@test length(x2y_dict) == 40

# Constants
c_vec = @variables m
constants = Set(value.(c_vec))
controls = Set(value.(u_vec))
iscoeff(x) = (x isa Number) || (x in constants)
isconstorcontrol(x) = iscoeff(x) || (x in controls)

# Dynamics of the original state
quat = UnitQuaternion(q, false)
A = SMatrix(quat)
rdot = A*v_
qdot = lmult(q)*SA[0, ω[1], ω[2], ω[3]] / 2
vdot = F_ / m - (ω_ × v_)

rdot = map(rdot) do expr
    filtersubstitute(iscoeff, expand(expr), x2y_dict)
end
x0dot_vec = [rdot; qdot; vdot]

# Dynamics of extended states
qqdot = map(ij) do (i,j)
    filtersubstitute(isconstorcontrol, expand(q[i]*qdot[j] + qdot[i]*q[j]), x2y_dict)
end

qqvdot = map(ijk) do (i,j,k)
    expr = q[i]*q[j]*vdot[k] + q[i]*qdot[j]*v[k] + qdot[i]*q[j]*v[k]
    filtersubstitute(isconstorcontrol, expand(expr), x2y_dict)
end
ydot_vec = [qqdot; qqvdot]

# Create expanded state vector and control vector
x_vec = [x0_vec; y_vec]
xdot_vec = [x0dot_vec; ydot_vec]
nx = length(x_vec)

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
    coeff = Num(coeff)

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

# Get all of the coefficients
#   Stored as a vector tuples:
#     (val, ix, iu, row)
#   where
#     val is the nonzero cofficient
#     ix is the index of the state vector
#     iu is the index of the control vector
#     row is the row (state index)
coeffs = Tuple{Real,Int,Int,Int}[]
for i = 1:length(x_vec)
    e = xdot_vec[i]
    _coeffs = getcoeffs(value(e), stateinds, controlinds, constants)
    row_coeffs = map(_coeffs) do coeff
        (coeff...,i)
    end
    append!(coeffs, row_coeffs)
end

# Sort into A,B,C,D matrices
Acoeffs = Tuple{Real,Int,Int}[]
Bcoeffs = Tuple{Real,Int,Int}[]
Ccoeffs = [Tuple{Real,Int,Int}[] for i = 1:length(controlinds)]
Dcoeffs = Tuple{Real,Int,Int}[]
for coeff in coeffs
    val,ix,iu,row = coeff
    if ix > 0 && iu == 0 
        push!(Acoeffs, (val,row,ix))
    elseif ix == 0 && iu > 0
        push!(Bcoeffs, (val,row,iu))
    elseif ix > 0 && iu > 0
        push!(Ccoeffs[iu], (val,row,ix))
    elseif ix == 0 && iu == 0
        push!(Dcoeffs, (val,row))
    else
        error("Got unexpected coefficient")
    end
end

# Sort the coefficients by column, then by row
function coeff_lessthan(a,b)
    r1,c1 = a[2],a[3]
    r2,c2 = b[2],b[3]
    if c1 == c2
        return r1 < r2
    else
        return c1 < c2
    end
end
sort!(Acoeffs, lt=coeff_lessthan)
sort!(Bcoeffs, lt=coeff_lessthan)
map(Ccoeffs) do coeff
    sort!(coeff, lt=coeff_lessthan)
end
sort!(Dcoeffs, lt=coeff_lessthan)

# Convert to SparseArrays
function coeffstosparse(m, n, coeffs)
    v = getindex.(coeffs,1)
    r = getindex.(coeffs,2)
    c = getindex.(coeffs,3)
    sparse(r, c, v, m, n)
end
function coeffstosparse(m, coeffs)
    v = getindex.(coeffs,1)
    r = getindex.(coeffs,2)
    sparsevec(r, v, m)
end
Asym = coeffstosparse(nx, nx, Acoeffs)
Bsym = coeffstosparse(nx, nu, Bcoeffs)
Csym = map(x->coeffstosparse(nx, nx, x), Ccoeffs)
Dsym = coeffstosparse(nx, Dcoeffs)

# Function inputs
name = "se3_angvel"

## Build function to build the sparse arrays
nx = length(x_vec)
nu = length(u_vec)
nc = length(c_vec)

# Rename inputs to
@variables _x[1:n] _u[1:m] _c[1:p]
toargs = Dict(value(x_vec[i])=>value(_x[i]) for i = 1:nx)
merge!(toargs, Dict(value(u_vec[i])=>value(_u[i]) for i = 1:nu))
merge!(toargs, Dict(value(c_vec[i])=>value(_c[i]) for i = 1:nc))

# Generate function to evaluate the dynamics
xdot_sub = substitute(xdot_vec, toargs)
xdot_expr = map(enumerate(xdot_sub)) do (i,xdot)
    :(xdot[$i] = $xdot)
end
dynamics_function = quote
    function $(Symbol(name * "_dynamics!"))(xdot, x, u, constants)
        _x,_u,_c = x, u, constants
        $(xdot_expr...)
        return
    end
end
dynamics_function

# Generate function to expand the state vector from original states
y0_sub = substitute(y0_vec, toargs)
expand_expr = map(enumerate(y0_sub)) do (i,y) 
    :(y[$i] = $y)
end
expand_function = quote
    function $(Symbol(name * "_expand!"))(y, x)
        _x = x
        $(expand_expr...)
        return y
    end
end
expand_function

# Generate expressions from symbolics
function genexprs(A, subs)
    map(enumerate(A.nzval)) do (i,e)
        # Convert to expression
        e_sub = substitute(e, subs)

        # Convert to expression
        expr = Symbolics.toexpr(e_sub)
        :(nzval[$i] = $expr)
    end
end
Aexprs = genexprs(Asym, toargs)
Bexprs = genexprs(Bsym, toargs)
Cexprs = map(1:m) do i
    Cexpr = genexprs(Csym[i], toargs)
    quote
        nzval = C[$i].nzval
        $(Cexpr...)
    end
end
Dexprs = genexprs(Dsym, toargs)

# Create update functions
update_functions = quote
    function $(Symbol(name * "_updateA!"))(A, constants)
        _c = constants
        nzval = A.nzval
        $(Aexprs...)
        return A
    end
    function $(Symbol(name * "_updateB!"))(B, constants)
        _c = constants
        nzval = B.nzval
        $(Bexprs...)
        return B
    end
    function $(Symbol(name * "_updateC!"))(C, constants)
        _c = constants
        $(Cexprs...)
        return D
    end
    function $(Symbol(name * "_updateD!"))(D, constants)
        _c = constants
        nzval = D.nzval
        $(Dexprs...)
        return D
    end
end
update_functions

# Create functions to generate sparse arrays
Cmatgen_expr = map(Csym) do Ci
    quote
        SparseMatrixCSC(n, n,
            $(Ci.colptr), 
            $(Ci.rowval),
            zeros($(nnz(Ci)))
        )
    end
end
genmats_function = quote
    function $(name * "_genarrays")()
        n = $nx
        m = $nu
        A = SparseMatrixCSC(n, n,
            $(Asym.colptr), 
            $(Asym.rowval),
            zeros($(nnz(Asym)))
        )
        B = SparseMatrixCSC(n, m,
            $(Bsym.colptr), 
            $(Bsym.rowval),
            zeros($(nnz(Bsym)))
        )
        C = [$(Cmatgen_expr...)]
        D = SparseVector(n,
            $(Dsym.nzind),
            zeros($(nnz(Dsym)))
        )
        return A,B,C,D
    end
end
genmats_function

expr = build_se3_dynamics_functions(
    Asym, Bsym, Csym, Dsym, xdot_vec, x_vec, u_vec, collect(constants)
)
expr
Vector(constants)
collect(constants)
Symbolics.to_expr
Symbolics.toexpr.(Asym.nzval)
Symbolics.gen_controllable
a_expr = build_function(Asym.nzval, constants)
b_expr = build_function(Bsym.nzval, constants)
c_expr = build_function(Csym.nzval, constants)
d_expr = build_function(Dsym.nzval, constants)