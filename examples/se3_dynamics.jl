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

include("symbolics_utils.jl")

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

function qrot(q)
    w,x,y,z = q
    ww = (w * w)
    xx = (x * x)
    yy = (y * y)
    zz = (z * z)

    xw = (w * x)
    xy = (x * y)
    xz = (x * z)
    yw = (y * w)
    yz = (y * z)
    zw = (w * z)

    
    A11 = ww + xx - yy - zz
    A21 = 2 * (xy + zw)
    A31 = 2 * (xz - yw)
    A12 = 2 * (xy - zw)
    A22 = ww - xx + yy - zz
    A32 = 2 * (yz + xw)
    A13 = 2 * (xz + yw)
    A23 = 2 * (yz - xw)
    A33 = ww - xx - yy + zz

    SA[
        A11 A12 A13
        A21 A22 A23
        A31 A32 A33
    ]
end

## Original variables
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
ij = NTuple{2,Int}[]
ijk = NTuple{3,Int}[]
for j = 1:4, i = j:4
    push!(ij, (i,j))
    for k = 1:3
        push!(ijk, (i,j,k))
    end
end
qq0_vec = [q[i]*q[j] for (i,j) in ij]
qqv0_vec = [q[i]*q[j]*v[k] for (i,j,k) in ijk]
qq_vec = [qq[i,j] for (i,j) in ij]
qqv_vec = [qqv[i,j,k] for (i,j,k) in ijk]
y0_vec = [qq0_vec; qqv0_vec]
y_vec = [qq_vec; qqv_vec]

# Create dictionary of substitutions, converting state to extended state
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

## Get all of the coefficients
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
sort!(Acoeffs, lt=coeff_lessthan)
sort!(Bcoeffs, lt=coeff_lessthan)
map(Ccoeffs) do coeff
    sort!(coeff, lt=coeff_lessthan)
end
sort!(Dcoeffs, lt=coeff_lessthan)

# Convert to SparseArrays
Asym = coeffstosparse(nx, nx, Acoeffs)
Bsym = coeffstosparse(nx, nu, Bcoeffs)
Csym = map(x->coeffstosparse(nx, nx, x), Ccoeffs)
Dsym = coeffstosparse(nx, Dcoeffs)

##################################
## Build functions
##################################

# Function inputs
name = "se3_angvel"

## Build function to build the sparse arrays
nx = length(x_vec)
nu = length(u_vec)
nc = length(c_vec)

# Rename inputs to input argument vectors 
@variables _x[1:nx] _u[1:nu] _c[1:nc]
toargs = Dict(value(x_vec[i])=>value(_x[i]) for i = 1:nx)
merge!(toargs, Dict(value(u_vec[i])=>value(_u[i]) for i = 1:nu))
merge!(toargs, Dict(value(c_vec[i])=>value(_c[i]) for i = 1:nc))

# Generate function to evaluate the dynamics
xdot_sub = substitute(xdot_vec, toargs)
xdot_expr = map(enumerate(xdot_sub)) do (i,xdot)
    :(xdot[$i] = $(Symbolics.toexpr(xdot)))
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
x_sub = substitute([x0_vec; y0_vec], toargs)
x_sub[end-6:end]
expand_expr = map(enumerate(x_sub)) do (i,y) 
    :(y[$i] = $(Symbolics.toexpr(y)))
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
Cexprs = map(1:nu) do i
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
        return C
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
    function $(Symbol(name * "_genarrays"))()
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


## Test generated functions
eval(dynamics_function)
eval(expand_function)
eval(update_functions)
eval(genmats_function)

x = [randn(3); normalize(randn(4)); randn(3)]
y = zeros(nx)
se3_angvel_expand!(y, x)
expand_function

## Text expand function
eval(expand_function)
let
    r = randn(3)
    q = normalize(randn(4))
    v = randn(3)
    x = [r; q; v]
    y = zeros(nx)
    se3_angvel_expand!(y, x)
    @test y[1:10] == x
    @test y[11] == q[1]^2
    @test y[12] == q[1]*q[2]
    @test y[end] == q[4]^2 * v[3] 
end

## Test dynamics
eval(dynamics_function)
let 
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

    # Create expanded state vector
    y = zeros(nx)
    se3_angvel_expand!(y, x)
    ydot = zero(y)

    # Test dynamics
    se3_angvel_dynamics!(ydot, y, u, constants)
    xdot = [qrot(q) * v; 0.5*lmult(q)*[0; ω]; F/m - (ω × v)]
    @test xdot ≈ ydot[1:10]
end

## Test bilinear dynamics
eval(genmats_function)
eval(update_functions)
let 
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

    # Create expanded state vector
    y = zeros(nx)
    se3_angvel_expand!(y, x)
    ydot = zero(y)

    # Test dynamics
    se3_angvel_dynamics!(ydot, y, u, constants)
    xdot = [qrot(q) * v; 0.5*lmult(q)*[0; ω]; F/m - (ω × v)]
    @test xdot ≈ ydot[1:10]

    # Build matrices
    A,B,C,D = se3_angvel_genarrays()
    se3_angvel_updateA!(A, constants)
    se3_angvel_updateB!(B, constants)
    se3_angvel_updateC!(C, constants)
    se3_angvel_updateD!(D, constants)
    ydot2 = A*y + B*u + sum(u[i]*C[i]*y for i = 1:nu) + D
    @test ydot2 ≈ ydot
    @test ydot2[1:10] ≈ xdot
end