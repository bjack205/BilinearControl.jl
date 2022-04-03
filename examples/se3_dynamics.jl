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
include("rotation_utils.jl")


function se3_angvel_dynamics()
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

    # Constants
    c_vec = @variables m
    constants = Set(value.(c_vec))

    # Functions for checking coefficients
    controls = Set(value.(u_vec))
    iscoeff(x) = (x isa Number) || (x in constants)
    isconstorcontrol(x) = iscoeff(x) || (x in controls)

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
    y0_vec = [qq0_vec; qqv0_vec]  # in terms of original states
    y_vec = [qq_vec; qqv_vec]     # new variables

    # Create dictionary of substitutions, converting state to extended state
    x2y_dict = Dict(value(x)=>value(y) for (x,y) in zip(y0_vec,y_vec))

    # Dynamics of the original state
    quat = UnitQuaternion(q, false)
    A = SMatrix(quat)
    rdot = A*v_
    qdot = lmult(q)*SA[0, ω[1], ω[2], ω[3]] / 2
    vdot = F_ / m - (ω_ × v_)

    # Substitute extended states into dynamics
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

    return xdot_vec, x_vec, u_vec, c_vec, y0_vec
end


function build_symbolic_matrices(xdot_vec, x_vec, u_vec, constants)
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
    nx = length(x_vec)
    nu = length(u_vec)
    Asym = coeffstosparse(nx, nx, Acoeffs)
    Bsym = coeffstosparse(nx, nu, Bcoeffs)
    Csym = map(x->coeffstosparse(nx, nx, x), Ccoeffs)
    Dsym = coeffstosparse(nx, Dcoeffs)

    return Asym, Bsym, Csym, Dsym
end

function build_bilinear_dynamics_functions(name::AbstractString, xdot_vec, x_vec, u_vec, 
                                           c_vec, y0_vec; filename::AbstractString="")
    nx = length(x_vec)
    nu = length(u_vec)
    nc = length(c_vec)
    nx0 = nx - length(y0_vec)

    constants = Set(value.(c_vec))
    
    # Rename states, variables, and constants to _x, _u, and _c vectors 
    @variables _x[1:nx] _u[1:nu] _c[1:nc]
    toargs = Dict(value(x_vec[i])=>value(_x[i]) for i = 1:nx)
    merge!(toargs, Dict(value(u_vec[i])=>value(_u[i]) for i = 1:nu))
    merge!(toargs, Dict(value(c_vec[i])=>value(_c[i]) for i = 1:nc))

    # Generate function to expand the state vector from original states
    x0_vec = x_vec[1:nx0]
    x_sub = substitute([x0_vec; y0_vec], toargs)
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

    # Get sparse arrays of symbolic expressions
    Asym, Bsym, Csym, Dsym = build_symbolic_matrices(xdot_vec, x_vec, u_vec, constants)

    # Generate expressions from symbolics in 
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

    # Put all the functions in a single expression that can be saved to a file
    allfunctions = quote
        $expand_function
        $dynamics_function
        $update_functions
        $genmats_function
    end

    # (optional) save to file
    if !isempty(filename)
        write(filename, string(allfunctions))
    end

    return allfunctions
end



## Test generated functions
name = "se3_angvel"
xdot_vec, x_vec, u_vec, c_vec, y0_vec = se3_angvel_dynamics()
allfunctions = build_bilinear_dynamics_functions(
    name, xdot_vec, x_vec, u_vec, c_vec, y0_vec;
    filename=joinpath(@__DIR__,"se3_angvel_dynamics.jl")
)
eval(allfunctions)

## Test expand function
let nx = 50
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
let nx = 50
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
let nx = 50, nu = 6
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
