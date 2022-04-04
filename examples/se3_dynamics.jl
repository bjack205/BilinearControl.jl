include("symbolics_utils.jl")
include("rotation_utils.jl")

"""
Build the bilinear dynamics for SE(3) with 6DOF force / torque inputs. 
The state is `[r; q; v; ω]`, the control is `[F; τ]` with dynamics:

```julia
A(q) v               # rotate body velocity into world frame
lmult(q)*[0; ω] / 2  # quaternion kinematics
F/m - ω × v          # linear acceleration
Jinv*(τ - ω × Jω)    # Euler's equation
```

Note that the linear and angular velocities are both defined in the body frame.

Returns the following arguments:
- `xdot_sym` a vector of symbolic expressions for the expanded state derivative
- `x_vec` a vector of the symbolic variables defining the expanded state
- `u_vec` a vector of the symbolic variables defining the control vector
- `c_sym` a vector of the symbolic constants used in the dynamics
- `y0_sym` a symbolic vector of the expanded states, in terms of the original state variables
"""
function se3_symbolic_dynamics()
    ## Original variables
    @variables r[1:3] q[1:4] v[1:3] ω[1:3] F[1:3] τ[1:3]
    r_ = SA[r...]
    q_ = SA[q...]
    v_ = SA[v...]
    ω_ = SA[ω...]
    F_ = SA[F...]
    τ_ = SA[τ...]

    x0_sym = [r_; q_; v_; ω_]
    u_sym = [F_; τ_]

    # Constants
    c_sym = @variables m J1 J2 J3
    J = Diagonal(SA[J1, J2, J3])
    Jinv = inv(J)

    # Functions for checking coefficients
    constants = Set(value.(c_sym))
    controls = Set(value.(u_sym))
    iscoeff(x) = (x isa Number) || (x in constants)
    isconstorcontrol(x) = iscoeff(x) || (x in controls)

    # Compound states
    qq0, qq = createcompoundstates(q,q)
    ww0, ww = createcompoundstates(ω, ω)

    qw0, qw = createcompoundstates(q, ω)
    vw0, vw = createcompoundstates(v, ω)

    wwq0, wwq = createcompoundstates(ww, q, ww0)
    wwv0, wwv = createcompoundstates(ww, v, ww0)
    qqw0, qqw = createcompoundstates(qq, ω, qq0)
    qqv0, qqv = createcompoundstates(qq, v, qq0)

    s0_sym = [qq0; ww0; qw0; vw0; wwq0; wwv0; qqw0; qqv0]
    s_sym  = [qq;  ww;  qw;  vw;  wwq;  wwv;  qqw;  qqv]

    # Create dictionary of substitutions, converting state to extended state
    x2s_dict = Dict(value(x)=>value(y) for (x,y) in zip(s0_sym,s_sym))

    # Original dynamics
    quat = UnitQuaternion(q, false)
    A = SMatrix(quat)

    rdot = A*v_
    qdot = lmult(q)*SA[0, ω[1], ω[2], ω[3]] / 2
    vdot = F_ / m - (ω_ × v_)
    ωdot = Jinv * (τ_ - ω_ × (J * ω_))

    x0dot_sym = [rdot; qdot; vdot; ωdot]

    # Dynamics of extended states
    qqdot = map(qq) do expr 
        i,j = getindices(expr)
        # filtersubstitute(isconstorcontrol, expand(q[i]*qdot[j] + qdot[i]*q[j]), x2s_dict)
        expand(q[i]*qdot[j] + qdot[i]*q[j])
    end

    wwdot = map(ww) do expr 
        i,j = getindices(expr)
        # filtersubstitute(isconstorcontrol, expand(ω[i]*ωdot[j] + ωdot[i]*ω[j]), x2s_dict)
        # expand(ω[i]*ωdot[j] + ωdot[i]*ω[j])
        expand(ω[i]*τ[j]/J[j,j] + ω[j]*τ[i]/J[i,i])  # dropping higher ω terms
    end
    wwdot

    qwdot = map(qw) do expr 
        i,j = getindices(expr)
        # filtersubstitute(isconstorcontrol, expand(q[i]*ωdot[j] + qdot[i]*ω[j]), x2s_dict)
        expand(q[i]*ωdot[j] + qdot[i]*ω[j])
    end

    vwdot = map(vw) do expr 
        i,j = getindices(expr)
        # filtersubstitute(isconstorcontrol, expand(v[i]*ωdot[j] + vdot[i]*ω[j]), x2s_dict)
        expand(v[i]*ωdot[j] + vdot[i]*ω[j])
    end
    vwdot

    wwqdot = map(wwq) do expr 
        i,j,k = getindices(expr)
        expand(
            # expand(ωdot[i]*ω[j]*q[k]) + expand(ω[i]*ωdot[j]*q[k]) + expand(ω[i]*ω[j]*qdot[k])
            expand(τ[i]*ω[j]*q[k]/J[i,i]) + expand(ω[i]*τ[j]*q[k]/J[j,j])  # dropping higher ω terms
        )
    end

    wwvdot = map(wwv) do expr 
        i,j,k = getindices(expr)
        expand(
            # expand(ωdot[i]*ω[j]*v[k]) + expand(ω[i]*ωdot[j]*v[k]) + expand(ω[i]*ω[j]*vdot[k])
            expand(τ[i]*ω[j]*v[k]/J[i,i]) + expand(ω[i]*τ[j]*v[k]/J[j,j]) + expand(ω[i]*ω[j]*F[k]/m)  # dropping higher terms
        )
    end

    qqwdot = map(qqw) do expr 
        i,j,k = getindices(expr)
        expand(
            # expand(qdot[i]*q[j]*ω[k]) + expand(q[i]*qdot[j]*ω[k]) + expand(q[i]*q[j]*ωdot[k])
            expand(q[i]*q[j]*τ[k]/J[k,k])  # dropping higher terms
        )
    end

    qqvdot = map(qqv) do expr 
        i,j,k = getindices(expr)
        expand(
            # expand(qdot[i]*q[j]*v[k]) + expand(q[i]*qdot[j]*v[k]) + expand(q[i]*q[j]*vdot[k])
            expand(q[i]*q[j]*F[k]/m)
        )
    end

    # Replace compound states with their symbolic variables
    s0dot_sym = [qqdot; wwdot; qwdot; vwdot; wwqdot; wwvdot; qqwdot; qqvdot]

    # Expanded state vector
    xdot_sym = map([x0dot_sym; s0dot_sym]) do expr
        filtersubstitute(isconstorcontrol, expand(expr), x2s_dict)
    end
    x_sym = [x0_sym; s_sym]

    return xdot_sym, x_sym, u_sym, c_sym, s0_sym
end


"""
Build the bilinear dynamics for SE(3) with 3DOF force and 3DOF angular velocity inputs. 
This mapping is exact.

The state is `[r; q; v]`, the control is `[F; ω]` with dynamics:

```julia
A(q) v               # rotate body velocity into world frame
lmult(q)*[0; ω] / 2  # quaternion kinematics
F/m - ω × v          # linear acceleration
```

Note that the linear and angular velocities are both defined in the body frame.

Returns the following arguments:
- `xdot_sym` a vector of symbolic expressions for the expanded state derivative
- `x_vec` a vector of the symbolic variables defining the expanded state
- `u_vec` a vector of the symbolic variables defining the control vector
- `c_sym` a vector of the symbolic constants used in the dynamics
- `y0_sym` a symbolic vector of the expanded states, in terms of the original state variables
"""
function se3_angvel_symbolic_dynamics()
    ## Original variables
    @variables r[1:3] q[1:4] v[1:3] ω[1:3] F[1:3]
    r_ = SA[r...]
    q_ = SA[q...]
    v_ = SA[v...]
    ω_ = SA[ω...]
    F_ = SA[F...]

    # Create original state and control vectors
    x0_sym = [r; q; v]
    u_sym = [F..., ω...]
    nx0 = length(x0_sym)
    nu = length(u_sym)

    # Constants
    c_sym = @variables m
    constants = Set(value.(c_sym))

    # Functions for checking coefficients
    controls = Set(value.(u_sym))
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
    qq0_sym = [q[i]*q[j] for (i,j) in ij]
    qqv0_sym = [q[i]*q[j]*v[k] for (i,j,k) in ijk]
    qq_sym = [qq[i,j] for (i,j) in ij]
    qqv_sym = [qqv[i,j,k] for (i,j,k) in ijk]
    s0_sym = [qq0_sym; qqv0_sym]  # in terms of original states
    s_sym = [qq_sym; qqv_sym]     # new variables

    # Create dictionary of substitutions, converting state to extended state
    x2s_dict = Dict(value(x)=>value(s) for (x,s) in zip(s0_sym,s_sym))

    # Dynamics of the original state
    quat = UnitQuaternion(q, false)
    A = SMatrix(quat)
    rdot = A*v_
    qdot = lmult(q)*SA[0, ω[1], ω[2], ω[3]] / 2
    vdot = F_ / m - (ω_ × v_)

    # Substitute extended states into dynamics
    rdot = map(rdot) do expr
        filtersubstitute(iscoeff, expand(expr), x2s_dict)
    end
    x0dot_sym = [rdot; qdot; vdot]

    # Dynamics of extended states
    qqdot = map(ij) do (i,j)
        filtersubstitute(isconstorcontrol, expand(q[i]*qdot[j] + qdot[i]*q[j]), x2s_dict)
    end

    qqvdot = map(ijk) do (i,j,k)
        expr = q[i]*q[j]*vdot[k] + q[i]*qdot[j]*v[k] + qdot[i]*q[j]*v[k]
        filtersubstitute(isconstorcontrol, expand(expr), x2s_dict)
    end
    sdot_sym = [qqdot; qqvdot]

    # Create expanded state vector and control vector
    x_sym = [x0_sym; s_sym]
    xdot_sym = [x0dot_sym; sdot_sym]

    return xdot_sym, x_sym, u_sym, c_sym, s0_sym
end

"""
Takes symbolic vectors defining the dynamics of a bilinear system, and extracts 
out the coefficients to form a bilinear system of the form:

```math
A x + B u + \\sum_{i = 1}^m u_i C_i x + D
```
"""
function build_symbolic_matrices(xdot_sym, x_sym, u_sym, c_sym)
    # Store in a dictionary for fast look-ups
    stateinds = Dict(value(x_sym[i])=>i for i in eachindex(x_sym))
    controlinds = Dict(value(u_sym[i])=>i for i in eachindex(u_sym))
    constants = Set(value.(c_sym))

    ## Get all of the coefficients
    #   Stored as a vector tuples:
    #     (val, ix, iu, row)
    #   where
    #     val is the nonzero cofficient
    #     ix is the index of the state vector
    #     iu is the index of the control vector
    #     row is the row (state index)
    coeffs = Tuple{Real,Int,Int,Int}[]
    for i = 1:length(x_sym)
        e = xdot_sym[i]
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
    nx = length(x_sym)
    nu = length(u_sym)
    Asym = coeffstosparse(nx, nx, Acoeffs)
    Bsym = coeffstosparse(nx, nu, Bcoeffs)
    Csym = map(x->coeffstosparse(nx, nx, x), Ccoeffs)
    Dsym = coeffstosparse(nx, Dcoeffs)

    return Asym, Bsym, Csym, Dsym
end

"""
Builds a Julia expression containing definitions for the following functions:

    <name>_expand!(y,x)

Expand the original state vector `x` to the expanded state vector `y`.

    <name>_dynamics!(ydot, y, u, c)

Evaluate the continuous dynamics with state `y`, control `u`, and constants `c`, saving 
the output in `ydot`.

    <name>_updateA!(A, c)
    <name>_updateB!(A, c)
    <name>_updateC!(A, c)
    <name>_updateD!(A, c)

Update the bilinear matrices using the vector of constants `c`.

    A,B,C,D = <name>_genarrays()

Generate the sparse bilinear matrices, initialized with the correct sparsity structure. 
Must call `<name>_updateX!` to fill them in.
"""
function build_bilinear_dynamics_functions(name::AbstractString, xdot_sym, x_sym, u_sym, 
                                           c_sym, y0_sym; filename::AbstractString="")
    nx = length(x_sym)
    nu = length(u_sym)
    nc = length(c_sym)
    nx0 = nx - length(y0_sym)

    constants = Set(value.(c_sym))
    
    # Rename states, variables, and constants to _x, _u, and _c vectors 
    @variables _x[1:nx] _u[1:nu] _c[1:nc]
    toargs = Dict{Symbolics.Symbolic, Symbolics.Symbolic}(
        value(x_sym[i])=>value(_x[i]) for i = 1:nx
    )
    merge!(toargs, Dict(value(u_sym[i])=>value(_u[i]) for i = 1:nu))
    merge!(toargs, Dict(value(c_sym[i])=>value(_c[i]) for i = 1:nc))

    # Generate function to expand the state vector from original states
    x0_sym = x_sym[1:nx0]
    x_sub = substitute([x0_sym; y0_sym], toargs)
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
    xdot_sub = substitute(xdot_sym, toargs)
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
    Asym, Bsym, Csym, Dsym = build_symbolic_matrices(xdot_sym, x_sym, u_sym, constants)

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


