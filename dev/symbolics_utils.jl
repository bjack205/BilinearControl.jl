
"""
Splits a multiplication of symbolic terms into constant coefficients and the 
variables. The function `f` should return `true` for any term that should be 
considered constant. 
"""
function splitcoeff(f, expr::SymbolicUtils.Mul; recursive=false)
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

function splitcoeff(f, expr::SymbolicUtils.Div; recursive=false)
    @assert f(expr.den) "All denominators must be constant."
    expr_nocoeff, coeff = splitcoeff(f, expr.num)
    return (expr_nocoeff, coeff/expr.den)
end

splitcoeff(f, expr; recursive=false) = f(expr) ? (1,expr) : (expr, 1)

"""
    filtersubstitute(f, expr, dict; [fold])

Make a substitution, allowing for substitution of factors within a multiplicative 
expression. For each multiplicative term, the constant coefficients are filtered out
and the resulting expression is checked against the entries in `dict` for a match.

This is used to substitute compound variables that show up in larger expressions, e.g.

    2*x*y*c 

Can become  

    2*z*c

If `dict` contains the pair `x*y=>z` and `f` is the `getcoeff` function below:

    constants = Set([c])
    iscoeff(x) = x isa Real || x in constants 

"""
function filtersubstitute(f, expr, dict; fold=true)
    expr_nocoeff, coeff = splitcoeff(f, expr)
    haskey(dict, expr_nocoeff) && return coeff*dict[expr_nocoeff]
    if istree(expr) && !(expr isa Symbolics.Mul)  # don't split a Mul object
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
                x->filtersubstitute(f, x, dict; fold=fold), 
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

"""
    getcoeffs(expr, stateinds, controlinds, constants)

Returns a vector of the constant coefficients for a bilinear expression.
Each tuple has the form:

    (val, ix, iu, row)

where
* `val`` is the nonzero cofficient
* 'ix' is the index of the state vector. `0` if independent of the state.
* 'iu' is the index of the control vector. `0` if independent of the control.
* 'row' is the row (state index)

The `stateinds` and `controlinds` inputs are dictionaries that map state and control 
symbolic variables to their indices in the state and control vectors. The `constants` 
argument should be a collection (usually a `Set` for efficiency) that contains any 
expressions that should be considered constant.
"""
function getcoeffs(expr::SymbolicUtils.Symbolic, stateinds, controlinds, constants)
    iscoeff(x) = (x isa Real) || (x in constants)
    iscoefforcontrol(x) = iscoeff(x) || haskey(controlinds, x)

    # Extract off either a control or coefficient
    statevar, controlcoeff = splitcoeff(iscoefforcontrol, expr)

    # Split control from coefficient
    controlvar, coeff = splitcoeff(iscoeff, controlcoeff)
    coeff = Num(coeff)
    # @show statevar, controlvar, coeff

    if haskey(stateinds, statevar) 
        stateindex = stateinds[statevar]
        if haskey(controlinds, controlvar)
            # C matrix coefficient (bilinear)
            controlindex = controlinds[controlvar]
            return [(coeff, stateindex, controlindex)]
        else
            # A matrix coefficient (state only)
            @assert controlvar == 1 
            return [(coeff, stateindex, 0)]
        end
    elseif haskey(controlinds, controlvar)
        # B matrix coefficient (control only)
        @assert statevar == 1
        controlindex = controlinds[controlvar]
        return [(coeff, 0, controlindex)]
    else
        # D vector coefficient (constant)
        isone(expr) = expr === one(expr) 
        if isone(statevar) && isone(controlvar)
            return [(coeff, 0, 0)]
        else
            return Tuple{Real,Int,Int}[] 
        end
    end
end

getcoeffs(expr::Real, args...) = (expr, 0, 0)

function getcoeffs(expr::SymbolicUtils.Add, args...)
    eargs = SymbolicUtils.unsorted_arguments(expr)
    coeffs = Tuple{Real,Int,Int}[]
    for i = 1:length(eargs)
        append!(coeffs, getcoeffs(eargs[i], args...))
    end
    coeffs
end

function getcoeffs(expr::SymbolicUtils.Div, stateinds, controlinds, constants)
    iscoeff(x) = (x isa Real) || (x in constants)
    @assert iscoeff(expr.den) "All denominators must be constant, got $(expr.den)."
    coeffs = getcoeffs(expr.num, stateinds, controlinds, constants)
    map(coeffs) do (v,r,c) 
        (v / expr.den, r, c)
    end
end

"""
    coeff_lessthan(a,b)

Comparison operation for the tuples output by [`getcoeffs`](@ref). Sorts 
elements by columns, then row, so that they can be stored in a SparseMatrixCSC.
"""
function coeff_lessthan(a,b)
    r1,c1 = a[2],a[3]
    r2,c2 = b[2],b[3]
    if c1 == c2
        return r1 < r2
    else
        return c1 < c2
    end
end

"""
    coeffstosparse(m, n, coeffs)
    coeffstosparse(m, coeffs)

Convert of vector of coefficients output by [`getcoeffs`](@ref) to a 
`SparseMatrixCSC{Num,Int}` of size `(m,n)`.
"""
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

function getindices(sym::Symbolics.Term)
    @assert operation(sym) === Base.getindex
    Int.(arguments(sym)[2:end])
end
getindices(num::Num) = getindices(value(num))

function createcompoundstates(x,y, x0=x, y0=y)
    @assert length(x0) == length(x)
    @assert length(y0) == length(y)
    n = length(x)
    m = length(y)
    x_parent = value(Symbolics.getparent(value(x[1])))
    y_parent = value(Symbolics.getparent(value(y[1])))
    xname = Symbolics.getname(x_parent)
    yname = Symbolics.getname(y_parent)
    x_shape = Symbolics.getmetadata(x_parent, Symbolics.ArrayShapeCtx)
    y_shape = Symbolics.getmetadata(y_parent, Symbolics.ArrayShapeCtx)
    xyname = Symbol(string(xname) * string(yname))
    aresame = xname == yname

    xy, = @variables $xyname[x_shape...,y_shape...]

    ij = NTuple{2,Int}[]
    for j = 1:m
        i0 = aresame ? j : 1
        for i = i0:n
            push!(ij, (i,j))
        end
    end
    xy0_vec = [x0[i]*y0[j] for (i,j) in ij]
    xy_vec = [xy[getindices(x[i])..., getindices(y[j])...] for (i,j) in ij]
    return xy0_vec, xy_vec
end