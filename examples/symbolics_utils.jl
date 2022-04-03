
"""
Splits a multiplication of symbolic terms into constant coefficients and the 
variables. The function `f` should return `true` for any term that should be 
considered constant. 
"""
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