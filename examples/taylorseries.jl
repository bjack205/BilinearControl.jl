using Symbolics
using LinearAlgebra
using SparseArrays
using Symbolics
using Symbolics.SymbolicUtils
using Symbolics: value, istree

getpow(num::Num) = getpow(Symbolics.value(num))
getpow(x::Real) = 0
getpow(::Union{<:SymbolicUtils.Term,SymbolicUtils.Sym}) = 1
getpow(sym::SymbolicUtils.Pow{<:Any,<:Any,<:Integer}) = getpow(sym.base) * sym.exp
getpow(sym::SymbolicUtils.Pow) = error("Expression has a non-integer power") 
getpow(sym::SymbolicUtils.Mul) = mapreduce(getpow, +, arguments(sym))
getpow(sym::SymbolicUtils.Add) = maximum(getpow, arguments(sym))

"""
    taylorexpansion(f::Function, nargs, order=:x)

Create a symbolic Taylor expansion of `f` to order `order`, where `f` has `nargs` arguments.
The resulting expression will contain `Symbolics` array variables of length `nargs`, with
names `name` and `name0` (e.g. `x` and `x0`). 
"""
function taylorexpansion(f::Function, nargs, order; name::Symbol=:x)
    name0 = Symbol(string(name) * "0")
    x, x0 = @variables $name[nargs] $name0[nargs]
    f0 = f(x0...)

    # Assign first term in the series
    e = f0
    prevterms = [f0]
    for k = 1:order
        newterms = Num[]
        for term in prevterms
            for i = 1:nargs
                ddx = Differential(x0[i])(term)
                push!(newterms, ddx)
                e += ddx/factorial(k) * (x[i] - x0[i])^k
            end
        end
        prevterms = copy(newterms)
    end
    return e
end

"""
    taylorexpand(f::Symbolics.Symbolic, vars, vars0, order)
"""
function taylorexpand(f::SymbolicUtils.Term, vars, vars0, order)
    # Check if the term is one of the variables (e.g. a dependent variable)
    for i = 1:length(vars)
        if f === value(vars[i])
            return f
        end
    end

    # Generate a symbolic expression representing the Taylor expansion of the called function
    op = operation(f)
    args = arguments(f)
    nargs = length(args)
    f_approx = expand_derivatives(taylorexpansion(op, nargs, order, name=:_x))
    @variables _x[nargs] _x0[nargs]

    # Taylor expand the arguments
    subs = Pair.(vars, vars0)
    for i = 1:nargs
        # Taylor expand the argument
        arg_approx = taylorexpand(args[i], vars, vars0, order)

        # Substitute vars0 into original argument expression
        arg0 = substitute(args[i], subs)

        # Replace arguments to Taylor series function with argument expressions
        f_approx = substitute(f_approx, Dict(_x[i]=>arg_approx, _x0[i]=>arg0))
    end
    return f_approx
end

taylorexpand(num::Num, args...) = taylorexpand(value(num), args...) 
taylorexpand(x::Number, args...) = x
taylorexpand(sym::SymbolicUtils.Sym, args...) = sym

function taylorexpand(sym::SymbolicUtils.Add, args...)
    mapreduce(+, arguments(sym)) do arg
        taylorexpand(arg, args...)
    end
end

function taylorexpand(sym::SymbolicUtils.Mul, args...)
    mapreduce(*, arguments(sym)) do arg
        taylorexpand(arg, args...) 
    end
end

function taylorexpand(sym::SymbolicUtils.Pow{<:Real,<:Any,<:Real}, args...)
    taylorexpand(sym.base, args...)^sym.exp
end

function taylorexpand(sym::SymbolicUtils.Div, args...)
    newterm = SymbolicUtils.Term((x,y)->x/y, [sym.num, sym.den])
    taylorexpand(newterm, args...)
end

"""
    getconstant(e, vars)

Return a version of the sybolic expression `e` that is constant with respect 
to the variables in `vars`. The resulting expression can still have other variables 
inside of it, as long as they aren't any of the ones given in `vars`. For a summation 
of terms, the non-constant terms are dropped.
"""
getconstant(e::Real, vars) = e

function getconstant(e::SymbolicUtils.Symbolic, vars)
    termvars = Symbolics.get_variables(e)
    isconstant = !any(termvars) do tvar
        any(vars) do var
            hash(value(var)) === hash(value(tvar))
        end
    end
    if isconstant
        return e
    else
        return 0
    end
end

function getconstant(e::SymbolicUtils.Add, vars)
    mapreduce(+, arguments(e)) do arg
        getconstant(arg, vars)
    end
end

"""
    getdifferential(var::SymbolicUtils.Symbolic)

Get the differential for the variable `var`. For compound variables (i.e. multiplied 
variables, not powers of the same variable), the result is the composition of the individual
differentials. The effect of applying the resulting differential is the derivative with 
respect to the joint term.

If `D` is the output of `getdifferential`, then

    getdifferential(x*y)(x^2 + 2x*y) = 2
    getdifferential(x^2)(x^2 + 2x*y) = 1
    getdifferential(x*y^2)(x*(3y^2 + 4y) + y*x*(4 - 2y)) = 3 - 2 = 1

Note you will usually need to call `Symbolics.value` prior to calling this function.
"""
getdifferential(var::SymbolicUtils.Mul) = mapreduce(Differential, *, arguments(value(var)))
getdifferential(var::SymbolicUtils.Pow) = Differential(var)
getdifferential(var::SymbolicUtils.Sym) = Differential(var)
getdifferential(var::SymbolicUtils.Term) = Differential(var)

"""
    getcoeffs(exprs, var, basevars)

Get the linear coefficients with respect to `var` for each symbolic expression in `exprs`.
Any symbolic variable not in `basevars` is considered a constant and will be included in
the expression for the coefficients.
"""
function getcoeffs(exprs::Vector{Num}, var, basevars)
    rowvals = Int[]
    terms = Num[]
    D = getdifferential(value(var))
    for (i,e) in enumerate(exprs)
        # Expand the expression to get all terms as multiplications
        e_expanded = Symbolics.expand(e)

        # Take the derivative with respect to the current variable
        dvar = Symbolics.expand(expand_derivatives(D(e_expanded)))

        # Extract out the constant part of the expression
        coeff = getconstant(value(dvar), basevars)

        # If it's not zero, add to results
        if hash(coeff) != hash(Num(0))
            push!(rowvals, i)
            push!(terms, coeff)
        end
    end
    return terms, rowvals 
end

function _buildsparsematrix(exprs, vars, basevars)
    n = length(exprs)  # number of rows
    m = length(vars)   # number of columns
    nzval = Num[]
    rowval = Int[]
    colptr = zeros(Int, m+1)
    colptr[1] = 1
    for i = 1:m
        coeffs, rvals = getcoeffs(exprs, vars[i], basevars)
        nterms = length(coeffs)
        colptr[i+1] = colptr[i] + nterms
        append!(nzval, coeffs)
        append!(rowval, rvals)
    end
    return SparseMatrixCSC(n, m, colptr, rowval, nzval)
end

function getAsym(ydot, y)
    basevars = filter(x->getpow(x)==1, y)
    _buildsparsematrix(ydot, y, basevars)
end

function getAsym(ydot, y, u)
    basevars = filter(x->getpow(x)==1, y)
    append!(basevars, u)  # must be constant wrt to both original state and control
    _buildsparsematrix(ydot, y, basevars)
end

function getB(ydot, y, u)
    basevars = filter(x->getpow(x)==1, y)
    append!(basevars, u)  # must be constant wrt to both original state and control
    _buildsparsematrix(ydot, u, basevars)
end

function getC(ydot, y, u)
    basevars = filter(x->getpow(x)==1, y)
    append!(basevars, u)  # must be constant wrt to both original state and control
    map(u) do uk
        # Differentiate the dynamics wrt the current control variable
        dydotdu = Differential(uk).(ydot)

        # Get the coefficients now that the current control has been differentiated out
        _buildsparsematrix(dydotdu, y, basevars)
    end
end

function buildstatevector(x, order)
    iters = ceil(Int, log2(order))
    @show iters
    y = copy(x)
    for i = 1:iters
        y_ = trilvec(y*y')
        y = unique([y; y_])
    end
    filter(x->getpow(x) <= order, y)
end


function trilvec(A::AbstractMatrix)
    n = minimum(size(A))
    numel = n * (n + 1) ÷ 2
    v = zeros(eltype(A), numel)
    cnt = 1
    for j = 1:size(A,2)
        for i = j:size(A,1)
            v[cnt] = A[i,j]
            cnt += 1
        end
    end
    return v
end



function build_expanded_vector_function(y)
    vars = filter(x->getpow(x)==1, y)
    n0 = length(vars)
    @variables x[n0]
    subs = Dict(y[i]=>x[i] for i = 1:n0)
    exprs = map(enumerate(y)) do (i,yk)
        y_ = substitute(yk, subs)
        y_expr = Symbolics.toexpr(y_)
        :(y[$i] = $y_expr) 
    end
    
    quote
        function expand!(y, x)
            $(exprs...)
            return y
        end
    end
end

function build_bilinear_dynamics_functions(Asym, Bsym, Csym, vars0, controls)
    n0 = length(vars0)
    m = length(controls)

    # Rename states and controls to _x, _u array variables
    @variables _x0[n0] _u[m]  # use underscore to avoid potential naming conflicts
    subs = Dict(vars0[i]=>_x0[i] for i = 1:n0)
    merge!(subs, Dict(controls[i]=>_u[i] for i = 1:m))

    function genexprs(A, subs)
        map(enumerate(A.nzval)) do (i,e)
            # Substitute in new state and control variable names 
            e_sub = substitute(e, subs)

            # Convert to expression
            expr = Symbolics.toexpr(e_sub)
            :(nzval[$i] = $expr)
        end
    end
    Aexprs = genexprs(Asym, subs)
    Bexprs = genexprs(Bsym, subs)
    Cexprs = map(1:m) do i
        Cexpr = genexprs(Csym[i], subs)
        quote
            nzval = C[$i].nzval
            $(Cexpr...)
        end
    end
    updateA! = quote
        function (A, x0, y, u)
            _x0 = x0
            _u = u
            nzval = A.nzval
            $(Aexprs...)
            return A
        end
    end
    updateB! = quote
        function (B, x0, y, u)
            _x0 = x0
            _u = u
            nzval = B.nzval
            $(Bexprs...)
            return B
        end
    end
    updateC! = quote
        function (C, x0, y, u)
            _x0 = x0
            _u = u
            $(Cexprs...)
            return C
        end
    end
    return updateA!, updateB!, updateC!
end