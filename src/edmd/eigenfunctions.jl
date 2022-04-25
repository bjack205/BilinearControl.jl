function hermite(x::Vector{Float64}; order::Int64 = 0)

    if order == 0
        return Float64[]
    end
    
    T0 = ones(length(x))
    T1 = 2 .* x

    hermite_poly = [T0]
    push!(hermite_poly, T1)

    for p in 2:order

        next_T = (2 .* x .* hermite_poly[p]) - 2 .* p .* hermite_poly[p-1]
        push!(hermite_poly, next_T)

    end

    hermite_poly = reduce(vcat, hermite_poly[3:end])

    return hermite_poly
    
end

function chebyshev(x::Vector{Float64}; order::Int64 = 0)

    if order == 0
        return Float64[]
    end
    
    T0 = ones(length(x))
    T1 = x

    chebyshev_poly = [T0]
    push!(chebyshev_poly, T1)

    for p in 2:order

        next_T = (2 .* x .* chebyshev_poly[p]) - chebyshev_poly[p-1]
        push!(chebyshev_poly, next_T)

    end

    chebyshev_poly = reduce(vcat, chebyshev_poly[3:end])

    return chebyshev_poly
    
end

function monomial(x::Vector{Float64}; order::Int64 = 0)

    if order == 0
        return Float64[]
    end

    monomials = [x]
    row_start_ind = ones(length(x))

    for p in 2:order

        prev_row_start_ind = row_start_ind

        mono_combinations = x * monomials[p-1]'
        mono_permutations = mono_combinations[1, :]
        row_start_ind = [size(mono_combinations)[2]]

        for i in 2:size(mono_combinations)[1]

            row_start_ind = vcat(row_start_ind, Int(row_start_ind[i-1] - prev_row_start_ind[i-1]))
            current_row_permutations = mono_combinations[i, end-row_start_ind[i]+1:end]
            mono_permutations = vcat(mono_permutations, current_row_permutations)

        end

        push!(monomials, mono_permutations)
    
    end

    monomials = reduce(vcat, monomials[2:end])

    return monomials
        
end

state(xk::Vector{Float64}) = xk
sine(xk::Vector{Float64}) = sin.(xk)
cosine(xk::Vector{Float64}) = cos.(xk)

function build_eigenfunctions(X::Vector{Vector{Float64}}, U::Vector{Vector{Float64}}, 
                              function_list::Vector{String}, order_list::Vector{Int64})

    n = length(X[1])
    num_func = length(function_list)

    Z = Vector{Float64}[]
    Zu = Vector{Float64}[]
    
    for k in 1:length(X)
        
        xk = X[k]
        zk = 1
        
        for i in 1:num_func

            func = function_list[i]
            order = order_list[i]

            a = eval(Symbol(func))
            
            if order == 0
                func_eval = a(xk)
            else
                func_eval = a(xk; order)
            end
            
            zk = vcat(zk, func_eval)

        end
        
        push!(Z, zk)

        if k < length(X)
            uk = U[k]
            zu = vcat(zk, vec(zk*uk'))
            push!(Zu, zu)
        end

    end

    z0 = Z[1]

return Z, Zu, z0

end