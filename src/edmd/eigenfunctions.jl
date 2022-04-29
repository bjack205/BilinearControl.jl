function hermite(x::AbstractVector{<:AbstractFloat}; order::Int64 = 0)

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

function chebyshev(x::AbstractVector{<:AbstractFloat}; order::Int64 = 0)

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

function monomial(x::AbstractVector{<:AbstractFloat}; order::Int64 = 0)

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

state(xk::AbstractVector{<:AbstractFloat}) = xk
sine(xk::AbstractVector{<:AbstractFloat}) = sin.(xk)
cosine(xk::AbstractVector{<:AbstractFloat}) = cos.(xk)

state_transform(z::AbstractVector{<:AbstractFloat}, g) = g * z

function koopman_transform(x::AbstractVector{<:AbstractFloat}, function_list::Vector{String}, 
    order_list::Vector{Int64})

    num_func = length(function_list)
    z = 1

    for i in 1:num_func

        func = function_list[i]
        order = order_list[i]

        a = eval(Symbol(func))
        
        if order == 0
            func_eval = a(x)
        else
            func_eval = a(x; order)
        end
        
        z = vcat(z, func_eval)

    end

    return z

end

function build_eigenfunctions(X::VecOrMat{<:AbstractVector{<:AbstractFloat}}, 
                              U::VecOrMat{<:AbstractVector{<:AbstractFloat}}, 
                              function_list::Vector{String}, order_list::Vector{Int64})

    n = length(X[1])
    num_func = length(function_list)

    Z = Vector{Float64}[]
    Zu = Vector{Float64}[]
    
    kf(x) = koopman_transform(x, function_list, order_list)

    Z = map(kf, X)
    Zu = map(zip(CartesianIndices(U), U)) do (cind,u)
        vcat(Z[cind], vec(Z[cind]*u')) 
    end

    # for k in 1:length(X)
        
    #     xk = X[k]
    #     zk = kf(xk)
        
    #     push!(Z, zk)

    #     if k < length(X)
    #         uk = U[k]
    #         zu = vcat(zk, vec(zk*uk'))
    #         push!(Zu, zu)
    #     end

    # end

    # z0 = Z[1]

    return Z, Zu, kf

end