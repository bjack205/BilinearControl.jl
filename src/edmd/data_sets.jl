function pendulum_test_data(num_knots, m, U_type::String)

    if U_type == "constant"
        u0 = 2 .* ones(m)
    elseif U_type == "random"
        u0 = 0.01 .* ones(m)
    end

    U = [u0]

    for k in 1:(num_knots-1)

        if U_type == "constant"
            u = 2 .* ones(m)
        elseif U_type == "random"
            u = -5 .+ (10*rand(Float64, m))
        end

        push!(U, u)

    end
    
    return U

end

function create_training_U(num_knots, m)

    u0 = -3 .* ones(m)
    U = [u0]

    for k in 1:(num_knots-1)

        u = -3 .+ (6*rand(Float64, m))
        push!(U, u)

    end

    # for k in (Int(round((num_knots-1)/2) + 1)) : (num_knots - 1)

    #     u = 2 .* ones(m)
    #     push!(U, u)
    # end
    
    # return U

    # for k in 1 : (Int(round((num_knots-1)/2)))

    #     u = -3 .+ (6*rand(Float64, m))
    #     push!(U, u)

    # end

    # for k in (Int(round((num_knots-1)/2) + 1)) : (num_knots - 1)

    #     u = -2 .* ones(m)
    #     push!(U, u)
    # end
    
    return U

end