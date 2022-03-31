evec(i,n) = evec(Float64, i, n)
function evec(::Type{T}, i::Integer, n::Integer) where T
    e = zeros(T,n)
    e[i] = 1
    return e
end