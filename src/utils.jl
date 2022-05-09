normsquared(x) = dot(x,x)

evec(i,n) = evec(Float64, i, n)
function evec(::Type{T}, i::Integer, n::Integer) where T
    e = zeros(T,n)
    e[i] = 1
    return e
end

function getnzind(A, i0, i1)
    if !(1 <= i0 <= size(A, 1) && 1 <= i1 <= size(A, 2)); throw(BoundsError()); end
    r1 = Int(SparseArrays.getcolptr(A)[i1])
    r2 = Int(SparseArrays.getcolptr(A)[i1+1]-1)
    (r1 > r2) && return zero(i0)  # empty column
    r1 = searchsortedfirst(rowvals(A), i0, r1, r2, Base.Order.Forward)
    ((r1 > r2) || (rowvals(A)[r1] != i0)) ? zero(i0) : r1
end

function getnzindsA(A, B::SparseMatrixCSC{<:Any,Ti}) where Ti
    size(A) == size(B) || throw(DimensionMismatch("A and B must have the same size to extract nonzero indices."))
    nzinds = zeros(Ti, nnz(B))
    ind = 1
    for c in axes(A,2)
        for i in nzrange(B, c)
            # Find the index into the nonzeros vector of A
            r = rowvals(B)[i]
            nzind = getnzind(A, r, c) 
            nzind == 0 && throw(ArgumentError("The nonzeros of B must be a subset of those of A. Got structural zero in A at ($r, $c)."))
            nzinds[ind] = nzind
            ind += 1
        end
    end
    @assert ind == nnz(B) + 1
    return nzinds 
end
getnzindsA(A, B) = Vector{Int}[]

function getnzindsB(B, C::SparseMatrixCSC{<:Any,Ti}, col) where Ti
    size(B,1) == size(C,1) || throw(DimensionMismatch("B and C must have the same number of rows.")) 
    nzinds = zeros(Ti, nnz(C))
    ind = 1
    for c in axes(C,2)
        for i in nzrange(C, c)
            r = rowvals(C)[i]
            nzind = getnzind(B, r, col)
            nzind == 0 && throw(ArgumentError("Expected nonzero entry in B at ($r, $col) does not exist."))
            nzinds[ind] = nzind
            ind += 1
        end
    end
    @assert ind == nnz(C) + 1
    return nzinds
end

getnzindsB(B, C, col) = Vector{Int}[]

tovecs(x, n) = collect(eachcol(reshape(x, n, :)))

matdensity(A) = nnz(sparse(A)) / length(A)

@userplot PlotStates 

@recipe function f(ps::PlotStates; inds=1:length(ps.args[end][1]))
    Xvec = ps.args[end]
    if length(ps.args) == 1
        times = 1:length(Xvecs)
    else
        times = ps.args[1]
    end
    Xmat = reduce(hcat,Xvec)[inds,:]'
    (times,Xmat)
end