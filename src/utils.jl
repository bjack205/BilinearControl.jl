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
            nzind == 0 && throw(ArgumentError("The nonzeros of B must be a subset of those of A."))
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

"""
    findmatches(a,b)

Get the indices into the sorted `a` and `b` vectors for which the elements match. 

# Example
```
a = [1,3,5,6]
b = [1,2,3,6,9]
findmatches(a,b) == [(1,1), (2,3), (4,4)]
```
"""
function findmatches(a,b)
    ia = 1
    ib = 1
    matches = Tuple{Int,Int}[]
    for i = 1:length(a) + length(b)
        if ia > length(a) || ib > length(b)
            break
        end

        if a[ia] == b[ib]
            push!(matches, (ia,ib))
            ia += 1
            ib += 1
        elseif a[ia] < b[ib]
            ia += 1
        elseif a[ia] > b[ib]
            ib += 1
        else
            error("This shouldn't be reached!")
        end
    end
    matches
end

"""
Creates a cache for computing `C = A'A` more efficiently, where `C` is a sparse 
matrix already initialized with the correct sparsity structure.

Returns a dictionary whose keys are the `(i,j)` tuples of nonzero entries in `C`, 
with values equal to `(ij,ji,matches)` where `ij` and `ji` are the indices into the nonzeros 
vector of `C` corresponding to `C[i,j]` and `C[j,i]`, respectively. The `matches`
entry is a vector of `NTuple{2,Int}`, which provide the indices into the nonzero 
vector of `A` for which column `i` and `j` have the same rows.
"""
function AtAcache(A)
    m,n = size(A)
    rv = rowvals(A)
    cache = Dict{Tuple{Int,Int},Tuple{Int,Int,Vector{NTuple{2,Int}}}}()
    AtA = A'A + I
    for i = 1:n
        rowi = view(rv, nzrange(A, i))
        for j = i+1:n
            ij = getnzind(AtA, i, j)
            ji = getnzind(AtA, j, i)
            if ij == 0 && ji == 0
                continue
            end

            rowj = view(rv, nzrange(A, j))

            # Find indices of entries with matching rows
            matches = findmatches(rowi, rowj)

            # Convert to nonzeros indices
            nzinds = map(matches) do (ri,rj) 
                (nzrange(A,i)[ri], nzrange(A,j)[rj])
            end
            cache[(i,j)] = (ij, ji, nzinds)
        end
        ii = getnzind(AtA, i, i)
        if ii > 0
            nzinds = map(i->(i,i), nzrange(A, i))
            cache[(i,i)] = (ii,ii,nzinds)
        end
    end
    cache, AtA
end

"""
Calculates `B = Q + ρ A'A`, given a cache computed via [`AtAcache`](@ref), where 
`A` is sparse, `Q` is a diagonal matrix, and `ρ` is a scalar.
"""
function QAtA!(B, Q, A, ρ, cache)
    nzv = nonzeros(A)
    nzvB = nonzeros(B)

    for (idx,nz) in pairs(cache)
        i,j = idx
        tmp = 0.0
        ij,ji,matches = nz 
        for (ii,jj) in matches 
            tmp += nzv[ii] * nzv[jj]
        end
        nzvB[ij] = ρ*tmp
        if i == j
            nzvB[ij] += Q[i,i]
        else
            nzvB[ji] = ρ*tmp
        end
    end
    B
end