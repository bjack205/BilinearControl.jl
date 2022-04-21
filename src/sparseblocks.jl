# Copyright [2022] @ Brian Jackson

using DataStructures, SparseArrays

"""
    BlockID
A simple identifier for a contiguous block in a matrix, specified by its top-left corner
and dimensions. Can be created from contiguous pairs of indices.
"""
struct BlockID
    row::Int
    col::Int
    m::Int  
    n::Int
end
Base.length(block::BlockID) = block.m * block.n
Base.size(block::BlockID) = (block.m, block.n)

BlockID(i1::Integer, i2::Integer) = BlockID(i1:i1, i2:i2)

function BlockID(i1::UnitRange, i2::UnitRange)
    BlockID(i1[1], i2[1], length(i1), length(i2))
end
 
function BlockID(i1::AbstractVector{<:Integer}, i2::AbstractVector{<:Integer})
    mblk = length(i1)
    nblk = length(i2)
    if (i1[end] != i1[1] + mblk - 1) || (i2[end] != i2[1] + nblk -1)
        @warn "Block indices should be contiguous."
    end
    BlockID(i1[1]:i1[end], i2[1]:i2[end])
end

"""
    BlockViews
A simple type to make it easier to work with ordered vectors of nonzeros. 
These often appear in solver interfaces (such as `MathOptInterface`), where the
user has to fill in a the vector of nonzero entries in a sparse Jacobian or
Hessian, and provide the sparsity pattern ahead of time. Dealing with this format is
often unintuitive and error-prone. Ideally we'd like to use cartesian-style indexing to 
fill in our non-zero entries, similar to how we would with a `SparseMatrix`. This 
simple type, together with `NonzerosVector`, allows this type of behavior.

# Usage
The `BlockViews` type is meant to be initialized once and stored, and then passed to 
the constructor of a `NonzerosVector` as needed. Create a new `BlockViews` type by 
passing in the size of the sparse array you want to emulate:
    blocks = BlockViews(m, n)
Then specify the blocks of non-zero entries using `setblock!`:
    setblock!(blocks, 1:3, 4:6)
Note that these blocks must have contiguous indices. Then, when filling in the nonzero 
blocks, create a `NonzerosVector` and use indexing as normal:
    nzv = NonzerosVector(v, blocks)
    nzv[1:3, 4:6] = randn(3,3)
Note that the indices must match exactly with the ones used to initialize the blocks in
`setblock!`. To have overlapping blocks, just create a new block. Note that this will not 
share the elements of the underlying vector, but since interfaces like `MathOptInterface` 
support repeated entries, this is a clean solution.
Interfaces like `MathOptInterface` also need a list of pairs of row, column entries in the 
order of the nonzero vector. This can be extracted from a `BlockViews` via [`getrc`](@ref).
This information allows a `NonzerosVector` to be cast to a sparse matrix via the `sparse`
method:
    A = sparse(nzv)

# Other
If you want to find which block an element of the nonzeros vector belongs to, use 
[`findindexblock`](@ref).
"""
mutable struct BlockViews
    m::Int  # number of rows
    n::Int  # number of columns
    blk2vec::Dict{BlockID,UnitRange{Int}}
    vec2blk::OrderedDict{UnitRange{Int},BlockID}
    len::Int
    initializing::Bool
    function BlockViews(m::Int, n::Int)
        new(m, n, Dict{BlockID,UnitRange{Int}}(), OrderedDict{UnitRange{Int},BlockID}(), 0, false)
    end
end

getblock(blocks::BlockViews, v::AbstractVector, i1, i2) = 
    getblock(blocks, v, BlockID(i1, i2))
getblock(blocks::BlockViews, v::AbstractVector, ::Colon, i2) = 
    getblock(blocks, v, BlockID(1:blocks.m, i2))
getblock(blocks::BlockViews, v::AbstractVector, i1, ::Colon) = 
    getblock(blocks, v, BlockID(i1, 1:blocks.n))

function getblock(blocks::BlockViews, v::AbstractVector, block::BlockID)
    if blocks.initializing
        setblock!(blocks, block)
        v = zeros(length(block))   # initialize a temporary array while initializing
        vecview = view(v, 1:length(block))
    else
        vecview = view(v, blocks.blk2vec[block])
    end
    reshape(vecview, block.m, block.n)
end

setblock!(blocks::BlockViews, i1::AbstractVector, i2::AbstractVector) = 
    setblock!(blocks, BlockID(i1, i2)) 

setblock!(blocks::BlockViews, i1::Integer, i2::Integer) = 
    setblock!(blocks, BlockID(i1:i1, i2:i2)) 

function setblock!(blocks::BlockViews, block::BlockID)
    if !haskey(blocks.blk2vec, block)
        inds = blocks.len .+ (1:length(block))
        blocks.blk2vec[block] = inds
        blocks.vec2blk[inds] = block
        blocks.len += length(block) 
        if (block.row + block.m - 1) > blocks.m || (block.col + block.n - 1) > blocks.n || block.m < 1 || block.n < 1
            i1 = block.row:block.m-1
            i2 = block.col:block.n-1
            @warn "Invalid block of ($i1,$i2) in matrix of size($(blocks.m),$(blocks.n))."
        end
    end
    return blocks
end

@inline Base.getindex(blocks::BlockViews, i1::AbstractVector, i2::AbstractVector) = 
    blocks.blk2vec[BlockID(i1, i2)]    

function findindexblock(blocks::BlockViews, i::Integer)
    for (vec,blk) in pairs(blocks.vec2blk)
        if i ∈ vec
            return blk
        end
    end
    return nothing
end

function getrc(blocks::BlockViews)
    rc = Tuple{Int,Int}[]
    cnt = 1
    for (vec,blk) in blocks.vec2blk  # OrderedDict ensures this iterates by insertion order
        @assert vec[1] == cnt
        i1 = blk.row:blk.row+blk.m-1
        i2 = blk.col:blk.col+blk.n-1
        carts = CartesianIndices((i1,i2))
        append!(rc, Tuple.(carts))
        cnt += length(vec)
    end
    return rc
end

"""
    NonzerosVector
A container for a vector that specifies the nonzero entries in a sparse matrix, and one 
that supports indexing via cartesian ranges. This assumes a pre-specified sparsity 
structure, specified by [`BlockViews`](@ref). See the documentation for `BlockViews` for 
more details.
# Usage
With the structure specified by `BlockViews` object, a `NonzerosVector` is created by 
wrapping an existing `AbstractVector`:
    nzv = NonzerosVector(v, blocks)
The nonzero blocks can be written to via `Base.setindex!`, as if it were a `SparseArray`:
    nzv[1:3, 4:6] = randn(3,3)
Note that this uses the normal assignment operator and not the broadcasted assignment 
operator. To assign a scalar value to the elements, also use the normal assignment operator:
    nzv[1:3, 4:6] = 4 
    
The vector can be cast to a sparse matrix:
    A = sparse(nzv)
"""
struct NonzerosVector{T,V} <: AbstractVector{T}
    data::V
    blocks::BlockViews
    function NonzerosVector(vec::V, blocks::BlockViews) where V
        new{eltype(V), V}(vec, blocks)
    end
end
Base.size(v::NonzerosVector) = size(v.data)
Base.IndexStyle(::NonzerosVector) = IndexLinear()
@inline Base.getindex(v::NonzerosVector, i::Integer) = v.data[i]
@inline Base.setindex!(v::NonzerosVector, val, i::Integer) = v.data[i] = val

function Base.setindex!(v::NonzerosVector, val, i1::AbstractVector, i2::AbstractVector)
    blockview = getblock(v.blocks, v.data, i1, i2) 
    blockview .= val
    return blockview
end

const IndexRange = Union{Colon, AbstractVector{<:Integer}, <:Integer}
@inline Base.getindex(v::NonzerosVector, i1::IndexRange, i2::IndexRange) = view(v, i1, i2) 
@inline Base.dotview(v::NonzerosVector, i1::IndexRange, i2::IndexRange) = view(v, i1, i2)
function Base.view(v::NonzerosVector, i1::IndexRange, i2::IndexRange)
    blockview = getblock(v.blocks, v.data, i1, i2)
    return blockview
end
Base.getindex(v::NonzerosVector, i1::Integer, i2::Integer) = v[i1:i1, i2:i2]
function Base.setindex!(v::NonzerosVector, val, i1::Integer, i2::Integer)
    blockview = getblock(v.blocks, v.data, i1:i1, i2:i2)
    blockview[1] = val
end

function SparseArrays.sparse(v::NonzerosVector)
    rc = getrc(v.blocks)
    r = [idx[1] for idx in rc]
    c = [idx[2] for idx in rc]
    sparse(r, c, v.data, v.blocks.m, v.blocks.n)
end
