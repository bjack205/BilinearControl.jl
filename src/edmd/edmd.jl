module EDMD

export learn_bilinear_model, build_eigenfunctions, state_transform, koopman_transform, state, sine, cosine, hermite, chebyshev, monomial

using LinearAlgebra
using Convex
using COSMO
import QDLDL
using SparseArrays

include("eigenfunctions.jl")
include("regression.jl")

end