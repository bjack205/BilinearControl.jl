module edmd

export learn_bilinear_model, build_eigenfunctions, state, sine, cosine, hermite, chebyshev, monomial

using LinearAlgebra
using Convex
using SCS

include("eigenfunctions.jl")
include("regression.jl")

end