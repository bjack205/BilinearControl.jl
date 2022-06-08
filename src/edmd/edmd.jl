module EDMD

export rls_qr, rls_chol, learn_bilinear_model, build_eigenfunctions, state_transform, koopman_transform, state, sine, cosine, hermite, chebyshev, monomial

using LinearAlgebra
using Convex
using COSMO
import QDLDL
using SparseArrays
using LazyArrays
import RobotDynamics as RD

include("eigenfunctions.jl")
include("regression.jl")

end