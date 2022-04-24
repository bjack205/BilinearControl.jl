module edmd

export learn_bilinear_model, build_eigenfunctions

using LinearAlgebra
using Convex
using SCS

include("eigenfunctions.jl")
include("regression.jl")