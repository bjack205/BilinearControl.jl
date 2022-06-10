module EDMD

export
    rls_qr,
    rls_chol, 
    learn_bilinear_model, 
    build_eigenfunctions, 
    state_transform, 
    koopman_transform, 
    state, 
    sine, 
    cosine, 
    hermite, 
    chebyshev, 
    monomial, 
    create_data

export EDMDModel, LQRController

export run_eDMD, run_jDMD

using BilinearControl
using LinearAlgebra
using Convex
using COSMO
import QDLDL
using SparseArrays
using LazyArrays
using ForwardDiff, FiniteDiff
using Statistics
import RobotDynamics
import RobotDynamics as RD

include("edmd_utils.jl")
include("eigenfunctions.jl")
include("edmd_model.jl")
include("regression.jl")


end