# module EDMD

# export
#     rls_qr,
#     rls_chol, 
#     learn_bilinear_model, 
#     build_eigenfunctions, 
#     state_transform, 
#     koopman_transform, 
#     state, 
#     sine, 
#     cosine, 
#     hermite, 
#     chebyshev, 
#     monomial, 
#     create_data,
#     simulatewithcontroller,
#     simulate

# export EDMDModel, LQRController, TrackingMPC, TrackingMPC_no_OSQP

# export run_eDMD, run_jDMD

# using BilinearControl
# using LinearAlgebra
# import QDLDL
# using SparseArrays
# using LazyArrays
# using ForwardDiff, FiniteDiff
# using Statistics
# using ProgressMeter
# import RobotDynamics
# import RobotDynamics as RD

# using BilinearControl: AbstractController

include("edmd_utils.jl")
include("eigenfunctions.jl")
include("edmd_model.jl")
include("regression.jl")
include("mpc.jl")


# end