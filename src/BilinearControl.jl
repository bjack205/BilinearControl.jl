module BilinearControl

export BilinearADMM, Problems

export extractstatevec, extractcontrolvec

using LinearAlgebra
using SparseArrays
using StaticArrays
using OSQP
import COSMO
import IterativeSolvers 
import RobotDynamics as RD
import TrajectoryOptimization as TO
import COSMOAccelerators
import Ipopt
import MathOptInterface as MOI

import TrajectoryOptimization: state_dim, control_dim

include("utils.jl")
include("bilinear_constraint.jl")
include("bilinear_model.jl")
include("problem.jl")
include("admm.jl")
include("trajopt_interface.jl")
include("mpc.jl")

# include("sparseblocks.jl")
include("moi.jl")

include(joinpath(@__DIR__,"..","examples","Problems.jl"))

end # module
