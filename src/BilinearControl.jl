module BilinearControl

export BilinearADMM

export extractstatevec, extractcontrolvec

using LinearAlgebra
using SparseArrays
using OSQP
import RobotDynamics as RD
import TrajectoryOptimization as TO

import TrajectoryOptimization: state_dim, control_dim

include("utils.jl")
include("bilinear_constraint.jl")
include("bilinear_model.jl")
include("problem.jl")
include("admm.jl")
include("trajopt_interface.jl")

end # module
