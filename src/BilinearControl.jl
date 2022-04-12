module BilinearControl

export BilinearADMM

export extractstatevec, extractcontrolvec

using LinearAlgebra
using SparseArrays
using StaticArrays
using OSQP
import IterativeSolvers 
import RobotDynamics as RD
import TrajectoryOptimization as TO
import COSMOAccelerators

import TrajectoryOptimization: state_dim, control_dim

include("utils.jl")
include("bilinear_constraint.jl")
include("bilinear_model.jl")
include("problem.jl")
include("admm.jl")
include("trajopt_interface.jl")

function loadexamples()
    @eval include(joinpath(@__DIR__,"..","examples","Problems.jl"))
end

end # module
