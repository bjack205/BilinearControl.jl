module BilinearControl

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")

export Problems
export LQRController, TrackingMPC

export resetcontroller!, simulatewithcontroller, simulate

using LinearAlgebra
using SparseArrays
using StaticArrays
using OSQP
using RecipesBase
using ForwardDiff, FiniteDiff
using Statistics
using ProgressMeter
using Polynomials
using JLD2
using RobotZoo
using Rotations

import RobotDynamics
import RobotDynamics as RD
import TrajectoryOptimization as TO

include("utils.jl")

include("gen_controllable.jl")

include("controllers.jl")
include("edmd/edmd.jl")
# include(joinpath(EXAMPLES_DIR,"problems.jl"))
include("problems.jl")

export run_eDMD, run_jDMD


end # module
