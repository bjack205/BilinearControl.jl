module BilinearControl

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")

export Problems

using LinearAlgebra
using SparseArrays
using StaticArrays
using OSQP
using RecipesBase
import RobotDynamics as RD
import TrajectoryOptimization as TO

import MathOptInterface as MOI

import RobotDynamics: state_dim, control_dim

import TrajectoryOptimization: state_dim, control_dim

include("utils.jl")

include("gen_controllable.jl")

include("edmd/edmd.jl")
include(joinpath(EXAMPLES_DIR,"problems.jl"))


end # module
