module BilinearControl

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")

export EDMDModel, ProjectedEDMDModel, LQRController, TrackingMPC, Visualizer, TrackingMPC_no_OSQP

export resetcontroller!, simulatewithcontroller, simulate
export set_airplane!, set_cartpole!, set_quadrotor!, visualize!

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
using MeshCat

import RobotDynamics
import RobotDynamics as RD
import TrajectoryOptimization as TO

include("utils.jl")

include("gen_controllable.jl")

include("controllers.jl")
include("edmd/edmd.jl")
# include(joinpath(EXAMPLES_DIR,"problems.jl"))
include("problems.jl")
include("visualization/visualization.jl")

export run_eDMD, run_jDMD


end # module
