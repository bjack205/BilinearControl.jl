module Problems

using TrajectoryOptimization
using LinearAlgebra
using RobotZoo
using StaticArrays
using RobotDynamics
using Rotations
using BilinearControl
using COSMO
using JLD2
using ForwardDiff, FiniteDiff
const RD = RobotDynamics
const TO = TrajectoryOptimization

const FIGDIR = joinpath(dirname(pathof(BilinearControl)), "..", "images")
const VISDIR = joinpath(@__DIR__, "visualization/") 

import BilinearControl: DiscreteLinearModel

# Include models
model_dir = joinpath(@__DIR__, "models")
include(joinpath(model_dir, "rotation_utils.jl"))
include(joinpath(model_dir, "dubins_model.jl"))
include(joinpath(model_dir, "se3_models.jl"))
include(joinpath(model_dir, "attitude_model.jl"))
include(joinpath(model_dir, "se3_force_model.jl"))
include(joinpath(model_dir, "quadrotor_model.jl"))
include(joinpath(model_dir, "integrator_models.jl"))
include(joinpath(model_dir, "swarm_model.jl"))
include(joinpath(model_dir, "cartpole_model.jl"))
include(joinpath(model_dir, "edmd_model.jl"))
include("learned_models/edmd_utils.jl")

# Problem constructors
include("problems.jl")

# Export models
export 
    BilinearDubins,
    AttitudeDynamics,
    SO3Dynamics,
    SE3Kinematics,
    Se3ForceDynamics,
    QuadrotorSE23,
    QuadrotorRateLimited,
    FullAttitudeDynamics,
    ConsensusDynamics,
    DoubleIntegrator,
    Swarm,
    BilinearCartpole,
    EDMDModel,
    EDMDErrorModel,
    Cartpole2

export BilinearMPC

export expandstate, originalstate, originalstatedim

end