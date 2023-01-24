const DATADIR = joinpath(dirname(pathof(BilinearControl)), "..", "data")
const FIGDIR = joinpath(dirname(pathof(BilinearControl)), "..", "images")
const VISDIR = joinpath(@__DIR__, "visualization/") 

# Include models
model_dir = joinpath(@__DIR__, "models")
include(joinpath(model_dir, "cartpole_model.jl"))
include(joinpath(model_dir, "rex_full_quadrotor_model.jl"))
include(joinpath(model_dir, "rex_planar_quadrotor_model.jl"))
include(joinpath(model_dir, "airplane_model.jl"))