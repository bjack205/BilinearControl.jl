using BilinearControl
import BilinearControl.TO
import BilinearControl.RD
using BilinearControl.TO
using BilinearControl.RD
using FiniteDiff
using LinearAlgebra
using Random
using StaticArrays
using Test
using Statistics

include("models/dubins_model.jl")
include("models/attitude_model.jl")
include("models/se3_models.jl")
include("gen_controllable.jl")
using Main.RandomLinearModels

include("dynamics_tests.jl")

@testset "Random Systems" begin
    include("randomsystems_test.jl")
end

@testset "Dubins Example" begin
    Random.seed!(2)
    include("dubins_test.jl")
end

@testset "Attitude Example" begin
    include("attitude_test.jl")
end

@testset "SE(3) Examples" begin
    include("se3_kinematics_test.jl")
    include("se3_force_test.jl")
end


# @testset "SE3 Examples" begin
#     include("se3_test.jl")
# end