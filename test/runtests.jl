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
using Rotations

# Add the problems module
include(joinpath(@__DIR__, "..", "examples", "Problems.jl"))
using .Problems

include("gen_controllable.jl")
using Main.RandomLinearModels

##
include("dynamics_tests.jl")

@testset "Dubins Example" begin
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