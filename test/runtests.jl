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

include("gen_controllable.jl")
using Main.RandomLinearModels
using BilinearControl.Problems
using BilinearControl.Problems: qrot, skew

##
include("dynamics_tests.jl")

@testset "Dubins Example" begin
    include("dubins_test.jl")
    include("mpc_test.jl")
end

@testset "Attitude Example" begin
    include("attitude_test.jl")
end

@testset "SE(3) Examples" begin
    include("se3_kinematics_test.jl")
    include("se3_force_test.jl")
end

@testset "Quadrotor Example" begin
    include("quadrotor_test.jl")
end
