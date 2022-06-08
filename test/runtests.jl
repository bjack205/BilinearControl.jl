using BilinearControl
import BilinearControl.TO
import BilinearControl.RD
using BilinearControl.TO
using BilinearControl.RD
using FiniteDiff
using LinearAlgebra
using Random
using StaticArrays
using SparseArrays
using SuiteSparse
using Test
using Statistics
using Rotations
using RobotZoo
using JLD2

using BilinearControl.RandomLinearModels
using BilinearControl.Problems
using BilinearControl.Problems: qrot, skew
using BilinearControl: getA, getB, getC, getD

##
include("dynamics_tests.jl")

@testset "LQR" begin
    include("lqr_test.jl")
end

@testset "Linear ADMM" begin
    include("linear_admm_test.jl")
end

@testset "EDMD" begin
    include("edmd_test.jl")
end

@testset "Swarm Model" begin
    include("swarm_test.jl")
end

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

@testset "Rex Quadrotor" begin
    include("rex_quadrotor_dynamics_test.jl")
end

@testset "RLS" begin
    include("rls_test.jl")
end

# @testset "MOI" begin
#     include("nlp_test.jl")
# end
