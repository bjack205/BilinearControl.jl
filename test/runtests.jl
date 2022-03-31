using BilinearControl
import BilinearControl.TO
import BilinearControl.RD
using BilinearControl.TO
using FiniteDiff
using LinearAlgebra
using Random
using StaticArrays
using Test
using Statistics

include("dubins_model.jl")
include("gen_controllable.jl")
using Main.RandomLinearModels

include("dynamics_tests.jl")

@testset "Dubins Example" begin
    include("dubins_test.jl")
end