import Pkg; Pkg.activate(@__DIR__)

using BilinearControl
using BilinearControl.Problems
import RobotDynamics as RD
import TrajectoryOptimization as TO
using LinearAlgebra
using StaticArrays
using Statistics
using Rotations
using BilinearControl: getA, getB, getC, getD

##
h = 0.02
data = Problems.generate_linear_models(h=h)
model = DiscreteLinearModel(
    data["dubins"]["A"],
    data["dubins"]["B"],
    data["dubins"]["C"],
    data["dubins"]["d"],
)

x0 = zeros(4)

qp = BilinearControl.TOQP(model, , x0)