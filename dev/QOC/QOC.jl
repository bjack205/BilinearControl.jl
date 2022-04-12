module QOC

export TwoQubit, sqrtiSWAP, TwoQubitBase, ControlDerivative

using RobotDynamics
using ForwardDiff
using FiniteDiff
using LinearAlgebra
const RD = RobotDynamics

include("bilinear_dynamics.jl")
include("quantum_hamiltonions.jl")
include("dynamics.jl")
end