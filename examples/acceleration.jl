import Pkg; Pkg.activate(@__DIR__)
using BilinearControl
using BilinearControl.RD
using BilinearControl.TO
import BilinearControl.RD
import BilinearControl.TO
using JLD2

# using Altro
using TrajectoryOptimization
using LinearAlgebra
using RobotZoo
using StaticArrays
using Test
using COSMOAccelerators
# using Plots

include("problems.jl")

## Run without acceleration
prob = builddubinsproblem(BilinearDubins())
prob = buildse3problem()
X = extractstatevec(prob)
U = extractcontrolvec(prob)
p = 3624
aa = AndersonAccelerator{Float64, Type1, RestartedMemory, NoRegularizer}(p+length(U))
admm = BilinearADMM(prob, acceleration=aa)
# admm = BilinearADMM(prob)
admm.opts.z_solver = :osqp
BilinearControl.solve(admm, X, U, verbose=true, max_iters=200)

probs = [
    "dubins (turn90)" => builddubinsproblem(BilinearDubins(), scenario=:turn90),
    "dubins (parallelpark)" => builddubinsproblem(BilinearDubins(), scenario=:parallelpark),
    "quaternion kinematics" => buildattitudeproblem(Val(2)),
    "rotmat kinematics" => buildso3problem(Val(2)),
    "se3" => buildse3problem()
]
accelerations = [
    EmptyAccelerator
    AndersonAccelerator{Float64, Type2{QRDecomp}, RestartedMemory, NoRegularizer}
    AndersonAccelerator{Float64, Type2{NormalEquations}, RestartedMemory, NoRegularizer}
    AndersonAccelerator{Float64, Type1, RestartedMemory, NoRegularizer}
    AndersonAccelerator{Float64, Type1, RestartedMemory, TikonovRegularizer}
    AndersonAccelerator{Float64, Type1, RollingMemory, TikonovRegularizer}
]
accel_labels = [
    "none",
    "QR",
    "NormalEq",
    "Broyden",
    "Broyden-Tik",
    "Brodyen-Rolling"
]
data = map(probs) do (name,prob)
    println("Solving $name")
    stats = map(accelerations) do aatype
        println("  solving with $aatype")
        admm = BilinearADMM(prob)
        X = extractstatevec(prob)
        U = extractcontrolvec(prob)
        p = length(admm.w)
        aa = aatype(p + length(U))
        admm = BilinearADMM(prob, acceleration=aa)
        admm.opts.x_solver = :osqp
        admm.opts.z_solver = :osqp
        BilinearControl.solve(admm, X, U, max_iters=500)
        admm.stats
    end
    
end
jldsave("accleration_data.jld2"; data=data)

using Plots
for i = 1:length(data)
    iters = map(data[i]) do stats 
        length(stats.cost) 
    end
    p = bar(1:6, iters, title=probs[i].first, label="", xticks=(1:6, accel_labels))
    annotate!(1:6, iters, string.(iters), :bottom)
    display(p)
end
using Printf
p = bar(1:6, iters, title=probs[1].first, xticks=(1:6, accel_labels), label="")
annotate!(1:6, iters, string.(iters), :bottom)