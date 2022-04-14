import Pkg; Pkg.activate(@__DIR__)

using BilinearControl
using BilinearControl.Problems
using COSMO
import RobotDynamics as RD
import TrajectoryOptimization as TO
using LinearAlgebra
using StaticArrays
using Statistics
using Rotations
using BilinearControl.Problems: qrot, skew
using SparseArrays
using Test

using BilinearControl: getA, getB, getC, getD

## Visualization 
using MeshCat
vis = Visualizer()
open(vis)
visdir = joinpath(@__DIR__, "../examples/visualization")
include(joinpath(visdir, "visualization.jl"))
setquadrotor!(vis)

struct Glideslope <: TO.StateConstraint
    n::Int
    α::Float64  # tan(θ) where θ is the half of the cone angle
    xind::Int
    yind::Int
    zind::Int
    function Glideslope(n, α, xind=1, yind=2, zind=3)
        new(n, α, xind, yind, zind)
    end
end

RD.output_dim(::Glideslope) = 3
RD.state_dim(con::Glideslope) = con.n

function RD.evaluate(con::Glideslope, x)
    px = x[con.xind]
    py = x[con.yind]
    pz = x[con.zind]
    SA[px,py, con.α*pz]
end

function RD.jacobian!(con::Glideslope, J, c, x)
    J .= 0
    J[1, con.xind] = 1.0
    J[2, con.yind] = 1.0
    J[3, con.zind] = con.α
    return
end

Base.copy(con::Glideslope) = Glideslope(con.n, con.xind, con)

function QuadrotorLanding(; tf=3.0, N=101, θ_glideslope=NaN)
    model = QuadrotorRateLimited()
    n = RD.state_dim(model)

    # Discretization
    h = tf / (N-1)

    # Initial and Final states
    x0 = [3; 3; 5.0; vec(I(3)); 0; 0; -3; zeros(3)]
    xf = [0; 0; 0.0; vec(I(3)); zeros(3); zeros(3)]
    uhover = [0,0,0,model.mass*model.gravity]

    # Build bilinear constraint matrices
    Abar,Bbar,Cbar,Dbar = BilinearControl.buildbilinearconstraintmatrices(
        model, x0, xf, h, N
    )

    # Build cost
    Q = Diagonal([fill(1e-2, 3); fill(1e-2, 9); fill(1e-2, 3); fill(1e-1, 3)])
    Qf = Q*(N-1)
    R = Diagonal([fill(1e-2,3); 1e-2])
    Qbar = Diagonal(vcat([diag(Q) for i = 1:N-1]...))
    Qbar = Diagonal([diag(Qbar); diag(Qf)])
    Rbar = Diagonal(vcat([diag(R) for i = 1:N]...))
    q = repeat(-Q*xf, N)
    r = repeat(-R*uhover, N)
    c = 0.5*sum(dot(xf,Q,xf) for k = 1:N-1) + 0.5*dot(xf,Qf,xf) + 
        0.5*sum(dot(uhover,R,uhover) for k = 1:N)

    # Initial guess
    X = repeat(x0, N)
    U = repeat(uhover, N)

    # Constraints
    Nx = length(X)
    if !isnan(θ_glideslope)
        α = tan(θ_glideslope)
        b = zeros(3)
        constraints = map(1:N-1) do k
            A = spzeros(3, Nx)
            for i = 1:3
                A[1,(k-1)*n + 3] = α
                A[2,(k-1)*n + 1] = 1.0
                A[3,(k-1)*n + 1] = 1.0
            end
            COSMO.Constraint(A, b, COSMO.SecondOrderCone)
        end
    else
        constraints = COSMO.Constraint{Float64}[]
    end

    admm = BilinearADMM(Abar,Bbar,Cbar,Dbar, Qbar,q,Rbar,r,c, constraints=constraints)
    admm.x .= X
    admm.z .= U
    admm.opts.penalty_threshold = 1e2
    BilinearControl.setpenalty!(admm, 1e4)
    admm
end

# Solve without glideslope
tf = 3.0
N = 101
model = QuadrotorRateLimited()
θ_glideslope = deg2rad(45.0)
admm = QuadrotorLanding(tf=tf, N=N, θ_glideslope=θ_glideslope*NaN)
length(admm.constraints) == N-1
BilinearControl.setpenalty!(admm, 1e4)
X = copy(admm.x)
U = copy(admm.z)
admm.opts.x_solver = :osqp
Xsol, Usol = BilinearControl.solve(admm, X, U, verbose=true)

admm2 = QuadrotorLanding(tf=tf, N=N, θ_glideslope=θ_glideslope)
admm.opts.x_solver = :cosmo
Xsol2, Usol2 = BilinearControl.solve(admm2, X, U, verbose=true)

Xs = collect(eachcol(reshape(Xsol, RD.state_dim(model), :)))
X2s = collect(eachcol(reshape(Xsol2, RD.state_dim(model), :)))
α = tan(θ_glideslope)
socerr = map(Xs) do x
    norm(SA[x[1], x[2]]) - α*x[3]
end
maximum(socerr)
socerr2 = map(X3s) do x
    norm(SA[x[1], x[2]]) - α*x[3]
end
maximum(socerr2)
visualize!(vis, model, tf, Xs)
visualize!(vis, model, tf, X2s)
admm.constraints[1].A*Xsol
Xsol[1:3]

using Colors
function comparison(vis, model, tf, X1, X2)
    delete!(vis["robot"])
    setquadrotor!(vis["quad1"])
    setquadrotor!(vis["quad2"], color=RGBA(0,0,1,0.5))
    N = length(X1)
    fps = Int(floor((N-1)/tf))
    anim = MeshCat.Animation(fps)
    for k = 1:N
        atframe(anim, k) do 
            visualize!(vis["quad1"], model, X1[k])
            visualize!(vis["quad2"], model, X2[k])
        end
    end
    setanimation!(vis, anim)
end
comparison(vis, model, tf, X2s, Xs)

admm.stats.x_solve_residual
coneheight = 4.5
r = tan(θ_glideslope) * coneheight
soc = Cone(Point(0,0,coneheight), Point(0,0,0.), r)
mat = MeshPhongMaterial(color=RGBA(1.0,0,0,0.2))
setobject!(vis["soc"], soc, mat)