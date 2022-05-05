import Pkg; Pkg.activate(@__DIR__)

using BilinearControl
using BilinearControl.RD
using BilinearControl.TO
using Test
using FiniteDiff
using LinearAlgebra
using Statistics
import BilinearControl.RD
include(joinpath(@__DIR__, "..", "test", "models", "attitude_model.jl"))

using BilinearControl: getA, getB, getC, getD
Nu = 2
function attitude_dynamics_test(::Val{Nu}) where Nu
    model = AttitudeDynamics{Nu}()
    n,m = RD.dims(model)
    @test n == 4
    @test m == Nu 
    x,u = rand(model) 
    @test norm(x) ≈ 1
   
    # Test dynamics match expeccted
    xdot = zeros(4)
    RD.dynamics!(model, xdot, x, u)
    if Nu == 3
        @test xdot ≈ 0.5*[
            -x[2]*u[1] - x[3]*u[2] - x[4]*u[3]
            x[1]*u[1] - x[4]*u[2] + x[3]*u[3]
            x[4]*u[1] + x[1]*u[2] - x[2]*u[3]
            -x[3]*u[1] + x[2]*u[2] + x[1]*u[3]
        ]
    else
        @test xdot ≈ 0.5*[
            -x[2]*u[1] - x[3]*u[2]
            x[1]*u[1] - x[4]*u[2]
            x[4]*u[1] + x[1]*u[2]
            -x[3]*u[1] + x[2]*u[2]
        ]
    end

    # Test dynamics match bilinear dynamics
    A,B,C,D = getA(model), getB(model), getC(model), getD(model)
    @test xdot ≈ A*x + B*u + sum(u[i]*C[i]*x for i = 1:m) + D

    # Test custom Jacobian
    J = zeros(n, n+m)
    RD.jacobian!(model, J, xdot, x, u)
    Jfd = zero(J)
    FiniteDiff.finite_difference_jacobian!(
        Jfd, (y,z)->RD.dynamics!(model, y, z[1:4], z[5:end]), Vector([x;u])
    )
    @test Jfd ≈ J
end

function buildattitudeproblem(::Val{Nu}) where Nu
    # Model
    model = AttitudeDynamics{Nu}()
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

    # Discretization
    tf = 3.0
    N = 301

    # Dimensions
    nx = RD.state_dim(model)
    nu = RD.control_dim(model)

    # Initial and final conditions
    x0 = [1,0,0,0]
    # xf = [0.382683, 0.412759, 0.825518, 0.0412759]
    xf = [0,0,0,1.0]

    # Objective
    Q = Diagonal(fill(1e-1, nx))
    R = Diagonal(fill(2e-2, nu))
    Qf = Diagonal(fill(100.0, nx))
    obj = LQRObjective(Q,R,Qf,xf,N)

    # Goal state
    cons = ConstraintList(nx, nu, N)
    goalcon = GoalConstraint(xf)  # only constraint the original states
    add_constraint!(cons, goalcon, N)

    # Initial Guess
    U0 = [fill(0.1,Nu) for k = 1:N-1] 

    # Build the problem
    Problem(dmodel, obj, x0, tf, xf=xf, constraints=cons, U0=U0)
end

attitude_dynamics_test(Val(Nu))
prob = buildattitudeproblem(Val(Nu))
rollout!(prob)

A,B,C,D = BilinearControl.buildbilinearconstraintmatrices(
    prob.model[1].continuous_dynamics, prob.x0, prob.xf, prob.Z[1].dt, prob.N
)
X = vcat(Vector.(states(prob))...)
U = vcat(Vector.(controls(prob))...)
c1 = A*X + B*U + sum(U[i] * C[i] * X for i = 1:length(U)) + D
c2 = BilinearControl.evaluatebilinearconstraint(prob)
@test c1 ≈ c2

# Solve use ADMM
Q,q,R,r,c = BilinearControl.buildcostmatrices(prob)
admm = BilinearADMM(A,B,C,D, Q,q,R,r,c)
admm.opts.penalty_threshold = 1e4
BilinearControl.setpenalty!(admm, 1e3)
Xsol, Usol = BilinearControl.solve(admm, X, U, max_iters=300)

# Reshape the solution vectors
n,m = RD.dims(prob.model[1])
Xs = collect(eachcol(reshape(Xsol, n, :)))
Us = collect(eachcol(reshape(Usol, m, :)))
Zsol = SampledTrajectory(Xs,Us, tf=prob.tf)

# Test that it got to the goal
@test abs(Xs[end]'prob.xf - 1.0) < 1e-5

# Test that the quaternion norms are preserved
@test all(x->abs(x-1) < 2e-3, norm.(Xs))

# Check that the control signals are smooth 
Us = reshape(Usol, m, :)
@test all(x->x< 2e-2, mean(diff(Us, dims=2), dims=2))


## Visualization
using Plots
plot(reshape(Usol, m, :)')

using MeshCat, GeometryBasics, Colors, Rotations, CoordinateTransformations
vis = Visualizer()
render(vis)

"""
    setbackgroundimage(vis, imagefile, [d])

Set an image as a pseudo-background image by placing an image on planes located a distance
`d` from the origin.
"""
function setbackgroundimage(vis, imagefile, d=50)

    # Create planes at a distance d from the origin
    d = 50
    px = Rect3D(Vec(+d,-d,-d), Vec(0.1,2d,2d))
    nx = Rect3D(Vec(-d,-d,-d), Vec(0.1,2d,2d))
    py = Rect3D(Vec(-d,+d,-d), Vec(2d,0.1,2d))
    ny = Rect3D(Vec(-d,-d,-d), Vec(2d,0.1,2d))
    pz = Rect3D(Vec(-d,-d,+d), Vec(2d,2d,0.1))
    nz = Rect3D(Vec(-d,-d,-d), Vec(2d,2d,0.1))

    # Create a material from the image
    img = PngImage(imagefile)
    mat = MeshLambertMaterial(map=Texture(image=img))

    # Set the objects
    setobject!(vis["background"]["px"], px, mat)
    setobject!(vis["background"]["nx"], nx, mat)
    setobject!(vis["background"]["py"], py, mat)
    setobject!(vis["background"]["ny"], ny, mat)
    setobject!(vis["background"]["pz"], pz, mat)
    setobject!(vis["background"]["nz"], nz, mat)
    return vis 
end

function setendurance!(vis; scale=1/3)
    meshfile = joinpath(@__DIR__, "Endurance and ranger.obj")
    obj = MeshFileObject(meshfile)
    setobject!(vis["robot"]["geometry"], obj)
    settransform!(vis["robot"]["geometry"], compose(Translation((5.05, 6.38, 1) .* scale), LinearMap(I*scale * RotZ(deg2rad(140)))))
end

function setmesh!(vis, ::AttitudeDynamics)
    setbackgroundimage(vis, joinpath(@__DIR__, "stars.jpg"), 30)
    setendurance!(vis)
end

function visualize!(vis, ::AttitudeDynamics, tf, X)
    N = length(X)
    fps = Int(floor((N-1)/tf))
    anim = MeshCat.Animation(fps)
    for k = 1:N
        atframe(anim, k) do 
            settransform!(vis["robot"], LinearMap(UnitQuaternion(X[k])))
        end
    end
    setanimation!(vis, anim)
end

model = AttitudeDynamics()
setmesh!(vis, model)
visualize!(vis, model, prob.tf, Xs)