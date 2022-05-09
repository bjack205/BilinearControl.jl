using GeometryBasics, CoordinateTransformations, Rotations
using LinearAlgebra
using StaticArrays
using Colors
using MeshCat
import RobotDynamics as RD
import BilinearControl.Problems: orientation, translation

#############################################
# Endurance (satellite) 
#############################################

"""
    setbackgroundimage(vis, imagefile, [d])

Set an image as a pseudo-background image by placing an image on planes located a distance
`d` from the origin.
"""
function setbackgroundimage(vis, imagefile, d=50)

    # Create planes at a distance d from the origin
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
    setbackgroundimage(vis, joinpath(@__DIR__, "stars.jpg"), 30)
    setobject!(vis["robot"]["geometry"], obj)
    settransform!(vis["robot"]["geometry"], compose(Translation((5.05, 6.38, 1) .* scale), LinearMap(I*scale * RotZ(deg2rad(140)))))
end

#############################################
# Quadrotor 
#############################################

function setquadrotor!(vis;scale=1/8, color=colorant"black")
    meshfile = joinpath(@__DIR__, "AR_Drone.obj")
    obj = MeshFileGeometry(meshfile)
    mat = MeshPhongMaterial(color=color)
    setobject!(vis["robot"]["geometry"], obj, mat)
    settransform!(vis["robot"]["geometry"], compose(
        Translation(0,0,-10*scale), 
        LinearMap(scale*RotZ(-pi/2)*RotX(pi/2))
    ))
end

function defcolor(c1, c2, c1def, c2def)
    if !isnothing(c1) && isnothing(c2)
        c2 = c1
    else
        c1 = isnothing(c1) ? c1def : c1
        c2 = isnothing(c2) ? c2def : c2
    end
    c1,c2
end

#############################################
# Cartpole
#############################################

function set_cartpole!(vis0; model=RobotZoo.Cartpole(), 
        color=nothing, color2=nothing)
    vis = vis0["robot"]
    dim = Vec(0.1, 0.3, 0.1)
    rod = Cylinder(Point3f0(0,-10,0), Point3f0(0,10,0), 0.01f0)
    cart = Rect3D(-dim/2, dim)
    hinge = Cylinder(Point3f0(-dim[1]/2,0,dim[3]/2), Point3f0(dim[1],0,dim[3]/2), 0.03f0)
    c1,c2 = defcolor(color,color2, colorant"blue", colorant"red")

    pole = Cylinder(Point3f0(0,0,0),Point3f0(0,0,model.l),0.01f0)
    mass = HyperSphere(Point3f0(0,0,model.l), 0.05f0)
    setobject!(vis["rod"], rod, MeshPhongMaterial(color=colorant"grey"))
    setobject!(vis["cart","box"],   cart, MeshPhongMaterial(color=isnothing(color) ? colorant"green" : color))
    setobject!(vis["cart","hinge"], hinge, MeshPhongMaterial(color=colorant"black"))
    setobject!(vis["cart","pole","geom","cyl"], pole, MeshPhongMaterial(color=c1))
    setobject!(vis["cart","pole","geom","mass"], mass, MeshPhongMaterial(color=c2))
    settransform!(vis["cart","pole"], Translation(0.75*dim[1],0,dim[3]/2))
end

function visualize!(vis, model::Union{RobotZoo.Cartpole, Cartpole2}, x::AbstractVector)
    y = x[1]
    θ = x[2]
    q = expm((pi-θ) * @SVector [1,0,0])
    settransform!(vis["robot","cart"], Translation(0,-y,0))
    settransform!(vis["robot","cart","pole","geom"], LinearMap(UnitQuaternion(q)))
end

#############################################
# Pendulum  
#############################################

function set_pendulum!(vis0; model::RobotZoo.Pendulum=RobotZoo.Pendulum(),
    color=nothing, color2=nothing)
    
    vis = vis0["robot"]
    dim = Vec(0.1, 0.3, 0.1)
    cart = Rect3D(-dim/2, dim)
    hinge = Cylinder(Point3f0(-dim[1]/2,0,dim[3]/2), Point3f0(dim[1],0,dim[3]/2), 0.03f0)
    c1,c2 = defcolor(color,color2, colorant"blue", colorant"red")

    pole = Cylinder(Point3f0(0,0,0),Point3f0(0,0,model.len),0.01f0)
    mass = HyperSphere(Point3f0(0,0,model.len), 0.05f0)
    setobject!(vis["cart","box"],   cart, MeshPhongMaterial(color=isnothing(color) ? colorant"green" : color))
    setobject!(vis["cart","hinge"], hinge, MeshPhongMaterial(color=colorant"black"))
    setobject!(vis["cart","pole","geom","cyl"], pole, MeshPhongMaterial(color=c1))
    setobject!(vis["cart","pole","geom","mass"], mass, MeshPhongMaterial(color=c2))
    settransform!(vis["cart","pole"], Translation(0.75*dim[1],0,dim[3]/2))
end

function visualize!(vis, model::RobotZoo.Pendulum, x::AbstractVector)
    θ = x[1]
    q = expm((pi-θ) * @SVector [1,0,0])
    settransform!(vis["robot","cart","pole","geom"], LinearMap(UnitQuaternion(q)))
end

#############################################
# Generic Methods
#############################################

function visualize!(vis, model, tf, X)
    N = length(X)
    fps = Int(floor((N-1)/tf))
    anim = MeshCat.Animation(fps)
    for k = 1:N
        atframe(anim, k) do 
            visualize!(vis, model, X[k])
        end
    end
    setanimation!(vis, anim)
end

function visualize!(vis, model::RD.AbstractModel, x::AbstractVector)
    r = translation(model, x)
    q = orientation(model, x)
    visualize!(vis, r, q)
end

visualize!(vis, r::AbstractVector, q::AbstractVector) = settransform!(vis["robot"], compose(Translation(r), LinearMap(UnitQuaternion(q[1], q[2], q[3], q[4]))))
visualize!(vis, r::AbstractVector, q::Rotation{3}) = settransform!(vis["robot"], compose(Translation(r), LinearMap(q)))

translation(model, x) = SA[x[1], x[2], x[3]]
orientation(model, x) = UnitQuaternion(x[4], x[5], x[6], x[7])

translation(model::RD.DiscretizedDynamics, x) = translation(model.continuous_dynamics, x)
orientation(model::RD.DiscretizedDynamics, x) = orientation(model.continuous_dynamics, x)

