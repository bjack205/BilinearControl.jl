using GeometryBasics, CoordinateTransformations, Rotations
using LinearAlgebra
using StaticArrays
using Colors
using MeshCat
using RobotZoo


#############################################
# Quadrotor 
#############################################

# function setquadrotor!(vis;scale=1.0, color=colorant"black")
    # meshfile = joinpath(@__DIR__, "quadrotor_scaled.obj")
    # obj = MeshFileGeometry(meshfile)
    # mat = MeshPhongMaterial(color=color)
    # setobject!(vis["robot"]["geometry"], obj, mat)
    # settransform!(vis["robot"]["geometry"], compose(
    #     Translation(0,0,-10*scale), 
    #     LinearMap(scale*RotZ(-pi/2)*RotX(pi/2))
    # ))
    # urdf_folder = joinpath(@__DIR__, "..", "data", "meshes")
# end

function set_quadrotor!(vis, model::L;
    scaling=1.0, color=colorant"black"
    ) where {L <: Union{RobotZoo.Quadrotor, RobotZoo.PlanarQuadrotor, BilinearControl.RexQuadrotor, BilinearControl.RexPlanarQuadrotor}}
     
    urdf_folder = @__DIR__
    # if scaling != 1.0
    #     quad_scaling = 0.085 * scaling
    obj = joinpath(urdf_folder, "quadrotor_scaled.obj")
    if scaling != 1.0
        error("Scaling not implemented after switching to MeshCat 0.12")
    end
    robot_obj = MeshFileGeometry(obj)
    mat = MeshPhongMaterial(color=color)
    setobject!(vis["robot"]["geom"], robot_obj, mat)
    if hasfield(L, :ned)
        model.ned && settransform!(vis["robot"]["geom"], LinearMap(RotX(pi)))
    end
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

function visualize!(vis, model::L, x::AbstractVector) where {L <: Union{RobotZoo.PlanarQuadrotor, BilinearControl.RexPlanarQuadrotor}}
    py,pz = x[1], x[2]
    θ = x[3]
    settransform!(vis["robot"], compose(Translation(0,py,pz), LinearMap(RotX(-θ))))
end

function visualize!(vis, model::L, x::AbstractVector) where {L <: Union{BilinearControl.RexQuadrotor}}
    px, py,pz = x[1], x[2], x[3]
    rx, ry, rz = x[4], x[5], x[6]
    settransform!(vis["robot"], compose(Translation(px,py,pz), LinearMap(MRP(rx, ry, rz))))
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

function visualize!(vis, model::Union{RobotZoo.Cartpole, BilinearControl.Cartpole2}, x::AbstractVector)
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
# Airplane
#############################################
function set_airplane!(vis, ::BilinearControl.YakPlane; color=nothing, scale=0.15)
    meshfile = joinpath(DATADIR,"piper","piper_pa18.obj")
    # meshfile = joinpath(@__DIR__,"..","data","meshes","cirrus","Cirrus.obj")
    # meshfile = joinpath(@__DIR__,"..","data","meshes","piper","piper_scaled.obj")
    jpg = joinpath(DATADIR,"piper","piper_diffuse.jpg")
    if isnothing(color)
        img = PngImage(jpg)
        texture = Texture(image=img)
        # mat = MeshLambertMaterial(map=texture) 
        mat = MeshPhongMaterial(map=texture) 
    else
        mat = MeshPhongMaterial(color=color)
    end
    obj = MeshFileGeometry(meshfile)
    setobject!(vis["robot"]["geom"], obj, mat)
    settransform!(vis["robot"]["geom"], compose(Translation(0,0,0.07),LinearMap( RotY(pi/2)*RotZ(-pi/2) * scale)))
end
orientation(model::BilinearControl.YakPlane, x) = UnitQuaternion(MRP(x[4], x[5], x[6]))
# orientation(model::YakPlane, x) = UnitQuaternion(x[4:7])

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

function waypoints!(vis, model::RD.AbstractModel,  X, setmodel, inds)
    for i in inds
        setmodel(vis["waypoints/point$i"])
        visualize!(vis["waypoints/point$i"], model, X[i])
    end
end

visualize!(vis, r::AbstractVector, q::AbstractVector) = settransform!(vis["robot"], compose(Translation(r), LinearMap(UnitQuaternion(q[1], q[2], q[3], q[4]))))
visualize!(vis, r::AbstractVector, q::Rotation{3}) = settransform!(vis["robot"], compose(Translation(r), LinearMap(q)))

translation(model, x) = SA[x[1], x[2], x[3]]
orientation(model, x) = UnitQuaternion(x[4], x[5], x[6], x[7])

translation(model::RD.DiscretizedDynamics, x) = translation(model.continuous_dynamics, x)
orientation(model::RD.DiscretizedDynamics, x) = orientation(model.continuous_dynamics, x)


function traj3!(vis, X::AbstractVector{<:AbstractVector}; inds=SA[1,2,3], kwargs...)
    pts = [Point{3,Float32}(x[inds]) for x in X] 
    setobject!(vis["traj"], MeshCat.Line(pts, LineBasicMaterial(;kwargs...)))
end