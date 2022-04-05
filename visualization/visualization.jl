using GeometryBasics, CoordinateTransformations

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
    setobject!(vis["robot"]["geometry"], obj)
    settransform!(vis["robot"]["geometry"], compose(Translation((5.05, 6.38, 1) .* scale), LinearMap(I*scale * RotZ(deg2rad(140)))))
end

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