using MeshCat, Colors, GeometryBasics, Rotations, CoordinateTransformations

function defcolor(c1, c2, c1def, c2def)
    if !isnothing(c1) && isnothing(c2)
        c2 = c1
    else
        c1 = isnothing(c1) ? c1def : c1
        c2 = isnothing(c2) ? c2def : c2
    end
    c1,c2
end

function setdubins!(vis;
        color=nothing, color2=nothing, height=0.05, radius=0.2)
    radius = Float32(radius)
    body = Cylinder(Point3f0(0,0,0), Point3f0(0,0,height), radius)
    face = Rect3D(Vec(3radius/4, -radius/2, 0), Vec(radius/4, radius, height*1.1))
    c1,c2 = defcolor(color, color2, colorant"blue", colorant"yellow")
    setobject!(vis["geom"]["body"], body, MeshPhongMaterial(color=c1))
    setobject!(vis["geom"]["face"], face, MeshPhongMaterial(color=c2))
end

function setswarm!(vis, model::SwarmSE2{P}) where P
    for i = 1:P
        setdubins!(vis["swarm"]["robot$i"])
    end
end

function visualize!(vis, model::SwarmSE2{P}, x::AbstractVector) where P
    X = reshape(x, 4, P)
    for i = 1:P
        trans = Translation(X[1,i], X[2,i], 0.0)
        rot = LinearMap(RotZ(atan(X[4,i], X[3,i])))
        settransform!(vis["swarm"]["robot$i"], compose(trans, rot))
    end
end

