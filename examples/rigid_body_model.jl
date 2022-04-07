include("rigid_body_autogen.jl")

function Base.rand(model::RigidBodyDynamics)
    x = [
            (@SVector randn(3)); 
            vec(qrot(normalize(@SVector randn(4)))); 
            (@SVector randn(3));
            (@SVector randn(3));

    ]
    u = @SVector randn(6)
    expandstate(model,x), u
end
orientation(::RigidBodyDynamics, x) = RotMatrix{3}(SMatrix{3,3}(x[4:12]))
buildstate(model::RigidBodyDynamics, x::RBState) = expandstate(model, Vector(x))