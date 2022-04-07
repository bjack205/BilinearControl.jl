include("se3_integrator_autogen.jl")
include(joinpath(@__DIR__, "../test/models/rotation_utils.jl"))

function Base.rand(model::Se3IntegratorDynamics)
    x = [
            (@SVector randn(3)); 
            vec(qrot(normalize(@SVector randn(4)))); 
            (@SVector randn(3));
            (@SVector randn(3));

    ]
    u = @SVector randn(6)
    expandstate(model,x), u
end
orientation(::Se3IntegratorDynamics, x) = RotMatrix{3}(SMatrix{3,3}(x[4:12]))
buildstate(model::Se3IntegratorDynamics, x::RBState) = expandstate(model, Vector(x))