include("se3_force_autogen.jl")

function Base.rand(model::Se3ForceDynamics)
    x = [(@SVector randn(3)); vec(qrot(normalize(@SVector randn(4)))); (@SVector randn(3))]
    u = @SVector randn(6)
    expandstate(model,x), u
end

orientation(::Se3ForceDynamics, x) = RotMatrix{3}(SMatrix{3,3}(x[4:12]))
buildstate(::Se3ForceDynamics, x::RBState) = [x.r; vec(x.q); x.v]