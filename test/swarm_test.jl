
P = 4
model0 = DoubleIntegrator{2}(gravity=zeros(2))
n0,m0 = RD.dims(model0)
model = Swarm{4}(model0)
@test RD.state_dim(model) == P*n0
@test RD.control_dim(model) == P*m0

x,u = Vector.(rand(model))
xdot = zero(x)
RD.dynamics!(model, xdot, x, u)

Xdot = reshape(xdot, :, P)
X = reshape(x, :, P)
U = reshape(u, :, P)
@test all(1:P) do i
    xdot0 = zeros(n0)
    RD.dynamics!(model0, xdot0, X[:,i], U[:,i])
    Xdot[:,i] ≈ xdot0
end

A,B,C,D = getA(model), getB(model), getC(model), getD(model)
@test xdot ≈ A*x + B*u + sum(u[i]*C[i] * x for i = 1:length(u)) + D