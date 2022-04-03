
model0 = SE3AngVelDynamics(2.0)
model = SE3AngVelExpandedDynamics(2.0)
n,m = RD.dims(model)
x,u = rand(model)
xdot = zero(x)
RD.dynamics!(model, xdot, x, u)
J = zeros(n,n+m)

x0 = SVector{10}(x[1:10])
u0 = SVector{6}(u)
x0dot = RD.dynamics(model0, x0, u0)
@test xdot[1:10] ≈ x0dot

RD.jacobian!(model, J, xdot, x, u)

Jfd = zero(J)
FiniteDiff.finite_difference_jacobian!(
    Jfd, (y,z)->RD.dynamics!(model, y, z[1:50], z[51:end]), [x; u]
)
@test Jfd ≈ J rtol=1e-8

J0 = zeros(10, 10+m)
z0 = KnotPoint(x0,u0,0.0,NaN)
x0tmp = zeros(10)
RD.jacobian!(RD.InPlace(), RD.ForwardAD(), model0, J0, x0tmp, z0) 

## Compare simulations
dmodel0 = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model0)
dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

times = range(0,5,step=1e-3)
X0 = [copy(x0) for t in times]
X = [copy(x) for t in times]
for i = 1:length(times) - 1
    h = times[i+1] - times[i]
    z0 = KnotPoint(X0[i], u0, times[i], h)
    X0[i+1] = RD.discrete_dynamics(dmodel0, z0)

    RD.discrete_dynamics!(dmodel, X[i+1], X[i], u, times[i], h)
end

X0_ = [x[1:10] for x in X]
norm(X0 - X0_, Inf)
norm(X0[end] - X0_[end], Inf)