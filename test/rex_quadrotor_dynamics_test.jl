using BilinearControl.Problems
using Random
using Test
using BenchmarkTools
using StaticArrays

model = RexQuadrotor()
@test RD.dims(model) == (12,4,12)

dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
t,dt = 1.1,0.1

r0 = [-0.02, 0.17, 1.70]
q0 = [0.0, 0.0, 0.0]
v0 = [0.0; 0.0; 0.0]
ω0 = [0.0; 0.0; 0.0]
x = [r0; q0; v0; ω0]

u = Problems.trim_controls(model)

n,m = RD.dims(model)
xdot = zeros(n)
z = KnotPoint(x, u, t, dt)

@test xdot ≈ RD.dynamics(model, x, u) atol = 1e-10
@test Problems.dynamics(model, x, u) ≈ RD.dynamics(model, x, u) atol = 1e-10
@test Problems.dynamics_rk4(model, x, u, dt) ≈ RD.discrete_dynamics(dmodel, z) atol = 1e-10

∇c1  = zeros(n,n+m)
∇c2  = zeros(n,n+m)

RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), model, ∇c1, xdot, z)
RD.jacobian!(RD.StaticReturn(), RD.FiniteDifference(), model, ∇c2, xdot, z)
@test ∇c1 ≈ ∇c2 atol=1e-5

RD.jacobian!(RD.InPlace(), RD.ForwardAD(), model, ∇c2, xdot, z)
RD.jacobian!(RD.InPlace(), RD.FiniteDifference(), model, ∇c1, xdot, z)
@test ∇c1 ≈ ∇c2 atol=1e-5

model = RexPlanarQuadrotor()
@test RD.dims(model) == (6,2,6)

dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
t,dt = 1.1,0.1

u = Problems.trim_controls(model)

n,m = RD.dims(model)

x = zeros(n)
xdot = zeros(n)

z = KnotPoint(x, u, t, dt)

@test xdot ≈ RD.dynamics(model, x, u) atol = 1e-10

RD.dynamics!(model, xdot, x, u)

@test zeros(n) ≈ xdot atol = 1e-10

∇c1  = zeros(n,n+m)
∇c2  = zeros(n,n+m)

RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), model, ∇c1, xdot, z)
RD.jacobian!(RD.StaticReturn(), RD.FiniteDifference(), model, ∇c2, xdot, z)
@test ∇c1 ≈ ∇c2 atol=1e-5

RD.jacobian!(RD.InPlace(), RD.ForwardAD(), model, ∇c2, xdot, z)
RD.jacobian!(RD.InPlace(), RD.FiniteDifference(), model, ∇c1, xdot, z)
@test ∇c1 ≈ ∇c2 atol=1e-5

model = Quadrotor()
@test RD.dims(model) == (12,4,12)

dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
t,dt = 1.1,0.1

u = Problems.trim_controls(model)

n,m = RD.dims(model)

x = zeros(n)
xdot = zeros(n)

z = KnotPoint(x, u, t, dt)

@test xdot ≈ RD.dynamics(model, x, u) atol = 1e-10

RD.dynamics!(model, xdot, x, u)

@test zeros(n) ≈ xdot atol = 1e-10

∇c1  = zeros(n,n+m)
∇c2  = zeros(n,n+m)

RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), model, ∇c1, xdot, z)
RD.jacobian!(RD.StaticReturn(), RD.FiniteDifference(), model, ∇c2, xdot, z)
@test ∇c1 ≈ ∇c2 atol=1e-5

RD.jacobian!(RD.InPlace(), RD.ForwardAD(), model, ∇c2, xdot, z)
RD.jacobian!(RD.InPlace(), RD.FiniteDifference(), model, ∇c1, xdot, z)
@test ∇c1 ≈ ∇c2 atol=1e-5