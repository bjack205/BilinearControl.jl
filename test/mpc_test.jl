
@testset "MPC Functions" begin
## Generate reference trajectory (a Scotty dog)
include(joinpath(@__DIR__, "..", "examples", "scotty.jl"))
scotty_interp = generate_scotty_trajectory(scale=0.01)
Nref = 501
tref = 50  # time of entire trajectory
dt = tref / (Nref - 1)
s = range(0, knots(scotty_interp).knots[end], length=Nref)
scotty = scotty_interp.(s)

Zref = RD.SampledTrajectory(map(1:Nref) do k
    p = scotty[k]
    if k < Nref
        pn = scotty[k+1]
        i = normalize(SA[pn[1] - p[1], pn[2] - p[2]])
        z = SA[p[1], p[2], i[1], i[2], 0.0, 0.0]
    else
        pp = scotty[k-1]
        i = normalize(SA[p[1] - pp[1], p[2] - pp[2]])
        z = SA[p[1], p[2], i[1], i[2], 0.0, 0.0]
    end
    RD.KnotPoint{4,2}(4, 2, z, (k-1)*dt, dt)
end)

##  Solve first problem
prob = Problems.DubinsMPCProblem(Zref)
n,m = RD.dims(prob.model[1])

## Generate Solver
admm = BilinearADMM(prob)
Q = prob.obj[1].Q
@test admm.q[1:4] ≈ -Q*RD.state(Zref[1])
X = extractstatevec(prob)
U = extractcontrolvec(prob)
admm.opts.penalty_threshold = 1e2
Xsol, Usol = BilinearControl.solve(admm, X, U, verbose=false)
@test admm.stats.iterations == 43
J0 = BilinearControl.cost(admm, Xsol, Usol)

## Advance one step
# Update cost
Q = prob.obj[1].Q
q0 = copy(admm.q)
BilinearControl.updatetrajectory!(admm, Zref, Q, 2)
@test admm.q[1:4] ≈ -Q*RD.state(Zref[2])
@test admm.q[5:8] ≈ -Q*RD.state(Zref[3])
@test admm.q ≉ q0
J1 = BilinearControl.cost(admm, Xsol, Usol)
@test J1 > J0

# Update initial state
BilinearControl.setinitialstate!(admm, RD.state(Zref[2]))
@test admm.d[1:4] ≈ RD.state(Zref[2])

# Shift the trajectories
x0 = copy(admm.x)
z0 = copy(admm.z)
BilinearControl.shiftfill!(admm, 4, 2)
@test x0 ≉ admm.x
@test z0 ≉ admm.z
@test admm.x[1:end-n] ≈ x0[n+1:end]
@test admm.z[1:end-m] ≈ z0[m+1:end]
@test admm.x[end-n+1:end] ≈ x0[end-n+1:end]
@test admm.z[end-m+1:end] ≈ z0[end-m+1:end]
J2 = BilinearControl.cost(admm, Xsol, Usol)
@test J0 < J2 < J1

## Solve updated problem
X1,U1 = copy(Xsol), copy(Usol)
Xsol2, Usol = BilinearControl.solve(admm, verbose=false)
@test admm.stats.iterations == 8
end