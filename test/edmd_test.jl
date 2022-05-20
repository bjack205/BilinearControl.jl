
function simulate_bilinear(F, B, C, g, x0, z0, U)
    
    x = x0
    z = z0
    Z = [z]
    X = [x]

    for k in 1:length(U)

        u = U[k]
        
        z = F * z + B * u + (C[1] * z) .* u
        x = g * z

        push!(Z, z)
        push!(X, x)
        
    end

    return X, Z
end

## Load Reference Trajectory from file
const datadir = joinpath(@__DIR__, "..", "data")
ref_traj = load(joinpath(datadir, "cartpole_reference_trajectory.jld2"))
N = 601  # QUESTION: why not use entire trajectory?
X_ref = ref_traj["X_sim"][1:N]
U_ref = ref_traj["U_sim"][1:N-1]
T_ref = ref_traj["T_sim"][1:N]

## Define model
dt = T_ref[2] - T_ref[1]
tf = T_ref[end]

# Initial condition
x0 = copy(X_ref[1])

# Define the model
model = RobotZoo.Cartpole(1.0, 0.2, 0.5, 9.81)
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
n,m = RD.dims(model)

## Learn the bilinear dynamics
Z_sim, Zu_sim, kf = BilinearControl.EDMD.build_eigenfunctions(
    X_ref, U_ref, ["state", "sine", "cosine"], [0,0,0]
)
F, B, C, g = BilinearControl.EDMD.learn_bilinear_model(
    X_ref, Z_sim, Zu_sim, ["lasso", "lasso"]; 
    edmd_weights=[0.0], mapping_weights=[0.0]
)

# Check the koopman transform function
@test all(k->Z_sim[k] ≈ kf(X_ref[k]), eachindex(Z_sim))

## Compare solutions
z0 = kf(x0)
bi_X, bi_Z = simulate_bilinear(F, B, C, g, x0, z0, U_ref)

# Test that the discrete dynamics match
@test all(1:length(U_ref)) do k
    h = dt
    z = bi_Z[k]
    x = g * bi_Z[k]
    u = U_ref[k]
    xn0 = RD.discrete_dynamics(dmodel, x, U_ref[k], 0, h)
    zn = F*z + B*u + C[1]*z * u[1]
    xn_bilinear = g*zn
    norm(xn0 - xn_bilinear) < 5e-2 
end

# Test that the trajectories are similar
@test norm(bi_X - X_ref, Inf) < 0.2

## Load Bilinear Cartpole Model
# model_bilinear = Problems.BilinearCartpole()
model_bilinear = Problems.EDMDModel(F,B,C,g,kf,dt, "cartpole")
@test RD.discrete_dynamics(model_bilinear, bi_Z[1], U_ref[1], 0.0, dt) ≈ 
    bi_Z[2]

n2,m2 = RD.dims(model_bilinear)
J = zeros(n2, n2+m2)
y = zeros(n2)
z2 = KnotPoint(n2,m2,[bi_Z[1]; U_ref[1]], 0.0, dt)
RD.jacobian!(RD.InPlace(), RD.UserDefined(), model_bilinear, J, y, z2)
@test J ≈ [F+B*U_ref[1][1]+C[1]*U_ref[1][1] C[1]*bi_Z[1]]

Jfd = zero(J)
FiniteDiff.finite_difference_jacobian!(Jfd, 
    (y,x)->RD.discrete_dynamics!(model_bilinear, y, x[1:n2], x[n2+1:end], 0.0, dt),
    z2.z
)
@test Jfd ≈ J

@test BilinearControl.Problems.expandstate(model_bilinear, X_ref[1]) ≈ Z_sim[1]

## Test Bilinear MPC on Pendulum
using BilinearControl.Problems: simulatewithcontroller
model = RobotZoo.Pendulum()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
model_bilinear = Problems.EDMDModel(
    joinpath(Problems.DATADIR, "pendulum_eDMD_data.jld2"), name="pendulum"
)
lqr_datafile = joinpath(Problems.DATADIR, "pendulum_lqr_trajectories.jld2")
altro_datafile = joinpath(Problems.DATADIR, "pendulum_altro_trajectories.jld2")

# Stabilizing controller
xe = [pi,0]
ue = [0.0] 
N = 1001
dt = model_bilinear.dt
Xref = [copy(xe) for k = 1:N]
Uref = [copy(ue) for k = 1:N]
tref = range(0,length=N,step=dt)
Nmpc = 21
Qmpc = Diagonal([10.0,0.1])
Rmpc = Diagonal([1e-4])
ctrl_mpc = BilinearMPC(
    model_bilinear, Nmpc, Xref[1], Qmpc, Rmpc, Xref, Uref, tref
)
for x0 in ([pi-pi/4, 1.2], [pi+pi/3], -2.2)
    x0 = [pi-pi/4, 1.2]
    tsim = 1.0
    Xmpc, Umpc = simulatewithcontroller(
        dmodel, ctrl_mpc, x0, tsim, dt
    )
    @test norm(Xmpc[end] - xe) < 1e-3
end

# Trajectory following
X_test = load(altro_datafile, "X_test")
U_test = load(altro_datafile, "U_test")
Nmpc = 21
x_max = [20pi, 1000]
u_max = [200.]
x_min = -x_max
u_min = -u_max 

for test_idx = 1:5
    local x0 = copy(X_test[:,test_idx])
    xf = [pi,0]
    local Xref = X_test[:,test_idx] 
    local Uref = U_test[:,test_idx] 
    local tref = range(0,length=length(Xref),step=dt)
    Xref[end] .= xf
    push!(Uref, zeros(m))

    local ctrl_mpc = BilinearMPC(
        model_bilinear, Nmpc, x0, Qmpc, Rmpc, Xref, Uref, tref;
        x_max, x_min, u_max, u_min
    )

    # Run controller
    t_sim = 5.0
    time_sim = range(0,t_sim, step=dt)
    Xsim,Usim = simulatewithcontroller(dmodel, ctrl_mpc, Xref[1], t_sim, dt)
    @test norm(Xsim[1:length(Xref)] - Xref) < 2.0
    @test norm(Xsim[end] - xf) < 1e-1
end

#############################################
## Test MPC controller
#############################################
n,m = RD.dims(model_bilinear)
n0 = originalstatedim(model_bilinear)

test_idx = 1
Xref = X_test[:,test_idx] 
Uref = U_test[:,test_idx] 
tref = range(0,length=length(Xref),step=dt)
Xref[end] .= xe
push!(Uref, ue) 
ctrl_mpc = BilinearMPC(
    model_bilinear, Nmpc, x0, Qmpc, Rmpc, Xref, Uref, tref;
    x_max, x_min, u_max, u_min
)

Nx = Nmpc*n0
Ny = Nmpc*n
Nu = (Nmpc-1)*m
Nd = Nmpc*n

@test size(ctrl_mpc.A,2) ≈ Ny+Nu
@test size(ctrl_mpc.A,1) == Nmpc*n + (Nmpc-1)*(n0+m)

# Generate random input
X = [randn(n0) for k = 1:Nmpc]
U = [randn(m) for k = 1:Nmpc-1]
Y = map(x->expandstate(model_bilinear, x), X)

# Convert to vectors
Yref = map(x->expandstate(model_bilinear, x), Xref)
x̄ = reduce(vcat, Xref[1:Nmpc])
ū = reduce(vcat, Uref[1:Nmpc-1])
ȳ = reduce(vcat, Yref[1:Nmpc])
z̄ = [ȳ;ū]
x = reduce(vcat, X)
u = reduce(vcat, U)
y = reduce(vcat, Y)
z = [y;u]

# Test cost
J = 0.5 * dot(z, ctrl_mpc.P, z) + dot(ctrl_mpc.q,z) + sum(ctrl_mpc.c)
@test J ≈ sum(1:Nmpc) do k
    J = 0.5 * (X[k]-Xref[k])'Qmpc*(X[k]-Xref[k])
    if k < Nmpc
        J += 0.5 * (U[k] - Uref[k])'Rmpc*(U[k] - Uref[k])
    end
    J
end

# Test dynamics constraint
c = mapreduce(vcat, 1:Nmpc-1) do k
    J = zeros(n,n+m)
    yn = zeros(n)
    z̄ = RD.KnotPoint(Yref[k], Uref[k], tref[k], dt)
    RD.jacobian!(RD.InPlace(), RD.UserDefined(), model_bilinear, J, yn, z̄)
    A = J[:,1:n]
    B = J[:,n+1:end]
    dy = Y[k] - Yref[k]
    du = U[k] - Uref[k]
    dyn = Y[k+1] - Yref[k+1]
    RD.discrete_dynamics(model_bilinear, z̄) - Yref[k+1] + A*dy + B*du - dyn
end
c = [expandstate(model_bilinear, Xref[1]) - Y[1]; c]
ceq = Vector(ctrl_mpc.A*z - ctrl_mpc.l)[1:Nmpc*n]
@test c ≈ ceq

# Test bound constraints
G = model_bilinear.g
clo = [
    mapreduce(vcat, 1:Nmpc-1) do k
        G*Y[k+1] - (-x_max)
    end
    mapreduce(vcat, 1:Nmpc-1) do k
        U[k] - (-u_max)
    end
]
chi = [
    mapreduce(vcat, 1:Nmpc-1) do k
        G*Y[k+1] - x_max
    end
    mapreduce(vcat, 1:Nmpc-1) do k
        U[k] - u_max
    end
]
@test clo ≈ (ctrl_mpc.A*z - ctrl_mpc.l)[Nmpc*n+1:end]
@test chi ≈ (ctrl_mpc.A*z - ctrl_mpc.u)[Nmpc*n+1:end]
@test (ctrl_mpc.A*z)[Nd+1:end] ≈ [x[n0+1:end]; u]

