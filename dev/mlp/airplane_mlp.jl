import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using JSON
using LinearAlgebra
using FiniteDiff
using ForwardDiff
using Distributions
using BilinearControl
using JLD2

include("mlp.jl")
include(joinpath(@__DIR__, "../../examples/airplane/airplane_utils.jl"))
include(joinpath(@__DIR__, "mlp_utils.jl"))

const AIRPLANE_DATAFILE_JSON = joinpath(@__DIR__, "airplane_data.json")

function airplane_gendata_mlp(;num_train=30)
    airplane_data = load(AIRPLANE_DATAFILE)
    good_cols = findall(x->isfinite(norm(x)), eachcol(airplane_data["X_train"]))
    num_train0 = size(airplane_data["X_train"], 2)
    X_train = airplane_data["X_train"][:,good_cols[1:num_train]]
    U_train = airplane_data["U_train"][:,good_cols[1:num_train]]
    X_ref = airplane_data["X_ref"][:,num_train0:end]
    U_ref = airplane_data["U_ref"][:,num_train0:end]
    T_ref = airplane_data["T_ref"]
    h = T_ref[2]
    t_sim = size(U_train, 1) * h

    # Generate Jacobians
    T_train = range(0,length=size(X_train,1), step=h)
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(BilinearControl.NominalAirplane())
    n,m = RD.dims(dmodel_nom)
    J_train = map(CartesianIndices(U_train)) do cind
        k = cind[1]
        x = X_train[cind]
        u = U_train[cind]
        xn = zero(x)
        z = RD.KnotPoint{n,m}(x,u,T_train[k],h)
        J = zeros(n,n+m)
        RD.jacobian!(
            RD.InPlace(), RD.ForwardAD(), dmodel_nom, J, xn, z 
        )
        J
    end

    # Flatten arrays
    states = reduce(hcat, @view X_train[1:end-1,:])
    inputs = reduce(hcat, U_train)
    nextstates = reduce(hcat, @view X_train[2:end,:])

    jacobians = zeros(n, n+m, length(J_train))
    for i in eachindex(J_train) 
        jacobians[:,:,i] = J_train[i]
    end

    # Save to JSON
    data = Dict(
        "name"=>"airplane",
        "h"=>h,
        "t_sim"=>t_sim,
        "num_lqr"=>0,
        "num_ref"=>num_train,
        "num_train"=>size(X_train,2),
        "num_test"=>size(X_ref,2),
        "states"=>states,
        "inputs"=>inputs,
        "nextstates"=>nextstates,
        "jacobians"=>jacobians,
        "state_reference"=>X_ref,
        "input_reference"=>U_ref,
    )
    open(AIRPLANE_DATAFILE_JSON, "w") do f
        JSON.print(f, data, 2)
    end
end

function gen_airplane_mpc_controller(model, Xref, Uref, Tref)
    u_trim = [41.66667789082778, 105.99999999471807, 74.65179381344494, 106.00000124622453]
    if length(Uref) < length(Xref)
        push!(Uref, u_trim)
    end

    Qk = Diagonal([fill(1e0, 3); fill(1e1, 3); fill(1e-1, 3); fill(2e-1, 3)])
    Rk = Diagonal(fill(1e-3,4))
    Qf = Diagonal([fill(1e-2, 3); fill(1e0, 3); fill(1e1, 3); fill(1e1, 3)]) * 10
    u_trim = [41.66667789082778, 105.99999999471807, 74.65179381344494, 106.00000124622453]
    xmax = [fill(0.5,3); fill(1.0, 3); fill(0.5, 3); fill(10.0, 3)]
    xmin = -xmax
    umin = fill(0.0, 4) - u_trim
    umax = fill(255.0, 4) - u_trim
    Nt = 21

    BilinearControl.LinearMPC(model, Xref, Uref, Tref, Qk, Rk, Qf; Nt=Nt,
        xmin,xmax,umin,umax
    )
end

function test_mlp_models(mlp, mlp_jac)
    # Models
    model_nom = BilinearControl.NominalAirplane()
    model_real = BilinearControl.SimulatedAirplane()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom)
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real)

    data = JSON.parsefile(AIRPLANE_DATAFILE_JSON)
    h = data["h"]
    t_sim = data["t_sim"]
    X_train = data["states"]
    X_ref = data["state_reference"]
    U_ref = data["input_reference"]
    X_ref = reshape(mapreduce(x->Vector{Float64}.(x),vcat, X_ref), :, length(X_ref))
    U_ref = reshape(mapreduce(x->Vector{Float64}.(x),vcat, U_ref), :, length(U_ref))
    X_ref0 = X_ref[:,end-9:end]
    U_ref0 = U_ref[:,end-9:end]
    T_ref = range(0,step=h,length=size(X_ref,1))

    # Allocate result vectors
    num_test = 10
    err_nom = zeros(num_test) 
    err_mlp = zeros(num_test) 
    err_jac = zeros(num_test) 

    # Run MPC on each trajectory
    for i = 1:num_test
        X_ref = X_ref0[:,i]
        U_ref = U_ref0[:,i]
        N = length(X_ref)

        # mpc_nom = BilinearControl.LinearMPC(dmodel_nom, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
        #     xmin,xmax,umin,umax
        # )
        # mpc_mlp = BilinearControl.LinearMPC(mlp, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
        #     xmin,xmax,umin,umax
        # )
        # mpc_jac = BilinearControl.LinearMPC(mlp_jac, X_ref, U_ref, T_ref, Qk, Rk, Qf; Nt=Nt,
        #     xmin,xmax,umin,umax
        # )
        mpc_nom = gen_airplane_mpc_controller(dmodel_nom, X_ref0[:,i], U_ref0[:,i], T_ref)
        mpc_mlp = gen_airplane_mpc_controller(mlp, X_ref0[:,i], U_ref0[:,i], T_ref)
        mpc_jac = gen_airplane_mpc_controller(mlp_jac, X_ref0[:,i], U_ref0[:,i], T_ref)

        x0 = copy(X_ref0[1,i])
        X_nom, U_nom = BilinearControl.simulatewithcontroller(dmodel_real, mpc_nom, x0, t_sim, h)
        X_jac, U_jac = BilinearControl.simulatewithcontroller(dmodel_real, mpc_jac, x0, t_sim, h)
        X_mlp, U_mlp = BilinearControl.simulatewithcontroller(dmodel_real, mpc_mlp, x0, t_sim, h)
        @show X_mlp[end]
        err_nom[i] = norm(X_nom - X_ref) / N
        err_mlp[i] = norm(X_mlp - X_ref) / N
        err_jac[i] = norm(X_jac - X_ref) / N
    end
    Dict(:nominal=>err_nom, :mlp=>err_mlp, :mlp_jac=>err_jac)
end

##
gen_airplane_data(num_train=2000)
airplane_gendata_mlp(num_train=2000)
airplane_data = load(AIRPLANE_DATAFILE)

model_nom = BilinearControl.NominalAirplane()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom) 
model_real = BilinearControl.SimulatedAirplane()
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real) 

data = JSON.parsefile(AIRPLANE_DATAFILE_JSON)
h = data["h"]
t_sim = data["t_sim"]
X_train = data["states"]
X_ref = data["state_reference"]
U_ref = data["input_reference"]
X_ref = reshape(mapreduce(x->Vector{Float64}.(x),vcat, X_ref), :, length(X_ref))
U_ref = reshape(mapreduce(x->Vector{Float64}.(x),vcat, U_ref), :, length(U_ref))
X_test_swingup_ref = X_ref[:,end-9:end]
U_test_swingup_ref = U_ref[:,end-9:end]
T_ref = range(0,step=h,length=size(X_ref,1))


modelfile = joinpath(@__DIR__, "airplane_model.json")
modelfile_jac = joinpath(@__DIR__, "airplane_model_jacobian.json")

train_model(AIRPLANE_DATAFILE_JSON, modelfile, 
    epochs=300, alpha=0.01, hidden=64, verbose=true
)

## 
mlp = MLP(modelfile)
mlp_jac = MLP(modelfile_jac)


i = 2
mpc = gen_airplane_mpc_controller(dmodel_nom, X_ref[:,i], U_ref[:,i], T_ref)
mpc_mlp = gen_airplane_mpc_controller(mlp, X_ref[:,i], U_ref[:,i], T_ref)
mpc_mlp_jac = gen_airplane_mpc_controller(mlp_jac, X_ref[:,i], U_ref[:,i], T_ref)

x0 = copy(X_ref[1,i])
Xmpc, Umpc, Tmpc = BilinearControl.simulatewithcontroller(dmodel_real, mpc, x0, t_sim, h)
Xmlp_jac, Umlp_jac = BilinearControl.simulatewithcontroller(dmodel_real, mpc_mlp_jac, x0, t_sim, h)
Xmlp, Umlp = BilinearControl.simulatewithcontroller(dmodel_real, mpc_mlp, x0, t_sim, h)

using Plots
plotstates(T_ref, X_ref[:,i], inds=1:3, lw=1, label=["ref" ""], c=:black, s=:dash, legend=:bottomright)
plotstates!(Tmpc, Xmpc, inds=1:3, lw=2, label=["nom" ""], c=[1 2])
plotstates!(Tmpc, Xmlp_jac, inds=1:2, c=[1 2], s=:dash, label=["mlp_jac" ""])
plotstates!(Tmpc, Xmlp, inds=1:2, c=[1 2], s=:dot, label=["mlp" ""])
Xmpc[end]

##
res = test_mlp_models(mlp, mlp_jac)
res