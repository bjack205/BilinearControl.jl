import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using JSON
using LinearAlgebra
using FiniteDiff
using ForwardDiff
using Distributions
using BilinearControl
using JLD2

include("mlp.jl")
include(joinpath(@__DIR__, "../../examples/cartpole/cartpole_utils.jl"))


##
function cartpole_gendata_mlp(;num_lqr=50, num_swingup=50)
    # X_train, U_train, X_test, U_test, X_ref, U_ref, metadata = generate_cartpole_data(;
    #     num_lqr, num_swingup,
    #     save_to_file=false,
    # )
    altro_lqr_traj = load(CARTPOLE_DATAFILE)
    X_train_lqr = altro_lqr_traj["X_train_lqr"][:,1:num_lqr]
    U_train_lqr = altro_lqr_traj["U_train_lqr"][:,1:num_lqr]
    X_train_swingup = altro_lqr_traj["X_train_swingup"][:,1:num_swingup]
    U_train_swingup = altro_lqr_traj["U_train_swingup"][:,1:num_swingup]
    X_train = [X_train_lqr X_train_swingup]
    U_train = [U_train_lqr U_train_swingup]

    X_test = altro_lqr_traj["X_test_swingup"]
    X_ref = altro_lqr_traj["X_ref"]
    U_ref = altro_lqr_traj["U_ref"]

    t_sim = altro_lqr_traj["t_sim"]
    h = altro_lqr_traj["dt"]

    n = length(X_train[1])
    m = length(U_train[1])
    T_train = range(0,t_sim, step=h)
    model = RD.DiscretizedDynamics{RD.RK4}(BilinearControl.NominalCartpole())

    goodinds = findall(x->abs(x[2]-pi) < deg2rad(45), X_train[end,:])
    X_train = X_train[:,goodinds]
    U_train = U_train[:,goodinds]
    num_lqr = count(x->x <= num_lqr, goodinds)
    num_swingup = length(goodinds) - num_lqr

    # Flatten arrays
    states = reduce(hcat, @view X_train[1:end-1,:])
    inputs = reduce(hcat, U_train)
    nextstates = reduce(hcat, @view X_train[2:end,:])

    # Generate Jacobians
    J_train = map(CartesianIndices(U_train)) do cind
        k = cind[1]
        x = X_train[cind]
        u = U_train[cind]
        xn = zero(x)
        z = RD.KnotPoint{n,m}(x,u,T_train[k],h)
        J = zeros(n,n+m)
        RD.jacobian!(
            RD.InPlace(), RD.ForwardAD(), model, J, xn, z 
        )
        J
    end

    jacobians = zeros(n, n+m, length(J_train))
    for i in eachindex(J_train) 
        jacobians[:,:,i] = J_train[i]
    end

    # Save to JSON
    data = Dict(
        "name"=>"cartpole",
        "h"=>h,
        "t_sim"=>t_sim,
        "num_lqr"=>num_lqr,
        "num_ref"=>num_swingup,
        "num_train"=>size(X_train,2),
        "num_test"=>size(X_test,2),
        "states"=>states,
        "inputs"=>inputs,
        "nextstates"=>nextstates,
        "jacobians"=>jacobians,
        "state_reference"=>X_ref,
        "input_reference"=>U_ref,
    )
    open(joinpath(@__DIR__, "cartpole_data.json"), "w") do f
        JSON.print(f, data, 2)
    end
end

# tracking error analysis
function test_mlp_models(mlp, mlp_jac, X_test, U_test, h, t_sim)
    # models
    model_nom = BilinearControl.NominalCartpole()
    dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom) 
    model_real = BilinearControl.SimulatedCartpole()
    dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real) 

    xe = [0,pi,0,0]
    ue = [0.0]
    N_test = size(X_test, 2) 
    T_sim = range(0, t_sim, step=h)
    N_sim = length(T_sim) 
    N_ref = size(X_test, 1)
    T_ref = range(0, length=N_ref, step=h)
    map(1:N_test) do i
        X_ref = deepcopy(X_test[:,i])
        U_ref = deepcopy(U_test[:,i])
        X_ref[end] .= xe
        push!(U_ref, ue)

        N_ref = length(T_ref)
        X_ref_full = [X_ref; [copy(xe) for i = 1:N_sim - N_ref]]

        mpc = gen_mpc_controller(dmodel_nom, X_ref, U_ref, T_ref)
        mpc_mlp = gen_mpc_controller(mlp, X_ref, U_ref, T_ref)
        mpc_mlp_jac = gen_mpc_controller(mlp_jac, X_ref, U_ref, T_ref)

        x0 = zeros(4)
        Xmpc, Umpc, Tmpc = BilinearControl.simulatewithcontroller(dmodel_real, mpc, x0, t_sim, h)
        Xmlp, Umlp = BilinearControl.simulatewithcontroller(dmodel_real, mpc_mlp, x0, t_sim, h)
        Xmlp_jac, Umlp_jac = BilinearControl.simulatewithcontroller(dmodel_real, mpc_mlp_jac, x0, t_sim, h)

        err_nom = norm(Xmpc - X_ref_full) / N_sim
        err_mlp = norm(Xmlp - X_ref_full) / N_sim
        err_mlp_jac = norm(Xmlp_jac - X_ref_full) / N_sim

        (; err_nom, err_mlp, err_mlp_jac)
    end
end

function gen_mpc_controller(model, Xref, Uref, Tref)
    Xref[end] = [0,pi,0,0]
    if length(Uref) < length(Xref)
        push!(Uref, zeros(1))
    end

    Qmpc = Diagonal(fill(1e-0,4))
    Rmpc = Diagonal(fill(1e-3,1))
    Qfmpc = Diagonal(fill(1e2,4))
    Nt = 41
    TrackingMPC(model, Xref, Uref, collect(Tref), Qmpc, Rmpc, Qfmpc; Nt=Nt)
end

function train_model(datafile, outfile; epochs=100, alpha=0.5, hidden=32)
    mainfile = joinpath(@__DIR__, "main.py")
    outbase,ext = splitext(outfile)
    outfile_jac = outbase * "_jacobian" * ext
    cmd = `python3 $mainfile $datafile -o $outfile --epochs $epochs --alpha $alpha --hidden $hidden`
    cmd_jac = `python3 $mainfile $datafile -o $outfile_jac --jacobian --epochs $epochs --alpha $alpha --hidden $hidden`
    run(cmd)
    run(cmd_jac)
end

function run_sample_efficiency_analysis()
    num_test = 10
    data = load(CARTPOLE_DATAFILE)
    X_ref = data["X_ref"][:,end-num_test+1:end]
    U_ref = data["U_ref"][:,end-num_test+1:end]
    h = data["dt"]
    t_sim = data["t_sim"]

    # sample_sizes = [25, 50, 75, 100, 125, 150, 175, 200]
    # sample_sizes = [25, 50, 75, 100, 125, 150, 175, 200] .+ 200
    sample_sizes = 25:25:400
    map(sample_sizes) do sample_size
        println("\n#############################################")
        println("## SAMPLE SIZE = ", sample_size)
        println("#############################################")
        # generate json data file
        cartpole_gendata_mlp(num_lqr=sample_size, num_swingup=sample_size)

        # call python to train the model using the json data file
        train_model(
            joinpath(@__DIR__,"cartpole_data.json"), 
            joinpath(@__DIR__,"cartpole_model.json"), 
            epochs=300,
            alpha=0.9
        )

        # build the models
        modelfile = joinpath(@__DIR__, "cartpole_model.json")
        modelfile_jac = joinpath(@__DIR__, "cartpole_model_jacobian.json")

        mlp = MLP(modelfile)
        mlp_jac = MLP(modelfile_jac)

        # run the analysis
        test_mlp_models(mlp, mlp_jac, X_ref, U_ref, h, t_sim)
    end
end
cartpole_res2 = run_sample_efficiency_analysis()
cartpole_res2
cartpole_res_combined = [cartpole_res; cartpole_res2]
sample_sizes_combined = [sample_sizes; sample_sizes .+ sample_sizes[end]]

jldsave(joinpath(@__DIR__, "cartpole_sample_efficiency.jld2"); 
    sample_sizes=sample_sizes_combined, alpha5=cartpole_res_combined, alpha9=cartpole_res2
)
res = jldopen(joinpath(@__DIR__, "cartpole_sample_efficiency.jld2"))
sample_sizes
sample_sizes_combined = res["sample_sizes"]
cartpole_res_combined = [res["alpha9"][1:7]; res["alpha5"][8:end]]

function getstats(results, field) 
    med = map(results) do res
        vals = filter(isfinite, getfield.(res, field))
        if isempty(vals)
             NaN
        else
            median(vals)
        end
    end
    up = map(results) do res 
        vals = filter(isfinite, getfield.(res, field))
        if isempty(vals)
            NaN
        else
            quantile(vals, 0.95)
        end
    end
    lo = map(results) do res 
        vals = filter(isfinite, getfield.(res, field))
        if isempty(vals) 
            NaN
        else
            quantile(vals, 0.05)
        end
    end
    cnt = map(results) do res 
        vals = getfield.(res, field)
        count(isfinite, vals)
    end
    (;median=med,up,lo,cnt)
end
nom = getstats(cartpole_res_combined, :err_nom)
mlp = getstats(cartpole_res_combined, :err_mlp)
mlp_jac = getstats(cartpole_res_combined, :err_mlp_jac)
mlp_inds = mlp.cnt .< 9
jac_inds = mlp_jac.cnt .< 9
setnan(x,i) = begin x2 = copy(x); x2[i] .= NaN; x2 end

using Plots
plot(sample_sizes_combined, nom.median, label="MPC", lw=2, c=:black, yscale=:log10, xlabel="training trajectories", ylabel="tracking error")
plot!(sample_sizes_combined, nom.up, label="", s=:dash, c=:black, yscale=:log10)
plot!(sample_sizes_combined, nom.lo, label="", s=:dash, c=:black, yscale=:log10)
plot!(sample_sizes_combined, setnan(mlp.median, mlp_inds), lw=2, label="MLP", c=1)
plot!(sample_sizes_combined, setnan(mlp.up, mlp_inds), s=:dash, label="", c=1)
plot!(sample_sizes_combined, setnan(mlp.lo, mlp_inds), s=:dash, label="", c=1)
plot!(sample_sizes_combined, setnan(mlp_jac.median, jac_inds), lw=2, label="JMLP", c=2)
plot!(sample_sizes_combined, setnan(mlp_jac.up, jac_inds), s=:dash, label="", c= 2)
plot!(sample_sizes_combined, setnan(mlp_jac.lo, jac_inds), s=:dash, label="", c= 2)

nom = getstats(cartpole_res2, :err_nom)
mlp = getstats(cartpole_res2, :err_mlp)
mlp_jac2 = getstats(res["alpha9"], :err_mlp_jac)
[sample_sizes_combined mlp_jac.cnt mlp_jac2.cnt]'
mlp_jac.median[end]
cartpole_res_combined[4]
mlp_jac
mlp_jac.median[end]
mlp

##
res = let sample_size = 300
    num_test = 10
    data = load(CARTPOLE_DATAFILE)
    X_ref = data["X_ref"][:,end-num_test+1:end]
    U_ref = data["U_ref"][:,end-num_test+1:end]
    h = data["dt"]
    t_sim = data["t_sim"]

    # generate json data file
    cartpole_gendata_mlp(num_lqr=sample_size, num_swingup=sample_size)

    # call python to train the model using the json data file
    train_model(
        joinpath(@__DIR__,"cartpole_data.json"), 
        joinpath(@__DIR__,"cartpole_model.json"), 
        epochs=300,
        alpha=0.1,
    )

    # build the models
    modelfile = joinpath(@__DIR__, "cartpole_model.json")
    modelfile_jac = joinpath(@__DIR__, "cartpole_model_jacobian.json")

    mlp = MLP(modelfile)
    mlp_jac = MLP(modelfile_jac)

    # run the analysis
    test_mlp_models(mlp, mlp_jac, X_ref, U_ref, h, t_sim)
end
sample_sizes_combined[end]
res
cartpole_res_combined[end]
median(getfield.(res, :err_mlp_jac))
median(getfield.(res, :err_mlp_jac))


##
X_train, U_train, X_test, U_test, X_ref, U_ref, metadata = generate_cartpole_data(;
    num_lqr=400, num_swingup=400,
    save_to_file=true,
)

@time cartpole_gendata_mlp(num_lqr=200, num_swingup=200)

model_nom = BilinearControl.NominalCartpole()
dmodel_nom = RD.DiscretizedDynamics{RD.RK4}(model_nom) 
model_real = BilinearControl.SimulatedCartpole()
dmodel_real = RD.DiscretizedDynamics{RD.RK4}(model_real) 

datafile = joinpath(@__DIR__, "cartpole_data.json")
data = JSON.parsefile(datafile)
h = data["h"]
t_sim = data["t_sim"]
X_ref = data["state_reference"]
U_ref = data["input_reference"]
X_ref = reshape(mapreduce(x->Vector{Float64}.(x),vcat, X_ref), :, length(X_ref))
U_ref = reshape(mapreduce(x->Vector{Float64}.(x),vcat, U_ref), :, length(U_ref))
X_test_swingup_ref = X_ref[:,end-9:end]
U_test_swingup_ref = U_ref[:,end-9:end]
T_ref = range(0,step=h,length=size(X_ref,1))

##
modelfile = joinpath(@__DIR__, "cartpole_model.json")
modelfile_jac = joinpath(@__DIR__, "cartpole_model_jacobian.json")
modeldata = JSON.parsefile(modelfile)
modeldata_jac = JSON.parsefile(modelfile_jac)

mlp = MLP(modelfile)
mlp_jac = MLP(modelfile_jac)

mpc = gen_mpc_controller(dmodel_nom, X_ref[:,1], U_ref[:,1], T_ref)
mpc_mlp = gen_mpc_controller(mlp, X_ref[:,1], U_ref[:,1], T_ref)
mpc_mlp_jac = gen_mpc_controller(mlp_jac, X_ref[:,1], U_ref[:,1], T_ref)

x0 = zeros(4)
Xmpc, Umpc, Tmpc = BilinearControl.simulatewithcontroller(dmodel_real, mpc, x0, tsim, h)
Xmlp_jac, Umlp_jac = BilinearControl.simulatewithcontroller(dmodel_real, mpc_mlp_jac, x0, tsim, h)
Xmlp_jac2, Umlp_jac2 = BilinearControl.simulatewithcontroller(dmodel_nom, mpc_mlp_jac, x0, tsim, h)
Xmlp, Umlp = BilinearControl.simulatewithcontroller(dmodel_real, mpc_mlp, x0, tsim, h)

using Plots
plotstates(T_ref, X_ref[:,1], inds=1:2, lw=1, label=["ref" ""], c=:black, s=:dash)
plotstates!(Tmpc, Xmpc, inds=1:2, lw=2, label=["nom" ""], c=[1 2])
plotstates!(Tmpc, Xmlp_jac, inds=1:2, ylim=(-3,5), c=[1 2], s=:dash, label=["mlp_jac" ""])
# plotstates!(Tmpc, Xmlp, inds=1:2, ylim=(-3,5), c=[1 2], s=:dashdot, label=["mlp" ""])
