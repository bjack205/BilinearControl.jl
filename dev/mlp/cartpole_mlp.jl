import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using JSON
using LinearAlgebra
using FiniteDiff
using ForwardDiff
using Distributions
using BilinearControl
using JLD2
using Plots

include("mlp.jl")
include(joinpath(@__DIR__, "../../examples/cartpole/cartpole_utils.jl"))
include(joinpath(@__DIR__, "mlp_utils.jl"))


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
    U_test = altro_lqr_traj["U_test_swingup"]
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

    states_test = reduce(hcat, @view X_test[1:end-1,:])
    inputs_test = reduce(hcat, U_test)
    nextstates_test = reduce(hcat, @view X_test[2:end,:])

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
    J_test = map(CartesianIndices(U_test)) do cind
        k = cind[1]
        x = X_test[cind]
        u = U_test[cind]
        xn = zero(x)
        z = RD.KnotPoint{n,m}(x,u,T_train[k],h)
        J = zeros(n,n+m)
        RD.jacobian!(
            RD.InPlace(), RD.ForwardAD(), model, J, xn, z 
        )
        J
    end

    jacobians = zeros(n, n+m, length(J_train))
    jacobians_test = zeros(n, n+m, length(J_test))
    for i in eachindex(J_train) 
        jacobians[:,:,i] = J_train[i]
    end
    for i in eachindex(J_test) 
        jacobians_test[:,:,i] = J_test[i]
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
        "states_test"=>states_test,
        "inputs_test"=>inputs_test,
        "nextstates_test"=>nextstates_test,
        "jacobians_test"=>jacobians_test,
        "state_reference"=>X_ref,
        "input_reference"=>U_ref,
    )
    open(joinpath(@__DIR__, "cartpole_data.json"), "w") do f
        JSON.print(f, data, 2)
    end
end

# tracking error analysis
function test_mlp_models(mlp, mlp_jac, X_test, U_test, h, t_sim, alpha)
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

        n,m = RD.dims(mlp)
        jac = zeros(n,n+m)
        jac0 = zeros(n,n+m)
        xn = zeros(n)
        loss = mapreduce(+,1:N_ref-1) do k
            z = RD.KnotPoint{n,m}(X_ref[k], U_ref[k], T_ref[k], h)
            xtrue = RD.discrete_dynamics(dmodel_real, z)
            dx = RD.discrete_dynamics(mlp, z) - xtrue
            0.5 * (1-alpha) * dot(dx,dx)
        end
        loss_jac = mapreduce(+,1:N_ref-1) do k
            z = RD.KnotPoint{n,m}(X_ref[k], U_ref[k], T_ref[k], h)
            xtrue = RD.discrete_dynamics(dmodel_real, z)
            dx = RD.discrete_dynamics(mlp_jac, z) - xtrue
            L1 = 0.5 * (1-alpha) * dot(dx, dx) 
            RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), mlp_jac, jac, xn, z)
            RD.jacobian!(RD.InPlace(), RD.ForwardAD(), dmodel_nom, jac0, xn, z)
            djac = vec(jac - jac0)
            L2  = 0.5 * alpha * dot(djac, djac) 
            L1 + L2
        end
        loss /= (N_ref - 1)
        loss_jac /= (N_ref - 1)

        err_nom = norm(Xmpc - X_ref_full) / N_sim
        err_mlp = norm(Xmlp - X_ref_full) / N_sim
        err_mlp_jac = norm(Xmlp_jac - X_ref_full) / N_sim

        (; err_nom, err_mlp, err_mlp_jac, loss, loss_jac)
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

function getstats(res, field) 
    vals = getfield.(res, field)
    vals_valid = filter(isfinite, vals)
    if isempty(vals_valid)
        (median=NaN, up=NaN, lo=NaN, cnt=0)
    else
        (
            median=median(vals_valid), 
            up=quantile(vals_valid,0.95), 
            lo=quantile(vals_valid,0.05),
            cnt=length(vals_valid),
        )
    end
end

function run_sample_efficiency_analysis(;alpha=0.9, epochs=300, sample_sizes=25:25:400)
    num_test = 10
    data = load(CARTPOLE_DATAFILE)
    X_ref = data["X_ref"][:,end-num_test+1:end]
    U_ref = data["U_ref"][:,end-num_test+1:end]
    h = data["dt"]
    t_sim = data["t_sim"]

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
            epochs=epochs,
            alpha=alpha,
        )

        # build the models
        modelfile = joinpath(@__DIR__, "cartpole_model.json")
        modelfile_jac = joinpath(@__DIR__, "cartpole_model_jacobian.json")
        modeldata = JSON.parsefile(modelfile)
        modeldata_jac = JSON.parsefile(modelfile_jac)
        alpha = modeldata_jac["alpha"]
        loss_train = Float64.(modeldata["loss"])
        loss_train_jac = Float64.(modeldata_jac["loss"])
        loss_valid = Float64.(modeldata["vloss"])
        loss_valid_jac = Float64.(modeldata_jac["vloss"])
        loss_test = Float64.(modeldata["tloss"])
        loss_test_jac = Float64.(modeldata_jac["tloss"])
        
        p = plot(loss_train, lw=2, label="train-std", xlabel="epochs", ylabel="loss")
        plot!(p, loss_train_jac, lw=2, label="train-jac")
        plot!(p, loss_valid, lw=2, c=1, s=:dash, label="valid-std")
        plot!(p, loss_valid_jac, lw=2, c=2, s=:dash, label="valid-jac")
        plot!(p, loss_test, lw=2, c=1, s=:dot, label="test-std")
        plot!(p, loss_test_jac, lw=2, c=2, s=:dot, label="test-jac")
        display(p)

        mlp = MLP(modelfile)
        mlp_jac = MLP(modelfile_jac)

        # run the analysis
        res = test_mlp_models(mlp, mlp_jac, X_ref, U_ref, h, t_sim, alpha)

        stats_mlp = getstats(res, :err_mlp)
        stats_mlp_jac = getstats(res, :err_mlp_jac)
        println("###############")
        println("## Results ")
        println("###############")
        println("err mlp = $(stats_mlp.median) ($(stats_mlp.cnt))")
        println("err jac = $(stats_mlp_jac.median) ($(stats_mlp_jac.cnt))")

        if stats_mlp_jac.cnt < 9
            error("JMLP didn't work")
        end

        (;test=res, loss_train, loss_train_jac, loss_valid, 
            loss_valid_jac, loss_test, loss_test_jac,
            stats_mlp, stats_mlp_jac
        )
    end
end

##

cartpole_alpha9 = run_sample_efficiency_analysis(alpha=0.9, sample_sizes=25:25:150)
cartpole_alpha8 = run_sample_efficiency_analysis(alpha=0.8, sample_sizes=175:25:225)
cartpole_alpha5 = run_sample_efficiency_analysis(alpha=0.5, sample_sizes=250:25:275)
cartpole_alpha4 = run_sample_efficiency_analysis(alpha=0.4, sample_sizes=300:25:350)
cartpole_alpha2 = run_sample_efficiency_analysis(alpha=0.2, sample_sizes=375:25:400)

jldsave(joinpath(@__DIR__, "cartpole_sample_efficiency.jld2"); 
    sample_sizes9= 25:25:150, alpha9=cartpole_alpha9, 
    sample_sizes8=175:25:225, alpha8=cartpole_alpha8, 
    sample_sizes5=250:25:275, alpha5=cartpole_alpha5, 
    sample_sizes4=300:25:350, alpha4=cartpole_alpha4, 
    sample_sizes2=375:25:400, alpha2=cartpole_alpha2, 
)
##
resfile = jldopen(joinpath(@__DIR__, "cartpole_sample_efficiency.jld2"))
sample_sizes = vcat([resfile["sample_sizes" * string(i)] for i in [9,8,5,4,2]]...)
cartpole_res = vcat([resfile["alpha" * string(i)] for i in [9,8,5,4,2]]...)
alphas = vcat([fill(i / 10, length(resfile["sample_sizes" * string(i)])) for i in [9,8,5,4,2]]...)
close(resfile)

loss_train = map(cartpole_res) do res
    res.loss_train[end]
end
loss_train_jac = map(cartpole_res) do res
    res.loss_train_jac[end]
end
loss_test = map(cartpole_res) do res
    res.loss_test[end]
end
loss_test_jac = map(cartpole_res) do res
    res.loss_test_jac[end]
end

##
mlp = (
    median=map(res->res.stats_mlp.median, cartpole_res),
    up=map(res->res.stats_mlp.up, cartpole_res),
    lo=map(res->res.stats_mlp.lo, cartpole_res),
    cnt=map(res->res.stats_mlp.cnt, cartpole_res),
) 
mlp_jac = (
    median=map(res->res.stats_mlp_jac.median, cartpole_res),
    up=map(res->res.stats_mlp_jac.up, cartpole_res),
    lo=map(res->res.stats_mlp_jac.lo, cartpole_res),
    cnt=map(res->res.stats_mlp_jac.cnt, cartpole_res),
) 
nom = (
    median=map(res->median(getfield.(res.test, :err_nom)), cartpole_res),
    up=map(res->quantile(getfield.(res.test, :err_nom),0.95), cartpole_res),
    lo=map(res->quantile(getfield.(res.test, :err_nom),0.05), cartpole_res),
    cnt=10,
)

mlp_inds = mlp.cnt .< 9
jac_inds = mlp_jac.cnt .< 9
function setnan(x,i)
    x2 = copy(x)
    x2[i] .= NaN
    # Remove singleton finite values
    for j = 2:length(x)-1
        if !isnan(x2[j]) && (isnan(x2[j-1]) && isnan(x2[j+1]))
            x2[j] = NaN
        end
    end
    x2
end

using Plots
plot(sample_sizes, nom.median, label="MPC", lw=2, c=:black, yscale=:log10, 
    xlabel="training trajectories", ylabel="tracking error", ylim=(0.03,1.2)
)
plot!(sample_sizes, nom.up, label="", s=:dash, c=:black, yscale=:log10)
plot!(sample_sizes, nom.lo, label="", s=:dash, c=:black, yscale=:log10)
plot!(sample_sizes, setnan(mlp.median, mlp_inds), lw=2, label="MLP", c=1)
plot!(sample_sizes, setnan(mlp.up, mlp_inds), s=:dash, label="", c=1)
plot!(sample_sizes, setnan(mlp.lo, mlp_inds), s=:dash, label="", c=1)
plot!(sample_sizes, setnan(mlp_jac.median, jac_inds), lw=2, label="JMLP", c=2)
plot!(sample_sizes, setnan(mlp_jac.up, jac_inds), s=:dash, label="", c= 2)
plot!(sample_sizes, setnan(mlp_jac.lo, jac_inds), s=:dash, label="", c= 2)
plot!(sample_sizes, alphas, label="alpha", s=:dash, c=:gray)

using PGFPlotsX
using LaTeXStrings
include(joinpath(@__DIR__, "../../examples/plotting_constants.jl"))
p_err = 
@pgf TikzPicture(
Axis(
    {
        xmajorgrids,
        ymajorgrids,
        ymode="log",
        xlabel = "Number of Training Trajectories",
        ylabel = "Tracking Error",
        legend_pos = "north west",
    },
    PlotInc({lineopts..., color=color_nominal, solid, thick}, 
        Coordinates(sample_sizes, nom.median)),
    PlotInc({lineopts..., "name_path=E", "black!20", "forget plot", solid, line_width=0.1}, 
        Coordinates(sample_sizes, nom.up)),
    PlotInc({lineopts..., "name_path=F","black!20", "forget plot", solid, line_width=0.1}, 
        Coordinates(sample_sizes, nom.lo)),
    PlotInc({lineopts..., color=color_eDMD, solid, thick}, 
        Coordinates(sample_sizes, setnan(mlp.median, mlp_inds))),
    PlotInc({lineopts..., "name_path=G", color="$(color_eDMD)!10", "forget plot", solid, line_width=0.1}, 
        Coordinates(sample_sizes, setnan(mlp.up, mlp_inds))),
    PlotInc({lineopts..., "name_path=H", color="$(color_eDMD)!10", "forget plot", solid, line_width=0.1}, 
        Coordinates(sample_sizes, setnan(mlp.lo, mlp_inds))),
    PlotInc({lineopts..., color=color_jDMD, solid, thick}, 
        Coordinates(sample_sizes, setnan(mlp_jac.median, jac_inds))),
    PlotInc({lineopts..., "name_path=I", color="$(color_jDMD)!10", "forget plot", solid, line_width=0.1}, 
        Coordinates(sample_sizes, setnan(mlp_jac.up, jac_inds))),
    PlotInc({lineopts..., "name_path=J",color="$(color_jDMD)!10", "forget plot", solid, line_width=0.1}, 
        Coordinates(sample_sizes, setnan(mlp_jac.lo, jac_inds))),
    Legend(["Nominal", "MLP", "JMLP"])
),
Axis(
    {
        "axis y line*"="right",
        "axis x line"="none",
        "ymax"=1.3,
        "ylabel"=L"\alpha"
    },
    PlotInc({"gray", "no_marks", "thick"}, Coordinates(sample_sizes, alphas))
)
);
pgfsave(joinpath(BilinearControl.FIGDIR, "cartpole_mlp.tikz"), p_err, include_preamble=false)

plot(sample_sizes, loss_train, lw=2, label="dyn-train",
    xlabel="training trajectories",
    ylabel="loss",
)
plot!(sample_sizes, loss_train_jac, lw=2, label="jac-train")
plot!(sample_sizes, loss_test, lw=2, s=:dash, c=1, label="dyn-test")
plot!(sample_sizes, loss_test_jac, lw=2, s=:dash, c=2, label="jac-test")
p_train = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel="Number of Training Trajectories",
        ylabel="Loss",
        legend_columns=2,
    },
    PlotInc({lineopts..., color=color_eDMD}, Coordinates(sample_sizes, loss_train)),
    PlotInc({lineopts..., color=color_eDMD, "dashed"}, Coordinates(sample_sizes, loss_test)),
    PlotInc({lineopts..., color=color_jDMD}, Coordinates(sample_sizes, loss_train_jac)),
    PlotInc({lineopts..., color=color_jDMD, "dashed"}, Coordinates(sample_sizes, loss_test_jac)),
    Legend("MLP-train", "MLP-test", "JMLP-train", "JMLP-test")
)
pgfsave(joinpath(BilinearControl.FIGDIR, "cartpole_train.tikz"), p_train, include_preamble=false)

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
res = let sample_size = 300, use_relu = false, alpha=0.6
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
        hidden=32,
        alpha=alpha,
        verbose=true;
        use_relu,
    )

    # build the models
    modelfile = joinpath(@__DIR__, "cartpole_model.json")
    modelfile_jac = joinpath(@__DIR__, "cartpole_model_jacobian.json")

    mlp = MLP(modelfile; use_relu)
    mlp_jac = MLP(modelfile_jac; use_relu)

    # run the analysis
    test_mlp_models(mlp, mlp_jac, X_ref, U_ref, h, t_sim, alpha)
end
getfield.(res,:err_mlp_jac)
modelfile = joinpath(@__DIR__, "cartpole_model.json")
modelfile_jac = joinpath(@__DIR__, "cartpole_model_jacobian.json")
modeldata = JSON.parsefile(modelfile)
modeldata_jac = JSON.parsefile(modelfile_jac)
using Plots
plot(modeldata["loss"], label="train")
plot!(modeldata["vloss"], label="validation")
sample_sizes_combined[10]
res
cartpole_res_combined[8]
median(filter(isfinite,getfield.(res, :err_mlp_jac)))
median(filter(isfinite,getfield.(cartpole_res_combined[8], :err_mlp_jac)))


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
