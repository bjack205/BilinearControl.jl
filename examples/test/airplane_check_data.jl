## Visualizer
model = Problems.NominalAirplane()
include(joinpath(Problems.VISDIR, "visualization.jl"))
vis = Visualizer()
delete!(vis)
set_airplane!(vis, model)
open(vis)

include("../airplane_constants.jl")

airplane_data = load(AIRPLANE_DATAFILE)
X_train = airplane_data["X_train"]
U_train = airplane_data["U_train"]
X_test = airplane_data["X_test"]
U_test = airplane_data["U_test"]
num_train = size(X_train,2)
num_test =  size(X_test,2)
T_ref = airplane_data["T_ref"]
t_ref = T_ref[end] 
X_train[end,:]

visualize!(vis, model_real, t_ref, X_train[:,7])

p = plot(xlabel="time (s)", ylabel="angle of attack")
for i = 1:num_train
    plot!(p, T_ref, map(rad2deg ∘ Problems.angleofattack, X_train[:,i]), label="")
end
p