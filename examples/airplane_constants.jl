const AIRPLANE_DATAFILE = joinpath(Problems.DATADIR, "airplane_trajectory_data.jld2")
const AIRPLANE_MODELFILE = joinpath(Problems.DATADIR, "airplane_trained_models.jld2")
const AIRPLANE_RESULTS = joinpath(Problems.DATADIR, "airplane_results.jld2")

function airplane_kf(x)
    p = x[1:3]
    q = x[4:6]
    mrp = MRP(x[4], x[5], x[6])
    R = Matrix(mrp)
    v = x[7:9]
    w = x[10:12]
    α = atan(v[3],v[1])  # angle of attack
    β = atan(v[2],v[1])  # side slip
    vbody = R'v
    speed = vbody'vbody
    [1; x; vec(R); vbody; speed; sin.(p); α; β; α^2; β^2; α^3; β^3; p × v; p × w; 
        w × w; 
        EDMD.chebyshev(x, order=[3,4])]
end
