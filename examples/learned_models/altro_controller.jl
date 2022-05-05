
mutable struct ALTROController{D} <: AbstractController
    genprob::Function
    distribution::D
    tvlqr::TVLQRController
    prob::TO.Problem{Float64}
    opts::Altro.SolverOptions{Float64}
    function ALTROController(genprob::Function, distribution::D; opts=Altro.SolverOptions()) where D
        params = rand(distribution)
        prob = genprob(params...)
        n = RD.state_dim(prob,1)
        m = RD.control_dim(prob,1)
        N = TO.horizonlength(prob)
        K = [zeros(m,n) for k = 1:N-1]
        Xref = RD.states(prob)
        Uref = RD.controls(prob)
        time = RD.gettimes(prob)
        tvlqr = TVLQRController(K, Xref, Uref, time)
        new{D}(genprob, distribution, tvlqr, prob, opts)
    end
end

function resetcontroller!(ctrl::ALTROController, x0)
    params = rand(ctrl.distribution)
    prob = ctrl.genprob(params...)
    TO.set_initial_state!(prob, x0)
    solver = Altro.ALTROSolver(prob, ctrl.opts)
    solve!(solver)
    status = Altro.status(solver)
    if status != Altro.SOLVE_SUCCEEDED
        @warn "ALTRO solve failed."
    end
    X = RD.states(solver)
    U = RD.controls(solver)
    t = RD.gettimes(prob)
    K = Altro.get_ilqr(solver).K
    N = TO.horizonlength(prob)
    ctrl.prob = prob
    resize!(ctrl.tvlqr.K, N-1)
    resize!(ctrl.tvlqr.xref, N)
    resize!(ctrl.tvlqr.uref, N-1)
    resize!(ctrl.tvlqr.time, N)
    copyto!(ctrl.tvlqr.K, K)
    copyto!(ctrl.tvlqr.xref, X)
    copyto!(ctrl.tvlqr.uref, U)
    copyto!(ctrl.tvlqr.time, t)
    ctrl
end

function getcontrol(ctrl::ALTROController, x, t)
    getcontrol(ctrl.tvlqr, x, t)
end