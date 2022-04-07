
function BilinearADMM(prob::TO.Problem)
    # Build bilinear matrices
    A,B,C,D = BilinearControl.buildbilinearconstraintmatrices(
        prob.model[1].continuous_dynamics, prob.x0, prob.xf, prob.Z[1].dt, prob.N
    )
    
    # Extract quadratic cost
    Q,q,R,r,c = BilinearControl.buildcostmatrices(prob)
    
    # Extract control bounds
    ucon = filter(x->x isa TO.BoundConstraint, TO.get_constraints(prob).constraints)
    if !isempty(ucon)
        n = RD.state_dim(prob, 1)
        ulo = TO.lower_bound(ucon[1])[n+1:end]
        uhi = TO.upper_bound(ucon[1])[n+1:end]
    else
        ulo = -Inf
        uhi = Inf
    end

    admm = BilinearADMM(A,B,C,D, Q,q,R,r,c, umin=ulo, umax=uhi)
    admm.opts.penalty_threshold = 1e4
    BilinearControl.setpenalty!(admm, 1e3)
    admm
end

extractstatevec(prob::TO.Problem) = vcat(Vector.(TO.states(prob))...)
extractcontrolvec(prob::TO.Problem) = vcat(Vector.(TO.controls(prob))...)
