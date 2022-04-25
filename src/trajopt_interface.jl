
function BilinearADMM(prob::TO.Problem; kwargs...)
    # Build bilinear matrices
    A,B,C,D = BilinearControl.buildbilinearconstraintmatrices(
        prob.model[1].continuous_dynamics, prob.x0, prob.xf, prob.Z[1].dt, prob.N
    )
    
    # Extract quadratic cost
    Q,q,R,r,c = BilinearControl.buildcostmatrices(prob)
    
    # Extract control bounds
    boundcons = filter(x->x isa TO.BoundConstraint, TO.get_constraints(prob).constraints)
    n,m = RD.dims(prob.model[1])
    ulo = fill(-Inf, m)
    uhi = fill(+Inf, m)
    xlo = fill(-Inf, n)
    xhi = fill(+Inf, n)
    for con in boundcons
        zlo = TO.lower_bound(con)
        zhi = TO.upper_bound(con)
        for i = 1:m
            xlo[i] = max(xlo[i], zlo[i])
            xhi[i] = min(xhi[i], zhi[i])
            ulo[i] = max(ulo[i], zlo[i+n])
            uhi[i] = min(uhi[i], zhi[i+n])
        end
    end

    admm = BilinearADMM(A,B,C,D, Q,q,R,r,c, umin=ulo, umax=uhi, xmin=xlo, xmax=xhi; kwargs...)
    admm.opts.penalty_threshold = 1e4
    BilinearControl.setpenalty!(admm, 1e3)
    admm
end

extractstatevec(prob::TO.Problem) = vcat(Vector.(TO.states(prob))...)
extractcontrolvec(prob::TO.Problem) = vcat(Vector.(TO.controls(prob))...)
