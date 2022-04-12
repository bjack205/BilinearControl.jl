function updatetrajectory!(solver::BilinearADMM, Xref, Q, Qf, kstart=1)
    nx = size(Q,1)
    q = reshape(solver.q, nx, :)
    N = size(q,2)
    c = 0.0
    for k = 1:N
        Q_ = k < N ? Q : Qf
        x = Xref[kstart - 1 + k]
        q[:,k] .= -Q_*x
        c += 0.5 * dot(x, Q_, x)
    end
    solver.c[] = c
    solver
end