function updatetrajectory!(solver::BilinearADMM, Zref, Q, kstart=1)
    nx = size(Q,1)
    q = reshape(solver.q, nx, :)
    N = size(q,2)
    c = 0.0
    Nref = length(Zref)
    for k = 1:N
        x = RD.state(Zref[min(kstart - 1 + k, Nref)])
        q[:,k] .= -Q*x
        c += 0.5 * dot(x, Q, x)
    end
    solver.c[] = c
    solver
end

function shiftfill!(solver::BilinearADMM, n, m)
    x = reshape(solver.x, n, :)
    z = reshape(solver.z, m, :)
    N = size(x,2)
    for k = 1:N-1
        x[:,k] .= x[:,k+1]
        if k < N-1
            z[:,k] .= z[:,k+1]
        end
    end
    solver
end

function setinitialstate!(solver, x0)
    for i = 1:length(x0)
        solver.d[i] = x0[i]
    end
    solver
end