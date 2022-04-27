using BilinearControl
using BilinearControl.Problems
import TrajectoryOptimization as TO
import RobotDynamics as RD
using LinearAlgebra
using OSQP
using Test

prob = Problems.Cartpole()
Z = prob.Z
constraints = prob.constraints
con_inds = [2]
jacvals = map(con_inds) do i
    con = constraints[i]
    inds = TO.constraintindices(constraints, i)
    sig = TO.functionsignature(constraints, i)
    diffmethod = TO.diffmethod(constraints, i)
    J = TO.gen_jacobian(con)
    c = zeros(RD.output_dim(con))
    vals = map(inds) do k
        TO.evaluate_constraint!(sig, con, c, Z[k]) 
        c
    end
    jacs = map(inds) do k
        TO.constraint_jacobian!(sig, diffmethod, con, J, c, Z[k]) 
        J
    end
    vals, jacs
end

vals = getindex.(jacvals, 1)
jacs = getindex.(jacvals, 2)

N = length(Z) 
h = map(1:N) do k
    con_inds_k = findall(con_inds) do i
        k in TO.constraintindices(constraints, i)
    end
    con_inds_k = con_inds[con_inds_k]
    inds = map(con_inds_k) do i
        inds = TO.constraintindices(constraints, i)
        searchsortedfirst(inds, k)
    end

    if !isempty(con_inds_k)
        return reduce(vcat, vals[i][j] for (i,j) in zip(eachindex(con_inds_k),inds))
    else
        return zeros(0)
    end
end
qp = BilinearControl.TOQP(prob)

n = state_dim(prob, 1)
m = control_dim(prob, 1)
N = BilinearControl.nhorizon(qp)
@test BilinearControl.num_equality(qp) == n
@test BilinearControl.num_inequality(qp) == 2m*(N-1)
@test BilinearControl.num_state_equality(qp) == n
@test BilinearControl.num_control_equality(qp) == 0
@test BilinearControl.num_state_inequality(qp) == 0
@test BilinearControl.num_control_inequality(qp) == 2m*(N-1)
@test all(x->x == -I(n), qp.C)
@test all(x->norm(x) == 0.0, qp.D)
@test all(x->length(x) == 0, qp.hx[1:end-1])
@test all(x->size(x,2) == n, qp.Hx)
@test all(x->size(x,2) == n, qp.Gx)
@test all(x->size(x,2) == m, qp.Hu)
@test all(x->size(x,2) == m, qp.Gu)
@test qp.Hx[end] ≈ I(n)
@test all(x->length(x) == 0, qp.hu)
@test all(x->length(x) == 0, qp.gx)
@test all(x->length(x) == 2, qp.gu)
@test all(x->x ≈ [I(m); -I(m)], qp.Gu)


## Solve with OSQP
qp = BilinearControl.TOQP(prob)
qp.Hx[1]
qp.Hu[1]
model = BilinearControl.setup_osqp(qp, eps_abs=1e-6, eps_rel=1e-6)
length(res.y) == BilinearControl.num_duals(qp)

res = OSQP.solve!(model)
Nx = BilinearControl.num_states(qp)

Xsol, Usol = BilinearControl.unpackprimals(qp, res.x)
λ,μ,ν = BilinearControl.unpackduals(qp, res.y)
@test BilinearControl.primal_feasibility(qp, Xsol, Usol) < 1e-6
@test BilinearControl.stationarity(qp, Xsol, Usol, λ, μ, ν) < 1e-3
@test BilinearControl.dual_feasibility(qp, λ, μ, ν) < 1e-6
@test BilinearControl.complementary_slackness(qp, Xsol, Usol, λ, μ, ν) < 1e-6
@test res.info.dua_res ≈ BilinearControl.stationarity(qp, Xsol, Usol, λ, μ, ν)