using BilinearControl: getA, getB, getC, getD

# Test formation of state sub-problem 
prob = Problems.QuadrotorProblem()
solver = BilinearADMM(prob)
A,C = getA(solver), getC(solver)

Ahat = solver.Ahat
x,z = solver.x, solver.z
BilinearControl.updateAhat!(solver, Ahat, z)
@test Ahat ≈ A + sum(z[i] * C[i] for i = 1:length(z))

ρ = BilinearControl.getpenalty(solver)
@test BilinearControl.updatePhat!(solver) ≈ (solver.Q + ρ*Ahat'Ahat)

function testupdateallocs(solver)
    allocs = @allocated BilinearControl.updateAhat!(solver, solver.Ahat, solver.z)
    allocs = @allocated BilinearControl.updatePhat!(solver)
end
@test testupdateallocs(solver) == 0
