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



## Controls solve
using BilinearControl: getnzind

function Bbar!(Bbar, B, C, x, cache)
    m,n = size(B)
    Bbar .= B
    for j = 1:n
        Bj = view(Bbar,:,j)
        Cj = C[j]

        rv = rowvals(Cj)
        nzv = nonzeros(Cj)
        fill!(Bj, zero(eltype(Bj)))
        for col in axes(Cj,2)
            for i in nzrange(Cj, col)
                Bj[rv[i]] += nzv[i] * x[col]
            end
        end
    end
    Bbar
end

function Bbar_coo!(Bbar, B, Ccoo, x, cache)
    m,n = size(B)
    # Bbar .= B
    Bbar .= 0
    for j = 1:n
        Bj = view(Bbar,:,j)

        for (r,c,v) in Ccoo[j]
            # Bj[r] += v * x[c]
            Bbar.nzval[r] += v * x[c]
        end
    end
    Bbar

end

model = prob.model[1].continuous_dynamics
B = sparse(getB(model))
C = sparse.(getC(model))
x1 = sparse(rand(model)[1])
Bbar0 = B + hcat([c*x1 for c in C]...)
Bbar = similar(Bbar0)

cache = BilinearControl.getnzindsA(Bbar, B)
# cache = map(1:size(B,2)) do j
#     map(nzrange(C[j],))
# end
Ccoo = map(enumerate(C)) do (j,c)
    map(zip(findnz(c)...)) do (r,c,v)
        r2 = getnzind(Bbar, r, j)
        (r2,c,v)
    end
end
Bbar = similar(Bbar0)
Bbar!(Bbar, B, C, x1, cache) ≈ Bbar0
Bbar_coo!(Bbar, B, Ccoo, x1, cache) ≈ Bbar0

view()