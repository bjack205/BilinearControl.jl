
using BilinearControl: getnzind, getnzindsA, getnzindsB
A = sparse([
    1 0 0 0
    1 1 0 0
    0 0 0 1
])
@test getnzind(A, 1,1) == 1
@test getnzind(A, 2,1) == 2
@test getnzind(A, 3,1) == 0
@test getnzind(A, 2,2) == 3
@test getnzind(A, 2,2) == 3
@test getnzind(A, 2,3) == 0
@test getnzind(A, 3,4) == 4

B = sparse([
    1 0 0 0
    0 1 0 0
    0 0 0 1
])
@test getnzindsA(A, B) == [1,3,4]
C = sparse([
    0 0 0 0
    0 1 0 0
    0 0 0 0
])
@test getnzindsA(A, C) == [3]
@test_throws DimensionMismatch getnzindsA(A, spzeros(4,4))
@test_throws ArgumentError getnzindsA(A, sparse(I,3,4))

C1 = sparse([
    1 0 0 0
    1 1 0 0
    1 0 0 0
    0 0 0 0
])
C2 = sparse([
    0 0 0 0
    0 0 0 0
    0 0 1 1
    1 0 0 1
])
C = [C1,C2]
x = sparse(ones(4))
B = [C1*x C2*x]
@test getnzindsB(B, C1, 1) == [1,2,3,2]
@test getnzindsB(B, C2, 2) == [5,4,4,5]
nzindsB = map(1:2) do i
    getnzindsB(B, C[i], i)
end
B2 = similar(B)
B2 .= 0
B2 ≈ B
for i = 1:2
    for c in axes(C[i],2)
        for j in nzrange(C[i], c)
            nzindB = nzindsB[i][j]
            nonzeros(B2)[nzindB] += nonzeros(C[i])[j] * x[c]
        end
    end
end
@test B2 ≈ B

# Methods for calculating AtA
using SparseArrays
a = [1,3,5,6]
b = [1,2,3,6,9]
@test BilinearControl.findmatches(a,b) == [(1,1), (2,3), (4,4)]

n = 10
N = 11 
A1 = sprandn(n,n,0.1)
A2 = sprandn(n,n,0.1)
A = kron(spdiagm(N-1,N,0=>ones(N-1)), A2) + kron(spdiagm(N-1,N,1=>ones(N-1)), A2)

cache,C = BilinearControl.AtAcache(A)
@test C ≉ A'A
Q = Diagonal(rand(N*n))
ρ = 2.0
BilinearControl.QAtA!(C, Q, A, ρ, cache)
@test C ≈ (Q + ρ*A'A)
BilinearControl.QAtA!(C, Q, A, ρ, cache)
@test C ≈ (Q + ρ*A'A)

testQAtAallocs(C,Q,A,ρ,cache) = @allocated BilinearControl.QAtA!(C, Q, A, ρ, cache)
testQAtAallocs(C,Q,A,ρ,cache)
testAtAallocs(C,A,cache) = @allocated BilinearControl.AtA!(C, A, cache)
@test testAtAallocs(C,A,cache) == 0

