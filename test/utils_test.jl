
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
getnzindsA(A, B) == [1,3,4]
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
B2 ≈ B