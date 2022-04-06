
using BilinearControl: getnzind, getnzinds
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
getnzinds(A, B) == [1,3,4]
C = sparse([
    0 0 0 0
    0 1 0 0
    0 0 0 0
])
@test getnzinds(A, C) == [3]
@test_throws DimensionMismatch getnzinds(A, spzeros(4,4))
@test_throws ArgumentError getnzinds(A, sparse(I,3,4))