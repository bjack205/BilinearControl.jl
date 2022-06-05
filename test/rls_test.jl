# test cholesky rank updates

A = Matrix{Float64}([1 0 0; 1 1 1; 1 2 4; 1 3 9; 1 4 16])
b = Vector{Float64}([0, 1, 4, 6, 9])

m, n = size(A)

x_true = A \ b
x_recursive = BilinearControl.EDMD.rls_chol(b, A)

@test size(x_recursive)[1] == n
@test x_true ≈ x_recursive rtol=1e-10

# test qr factorization

A = sparse(bitrand(100, 3)) .* randn(100, 3)

m, n = size(A)

QR_true = qr(A)

R, prow, pcol = BilinearControl.EDMD.qrr(A)

@test size(prow)[1] == m
@test size(pcol)[1] == n

@test prow*prow' == I
@test pcol*pcol' == I

@test R == QR_true.R
@test prow*A == A[QR_true.prow, :]
@test A*pcol == A[:, QR_true.pcol]
@test prow*A*pcol == A[QR_true.prow, QR_true.pcol]

# test qr square-root rls

A = randn(1000, 5)
b = randn(1000)

m, n = size(A)

x_true = A \ b

x_recursive = BilinearControl.EDMD.rls_qr(b, A)

@test size(x_recursive)[1] == n
@test x_true ≈ x_recursive rtol=1e-10

x_recursive = BilinearControl.EDMD.rls_qr(b, A; batchsize=12)

@test size(x_recursive)[1] == n
@test x_true ≈ x_recursive rtol=1e-10

x_recursive = BilinearControl.EDMD.rls_qr(b, A; batchsize=120)

@test size(x_recursive)[1] == n
@test x_true ≈ x_recursive rtol=1e-10

A = sparse(bitrand(1000, 5)) .* randn(1000, 5)
b = randn(1000)

m, n = size(A)

x_true = A \ b

x_recursive = BilinearControl.EDMD.rls_qr(b, A)

@test size(x_recursive)[1] == n
@test x_true ≈ x_recursive rtol=1e-10