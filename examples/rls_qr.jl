using SparseArrays
using LinearAlgebra
using BilinearControl.EDMD: rls_qr
function solve_neq(b,A; reg=zero(eltype(b)))
    (A'A + I*reg)\(A'b)
end

function solve_qr(b,A; reg=zero(eltype(b)))
    n = size(A,2)
    qr([A; I*sqrt(reg)]) \ [b; zeros(n)]
end

n = 100_000
m = 3_000
A = sprandn(n,m,0.1)
b = randn(n)

reg = 1e-4
@time x_rls = rls_qr(b, A; Q=reg, verbose=true)
@time x_neq = solve_neq(b,A;reg)
@time x_qr = solve_qr(b,A;reg) 