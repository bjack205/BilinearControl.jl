import Pkg; Pkg.activate(@__DIR__)
using Rotations
using StaticArrays
using LinearAlgebra
using ForwardDiff
using Rotations: skewmat

function rotmatkinematics(T,w)
    ŵ = SA[
        0 -w[6] w[5] w[1]
        w[6] 0 -w[4] w[2]
        -w[5] w[4] 0 w[3]
        0 0 0 0
    ]
    T*ŵ
end

function quatkinematics(x,w)
    v = SA[w[1],w[2],w[3]]
    ω = SA[w[4],w[5],w[6]]
    r = SA[x[1],x[2],x[3]]
    q = UnitQuaternion(x[4], x[5], x[6], x[7], false)
    rdot = q*v
    qdot = Rotations.kinematics(q, ω)
    return [rdot; qdot]
end

function implicitmidpoint(f,x,u, h; tol=1e-10)
    xn = copy(x)
    r(xn) = h*f((x+xn)/2, u) + x - xn
    ∇r(xn) = ForwardDiff.jacobian(r, xn)
    for i = 1:10
        res = r(xn)
        if norm(res) < tol
            return xn
        end

        xn += -(∇r(xn) \ res)
    end
    error("Newton solve didn't converge.")
end

function simulatequat(x0, w, times)
    X = [copy(x0) for t in times]
    for i = 1:length(times)-1 
        h = times[i+1] - times[i]
        X[i+1] = implicitmidpoint(quatkinematics, X[i], w, h)
    end
    return X
end

function simulaterotmat(T0, w, times)
    T = [copy(T0) for t in times]
    ŵ = SA[
        0 -w[6] w[5] w[1]
        w[6] 0 -w[4] w[2]
        -w[5] w[4] 0 w[3]
        0 0 0 0
    ]
    T = map(times) do t
        T0*exp(t*ŵ)
    end
    return T
end

function Ttox(T)
    q = UnitQuaternion(view(T, 1:3, 1:3))
    r = SA[T[1,4], T[2,4], T[3,4]]
    return [r; Rotations.params(q)]
end

function xtoT(x)
    r = SA[x[1],x[2],x[3]]
    q = UnitQuaternion(x[4], x[5], x[6], x[7], false)
    A = SMatrix(q)
    T = [[A r]; SA[0 0 0 1]]
end

x0 = [
    @SVector randn(3);
    normalize(@SVector randn(4))
]
T0 = xtoT(x0)
Ttox(T0) ≈ x0

w = SA[10,0.3,0.1, 0.1,0.2,0.5]
times = range(0,2,step=1e-4)
X = simulatequat(x0, w, times)
T = simulaterotmat(T0, w, times)
T2 = map(xtoT, X)
X2 = map(Ttox, T)
norm(T .- T2)


## SE(3) Dynamics
using Symbolics
@variables q[1:4] x[1:3] v[1:3] ω[1:3]

quat = UnitQuaternion(q, false)
A = SMatrix(quat) 
vs = SA[v[1], v[2], v[3]]
A*vs
A*v
rmult(q) * SA[0, ω[1], ω[2], ω[3]]