import Pkg; Pkg.activate(@__DIR__)

using Symbolics 
using Symbolics: value
using SymbolicUtils
using StaticArrays
using Rotations
using SparseArrays
using StaticArrays
using LinearAlgebra

include("symbolics_utils.jl")
include(joinpath(@__DIR__, "../test/models/rotation_utils.jl"))

@variables r[1:3] R[1:3,1:3] v[1:3] ω[1:3] F[1:3] τ[1:3]
r_ = SA[r...]
R_ = SMatrix{3,3}(R...) 
v_ = SA[v...]
ω_ = SA[ω...]
F_ = SA[F...]
τ_ = SA[τ...]

x0_sym = [r; vec(R); v; ω]
u_sym = [F; τ]

# Constants
c_sym = @variables m::Float64 J1::Float64 J2::Float64 J3::Float64
J = Diagonal([J1, J2, J3])
Jinv = Diagonal(inv.(diag(J))) 

# Compound states
Rv0, Rv = createcompoundstates(vec(R), v)
Rw0, Rw = createcompoundstates(vec(R), ω)
vw0, vw = createcompoundstates(v, ω)
ww0, ww = createcompoundstates(ω, ω)

s0_sym = [Rv0; Rw0; vw0; ww0]
s_sym = [Rv; Rw; vw; ww]

# Dynamics
rdot = R_*v_
Rdot = R_*skew(ω_)
vdot = F_ / m - ω_ × v_
ωdot = Jinv * (τ_ - ω_ × (J * ω_))

x0dot_sym = [rdot; vec(Rdot); vdot; ωdot]

# Compound state derivatives
Rvdot = map(Rv) do expr
    i,j,k = getindices(expr)
    # expand(Rdot[i,j] * v[k]) + expand(R[i,j] * vdot[k])
    expand(R[i,j] * F[k]/m)  # dropping R[i,j]*v[k]*ω[k] terms
end
Rvdot

Rwdot = map(Rw) do expr
    i,j,k = getindices(expr)
    # expand(Rdot[i,j] * v[k]) + expand(R[i,j] * ωdot[k])
    expand(R[i,j] * τ[k]/J[k,k])  # dropping R[i,j]*ω[k]*ω[l] terms
end
Rwdot

vwdot = map(vw) do expr 
    i,j = getindices(expr)
    # expand(v[i]*ωdot[j]) + expand(vdot[i]*ω[j])
    v[i]*τ[j]/J[j,j] + F[i]/m * ω[j]  # dropping v[i]*ω[j]*ω[k] terms
end
vwdot

wwdot = map(ww) do expr 
    i,j = getindices(expr)
    # expand(ωdot[i]*ω[j]) + expand(ω[i]*ωdot[j])
    expand(ω[i]*τ[j]/J[j,j] + ω[j]*τ[i]/J[i,i])  # dropping ω[i]*ω[j]*ω[k] terms
end
wwdot

s0dot_sym = [Rvdot; Rwdot; vwdot; wwdot]

xdot_sym, x_sym = build_compound_dynamics(x0dot_sym, x0_sym, s0dot_sym, s0_sym, s_sym)

build_bilinear_dynamics_functions(
    "rigid_body", xdot_sym, x_sym, u_sym, c_sym, s0_sym,
    filename=joinpath(@__DIR__, "rigid_body_autogen.jl")
)