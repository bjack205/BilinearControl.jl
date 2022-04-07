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
include("se3_dynamics.jl")
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
Rw0, Rw = createcompoundstates(vec(R), ω)
s0_sym = Rw0
s_sym = Rw

# Dynamics
rdot = v_
Rdot = R_*skew(ω_)
vdot = R_ * F_ / m   # velocity in world frame
ωdot = Jinv * τ_     # use integrator dynamics
x0dot_sym = [rdot; vec(Rdot); vdot; ωdot]

# Compound state dynamics
Rwdot = map(Rw) do expr
    i,j,k = getindices(expr)
    # expand(Rdot[i,j] * ω[k]) + expand(R[i,j] * ωdot[k])
    expand(R[i,j] * τ[k]/J[k,k])  # dropping R[i,j]*ω[k]*ω[l] terms
end

s0dot_sym = Rwdot

xdot_sym, x_sym = build_compound_dynamics(x0dot_sym, x0_sym, s0dot_sym, s0_sym, s_sym)

build_bilinear_dynamics_functions(
    "se3_integrator", xdot_sym, x_sym, u_sym, c_sym, s0_sym,
    filename=joinpath(@__DIR__, "se3_integrator_autogen.jl")
)