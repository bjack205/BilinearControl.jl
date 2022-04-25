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

@variables r[1:3] R[1:3,1:3] v[1:3] ω[1:3] F[1:3]
r_ = SA[r...]
R_ = SMatrix{3,3}(R...) 
v_ = SA[v...]
ω_ = SA[ω...]
F_ = SA[F...]

x0_sym = [r; vec(R); v]
u_sym = [F; ω]

# Constants
c_sym = @variables m

# Compound states
rv0, rv = createcompoundstates(vec(R), v)
s0_sym = rv0 
s_sym = rv

# Create dictionary of substitutions, converting state to extended state
x2s_dict = Dict(value(x)=>value(y) for (x,y) in zip(s0_sym,s_sym))

# Dynamics
rdot = R_*v_
Rdot = R_*skew(ω_)
vdot = F_ / m - ω_ × v_

x0dot_sym = [rdot; vec(Rdot); vdot]

# Compound state derivatives
rvdot = map(rv) do expr
    i,j,k = getindices(expr)
    Symbolics.expand(Rdot[i,j] * v[k]) + Symbolics.expand(R[i,j] * vdot[k])
end
s0dot_sym = rvdot

# Expanded state vector
x_sym = [x0_sym; s_sym]
xdot_sym = [x0dot_sym; s0dot_sym]

# Replace compound states with symbolic variables
constants = Set(value.(c_sym))
controls = Set(value.(u_sym))
iscoeff(x) = (x isa Number) || (x in constants)
isconstorcontrol(x) = iscoeff(x) || (x in controls)

xdot_sym = map(xdot_sym) do expr
    filtersubstitute(isconstorcontrol, Symbolics.expand(expr), x2s_dict)
end

A,B,C,D = build_symbolic_matrices(xdot_sym, x_sym, u_sym, c_sym)
expr = build_bilinear_dynamics_functions(
    "se3_force", xdot_sym, x_sym, u_sym, c_sym, s0_sym, 
    filename=joinpath(@__DIR__, "se3_force_autogen.jl")
)
