using BilinearControl: evec

function sqrtiSWAP()
    d = 1/sqrt(2)
    ComplexF64[
        1 0 0 0
        0 1d -d*im 0
        0 -im * d d 0 
        0 0 0 01
    ]
end

function real2complex(x::AbstractVector{T}) where T <: Real
    n = length(x)
    @assert iseven(n)
    m = n ÷ 2
    c = zeros(Complex{T}, m)
    for i = 1:m
        c[i] = Complex(x[i], x[i+m])
    end
    return c
end

function complex2real(c::AbstractVector{<:Complex{T}}) where T <: Real
    m = length(c)
    n = 2m
    x = zeros(T, n)
    for i = 1:m
        x[i] = real(c[i])
        x[i+m] = imag(c[i])
    end
    return x
end

function real2complex(A::AbstractMatrix{T}) where T <: Real
    n = size(A,1)
    @assert size(A,2) == n
    @assert iseven(n)
    m = n ÷ 2
    C = similar(A, Complex{T}, m, m)
    for j = 1:m, i = 1:m
        C[i,j] = Complex(A[i,j], A[i+m,j])
    end
    return C
end

function complex2real(C::AbstractMatrix{<:Complex{T}}) where T <: Real
    m = size(C,1)
    @assert size(C,2) == m
    n = 2m
    A = similar(C, T, n, n)
    for j = 1:m, i = 1:m
        r,c = real(C[i,j]), imag(C[i,j])
        A[i+0, j+0] = +r 
        A[i+m, j+0] = +c 
        A[i+0, j+m] = -c
        A[i+m, j+m] = +r
    end
    return A
end

"""
    paulimat(ax::Symbol)

Generate one of the Pauli Spin matrices. Axis `ax` must be one of `:x`, `:y`, or `:z`.
"""
function paulimat(ax::Symbol)
    if ax == :x
        return [0im 1; 1 0im]
    elseif ax == :y
        return [0 -1im; 1im 0]
    elseif ax == :z
        return [1 0im; 0im -1]
    else
        error("$ax not a recognized axis for Pauli matrices. Should be one of [:x,:y,:z].")
    end
end

function twospinhamiltonian(fq_1 = 1/2pi, fq_2 = 1/2pi)
    ω1 = fq_1 * 2pi
    ω2 = fq_2 * 2pi

    # Drift Hamiltonian
    I2 = I(2) 
    σz = paulimat(:z)
    σz_1 = kron(σz, I2)
    σz_2 = kron(I2, σz)
    Hdrift = σz_1 * ω1 / 2 + σz_2 * ω2 / 2 

    # Drive Hamiltonian
    σx = paulimat(:x)
    σx_1 = kron(σx, I2)
    σx_2 = kron(I2, σx)
    Hdrive = σx_1 * σx_2

    return sparse(Hdrift/im), sparse(Hdrive/1im)
end

function twospinproblem()
    # Dynamics 
    Hdrift0, Hdrive0 = twospinhamiltonian(2/2pi) 
    Hdrift = complex2real(Hdrift0)
    Hdrive = complex2real(Hdrive0)
    n = size(Hdrift, 1)
    m = 1
    model = BilinearDynamics(Hdrift, spzeros(n, m), [Hdrive])
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)

    # Discretization
    tf = 20.0
    dt = 0.2
    N = round(Int,tf/dt) + 1

    # Initial and final state
    U = sqrtiSWAP()
    ψ0 = ComplexF64[0,1,0,0]
    ψf = U*ψ0
    x0 = complex2real(ψ0)
    xf = complex2real(ψf)

    # Objective 
    Q = Diagonal(fill(1.0, n))
    R = Diagonal(fill(1e-0, m))
    obj = LQRObjective(Q,R,Q*(N-1), xf, N)

    # Constraints
    cons = ConstraintList(n, m, N)
    goalcon =  GoalConstraint(xf)
    add_constraint!(cons, goalcon, N)

    # Initial guess
    U0 = fill(0.1, m, N-1)

    # Problem
    Problem(dmodel, obj, x0, tf, xf=xf, constraints=cons, U0=U0)
end

function twospingateproblem()
    nq = 4  # number of qubits
    Hdrift0, Hdrive0 = twospinhamiltonian() 

    # Discretization
    tf = 20.0
    dt = 0.2
    N = round(Int,tf/dt) + 1

    # Dynamics
    Aim = blockdiag([Hdrift0 for i = 1:nq]...)
    Cim = blockdiag([Hdrive0 for i = 1:nq]...)
    A = complex2real(Aim)
    C = complex2real(Cim)
    n = size(A, 1) 
    m = 1
    model = BilinearDynamics(A, spzeros(n, m), [C])
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)
    
    # Initial and final states
    U = sqrtiSWAP()
    x0 = vcat(map(1:nq) do i
        ψi = evec(ComplexF64, i, nq)
        complex2real(ψi)
    end...)
    xf = vcat(map(1:nq) do i
        ψi = evec(ComplexF64, i, nq)
        complex2real(U*ψi)
    end...)

    # Objective 
    Q = Diagonal(fill(1.0, n))
    R = Diagonal(fill(1e-3, m))
    obj = LQRObjective(Q,R,Q*(N-1), xf, N)

    # Constraints
    cons = ConstraintList(n, m, N)
    goalcon =  GoalConstraint(xf)
    add_constraint!(cons, goalcon, N)

    # Initial guess
    U0 = fill(0.1, m, N-1)

    # Problem
    Problem(dmodel, obj, x0, tf, xf=xf, constraints=cons, U0=U0)

end