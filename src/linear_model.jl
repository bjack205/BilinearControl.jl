struct DiscreteLinearModel <: RD.DiscreteDynamics
    A::SparseMatrixCSC{Float64,Int}
    B::SparseMatrixCSC{Float64,Int}
    C::SparseMatrixCSC{Float64,Int}
    d::SparseVector{Float64,Int}
    CA::SparseMatrixCSC{Float64,Int}
    CB::SparseMatrixCSC{Float64,Int}
    Cd::SparseVector{Float64,Int}
    function DiscreteLinearModel(A,B,C,d)
        if C isa SMatrix
            F = lu(C)
        else
            F = factorize(C)
        end
        CA = -(F\A)
        CB = -(F\B)
        Cd = -(F\d)
        new(A,B,C,d, CA, CB, Cd)
    end
end

function RD.dynamics_error!(model::DiscreteLinearModel, z2::RD.KnotPoint, z1::RD.KnotPoint)
    x1,u1 = RD.state(z1), RD.control(z1)
    x2,u2 = RD.state(z2), RD.control(z2)
    xn .= model.d
    mul!(xn, model.A, x1, 1.0, 1.0)
    mul!(xn, model.B, u1, 1.0, 1.0)
    mul!(xn, model.C, x2, 1.0, 1.0)
    nothing
end

function RD.discrete_dynamics!(model::DiscreteLinearModel, xn, x, u, t, h)
    xn .= model.Cd
    mul!(xn, model.CA, x)
    mul!(xn, model.CB, u)
    nothing
end

getA(model::DiscreteLinearModel) = model.A
getB(model::DiscreteLinearModel) = model.B
getC(model::DiscreteLinearModel) = model.C
getD(model::DiscreteLinearModel) = model.d


"""
Use Implicit Midpoint to create a discrete linear model from a continuous 
(bi)linear one.
"""
function DiscreteLinearModel(model::RD.ContinuousDynamics, x2, x1, u1, h)
    n,m = length(x2), length(u1)
    A = getA(model)
    B = getB(model)
    C = getC(model)
    D = getD(model)
    xm = (x1+x2)/2

    isbilinear = norm(C) > 0
    if isbilinear
        Cx = h/2 * sum(u1[i] * C[i] for i = 1:m)
        Cu = h * hcat([C[i] * xm for i = 1:m]...)
    else
        Cx = zero(A)
        Cu = zero(B)
    end
    Ad = h/2 * A + I + Cx
    Bd = h * B + Cu
    Cd = h/2 * A - I + Cx
    d = h*(A*xm + B*u1 + sum(u1[i] * C[i] * xm for i = 1:m) + D) + x1 - x2
    DiscreteLinearModel(Ad, Bd, Cd, d)
end