using StaticArrays
import RobotDynamics as RD

struct BilinearDubins <: RD.ContinuousDynamics end
RD.state_dim(::BilinearDubins) = 5
RD.control_dim(::BilinearDubins) = 2
RD.default_diffmethod(::BilinearDubins) = RD.UserDefined()

function RD.dynamics(::BilinearDubins, x, u)
    cθ = x[4]
    sθ = x[5]
    v, ω = u
    SA[
        v * cθ 
        v * sθ
        ω
        -sθ
        +cθ
    ] 
end

function expand(::BilinearDubins, x)
    sθ,cθ = sincos(x[3])
    return SA[x[1], x[2], x[3], cθ, sθ]
end

function RD.jacobian!(::BilinearDubins, J, y, x, u)
    cθ = x[4]
    sθ = x[5]
    v, ω = u
    J .= 0
    J[1,4] = v
    J[1,6] = cθ 
    J[2,5] = v 
    J[2,6] = sθ 
    J[3,7] = 1
    J[4,5] = -1
    J[5,4] = +1
    return J
end

function getA(::BilinearDubins)
    return SA[
        0 0 0 0 0
        0 0 0 0 0
        0 0 0 0 0
        0 0 0 0 -1 
        0 0 0 1 0
    ]
end

function getB(::BilinearDubins)
    return SA[
        0 0; 0 0; 0 1; 0 0; 0 0
    ]
end

function getC(::BilinearDubins)
    C1 = SA[
        0 0 0 1 0
        0 0 0 0 1
        0 0 0 0 0
        0 0 0 0 0
        0 0 0 0 0
    ]
    C2 = @SMatrix zeros(5,5)
    return [C1,C2]
end

function getD(::BilinearDubins)
    return @SVector zeros(5)
end