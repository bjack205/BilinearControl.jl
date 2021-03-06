using StaticArrays

struct BilinearDubins <: RD.ContinuousDynamics end
RD.state_dim(::BilinearDubins) = 4
RD.control_dim(::BilinearDubins) = 2
RD.default_diffmethod(::BilinearDubins) = RD.UserDefined()
RD.default_signature(::BilinearDubins) = RD.InPlace()

function expandstate(model::BilinearDubins, x)
    return SA[x[1], x[2], cos(x[3]), sin(x[3])]
end

function RD.dynamics(::BilinearDubins, x, u)
    α = x[3]
    β = x[4]
    v, ω = u
    SA[
        v * α 
        v * β 
        -β * ω
        +α * ω
    ] 
end

function RD.dynamics!(model::BilinearDubins, xdot, x, u)
    xdot .= RD.dynamics(model, x, u)
end

function RD.jacobian!(::BilinearDubins, J, y, x, u)
    α = x[3]
    β = x[4]
    v, ω = u
    J .= 0
    J[1,3] = v
    J[1,5] = α
    J[2,4] = v 
    J[2,5] = β
    J[3,4] = -ω
    J[3,6] = -β 
    J[4,3] = ω
    J[4,6] = α 
    return J
end

BilinearControl.getA(::BilinearDubins) = @SMatrix zeros(4,4)

BilinearControl.getB(::BilinearDubins) = @SMatrix zeros(4,2)

function BilinearControl.getC(::BilinearDubins)
    C1 = SA_F64[
        0 0 1 0
        0 0 0 1
        0 0 0 0
        0 0 0 0
    ]
    C2 = SA_F64[
        0 0 0 0
        0 0 0 0
        0 0 0 -1 
        0 0 1 0
    ]

    return [C1,C2]
end

BilinearControl.getD(::BilinearDubins) = @SVector zeros(4)