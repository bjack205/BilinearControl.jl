
Base.@kwdef struct QuadrotorSE23 <: RD.ContinuousDynamics
    mass::Float64 = 2.0
    gravity::Float64 = 9.81
end

RD.state_dim(::QuadrotorSE23) = 15
RD.control_dim(::QuadrotorSE23) = 4

BilinearControl.Problems.translation(::QuadrotorSE23, x) = SVector{3}(x[1], x[2], x[3])
BilinearControl.Problems.orientation(::QuadrotorSE23, x) = RotMatrix{3}(x[4:12]...)


function Base.rand(::QuadrotorSE23)
    x = [
            @SVector randn(3);
            vec(qrot(normalize(@SVector randn(4))));
            @SVector randn(3)
    ]
    u = push((@SVector randn(3)), rand())
    x,u
end

function RD.dynamics(model::QuadrotorSE23, x, u)
    mass = model.mass
    g = model.gravity 
    R = SA[
        x[4] x[7] x[10]
        x[5] x[8] x[11]
        x[6] x[9] x[12]
    ]
    v = SA[x[13], x[14], x[15]]
    ω = SA[u[1], u[2], u[3]]
    Fbody = [0, 0, u[4]]

    rdot = v;
    Rdot = R * Rotations.skew(ω)
    vdot = R*Fbody ./ mass - [0,0,g]
    return [rdot; vec(Rdot); vdot]
end

function RD.jacobian!(model::QuadrotorSE23, J, xdot, x, u)
    R = SA[
        x[4] x[7] x[10]
        x[5] x[8] x[11]
        x[6] x[9] x[12]
    ]
    for i = 1:3
        J[i,12+i] = 1.0

        J[6+i,9+i] = +1.0 * u[1]
        J[9+i,6+i] = -1.0 * u[1]
        J[3+i,9+i] = -1.0 * u[2]
        J[9+i,3+i] = +1.0 * u[2]
        J[3+i,6+i] = +1.0 * u[3]
        J[6+i,3+i] = -1.0 * u[3]
        J[12+i,9+i] = 1/model.mass * u[4]

        J[3+i,17] = -R[i,3]
        J[3+i,18] = +R[i,2]
        J[6+i,16] = +R[i,3]
        J[6+i,18] = -R[i,1]
        J[9+i,16] = -R[i,2]
        J[9+i,17] = +R[i,1]
        
        J[12+i,19] = R[i,3] / model.mass 
    end
end

function BilinearControl.getA(::QuadrotorSE23)
    A = zeros(15,15)
    for i = 1:3
        A[i,12+i] = 1.0
    end
    A
end

BilinearControl.getB(::QuadrotorSE23) = zeros(15,4)

function BilinearControl.getC(model::QuadrotorSE23)
    m = model.mass
    C = [zeros(15,15) for i = 1:4]
    for i = 1:3
        C[1][6+i,9+i] = +1.0
        C[1][9+i,6+i] = -1.0
        C[2][3+i,9+i] = -1.0
        C[2][9+i,3+i] = +1.0
        C[3][3+i,6+i] = +1.0
        C[3][6+i,3+i] = -1.0
        C[4][12+i,9+i] = 1/m
    end
    C
end

function BilinearControl.getD(model::QuadrotorSE23)
    g = model.gravity 
    d = zeros(15)
    d[end] = -g
    d
end