
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

struct QuadrotorRateLimited{Order} <: RD.DiscreteDynamics
    mass::Float64
    gravity::Float64
    function QuadrotorRateLimited{O}(; mass=2.0, gravity=9.81) where O
        @assert O in (1,2) "Invalid parameter for quadrotor. Must be 1 or 2."
        new{O}(mass, gravity)
    end
end
QuadrotorRateLimited(;kwargs...) = QuadrotorRateLimited{1}(;kwargs...)

RD.state_dim(::QuadrotorRateLimited{1}) = 18
RD.state_dim(::QuadrotorRateLimited{2}) = 21 
RD.control_dim(::QuadrotorRateLimited) = 4

BilinearControl.Problems.translation(::QuadrotorRateLimited, x) = SVector{3}(x[1], x[2], x[3])
BilinearControl.Problems.orientation(::QuadrotorRateLimited, x) = RotMatrix{3}(x[4:12]...)

function Base.rand(::QuadrotorRateLimited{O}) where O
    x = [
            @SVector randn(3);
            vec(qrot(normalize(@SVector randn(4))));
            @SVector randn(3 * (O+1))
    ]
    u = push((@SVector randn(3)), rand())
    x,u
end


# function RD.dynamics(model::QuadrotorRateLimited, x, u)
function RD.dynamics_error(model::QuadrotorRateLimited{O}, z2::RD.KnotPoint, z1::RD.KnotPoint) where O
    x1 = RD.state(z1)
    x2 = RD.state(z2)
    u1 = RD.control(z1)
    u2 = RD.control(z2)

    xm = (x1 + x2) / 2
    h = RD.timestep(z1)
    xdot0 = let x = xm, u = u1
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
        [rdot; vec(Rdot); vdot]
    end
    dx0 = x1[1:15] - x2[1:15]
    ω1 = SA[u1[1], u1[2], u1[3]]
    α2 = SA[x2[16], x2[17], x2[18]]
    if O == 1
        ω2 = SA[u2[1], u2[2], u2[3]]
        return [h*xdot0 + dx0; h*α2 + ω1 - ω2]
    elseif O == 2
        # α1 = SA[x1[16], x1[17], x1[18]]
        ωp1 = SA[x1[19], x1[20], x1[21]]
        ωp2 = SA[x2[19], x2[20], x2[21]]
        return [h*xdot0 + dx0; h*α2 - (ωp2 - ωp1); ωp2 - ω1]
    end
end

function BilinearControl.getA(::QuadrotorRateLimited{1}, h)
    n = 18 
    A = zeros(n, 2n)
    for i = 1:3
        A[i,12+i] = h/2
        A[i,n+12+i] = h/2
        A[15+i,n+15+i] = h 
    end
    for i = 1:15
        A[i,i] = 1.0
        A[i,n+i] = -1.0
    end
    A
end

function BilinearControl.getA(::QuadrotorRateLimited{2}, h)
    n =  21 
    A = zeros(n, 2n)
    for i = 1:3
        A[i,12+i] = h/2
        A[i,n+12+i] = h/2
        A[15+i,15+i] = h     # α1
        A[15+i,18+i] = 1.0     # ωp1
        A[15+i,18+i+n] = -1.0  # ωp2
        A[18+i,18+i+n] = 1.0   # ωp2
    end
    for i = 1:15
        A[i,i] = 1.0
        A[i,n+i] = -1.0
    end
    A
end

function BilinearControl.getB(::QuadrotorRateLimited{1}, h)
    n,m = 18,4
    B = zeros(n,2m)
    for i = 1:3
        B[15+i,i] = 1.0
        B[15+i,m+i] = -1.0
    end
    B
end

function BilinearControl.getB(::QuadrotorRateLimited{2}, h)
    n,m = 21,4
    B = zeros(n,2m)
    for i = 1:3
        # B[15+i,i] = -1.0
        B[18+i,i] = -1.0
    end
    B
end

function BilinearControl.getC(model::QuadrotorRateLimited, h)
    n,m = RD.dims(model)
    C = [zeros(n,2n) for i = 1:2m]
    mass = model.mass
    for i = 1:3
        for j in (0,1)
            C[1][6+i,9+i+j*n] = +h*0.5
            C[1][9+i,6+i+j*n] = -h*0.5
            C[2][3+i,9+i+j*n] = -h*0.5
            C[2][9+i,3+i+j*n] = +h*0.5
            C[3][3+i,6+i+j*n] = +h*0.5
            C[3][6+i,3+i+j*n] = -h*0.5
            C[4][12+i,9+i+j*n] = h/2mass
        end
    end
    C
end

function BilinearControl.getD(model::QuadrotorRateLimited, h)
    g = model.gravity 
    d = zeros(RD.state_dim(model))
    d[15] = -g*h
    d
end