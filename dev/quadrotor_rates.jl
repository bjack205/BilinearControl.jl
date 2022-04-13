using BilinearControl.Problems: qrot, skew

Base.@kwdef struct QuadrotorRateLimited <: RD.DiscreteDynamics
    mass::Float64 = 2.0
    gravity::Float64 = 9.81
end

RD.state_dim(::QuadrotorRateLimited) = 18
RD.control_dim(::QuadrotorRateLimited) = 4

BilinearControl.Problems.translation(::QuadrotorRateLimited, x) = SVector{3}(x[1], x[2], x[3])
BilinearControl.Problems.orientation(::QuadrotorRateLimited, x) = RotMatrix{3}(x[4:12]...)

function Base.rand(::QuadrotorRateLimited)
    x = [
            @SVector randn(3);
            vec(qrot(normalize(@SVector randn(4))));
            @SVector randn(6)
    ]
    u = push((@SVector randn(3)), rand())
    x,u
end

# function RD.dynamics(model::QuadrotorRateLimited, x, u)
function RD.dynamics_error(model::QuadrotorRateLimited, z2::RD.KnotPoint, z1::RD.KnotPoint)
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
    α2 = SA[x2[16], x2[17], x2[18]]
    ω1 = SA[u1[1], u1[2], u1[3]]
    ω2 = SA[u2[1], u2[2], u2[3]]
    [h*xdot0 + dx0; h*α2 + ω1 - ω2]
end

function BilinearControl.getA(::QuadrotorRateLimited, h)
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

function BilinearControl.getB(::QuadrotorRateLimited, h)
    n,m = 18,4
    B = zeros(n,2m)
    for i = 1:3
        B[15+i,i] = 1.0
        B[15+i,m+i] = -1.0
    end
    B
end

function BilinearControl.getC(model::QuadrotorRateLimited, h)
    n,m = 18,4
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
    d = zeros(18)
    d[15] = -g*h
    d
end

## Test dynamics
model = QuadrotorRateLimited()
n,m = RD.dims(model)
r1,r2 = randn(3), randn(3) 
R1,R2 = qrot(normalize(randn(4))), qrot(normalize(randn(4)))
v1,v2 = randn(3), randn(3) 
α1,α2 = randn(3), randn(3) 
ω1,ω2 = randn(3), randn(3) 
F1,F2 = rand(), rand()

x1 = [r1; vec(R1); v1; α1]
x2 = [r2; vec(R2); v2; α2]
u1 = [ω1; F1]
u2 = [ω2; F2]

h = 0.1
z1 = RD.KnotPoint{n,m}(n,m,[x1;u1],0.0,h)
z2 = RD.KnotPoint{n,m}(n,m,[x2;u2],h,h)
err = RD.dynamics_error(model, z2, z1)

err[1:3] ≈ h * (v1 + v2) / 2 + r1 - r2
err[4:12] ≈ vec(h * (R1 + R2) /2 * skew(ω1) + R1 - R2)
err[13:15] ≈ h*( (R1 + R2) /2 * [0,0,F1]) / model.mass - 
    h*[0,0,model.gravity] + v1 - v2
err[16:18] ≈ h*α2 - (ω2 - ω1)


using BilinearControl: getA, getB, getC, getD
# Test dynamics match bilinear dynamics
A,B,C,D = getA(model,h), getB(model,h), getC(model,h), getD(model,h)
x12 = [x1;x2]
u12 = [u1;u2]
err2 = A*x12 + B*u12 + sum(u12[i]*C[i]*x12 for i = 1:length(u12)) + D
err ≈ err2