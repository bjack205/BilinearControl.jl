
"""
    Quadrotor{R}

A standard quadrotor model, with simple aerodynamic forces. The orientation is represent by
a general rotation `R`. The body z-axis point is vertical, so positive controls cause acceleration
in the positive z direction.

# Constructor
    Quadrotor(; kwargs...)
    Quadrotor{R}(; kwargs...)

where `R <: Rotation{3}` and defaults to `UnitQuaternion{Float64}` if omitted. The keyword arguments are
* `mass` - mass of the quadrotor, in kg (default = 0.5)
* `J` - inertia of the quadrotor, in kg⋅m² (default = `Diagonal([0.0023, 0.0023, 0.004])`)
* `gravity` - gravity vector, in kg/m² (default = [0,0,-9.81])
* `motor_dist` - distane between the motors, in m (default = 0.1750)
* `km` - motor torque constant (default = 0.0245)
* `kf` - motor force constant (default = 1.0)
"""

RD.@autodiff struct Quadrotor{R} <: RD.RigidBody{R}
    mass::Float64
    J::SMatrix{3,3,Float64,9}
    Jinv::SMatrix{3,3,Float64,9}
    gravity::SVector{3,Float64}
    motor_dist::Float64
    kf::Float64
    km::Float64
    bodyframe::Bool  # velocity in body frame?
    ned::Bool
    cross_A_x::Float64
    cross_A_y::Float64
    cross_A_z::Float64
    cd::SVector{3, Float64}
end
RD.control_dim(::Quadrotor) = 4

function Quadrotor{R}(;
        mass=0.5,
        J=Diagonal(@SVector [0.0023, 0.0023, 0.004]),
        gravity=SVector(0,0,-9.81),
        motor_dist=0.1750,
        kf=1.0,
        km=0.0245,
        bodyframe=false,
        ned=false,
        cross_A_x=0.25,
        cross_A_y=0.25,
        cross_A_z=0.5,
        cd=@SVector [0.0, 0.0, 0.0]
    ) where R
    @assert issymmetric(J)
    Quadrotor{R}(mass,J,inv(J),gravity,motor_dist,kf,km,bodyframe,ned,
                cross_A_x, cross_A_y, cross_A_z, cd)
end

(::Type{Quadrotor})(;kwargs...) = Quadrotor{MRP{Float64}}(;kwargs...)

@inline RobotDynamics.velocity_frame(model::Quadrotor) = model.bodyframe ? :body : :world

NominalQuadrotor() = Quadrotor{MRP{Float64}}()
SimulatedQuadrotor() = Quadrotor{MRP{Float64}}(; mass=0.75,
                                J=SMatrix{3,3,Float64,9}([0.0026 0.0003 0.0;
                                                        0.0003 0.0026 0.0;
                                                        0.0 0.0 0.005]),
                                motor_dist=0.2,
                                kf=0.95,
                                km=0.026,
                                cross_A_x = 0.3,
                                cross_A_y = 0.3,
                                cross_A_z = 0.65,
                                cd=@SVector [0.4, 0.4, 0.4])

function trim_controls(model::Quadrotor)
    Vector(fill(-model.gravity[3]*model.mass/4.0, size(model)[2]))
end

function RD.forces(model::Quadrotor, x, u)
    q = RD.orientation(model, x)
    kf = model.kf
    g = model.gravity
    m = model.mass
    cd = model.cd
    cross_A_x = model.cross_A_x
    cross_A_y = model.cross_A_y
    cross_A_z = model.cross_A_z

    w1 = u[1]
    w2 = u[2]
    w3 = u[3]
    w4 = u[4]

    F1 = max(0,kf*w1);
    F2 = max(0,kf*w2);
    F3 = max(0,kf*w3);
    F4 = max(0,kf*w4);
    F = @SVector [0., 0., F1+F2+F3+F4] #total rotor force in body frame
    if model.ned
        F = SA[0,0,-F[3]]
        g = -g
    end

    # add aero drag where 1.27 is air density (kg/m^3)
    df = -sign.(x[4:6]).*0.5.*1.27.*(x[4:6].^2).*
        cd.*[cross_A_x, cross_A_y, cross_A_z]

    f = m*g + q*F + df # forces in world frame
    return f
end

function RD.moments(model::Quadrotor, x, u)

    kf, km = model.kf, model.km
    L = model.motor_dist

    w1 = u[1]
    w2 = u[2]
    w3 = u[3]
    w4 = u[4]

    F1 = max(0,kf*w1);
    F2 = max(0,kf*w2);
    F3 = max(0,kf*w3);
    F4 = max(0,kf*w4);

    M1 = km*w1;
    M2 = km*w2;
    M3 = km*w3;
    M4 = km*w4;
    tau = @SVector [L*(F2-F4), L*(F3-F1), (M1-M2+M3-M4)] #total rotor torque in body frame
    if model.ned
        tau = SA[tau[1], -tau[2], -tau[3]]
    end
    return tau
end

function RD.wrenches(model::Quadrotor, x, u)
    F = RD.forces(model, x, u)
    M = RD.moments(model, x, u)
    return [F; M]

    q = RD.orientation(model, x)
    C = forceMatrix(model)
    mass, g = model.mass, model.gravity

    # Calculate force and moments
    w = max.(u, 0)  # keep forces positive
    fM = forceMatrix(model)*w
    f = fM[1]
    M = @SVector [fM[2], fM[3], fM[4]]
    e3 = @SVector [0,0,1]
    F = mass*g - q*(f*e3)
    return F,M
end

function forceMatrix(model::Quadrotor)
    kf, km = model.kf, model.km
    L = model.motor_dist
    @SMatrix [
        kf   kf   kf   kf;
        0    L*kf 0   -L*kf;
       -L*kf 0    L*kf 0;
        km  -km   km  -km;
    ]
end


RobotDynamics.inertia(model::Quadrotor) = model.J
RobotDynamics.inertia_inv(model::Quadrotor) = model.Jinv
RobotDynamics.mass(model::Quadrotor) = model.mass

function Base.zeros(model::Quadrotor{R}) where R
    x = RobotDynamics.build_state(model, zero(RBState))
    u = @SVector fill(-model.mass*model.gravity[end]/4, 4)
    return x,u
end
