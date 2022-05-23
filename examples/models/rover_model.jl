struct RoverKinematics <: RD.ContinuousDynamics
    width::Float64
    length::Float64
    height::Float64
    radius::Float64
    vxl::Float64
    vxr::Float64
    Aν::SMatrix{3,12,Float64,36}
    Aω::SMatrix{3,12,Float64,36}
    B::SMatrix{12,4,Float64,48}
    function RoverKinematics(;width=0.2, length=0.3, height=0.1, radius=0.1, vxl=0.0, vxr=0.0)
        v_constraints_l = zeros(12)
        v_constraints_r = zeros(12)
        v_constraints_l[1] = vxl
        v_constraints_l[7] = vxl
        v_constraints_r[4] = vxr
        v_constraints_r[10] = vxr
        w = width
        l = length
        r = radius
        h = height + radius
        
        # Maps wheel rates (rad/s) to contact point velocities
        B = SA[
           -r  0  0  0 
            0  0  0  0 
            0  0  0  0 
            0 -r  0  0 
            0  0  0  0 
            0  0  0  0 
            0  0 -r  0 
            0  0  0  0 
            0  0  0  0 
            0  0  0 -r 
            0  0  0  0 
            0  0  0  0 
        ]

        # Maps contact point velocities to angular velocity of the body
        d = l^2 + w^2
        Aω = SA[
               0    0 +1/w    0    0 +1/w    0    0 +1/w    0    0 -1/w
               0    0 -1/l    0    0 -1/l    0    0 +1/l    0    0 +1/l
            -w/d +l/d    0 +w/d +l/d    0 -w/d -l/d    0 +w/d -l/d    0
        ]

        # Maps contact point velocities to linear velocities of the body
        Aν = SA[
            1 0 -h/l 1 0 -h/l 1 0 +h/l 1 0 +h/l
            0 1 -h/w 0 1 +h/w 0 1 -h/w 0 1 +h/w
            0 0    1 0 0    1 0 0    1 0 0    1
        ]
        new(width, length, height, radius, vxl, vxr, Aν, Aω, B)
    end
end

RD.state_dim(::RoverKinematics) = 7
RD.control_dim(::RoverKinematics) = 4

function RD.dynamics(model::RoverKinematics, x, u)
    p = SA[x[1], x[2], x[3]]        # translation
    q = SA[x[4], x[5], x[6], x[7]]  # attitude
    quat = UnitQuaternion(q, true)

    # Get velocities of the contact points on wheels
    v_contacts_wheel = model.B * u 

    # Get contact point velocities for the body
    # vbody = vslip - vwheel
    v_slip = SA[model.vxl,0,0, model.vxr,0,0, model.vxl,0,0, model.vxr,0,0]
    v_contacts_body = v_slip - v_contacts_wheel  # QUESTION: wouldn't a 0-1 multiplier make more sense?

    # Convert contact point velocities in velocities of the body, in the body frame
    ν = model.Aν * v_contacts_body
    ω = model.Aω * v_contacts_body

    # Compute kinematics
    pdot = quat * ν
    qdot = Rotations.kinematics(quat, ω)

    return [pdot; qdot]
end
