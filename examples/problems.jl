using TrajectoryOptimization
using LinearAlgebra
using RobotZoo
using StaticArrays

model_dir = joinpath(@__DIR__, "..", "test", "models")
include(joinpath(model_dir, "dubins_model.jl"))

function builddubinsproblem(model=RobotZoo.DubinsCar(); 
        scenario=:turn90, N=101, ubnd=1.5
    )
    # model
    dmodel = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(model)
    n,m = RD.dims(model)

    tf = 3.
    dt = tf / (N-1)

    # cost
    d = 1.5
    x0 = @SVector [0., 0., 0.]
    if scenario == :turn90
        xf = @SVector [d, d,  deg2rad(90)]
    else
        xf = @SVector [0, d, 0.]
    end
    Qf = 100.0*Diagonal(@SVector ones(n))
    Q = (1e-2)*Diagonal(@SVector ones(n))
    R = (1e-2)*Diagonal(@SVector ones(m))

    if model isa BilinearDubins
        x0 = expandstate(model, x0)
        xf = expandstate(model, xf)
        Q = Diagonal([diag(Q)[1:2]; fill(Q[3,3]*1e-3, 2)]) 
        Qf = Diagonal([diag(Qf)[1:2]; fill(Qf[3,3]*1e-3, 2)]) 
    end

    # objective 
    obj = LQRObjective(Q*dt,R*dt,Qf,xf,N)

    # Initial Guess
    U = [@SVector fill(0.1,m) for k = 1:N-1]

    # constraints
    cons = ConstraintList(n,m,N)
    add_constraint!(cons, GoalConstraint(xf), N)
    add_constraint!(cons, BoundConstraint(n,m, u_min=-ubnd, u_max=ubnd), 1:N-1)

    if scenario == :parallelpark
        x_min = @SVector [-0.25, -0.1, -Inf]
        x_max = @SVector [0.25, d + 0.1, Inf]
        if model isa BilinearDubins
            x_min = push(x_min, -Inf) 
            x_max = push(x_max, +Inf) 
        end
        bnd_x = BoundConstraint(n,m, x_min=x_min, x_max=x_max)
        add_constraint!(cons, bnd_x, 2:N-1)
    end

    prob = Problem(dmodel, obj, x0, tf, xf=xf, U0=U, constraints=cons)
    rollout!(prob)

    return prob
end