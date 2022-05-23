import Pkg;
Pkg.activate(joinpath(@__DIR__, ".."));
using DataFrames
using CSV
using LinearAlgebra
using Interpolations

# Read in rosbag exports
imu = DataFrame(CSV.File(joinpath(@__DIR__, "imu-data.csv")))
joy = DataFrame(CSV.File(joinpath(@__DIR__, "joy.csv")))
quat = DataFrame(CSV.File(joinpath(@__DIR__, "quat.csv")))
vicon = DataFrame(CSV.File(joinpath(@__DIR__, "vicon-morphin-morphin.csv")))
wheel_vel = DataFrame(CSV.File(joinpath(@__DIR__, "wheel_velocity.csv")))
wheels = DataFrame(CSV.File(joinpath(@__DIR__, "wheels.csv")))

# Get the time range
msgsall = Dict(:imu => imu, :joy => joy, :quat => quat, :vicon => vicon, :wheel_vel => wheel_vel, :wheels => wheels)
timesall = Dict(kv.first => getproperty(kv.second, :Time) for kv in pairs(msgsall))
t0 = maximum(x->x[1], values(timesall))
tf = minimum(x->x[end], values(timesall))
dt = 0.01
times = range(0,tf-t0,step=dt)

# Export data using linear interpolation
function interpdata(df, field)
    LinearInterpolation(df.Time .- df.Time[1], df[:,field]).(times)
end
df = DataFrame()
df.time = times
df.angular_velocity_x = interpdata(imu, "angular_velocity.x")
df.angular_velocity_y = interpdata(imu, "angular_velocity.y")
df.angular_velocity_z = interpdata(imu, "angular_velocity.z")
df.linear_accel_x = interpdata(imu, "linear_acceleration.x") 
df.linear_accel_y = interpdata(imu, "linear_acceleration.y") 
df.linear_accel_z = interpdata(imu, "linear_acceleration.z") 
df.vicon_x = interpdata(vicon, "transform.translation.x") 
df.vicon_y = interpdata(vicon, "transform.translation.y") 
df.vicon_z = interpdata(vicon, "transform.translation.z") 
df.vicon_qw = interpdata(vicon, "transform.rotation.w")
df.vicon_qx = interpdata(vicon, "transform.rotation.x")
df.vicon_qy = interpdata(vicon, "transform.rotation.y")
df.vicon_qz = interpdata(vicon, "transform.rotation.z")
df.cmd_fl = interpdata(joy, "axes_1")
df.cmd_fr = interpdata(joy, "axes_2")
df.cmd_rl = interpdata(joy, "axes_3")
df.cmd_rr = interpdata(joy, "axes_4")
df.wheel_vel_fl = interpdata(wheel_vel, "data_0")
df.wheel_vel_fr = interpdata(wheel_vel, "data_1")
df.wheel_vel_rl = interpdata(wheel_vel, "data_2")
df.wheel_vel_rr = interpdata(wheel_vel, "data_3")
df.wheels = interpdata(wheels, "data_0")
df.wheels = interpdata(wheels, "data_1")
df.wheels = interpdata(wheels, "data_2")
df.wheels = interpdata(wheels, "data_3")
prod(size(df)) * sizeof(Float64) / 1e6  # size in MB
CSV.write(joinpath(@__DIR__, "rover_data_processed.csv"), df)