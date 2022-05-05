import Pkg; Pkg.activate(@__DIR__)
using BenchmarkTools
using Test

N = 2^20
x = fill(1.0f0, N)
y = fill(2.0f0, N)
y .+= x

@test all(y .== 3.0f0)

function sequential_add!(y, x)
    for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function parallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
sequential_add!(y, x)
@test all(y .== 3.0f0)

fill!(y, 2)
parallel_add!(y,x)
@test all(y .== 3.0f0)

@btime sequential_add!($y, $x)
@btime parallel_add!($y, $x)

using CUDA
x_d = CUDA.fill(1.0f0, N)
y_d = CUDA.fill(2.0f0, N)

y_d .+= x_d
@test all(Array(y_d) .== 3.0f0)

function add_broadcast!(y,x)
    CUDA.@sync y .+= x
    return
end
@btime add_broadcast!($y, $x)

function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function bench_gpu1!(y,x)
    CUDA.@sync begin
        @cuda gpu_add1!(y, x)
    end
end

fill!(y_d, 2)
@cuda gpu_add1!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)
@btime bench_gpu1!($y_d, $x_d)


function gpu_add2!(y, x)::Nothing
    index = threadIdx().x
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function bench_gpu2!(y,x)
    CUDA.@sync begin
        @cuda threads=1024 gpu_add2!(y, x)
    end
end
fill!(y_d, 2)
@cuda threads=1024 gpu_add2!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)
@btime bench_gpu2!($y_d, $x_d)

function gpu_add3!(y, x)::Nothing
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function bench_gpu3!(y,x; threads=256, blocks=ceil(Int,length(y)/threads))
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks gpu_add3!(y, x)
    end
end

fill!(y_d, 2)
nthreads = 256   
nblocks = ceil(Int, N / nthreads)
@cuda threads=nthreads blocks=nblocks gpu_add3!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)
@btime bench_gpu3!($y_d, $x_d, threads=$nthreads)

kernel = @cuda launch=false gpu_add3!(y_d, x_d)
config = launch_configuration(kernel.fun)
threads = min(N, config.threads)
blocks = cld(N, threads)

fill!(y_d, 2)
kernel(y_d, x_d; threads, blocks) 
@test all(Array(y_d) .== 3.0f0)

@btime kernel($y_d, $x_d; threads=$threads, blocks=$blocks)