begin
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:548 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:495 =#
        struct Se3ForceDynamics <: RD.ContinuousDynamics
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:496 =#
            m::Float64
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:497 =#
            A::SparseMatrixCSC{Float64, Int}
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:498 =#
            B::SparseMatrixCSC{Float64, Int}
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:499 =#
            C::Vector{SparseMatrixCSC{Float64, Int}}
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:500 =#
            D::SparseVector{Float64, Int}
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:502 =#
        function Se3ForceDynamics(m)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:502 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:503 =#
            (A, B, C, D) = se3_force_genarrays()
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:504 =#
            c = SA[m]
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:505 =#
            se3_force_updateA!(A, c)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:506 =#
            se3_force_updateB!(B, c)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:507 =#
            se3_force_updateC!(C, c)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:508 =#
            se3_force_updateD!(D, c)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:509 =#
            Se3ForceDynamics(m, A, B, C, D)
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:511 =#
        RD.state_dim(::Se3ForceDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:511 =#
                42
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:512 =#
        RD.control_dim(::Se3ForceDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:512 =#
                6
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:513 =#
        RD.default_diffmethod(::Se3ForceDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:513 =#
                RD.UserDefined()
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:514 =#
        RD.default_signature(::Se3ForceDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:514 =#
                RD.InPlace()
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:515 =#
        base_state_dim(::Se3ForceDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:515 =#
                15
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:517 =#
        getconstants(model::Se3ForceDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:517 =#
                SA[getfield(model, :m)]
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:519 =#
        expandstate!(model::Se3ForceDynamics, y, x) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:519 =#
                se3_force_expand!(y, x)
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:520 =#
        expandstate(model::Se3ForceDynamics, x) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:520 =#
                expandstate!(model, zeros(RD.state_dim(model)), x)
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:522 =#
        function RD.dynamics!(model::Se3ForceDynamics, xdot, x, u)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:522 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:523 =#
            c = getconstants(model)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:524 =#
            se3_force_dynamics!(xdot, x, u, c)
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:527 =#
        function RD.jacobian!(model::Se3ForceDynamics, J, y, x, u)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:527 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:528 =#
            (n, m) = RD.dims(model)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:529 =#
            Jx = view(J, :, 1:n)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:530 =#
            Ju = view(J, :, n + 1:n + m)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:531 =#
            Jx .= model.A
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:532 =#
            Ju .= model.B
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:533 =#
            for i = 1:length(u)
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:534 =#
                Jx .+= model.C[i] .* u[i]
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:535 =#
                Ju[:, i] .+= model.C[i] * x
            end
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:537 =#
            return nothing
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:540 =#
        BilinearControl.getA(model::Se3ForceDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:540 =#
                model.A
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:541 =#
        BilinearControl.getB(model::Se3ForceDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:541 =#
                model.B
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:542 =#
        BilinearControl.getC(model::Se3ForceDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:542 =#
                model.C
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:543 =#
        BilinearControl.getD(model::Se3ForceDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:543 =#
                model.D
            end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:549 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:380 =#
        function se3_force_expand!(y, x)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:380 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:381 =#
            _x = x
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:382 =#
            y[1] = (getindex)(_x, 1)
            y[2] = (getindex)(_x, 2)
            y[3] = (getindex)(_x, 3)
            y[4] = (getindex)(_x, 4)
            y[5] = (getindex)(_x, 5)
            y[6] = (getindex)(_x, 6)
            y[7] = (getindex)(_x, 7)
            y[8] = (getindex)(_x, 8)
            y[9] = (getindex)(_x, 9)
            y[10] = (getindex)(_x, 10)
            y[11] = (getindex)(_x, 11)
            y[12] = (getindex)(_x, 12)
            y[13] = (getindex)(_x, 13)
            y[14] = (getindex)(_x, 14)
            y[15] = (getindex)(_x, 15)
            y[16] = (*)((getindex)(_x, 4), (getindex)(_x, 13))
            y[17] = (*)((getindex)(_x, 5), (getindex)(_x, 13))
            y[18] = (*)((getindex)(_x, 6), (getindex)(_x, 13))
            y[19] = (*)((getindex)(_x, 7), (getindex)(_x, 13))
            y[20] = (*)((getindex)(_x, 8), (getindex)(_x, 13))
            y[21] = (*)((getindex)(_x, 9), (getindex)(_x, 13))
            y[22] = (*)((getindex)(_x, 10), (getindex)(_x, 13))
            y[23] = (*)((getindex)(_x, 11), (getindex)(_x, 13))
            y[24] = (*)((getindex)(_x, 12), (getindex)(_x, 13))
            y[25] = (*)((getindex)(_x, 4), (getindex)(_x, 14))
            y[26] = (*)((getindex)(_x, 5), (getindex)(_x, 14))
            y[27] = (*)((getindex)(_x, 6), (getindex)(_x, 14))
            y[28] = (*)((getindex)(_x, 7), (getindex)(_x, 14))
            y[29] = (*)((getindex)(_x, 8), (getindex)(_x, 14))
            y[30] = (*)((getindex)(_x, 9), (getindex)(_x, 14))
            y[31] = (*)((getindex)(_x, 10), (getindex)(_x, 14))
            y[32] = (*)((getindex)(_x, 11), (getindex)(_x, 14))
            y[33] = (*)((getindex)(_x, 12), (getindex)(_x, 14))
            y[34] = (*)((getindex)(_x, 4), (getindex)(_x, 15))
            y[35] = (*)((getindex)(_x, 5), (getindex)(_x, 15))
            y[36] = (*)((getindex)(_x, 6), (getindex)(_x, 15))
            y[37] = (*)((getindex)(_x, 7), (getindex)(_x, 15))
            y[38] = (*)((getindex)(_x, 8), (getindex)(_x, 15))
            y[39] = (*)((getindex)(_x, 9), (getindex)(_x, 15))
            y[40] = (*)((getindex)(_x, 10), (getindex)(_x, 15))
            y[41] = (*)((getindex)(_x, 11), (getindex)(_x, 15))
            y[42] = (*)((getindex)(_x, 12), (getindex)(_x, 15))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:383 =#
            return y
        end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:550 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:393 =#
        function se3_force_dynamics!(xdot, x, u, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:393 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:394 =#
            (_x, _u, _c) = (x, u, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:395 =#
            xdot[1] = (+)((+)((getindex)(_x, 16), (getindex)(_x, 28)), (getindex)(_x, 40))
            xdot[2] = (+)((+)((getindex)(_x, 17), (getindex)(_x, 29)), (getindex)(_x, 41))
            xdot[3] = (+)((+)((getindex)(_x, 18), (getindex)(_x, 30)), (getindex)(_x, 42))
            xdot[4] = (+)((*)((getindex)(_u, 6), (getindex)(_x, 7)), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 10)))
            xdot[5] = (+)((*)((getindex)(_u, 6), (getindex)(_x, 8)), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 11)))
            xdot[6] = (+)((*)((getindex)(_u, 6), (getindex)(_x, 9)), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 12)))
            xdot[7] = (+)((*)((getindex)(_u, 4), (getindex)(_x, 10)), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 4)))
            xdot[8] = (+)((*)((getindex)(_u, 4), (getindex)(_x, 11)), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 5)))
            xdot[9] = (+)((*)((getindex)(_u, 4), (getindex)(_x, 12)), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 6)))
            xdot[10] = (+)((*)((getindex)(_u, 5), (getindex)(_x, 4)), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 7)))
            xdot[11] = (+)((*)((getindex)(_u, 5), (getindex)(_x, 5)), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 8)))
            xdot[12] = (+)((*)((getindex)(_u, 5), (getindex)(_x, 6)), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 9)))
            xdot[13] = (+)((+)((/)((getindex)(_u, 1), (getindex)(_c, 1)), (*)((getindex)(_u, 6), (getindex)(_x, 14))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 15)))
            xdot[14] = (+)((+)((/)((getindex)(_u, 2), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 15))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 13)))
            xdot[15] = (+)((+)((/)((getindex)(_u, 3), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 13))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 14)))
            xdot[16] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 4)), (getindex)(_c, 1)), (*)((getindex)(_u, 6), (getindex)(_x, 19))), (*)((getindex)(_u, 6), (getindex)(_x, 25))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 22))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 34)))
            xdot[17] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 5)), (getindex)(_c, 1)), (*)((getindex)(_u, 6), (getindex)(_x, 20))), (*)((getindex)(_u, 6), (getindex)(_x, 26))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 23))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 35)))
            xdot[18] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 6)), (getindex)(_c, 1)), (*)((getindex)(_u, 6), (getindex)(_x, 21))), (*)((getindex)(_u, 6), (getindex)(_x, 27))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 24))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 36)))
            xdot[19] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 7)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 22))), (*)((getindex)(_u, 6), (getindex)(_x, 28))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 37))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 16)))
            xdot[20] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 8)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 23))), (*)((getindex)(_u, 6), (getindex)(_x, 29))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 17))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 38)))
            xdot[21] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 9)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 24))), (*)((getindex)(_u, 6), (getindex)(_x, 30))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 39))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 18)))
            xdot[22] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 10)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 16))), (*)((getindex)(_u, 6), (getindex)(_x, 31))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 19))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 40)))
            xdot[23] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 11)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 17))), (*)((getindex)(_u, 6), (getindex)(_x, 32))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 20))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 41)))
            xdot[24] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 12)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 18))), (*)((getindex)(_u, 6), (getindex)(_x, 33))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 21))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 42)))
            xdot[25] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 4)), (getindex)(_c, 1)), (*)((getindex)(_u, 6), (getindex)(_x, 28))), (*)((getindex)(_u, 4), (getindex)(_x, 34))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 31))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 16)))
            xdot[26] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 5)), (getindex)(_c, 1)), (*)((getindex)(_u, 6), (getindex)(_x, 29))), (*)((getindex)(_u, 4), (getindex)(_x, 35))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 17))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 32)))
            xdot[27] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 6)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 36))), (*)((getindex)(_u, 6), (getindex)(_x, 30))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 33))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 18)))
            xdot[28] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 7)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 31))), (*)((getindex)(_u, 4), (getindex)(_x, 37))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 19))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 25)))
            xdot[29] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 8)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 32))), (*)((getindex)(_u, 4), (getindex)(_x, 38))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 20))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 26)))
            xdot[30] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 9)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 33))), (*)((getindex)(_u, 4), (getindex)(_x, 39))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 21))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 27)))
            xdot[31] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 10)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 25))), (*)((getindex)(_u, 4), (getindex)(_x, 40))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 22))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 28)))
            xdot[32] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 11)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 41))), (*)((getindex)(_u, 5), (getindex)(_x, 26))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 23))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 29)))
            xdot[33] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 12)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 42))), (*)((getindex)(_u, 5), (getindex)(_x, 27))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 24))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 30)))
            xdot[34] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 4)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 16))), (*)((getindex)(_u, 6), (getindex)(_x, 37))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 25))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 40)))
            xdot[35] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 5)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 17))), (*)((getindex)(_u, 6), (getindex)(_x, 38))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 26))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 41)))
            xdot[36] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 6)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 18))), (*)((getindex)(_u, 6), (getindex)(_x, 39))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 27))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 42)))
            xdot[37] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 7)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 40))), (*)((getindex)(_u, 5), (getindex)(_x, 19))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 28))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 34)))
            xdot[38] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 8)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 41))), (*)((getindex)(_u, 5), (getindex)(_x, 20))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 29))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 35)))
            xdot[39] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 9)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 42))), (*)((getindex)(_u, 5), (getindex)(_x, 21))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 30))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 36)))
            xdot[40] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 10)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 22))), (*)((getindex)(_u, 5), (getindex)(_x, 34))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 31))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 37)))
            xdot[41] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 11)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 23))), (*)((getindex)(_u, 5), (getindex)(_x, 35))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 32))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 38)))
            xdot[42] = (+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 12)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 24))), (*)((getindex)(_u, 5), (getindex)(_x, 36))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 33))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 39)))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:396 =#
            return
        end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:551 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:427 =#
        function se3_force_updateA!(A, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:427 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:428 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:429 =#
            nzval = A.nzval
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:430 =#
            nzval[1] = 1
            nzval[2] = 1
            nzval[3] = 1
            nzval[4] = 1
            nzval[5] = 1
            nzval[6] = 1
            nzval[7] = 1
            nzval[8] = 1
            nzval[9] = 1
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:431 =#
            return A
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:433 =#
        function se3_force_updateB!(B, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:433 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:434 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:435 =#
            nzval = B.nzval
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:436 =#
            nzval[1] = (/)(1, (getindex)(_c, 1))
            nzval[2] = (/)(1, (getindex)(_c, 1))
            nzval[3] = (/)(1, (getindex)(_c, 1))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:437 =#
            return B
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:439 =#
        function se3_force_updateC!(C, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:439 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:440 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:441 =#
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:419 =#
                nzval = (C[1]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:420 =#
                nzval[1] = (/)(1, (getindex)(_c, 1))
                nzval[2] = (/)(1, (getindex)(_c, 1))
                nzval[3] = (/)(1, (getindex)(_c, 1))
                nzval[4] = (/)(1, (getindex)(_c, 1))
                nzval[5] = (/)(1, (getindex)(_c, 1))
                nzval[6] = (/)(1, (getindex)(_c, 1))
                nzval[7] = (/)(1, (getindex)(_c, 1))
                nzval[8] = (/)(1, (getindex)(_c, 1))
                nzval[9] = (/)(1, (getindex)(_c, 1))
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:419 =#
                nzval = (C[2]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:420 =#
                nzval[1] = (/)(1, (getindex)(_c, 1))
                nzval[2] = (/)(1, (getindex)(_c, 1))
                nzval[3] = (/)(1, (getindex)(_c, 1))
                nzval[4] = (/)(1, (getindex)(_c, 1))
                nzval[5] = (/)(1, (getindex)(_c, 1))
                nzval[6] = (/)(1, (getindex)(_c, 1))
                nzval[7] = (/)(1, (getindex)(_c, 1))
                nzval[8] = (/)(1, (getindex)(_c, 1))
                nzval[9] = (/)(1, (getindex)(_c, 1))
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:419 =#
                nzval = (C[3]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:420 =#
                nzval[1] = (/)(1, (getindex)(_c, 1))
                nzval[2] = (/)(1, (getindex)(_c, 1))
                nzval[3] = (/)(1, (getindex)(_c, 1))
                nzval[4] = (/)(1, (getindex)(_c, 1))
                nzval[5] = (/)(1, (getindex)(_c, 1))
                nzval[6] = (/)(1, (getindex)(_c, 1))
                nzval[7] = (/)(1, (getindex)(_c, 1))
                nzval[8] = (/)(1, (getindex)(_c, 1))
                nzval[9] = (/)(1, (getindex)(_c, 1))
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:419 =#
                nzval = (C[4]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:420 =#
                nzval[1] = -1
                nzval[2] = -1
                nzval[3] = -1
                nzval[4] = 1
                nzval[5] = 1
                nzval[6] = 1
                nzval[7] = -1
                nzval[8] = 1
                nzval[9] = -1
                nzval[10] = -1
                nzval[11] = -1
                nzval[12] = 1
                nzval[13] = 1
                nzval[14] = 1
                nzval[15] = -1
                nzval[16] = -1
                nzval[17] = -1
                nzval[18] = -1
                nzval[19] = -1
                nzval[20] = -1
                nzval[21] = -1
                nzval[22] = -1
                nzval[23] = -1
                nzval[24] = 1
                nzval[25] = -1
                nzval[26] = 1
                nzval[27] = -1
                nzval[28] = 1
                nzval[29] = -1
                nzval[30] = 1
                nzval[31] = 1
                nzval[32] = 1
                nzval[33] = 1
                nzval[34] = -1
                nzval[35] = 1
                nzval[36] = -1
                nzval[37] = 1
                nzval[38] = -1
                nzval[39] = 1
                nzval[40] = 1
                nzval[41] = 1
                nzval[42] = 1
                nzval[43] = 1
                nzval[44] = 1
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:419 =#
                nzval = (C[5]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:420 =#
                nzval[1] = 1
                nzval[2] = 1
                nzval[3] = 1
                nzval[4] = -1
                nzval[5] = -1
                nzval[6] = -1
                nzval[7] = 1
                nzval[8] = -1
                nzval[9] = 1
                nzval[10] = 1
                nzval[11] = 1
                nzval[12] = 1
                nzval[13] = 1
                nzval[14] = 1
                nzval[15] = 1
                nzval[16] = 1
                nzval[17] = 1
                nzval[18] = -1
                nzval[19] = 1
                nzval[20] = -1
                nzval[21] = 1
                nzval[22] = -1
                nzval[23] = 1
                nzval[24] = 1
                nzval[25] = 1
                nzval[26] = 1
                nzval[27] = -1
                nzval[28] = -1
                nzval[29] = -1
                nzval[30] = -1
                nzval[31] = 1
                nzval[32] = -1
                nzval[33] = 1
                nzval[34] = -1
                nzval[35] = 1
                nzval[36] = -1
                nzval[37] = -1
                nzval[38] = -1
                nzval[39] = -1
                nzval[40] = -1
                nzval[41] = -1
                nzval[42] = -1
                nzval[43] = -1
                nzval[44] = -1
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:419 =#
                nzval = (C[6]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:420 =#
                nzval[1] = -1
                nzval[2] = -1
                nzval[3] = -1
                nzval[4] = 1
                nzval[5] = 1
                nzval[6] = 1
                nzval[7] = -1
                nzval[8] = 1
                nzval[9] = -1
                nzval[10] = -1
                nzval[11] = -1
                nzval[12] = -1
                nzval[13] = -1
                nzval[14] = -1
                nzval[15] = 1
                nzval[16] = -1
                nzval[17] = 1
                nzval[18] = -1
                nzval[19] = 1
                nzval[20] = -1
                nzval[21] = -1
                nzval[22] = -1
                nzval[23] = -1
                nzval[24] = 1
                nzval[25] = -1
                nzval[26] = 1
                nzval[27] = -1
                nzval[28] = 1
                nzval[29] = -1
                nzval[30] = 1
                nzval[31] = 1
                nzval[32] = 1
                nzval[33] = 1
                nzval[34] = 1
                nzval[35] = 1
                nzval[36] = 1
                nzval[37] = 1
                nzval[38] = 1
                nzval[39] = -1
                nzval[40] = -1
                nzval[41] = -1
                nzval[42] = 1
                nzval[43] = 1
                nzval[44] = 1
            end
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:442 =#
            return C
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:444 =#
        function se3_force_updateD!(D, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:444 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:445 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:446 =#
            nzval = D.nzval
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:447 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:448 =#
            return D
        end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:552 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:463 =#
        function se3_force_genarrays()
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:463 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:464 =#
            n = 42
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:465 =#
            m = 6
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:466 =#
            A = SparseMatrixCSC(n, n, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 10], [1, 2, 3, 1, 2, 3, 1, 2, 3], zeros(9))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:471 =#
            B = SparseMatrixCSC(n, m, [1, 2, 3, 4, 4, 4, 4], [13, 14, 15], zeros(3))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:476 =#
            C = [begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:455 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], [16, 17, 18, 19, 20, 21, 22, 23, 24], zeros(9))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:455 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], [25, 26, 27, 28, 29, 30, 31, 32, 33], zeros(9))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:455 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], [34, 35, 36, 37, 38, 39, 40, 41, 42], zeros(9))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:455 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 9, 9, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 31, 32, 33, 35, 37, 39, 41, 43, 45], [10, 11, 12, 7, 8, 9, 15, 14, 22, 23, 24, 19, 20, 21, 34, 35, 36, 31, 37, 32, 38, 33, 39, 28, 40, 29, 41, 30, 42, 25, 26, 27, 28, 40, 29, 41, 30, 42, 31, 37, 32, 38, 33, 39], zeros(44))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:455 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 5, 6, 7, 8, 8, 9, 11, 13, 15, 16, 17, 18, 20, 22, 24, 25, 26, 27, 27, 27, 27, 28, 29, 30, 32, 34, 36, 37, 38, 39, 41, 43, 45], [10, 11, 12, 4, 5, 6, 15, 13, 22, 34, 23, 35, 24, 36, 37, 38, 39, 16, 40, 17, 41, 18, 42, 31, 32, 33, 25, 26, 27, 16, 40, 17, 41, 18, 42, 19, 20, 21, 22, 34, 23, 35, 24, 36], zeros(44))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:455 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 8, 9, 9, 11, 13, 15, 17, 19, 21, 22, 23, 24, 26, 28, 30, 32, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 45, 45, 45], [7, 8, 9, 4, 5, 6, 14, 13, 19, 25, 20, 26, 21, 27, 16, 28, 17, 29, 18, 30, 31, 32, 33, 16, 28, 17, 29, 18, 30, 19, 25, 20, 26, 21, 27, 22, 23, 24, 37, 38, 39, 34, 35, 36], zeros(44))
                    end]
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:477 =#
            D = SparseVector(n, Int64[], zeros(0))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:481 =#
            return (A, B, C, D)
        end
    end
end