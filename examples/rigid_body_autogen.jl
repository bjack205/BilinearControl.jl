begin
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:569 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:516 =#
        struct RigidBodyDynamics <: RD.ContinuousDynamics
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:517 =#
            m::Float64
            J1::Float64
            J2::Float64
            J3::Float64
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:518 =#
            A::SparseMatrixCSC{Float64, Int}
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:519 =#
            B::SparseMatrixCSC{Float64, Int}
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:520 =#
            C::Vector{SparseMatrixCSC{Float64, Int}}
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:521 =#
            D::SparseVector{Float64, Int}
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:523 =#
        function RigidBodyDynamics(m, J1, J2, J3)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:523 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:524 =#
            (A, B, C, D) = rigid_body_genarrays()
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:525 =#
            c = SA[m, J1, J2, J3]
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:526 =#
            rigid_body_updateA!(A, c)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:527 =#
            rigid_body_updateB!(B, c)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:528 =#
            rigid_body_updateC!(C, c)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:529 =#
            rigid_body_updateD!(D, c)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:530 =#
            RigidBodyDynamics(m, J1, J2, J3, A, B, C, D)
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:532 =#
        RD.state_dim(::RigidBodyDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:532 =#
                87
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:533 =#
        RD.control_dim(::RigidBodyDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:533 =#
                6
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:534 =#
        RD.default_diffmethod(::RigidBodyDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:534 =#
                RD.UserDefined()
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:535 =#
        RD.default_signature(::RigidBodyDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:535 =#
                RD.InPlace()
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:536 =#
        base_state_dim(::RigidBodyDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:536 =#
                18
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:538 =#
        getconstants(model::RigidBodyDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:538 =#
                SA[getfield(model, :m), getfield(model, :J1), getfield(model, :J2), getfield(model, :J3)]
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:540 =#
        expandstate!(model::RigidBodyDynamics, y, x) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:540 =#
                rigid_body_expand!(y, x)
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:541 =#
        expandstate(model::RigidBodyDynamics, x) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:541 =#
                expandstate!(model, zeros(RD.state_dim(model)), x)
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:543 =#
        function RD.dynamics!(model::RigidBodyDynamics, xdot, x, u)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:543 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:544 =#
            c = getconstants(model)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:545 =#
            rigid_body_dynamics!(xdot, x, u, c)
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:548 =#
        function RD.jacobian!(model::RigidBodyDynamics, J, y, x, u)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:548 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:549 =#
            (n, m) = RD.dims(model)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:550 =#
            Jx = view(J, :, 1:n)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:551 =#
            Ju = view(J, :, n + 1:n + m)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:552 =#
            Jx .= model.A
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:553 =#
            Ju .= model.B
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:554 =#
            for i = 1:length(u)
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:555 =#
                Jx .+= model.C[i] .* u[i]
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:556 =#
                Ju[:, i] .+= model.C[i] * x
            end
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:558 =#
            return nothing
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:561 =#
        BilinearControl.getA(model::RigidBodyDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:561 =#
                model.A
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:562 =#
        BilinearControl.getB(model::RigidBodyDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:562 =#
                model.B
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:563 =#
        BilinearControl.getC(model::RigidBodyDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:563 =#
                model.C
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:564 =#
        BilinearControl.getD(model::RigidBodyDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:564 =#
                model.D
            end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:570 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:401 =#
        function rigid_body_expand!(y, x)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:401 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:402 =#
            _x = x
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:403 =#
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
            y[16] = (getindex)(_x, 16)
            y[17] = (getindex)(_x, 17)
            y[18] = (getindex)(_x, 18)
            y[19] = (*)((getindex)(_x, 4), (getindex)(_x, 13))
            y[20] = (*)((getindex)(_x, 5), (getindex)(_x, 13))
            y[21] = (*)((getindex)(_x, 6), (getindex)(_x, 13))
            y[22] = (*)((getindex)(_x, 7), (getindex)(_x, 13))
            y[23] = (*)((getindex)(_x, 8), (getindex)(_x, 13))
            y[24] = (*)((getindex)(_x, 9), (getindex)(_x, 13))
            y[25] = (*)((getindex)(_x, 10), (getindex)(_x, 13))
            y[26] = (*)((getindex)(_x, 11), (getindex)(_x, 13))
            y[27] = (*)((getindex)(_x, 12), (getindex)(_x, 13))
            y[28] = (*)((getindex)(_x, 4), (getindex)(_x, 14))
            y[29] = (*)((getindex)(_x, 5), (getindex)(_x, 14))
            y[30] = (*)((getindex)(_x, 6), (getindex)(_x, 14))
            y[31] = (*)((getindex)(_x, 7), (getindex)(_x, 14))
            y[32] = (*)((getindex)(_x, 8), (getindex)(_x, 14))
            y[33] = (*)((getindex)(_x, 9), (getindex)(_x, 14))
            y[34] = (*)((getindex)(_x, 10), (getindex)(_x, 14))
            y[35] = (*)((getindex)(_x, 11), (getindex)(_x, 14))
            y[36] = (*)((getindex)(_x, 12), (getindex)(_x, 14))
            y[37] = (*)((getindex)(_x, 4), (getindex)(_x, 15))
            y[38] = (*)((getindex)(_x, 5), (getindex)(_x, 15))
            y[39] = (*)((getindex)(_x, 6), (getindex)(_x, 15))
            y[40] = (*)((getindex)(_x, 7), (getindex)(_x, 15))
            y[41] = (*)((getindex)(_x, 8), (getindex)(_x, 15))
            y[42] = (*)((getindex)(_x, 9), (getindex)(_x, 15))
            y[43] = (*)((getindex)(_x, 10), (getindex)(_x, 15))
            y[44] = (*)((getindex)(_x, 11), (getindex)(_x, 15))
            y[45] = (*)((getindex)(_x, 12), (getindex)(_x, 15))
            y[46] = (*)((getindex)(_x, 4), (getindex)(_x, 16))
            y[47] = (*)((getindex)(_x, 5), (getindex)(_x, 16))
            y[48] = (*)((getindex)(_x, 6), (getindex)(_x, 16))
            y[49] = (*)((getindex)(_x, 7), (getindex)(_x, 16))
            y[50] = (*)((getindex)(_x, 8), (getindex)(_x, 16))
            y[51] = (*)((getindex)(_x, 9), (getindex)(_x, 16))
            y[52] = (*)((getindex)(_x, 10), (getindex)(_x, 16))
            y[53] = (*)((getindex)(_x, 11), (getindex)(_x, 16))
            y[54] = (*)((getindex)(_x, 12), (getindex)(_x, 16))
            y[55] = (*)((getindex)(_x, 4), (getindex)(_x, 17))
            y[56] = (*)((getindex)(_x, 5), (getindex)(_x, 17))
            y[57] = (*)((getindex)(_x, 6), (getindex)(_x, 17))
            y[58] = (*)((getindex)(_x, 7), (getindex)(_x, 17))
            y[59] = (*)((getindex)(_x, 8), (getindex)(_x, 17))
            y[60] = (*)((getindex)(_x, 9), (getindex)(_x, 17))
            y[61] = (*)((getindex)(_x, 10), (getindex)(_x, 17))
            y[62] = (*)((getindex)(_x, 11), (getindex)(_x, 17))
            y[63] = (*)((getindex)(_x, 12), (getindex)(_x, 17))
            y[64] = (*)((getindex)(_x, 4), (getindex)(_x, 18))
            y[65] = (*)((getindex)(_x, 5), (getindex)(_x, 18))
            y[66] = (*)((getindex)(_x, 6), (getindex)(_x, 18))
            y[67] = (*)((getindex)(_x, 7), (getindex)(_x, 18))
            y[68] = (*)((getindex)(_x, 8), (getindex)(_x, 18))
            y[69] = (*)((getindex)(_x, 9), (getindex)(_x, 18))
            y[70] = (*)((getindex)(_x, 10), (getindex)(_x, 18))
            y[71] = (*)((getindex)(_x, 11), (getindex)(_x, 18))
            y[72] = (*)((getindex)(_x, 12), (getindex)(_x, 18))
            y[73] = (*)((getindex)(_x, 13), (getindex)(_x, 16))
            y[74] = (*)((getindex)(_x, 14), (getindex)(_x, 16))
            y[75] = (*)((getindex)(_x, 15), (getindex)(_x, 16))
            y[76] = (*)((getindex)(_x, 13), (getindex)(_x, 17))
            y[77] = (*)((getindex)(_x, 14), (getindex)(_x, 17))
            y[78] = (*)((getindex)(_x, 15), (getindex)(_x, 17))
            y[79] = (*)((getindex)(_x, 13), (getindex)(_x, 18))
            y[80] = (*)((getindex)(_x, 14), (getindex)(_x, 18))
            y[81] = (*)((getindex)(_x, 15), (getindex)(_x, 18))
            y[82] = (^)((getindex)(_x, 16), 2)
            y[83] = (*)((getindex)(_x, 16), (getindex)(_x, 17))
            y[84] = (*)((getindex)(_x, 16), (getindex)(_x, 18))
            y[85] = (^)((getindex)(_x, 17), 2)
            y[86] = (*)((getindex)(_x, 17), (getindex)(_x, 18))
            y[87] = (^)((getindex)(_x, 18), 2)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:404 =#
            return y
        end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:571 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:414 =#
        function rigid_body_dynamics!(xdot, x, u, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:414 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:415 =#
            (_x, _u, _c) = (x, u, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:416 =#
            xdot[1] = (+)((+)((getindex)(_x, 19), (getindex)(_x, 31)), (getindex)(_x, 43))
            xdot[2] = (+)((+)((getindex)(_x, 20), (getindex)(_x, 32)), (getindex)(_x, 44))
            xdot[3] = (+)((+)((getindex)(_x, 21), (getindex)(_x, 33)), (getindex)(_x, 45))
            xdot[4] = (+)((*)(-1, (getindex)(_x, 61)), (getindex)(_x, 67))
            xdot[5] = (+)((*)(-1, (getindex)(_x, 62)), (getindex)(_x, 68))
            xdot[6] = (+)((*)(-1, (getindex)(_x, 63)), (getindex)(_x, 69))
            xdot[7] = (+)((*)(-1, (getindex)(_x, 64)), (getindex)(_x, 52))
            xdot[8] = (+)((*)(-1, (getindex)(_x, 65)), (getindex)(_x, 53))
            xdot[9] = (+)((*)(-1, (getindex)(_x, 66)), (getindex)(_x, 54))
            xdot[10] = (+)((*)(-1, (getindex)(_x, 49)), (getindex)(_x, 55))
            xdot[11] = (+)((*)(-1, (getindex)(_x, 50)), (getindex)(_x, 56))
            xdot[12] = (+)((*)(-1, (getindex)(_x, 51)), (getindex)(_x, 57))
            xdot[13] = (+)((+)((/)((getindex)(_u, 1), (getindex)(_c, 1)), (*)(-1, (getindex)(_x, 78))), (getindex)(_x, 80))
            xdot[14] = (+)((+)((/)((getindex)(_u, 2), (getindex)(_c, 1)), (*)(-1, (getindex)(_x, 79))), (getindex)(_x, 75))
            xdot[15] = (+)((+)((/)((getindex)(_u, 3), (getindex)(_c, 1)), (*)(-1, (getindex)(_x, 74))), (getindex)(_x, 76))
            xdot[16] = (/)((+)((+)((*)((getindex)(_c, 3), (getindex)(_x, 86)), (*)((*)(-1, (getindex)(_c, 4)), (getindex)(_x, 86))), (getindex)(_u, 4)), (getindex)(_c, 2))
            xdot[17] = (/)((+)((+)((*)((getindex)(_c, 4), (getindex)(_x, 84)), (*)((*)(-1, (getindex)(_c, 2)), (getindex)(_x, 84))), (getindex)(_u, 5)), (getindex)(_c, 3))
            xdot[18] = (/)((+)((+)((*)((getindex)(_c, 2), (getindex)(_x, 83)), (*)((*)(-1, (getindex)(_c, 3)), (getindex)(_x, 83))), (getindex)(_u, 6)), (getindex)(_c, 4))
            xdot[19] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 4)), (getindex)(_c, 1))
            xdot[20] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 5)), (getindex)(_c, 1))
            xdot[21] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 6)), (getindex)(_c, 1))
            xdot[22] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 7)), (getindex)(_c, 1))
            xdot[23] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 8)), (getindex)(_c, 1))
            xdot[24] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 9)), (getindex)(_c, 1))
            xdot[25] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 10)), (getindex)(_c, 1))
            xdot[26] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 11)), (getindex)(_c, 1))
            xdot[27] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 12)), (getindex)(_c, 1))
            xdot[28] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 4)), (getindex)(_c, 1))
            xdot[29] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 5)), (getindex)(_c, 1))
            xdot[30] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 6)), (getindex)(_c, 1))
            xdot[31] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 7)), (getindex)(_c, 1))
            xdot[32] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 8)), (getindex)(_c, 1))
            xdot[33] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 9)), (getindex)(_c, 1))
            xdot[34] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 10)), (getindex)(_c, 1))
            xdot[35] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 11)), (getindex)(_c, 1))
            xdot[36] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 12)), (getindex)(_c, 1))
            xdot[37] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 4)), (getindex)(_c, 1))
            xdot[38] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 5)), (getindex)(_c, 1))
            xdot[39] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 6)), (getindex)(_c, 1))
            xdot[40] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 7)), (getindex)(_c, 1))
            xdot[41] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 8)), (getindex)(_c, 1))
            xdot[42] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 9)), (getindex)(_c, 1))
            xdot[43] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 10)), (getindex)(_c, 1))
            xdot[44] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 11)), (getindex)(_c, 1))
            xdot[45] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 12)), (getindex)(_c, 1))
            xdot[46] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 4)), (getindex)(_c, 2))
            xdot[47] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 5)), (getindex)(_c, 2))
            xdot[48] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 6)), (getindex)(_c, 2))
            xdot[49] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 7)), (getindex)(_c, 2))
            xdot[50] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 8)), (getindex)(_c, 2))
            xdot[51] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 9)), (getindex)(_c, 2))
            xdot[52] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 10)), (getindex)(_c, 2))
            xdot[53] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 11)), (getindex)(_c, 2))
            xdot[54] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 12)), (getindex)(_c, 2))
            xdot[55] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 4)), (getindex)(_c, 3))
            xdot[56] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 5)), (getindex)(_c, 3))
            xdot[57] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 6)), (getindex)(_c, 3))
            xdot[58] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 7)), (getindex)(_c, 3))
            xdot[59] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 8)), (getindex)(_c, 3))
            xdot[60] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 9)), (getindex)(_c, 3))
            xdot[61] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 10)), (getindex)(_c, 3))
            xdot[62] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 11)), (getindex)(_c, 3))
            xdot[63] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 12)), (getindex)(_c, 3))
            xdot[64] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 4)), (getindex)(_c, 4))
            xdot[65] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 5)), (getindex)(_c, 4))
            xdot[66] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 6)), (getindex)(_c, 4))
            xdot[67] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 7)), (getindex)(_c, 4))
            xdot[68] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 8)), (getindex)(_c, 4))
            xdot[69] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 9)), (getindex)(_c, 4))
            xdot[70] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 10)), (getindex)(_c, 4))
            xdot[71] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 11)), (getindex)(_c, 4))
            xdot[72] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 12)), (getindex)(_c, 4))
            xdot[73] = (+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 16)), (getindex)(_c, 1)), (/)((*)((getindex)(_u, 4), (getindex)(_x, 13)), (getindex)(_c, 2)))
            xdot[74] = (+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 16)), (getindex)(_c, 1)), (/)((*)((getindex)(_u, 4), (getindex)(_x, 14)), (getindex)(_c, 2)))
            xdot[75] = (+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 16)), (getindex)(_c, 1)), (/)((*)((getindex)(_u, 4), (getindex)(_x, 15)), (getindex)(_c, 2)))
            xdot[76] = (+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 17)), (getindex)(_c, 1)), (/)((*)((getindex)(_u, 5), (getindex)(_x, 13)), (getindex)(_c, 3)))
            xdot[77] = (+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 17)), (getindex)(_c, 1)), (/)((*)((getindex)(_u, 5), (getindex)(_x, 14)), (getindex)(_c, 3)))
            xdot[78] = (+)((/)((*)((getindex)(_u, 5), (getindex)(_x, 15)), (getindex)(_c, 3)), (/)((*)((getindex)(_u, 3), (getindex)(_x, 17)), (getindex)(_c, 1)))
            xdot[79] = (+)((/)((*)((getindex)(_u, 6), (getindex)(_x, 13)), (getindex)(_c, 4)), (/)((*)((getindex)(_u, 1), (getindex)(_x, 18)), (getindex)(_c, 1)))
            xdot[80] = (+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 18)), (getindex)(_c, 1)), (/)((*)((getindex)(_u, 6), (getindex)(_x, 14)), (getindex)(_c, 4)))
            xdot[81] = (+)((/)((*)((getindex)(_u, 6), (getindex)(_x, 15)), (getindex)(_c, 4)), (/)((*)((getindex)(_u, 3), (getindex)(_x, 18)), (getindex)(_c, 1)))
            xdot[82] = (/)((*)((*)(2, (getindex)(_u, 4)), (getindex)(_x, 16)), (getindex)(_c, 2))
            xdot[83] = (+)((/)((*)((getindex)(_u, 4), (getindex)(_x, 17)), (getindex)(_c, 2)), (/)((*)((getindex)(_u, 5), (getindex)(_x, 16)), (getindex)(_c, 3)))
            xdot[84] = (+)((/)((*)((getindex)(_u, 6), (getindex)(_x, 16)), (getindex)(_c, 4)), (/)((*)((getindex)(_u, 4), (getindex)(_x, 18)), (getindex)(_c, 2)))
            xdot[85] = (/)((*)((*)(2, (getindex)(_u, 5)), (getindex)(_x, 17)), (getindex)(_c, 3))
            xdot[86] = (+)((/)((*)((getindex)(_u, 6), (getindex)(_x, 17)), (getindex)(_c, 4)), (/)((*)((getindex)(_u, 5), (getindex)(_x, 18)), (getindex)(_c, 3)))
            xdot[87] = (/)((*)((*)(2, (getindex)(_u, 6)), (getindex)(_x, 18)), (getindex)(_c, 4))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:417 =#
            return
        end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:572 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:448 =#
        function rigid_body_updateA!(A, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:448 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:449 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:450 =#
            nzval = A.nzval
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:451 =#
            nzval[1] = 1
            nzval[2] = 1
            nzval[3] = 1
            nzval[4] = 1
            nzval[5] = 1
            nzval[6] = 1
            nzval[7] = 1
            nzval[8] = 1
            nzval[9] = 1
            nzval[10] = -1
            nzval[11] = -1
            nzval[12] = -1
            nzval[13] = 1
            nzval[14] = 1
            nzval[15] = 1
            nzval[16] = 1
            nzval[17] = 1
            nzval[18] = 1
            nzval[19] = -1
            nzval[20] = -1
            nzval[21] = -1
            nzval[22] = -1
            nzval[23] = -1
            nzval[24] = -1
            nzval[25] = 1
            nzval[26] = 1
            nzval[27] = 1
            nzval[28] = -1
            nzval[29] = 1
            nzval[30] = 1
            nzval[31] = -1
            nzval[32] = -1
            nzval[33] = 1
            nzval[34] = (+)((/)((*)(-1, (getindex)(_c, 3)), (getindex)(_c, 4)), (/)((getindex)(_c, 2), (getindex)(_c, 4)))
            nzval[35] = (+)((/)((*)(-1, (getindex)(_c, 2)), (getindex)(_c, 3)), (/)((getindex)(_c, 4), (getindex)(_c, 3)))
            nzval[36] = (+)((/)((*)(-1, (getindex)(_c, 4)), (getindex)(_c, 2)), (/)((getindex)(_c, 3), (getindex)(_c, 2)))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:452 =#
            return A
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:454 =#
        function rigid_body_updateB!(B, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:454 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:455 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:456 =#
            nzval = B.nzval
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:457 =#
            nzval[1] = (/)(1, (getindex)(_c, 1))
            nzval[2] = (/)(1, (getindex)(_c, 1))
            nzval[3] = (/)(1, (getindex)(_c, 1))
            nzval[4] = (/)(1, (getindex)(_c, 2))
            nzval[5] = (/)(1, (getindex)(_c, 3))
            nzval[6] = (/)(1, (getindex)(_c, 4))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:458 =#
            return B
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:460 =#
        function rigid_body_updateC!(C, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:460 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:461 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:462 =#
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:440 =#
                nzval = (C[1]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:441 =#
                nzval[1] = (/)(1, (getindex)(_c, 1))
                nzval[2] = (/)(1, (getindex)(_c, 1))
                nzval[3] = (/)(1, (getindex)(_c, 1))
                nzval[4] = (/)(1, (getindex)(_c, 1))
                nzval[5] = (/)(1, (getindex)(_c, 1))
                nzval[6] = (/)(1, (getindex)(_c, 1))
                nzval[7] = (/)(1, (getindex)(_c, 1))
                nzval[8] = (/)(1, (getindex)(_c, 1))
                nzval[9] = (/)(1, (getindex)(_c, 1))
                nzval[10] = (/)(1, (getindex)(_c, 1))
                nzval[11] = (/)(1, (getindex)(_c, 1))
                nzval[12] = (/)(1, (getindex)(_c, 1))
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:440 =#
                nzval = (C[2]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:441 =#
                nzval[1] = (/)(1, (getindex)(_c, 1))
                nzval[2] = (/)(1, (getindex)(_c, 1))
                nzval[3] = (/)(1, (getindex)(_c, 1))
                nzval[4] = (/)(1, (getindex)(_c, 1))
                nzval[5] = (/)(1, (getindex)(_c, 1))
                nzval[6] = (/)(1, (getindex)(_c, 1))
                nzval[7] = (/)(1, (getindex)(_c, 1))
                nzval[8] = (/)(1, (getindex)(_c, 1))
                nzval[9] = (/)(1, (getindex)(_c, 1))
                nzval[10] = (/)(1, (getindex)(_c, 1))
                nzval[11] = (/)(1, (getindex)(_c, 1))
                nzval[12] = (/)(1, (getindex)(_c, 1))
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:440 =#
                nzval = (C[3]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:441 =#
                nzval[1] = (/)(1, (getindex)(_c, 1))
                nzval[2] = (/)(1, (getindex)(_c, 1))
                nzval[3] = (/)(1, (getindex)(_c, 1))
                nzval[4] = (/)(1, (getindex)(_c, 1))
                nzval[5] = (/)(1, (getindex)(_c, 1))
                nzval[6] = (/)(1, (getindex)(_c, 1))
                nzval[7] = (/)(1, (getindex)(_c, 1))
                nzval[8] = (/)(1, (getindex)(_c, 1))
                nzval[9] = (/)(1, (getindex)(_c, 1))
                nzval[10] = (/)(1, (getindex)(_c, 1))
                nzval[11] = (/)(1, (getindex)(_c, 1))
                nzval[12] = (/)(1, (getindex)(_c, 1))
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:440 =#
                nzval = (C[4]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:441 =#
                nzval[1] = (/)(1, (getindex)(_c, 2))
                nzval[2] = (/)(1, (getindex)(_c, 2))
                nzval[3] = (/)(1, (getindex)(_c, 2))
                nzval[4] = (/)(1, (getindex)(_c, 2))
                nzval[5] = (/)(1, (getindex)(_c, 2))
                nzval[6] = (/)(1, (getindex)(_c, 2))
                nzval[7] = (/)(1, (getindex)(_c, 2))
                nzval[8] = (/)(1, (getindex)(_c, 2))
                nzval[9] = (/)(1, (getindex)(_c, 2))
                nzval[10] = (/)(1, (getindex)(_c, 2))
                nzval[11] = (/)(1, (getindex)(_c, 2))
                nzval[12] = (/)(1, (getindex)(_c, 2))
                nzval[13] = (/)(2, (getindex)(_c, 2))
                nzval[14] = (/)(1, (getindex)(_c, 2))
                nzval[15] = (/)(1, (getindex)(_c, 2))
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:440 =#
                nzval = (C[5]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:441 =#
                nzval[1] = (/)(1, (getindex)(_c, 3))
                nzval[2] = (/)(1, (getindex)(_c, 3))
                nzval[3] = (/)(1, (getindex)(_c, 3))
                nzval[4] = (/)(1, (getindex)(_c, 3))
                nzval[5] = (/)(1, (getindex)(_c, 3))
                nzval[6] = (/)(1, (getindex)(_c, 3))
                nzval[7] = (/)(1, (getindex)(_c, 3))
                nzval[8] = (/)(1, (getindex)(_c, 3))
                nzval[9] = (/)(1, (getindex)(_c, 3))
                nzval[10] = (/)(1, (getindex)(_c, 3))
                nzval[11] = (/)(1, (getindex)(_c, 3))
                nzval[12] = (/)(1, (getindex)(_c, 3))
                nzval[13] = (/)(1, (getindex)(_c, 3))
                nzval[14] = (/)(2, (getindex)(_c, 3))
                nzval[15] = (/)(1, (getindex)(_c, 3))
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:440 =#
                nzval = (C[6]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:441 =#
                nzval[1] = (/)(1, (getindex)(_c, 4))
                nzval[2] = (/)(1, (getindex)(_c, 4))
                nzval[3] = (/)(1, (getindex)(_c, 4))
                nzval[4] = (/)(1, (getindex)(_c, 4))
                nzval[5] = (/)(1, (getindex)(_c, 4))
                nzval[6] = (/)(1, (getindex)(_c, 4))
                nzval[7] = (/)(1, (getindex)(_c, 4))
                nzval[8] = (/)(1, (getindex)(_c, 4))
                nzval[9] = (/)(1, (getindex)(_c, 4))
                nzval[10] = (/)(1, (getindex)(_c, 4))
                nzval[11] = (/)(1, (getindex)(_c, 4))
                nzval[12] = (/)(1, (getindex)(_c, 4))
                nzval[13] = (/)(1, (getindex)(_c, 4))
                nzval[14] = (/)(1, (getindex)(_c, 4))
                nzval[15] = (/)(2, (getindex)(_c, 4))
            end
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:463 =#
            return C
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:465 =#
        function rigid_body_updateD!(D, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:465 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:466 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:467 =#
            nzval = D.nzval
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:468 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:469 =#
            return D
        end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:573 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:484 =#
        function rigid_body_genarrays()
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:484 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:485 =#
            n = 87
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:486 =#
            m = 6
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:487 =#
            A = SparseMatrixCSC(n, n, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 10, 10, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19, 19, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 28, 28, 28, 28, 29, 30, 31, 31, 32, 33, 34, 34, 34, 35, 36, 36, 37, 37], [1, 2, 3, 1, 2, 3, 1, 2, 3, 10, 11, 12, 7, 8, 9, 10, 11, 12, 4, 5, 6, 7, 8, 9, 4, 5, 6, 15, 14, 15, 13, 14, 13, 18, 17, 16], zeros(36))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:492 =#
            B = SparseMatrixCSC(n, m, [1, 2, 3, 4, 5, 6, 7], [13, 14, 15, 16, 17, 18], zeros(6))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:497 =#
            C = [begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:476 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13], [19, 20, 21, 22, 23, 24, 25, 26, 27, 73, 76, 79], zeros(12))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:476 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13], [28, 29, 30, 31, 32, 33, 34, 35, 36, 74, 77, 80], zeros(12))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:476 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13], [37, 38, 39, 40, 41, 42, 43, 44, 45, 75, 78, 81], zeros(12))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:476 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], [46, 47, 48, 49, 50, 51, 52, 53, 54, 73, 74, 75, 82, 83, 84], zeros(15))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:476 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], [55, 56, 57, 58, 59, 60, 61, 62, 63, 76, 77, 78, 83, 85, 86], zeros(15))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:476 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], [64, 65, 66, 67, 68, 69, 70, 71, 72, 79, 80, 81, 84, 86, 87], zeros(15))
                    end]
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:498 =#
            D = SparseVector(n, Int64[], zeros(0))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:502 =#
            return (A, B, C, D)
        end
    end
end