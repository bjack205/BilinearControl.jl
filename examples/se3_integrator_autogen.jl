begin
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:569 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:516 =#
        struct Se3IntegratorDynamics <: RD.ContinuousDynamics
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
        function Se3IntegratorDynamics(m, J1, J2, J3)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:523 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:524 =#
            (A, B, C, D) = se3_integrator_genarrays()
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:525 =#
            c = SA[m, J1, J2, J3]
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:526 =#
            se3_integrator_updateA!(A, c)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:527 =#
            se3_integrator_updateB!(B, c)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:528 =#
            se3_integrator_updateC!(C, c)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:529 =#
            se3_integrator_updateD!(D, c)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:530 =#
            Se3IntegratorDynamics(m, J1, J2, J3, A, B, C, D)
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:532 =#
        RD.state_dim(::Se3IntegratorDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:532 =#
                45
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:533 =#
        RD.control_dim(::Se3IntegratorDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:533 =#
                6
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:534 =#
        RD.default_diffmethod(::Se3IntegratorDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:534 =#
                RD.UserDefined()
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:535 =#
        RD.default_signature(::Se3IntegratorDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:535 =#
                RD.InPlace()
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:536 =#
        base_state_dim(::Se3IntegratorDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:536 =#
                18
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:538 =#
        getconstants(model::Se3IntegratorDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:538 =#
                SA[getfield(model, :m), getfield(model, :J1), getfield(model, :J2), getfield(model, :J3)]
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:540 =#
        expandstate!(model::Se3IntegratorDynamics, y, x) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:540 =#
                se3_integrator_expand!(y, x)
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:541 =#
        expandstate(model::Se3IntegratorDynamics, x) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:541 =#
                expandstate!(model, zeros(RD.state_dim(model)), x)
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:543 =#
        function RD.dynamics!(model::Se3IntegratorDynamics, xdot, x, u)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:543 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:544 =#
            c = getconstants(model)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:545 =#
            se3_integrator_dynamics!(xdot, x, u, c)
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:548 =#
        function RD.jacobian!(model::Se3IntegratorDynamics, J, y, x, u)
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
        BilinearControl.getA(model::Se3IntegratorDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:561 =#
                model.A
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:562 =#
        BilinearControl.getB(model::Se3IntegratorDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:562 =#
                model.B
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:563 =#
        BilinearControl.getC(model::Se3IntegratorDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:563 =#
                model.C
            end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:564 =#
        BilinearControl.getD(model::Se3IntegratorDynamics) = begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:564 =#
                model.D
            end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:570 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:401 =#
        function se3_integrator_expand!(y, x)
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
            y[19] = (*)((getindex)(_x, 4), (getindex)(_x, 16))
            y[20] = (*)((getindex)(_x, 5), (getindex)(_x, 16))
            y[21] = (*)((getindex)(_x, 6), (getindex)(_x, 16))
            y[22] = (*)((getindex)(_x, 7), (getindex)(_x, 16))
            y[23] = (*)((getindex)(_x, 8), (getindex)(_x, 16))
            y[24] = (*)((getindex)(_x, 9), (getindex)(_x, 16))
            y[25] = (*)((getindex)(_x, 10), (getindex)(_x, 16))
            y[26] = (*)((getindex)(_x, 11), (getindex)(_x, 16))
            y[27] = (*)((getindex)(_x, 12), (getindex)(_x, 16))
            y[28] = (*)((getindex)(_x, 4), (getindex)(_x, 17))
            y[29] = (*)((getindex)(_x, 5), (getindex)(_x, 17))
            y[30] = (*)((getindex)(_x, 6), (getindex)(_x, 17))
            y[31] = (*)((getindex)(_x, 7), (getindex)(_x, 17))
            y[32] = (*)((getindex)(_x, 8), (getindex)(_x, 17))
            y[33] = (*)((getindex)(_x, 9), (getindex)(_x, 17))
            y[34] = (*)((getindex)(_x, 10), (getindex)(_x, 17))
            y[35] = (*)((getindex)(_x, 11), (getindex)(_x, 17))
            y[36] = (*)((getindex)(_x, 12), (getindex)(_x, 17))
            y[37] = (*)((getindex)(_x, 4), (getindex)(_x, 18))
            y[38] = (*)((getindex)(_x, 5), (getindex)(_x, 18))
            y[39] = (*)((getindex)(_x, 6), (getindex)(_x, 18))
            y[40] = (*)((getindex)(_x, 7), (getindex)(_x, 18))
            y[41] = (*)((getindex)(_x, 8), (getindex)(_x, 18))
            y[42] = (*)((getindex)(_x, 9), (getindex)(_x, 18))
            y[43] = (*)((getindex)(_x, 10), (getindex)(_x, 18))
            y[44] = (*)((getindex)(_x, 11), (getindex)(_x, 18))
            y[45] = (*)((getindex)(_x, 12), (getindex)(_x, 18))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:404 =#
            return y
        end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:571 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:414 =#
        function se3_integrator_dynamics!(xdot, x, u, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:414 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:415 =#
            (_x, _u, _c) = (x, u, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:416 =#
            xdot[1] = (getindex)(_x, 13)
            xdot[2] = (getindex)(_x, 14)
            xdot[3] = (getindex)(_x, 15)
            xdot[4] = (+)((*)(-1, (getindex)(_x, 34)), (getindex)(_x, 40))
            xdot[5] = (+)((*)(-1, (getindex)(_x, 35)), (getindex)(_x, 41))
            xdot[6] = (+)((*)(-1, (getindex)(_x, 36)), (getindex)(_x, 42))
            xdot[7] = (+)((*)(-1, (getindex)(_x, 37)), (getindex)(_x, 25))
            xdot[8] = (+)((*)(-1, (getindex)(_x, 38)), (getindex)(_x, 26))
            xdot[9] = (+)((*)(-1, (getindex)(_x, 39)), (getindex)(_x, 27))
            xdot[10] = (+)((*)(-1, (getindex)(_x, 22)), (getindex)(_x, 28))
            xdot[11] = (+)((*)(-1, (getindex)(_x, 23)), (getindex)(_x, 29))
            xdot[12] = (+)((*)(-1, (getindex)(_x, 24)), (getindex)(_x, 30))
            xdot[13] = (/)((+)((+)((*)((getindex)(_u, 1), (getindex)(_x, 4)), (*)((getindex)(_u, 2), (getindex)(_x, 7))), (*)((getindex)(_u, 3), (getindex)(_x, 10))), (getindex)(_c, 1))
            xdot[14] = (/)((+)((+)((*)((getindex)(_u, 1), (getindex)(_x, 5)), (*)((getindex)(_u, 2), (getindex)(_x, 8))), (*)((getindex)(_u, 3), (getindex)(_x, 11))), (getindex)(_c, 1))
            xdot[15] = (/)((+)((+)((*)((getindex)(_u, 1), (getindex)(_x, 6)), (*)((getindex)(_u, 2), (getindex)(_x, 9))), (*)((getindex)(_u, 3), (getindex)(_x, 12))), (getindex)(_c, 1))
            xdot[16] = (/)((getindex)(_u, 4), (getindex)(_c, 2))
            xdot[17] = (/)((getindex)(_u, 5), (getindex)(_c, 3))
            xdot[18] = (/)((getindex)(_u, 6), (getindex)(_c, 4))
            xdot[19] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 4)), (getindex)(_c, 2))
            xdot[20] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 5)), (getindex)(_c, 2))
            xdot[21] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 6)), (getindex)(_c, 2))
            xdot[22] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 7)), (getindex)(_c, 2))
            xdot[23] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 8)), (getindex)(_c, 2))
            xdot[24] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 9)), (getindex)(_c, 2))
            xdot[25] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 10)), (getindex)(_c, 2))
            xdot[26] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 11)), (getindex)(_c, 2))
            xdot[27] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 12)), (getindex)(_c, 2))
            xdot[28] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 4)), (getindex)(_c, 3))
            xdot[29] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 5)), (getindex)(_c, 3))
            xdot[30] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 6)), (getindex)(_c, 3))
            xdot[31] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 7)), (getindex)(_c, 3))
            xdot[32] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 8)), (getindex)(_c, 3))
            xdot[33] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 9)), (getindex)(_c, 3))
            xdot[34] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 10)), (getindex)(_c, 3))
            xdot[35] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 11)), (getindex)(_c, 3))
            xdot[36] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 12)), (getindex)(_c, 3))
            xdot[37] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 4)), (getindex)(_c, 4))
            xdot[38] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 5)), (getindex)(_c, 4))
            xdot[39] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 6)), (getindex)(_c, 4))
            xdot[40] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 7)), (getindex)(_c, 4))
            xdot[41] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 8)), (getindex)(_c, 4))
            xdot[42] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 9)), (getindex)(_c, 4))
            xdot[43] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 10)), (getindex)(_c, 4))
            xdot[44] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 11)), (getindex)(_c, 4))
            xdot[45] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 12)), (getindex)(_c, 4))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:417 =#
            return
        end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:572 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:448 =#
        function se3_integrator_updateA!(A, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:448 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:449 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:450 =#
            nzval = A.nzval
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:451 =#
            nzval[1] = 1
            nzval[2] = 1
            nzval[3] = 1
            nzval[4] = -1
            nzval[5] = -1
            nzval[6] = -1
            nzval[7] = 1
            nzval[8] = 1
            nzval[9] = 1
            nzval[10] = 1
            nzval[11] = 1
            nzval[12] = 1
            nzval[13] = -1
            nzval[14] = -1
            nzval[15] = -1
            nzval[16] = -1
            nzval[17] = -1
            nzval[18] = -1
            nzval[19] = 1
            nzval[20] = 1
            nzval[21] = 1
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:452 =#
            return A
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:454 =#
        function se3_integrator_updateB!(B, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:454 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:455 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:456 =#
            nzval = B.nzval
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:457 =#
            nzval[1] = (/)(1, (getindex)(_c, 2))
            nzval[2] = (/)(1, (getindex)(_c, 3))
            nzval[3] = (/)(1, (getindex)(_c, 4))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:458 =#
            return B
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:460 =#
        function se3_integrator_updateC!(C, constants)
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
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:440 =#
                nzval = (C[2]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:441 =#
                nzval[1] = (/)(1, (getindex)(_c, 1))
                nzval[2] = (/)(1, (getindex)(_c, 1))
                nzval[3] = (/)(1, (getindex)(_c, 1))
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:440 =#
                nzval = (C[3]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:441 =#
                nzval[1] = (/)(1, (getindex)(_c, 1))
                nzval[2] = (/)(1, (getindex)(_c, 1))
                nzval[3] = (/)(1, (getindex)(_c, 1))
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
            end
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:463 =#
            return C
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:465 =#
        function se3_integrator_updateD!(D, constants)
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
        function se3_integrator_genarrays()
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:484 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:485 =#
            n = 45
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:486 =#
            m = 6
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:487 =#
            A = SparseMatrixCSC(n, n, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 13, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 22, 22, 22], [1, 2, 3, 10, 11, 12, 7, 8, 9, 10, 11, 12, 4, 5, 6, 7, 8, 9, 4, 5, 6], zeros(21))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:492 =#
            B = SparseMatrixCSC(n, m, [1, 1, 1, 1, 2, 3, 4], [16, 17, 18], zeros(3))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:497 =#
            C = [begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:476 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [13, 14, 15], zeros(3))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:476 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [13, 14, 15], zeros(3))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:476 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [13, 14, 15], zeros(3))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:476 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], [19, 20, 21, 22, 23, 24, 25, 26, 27], zeros(9))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:476 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], [28, 29, 30, 31, 32, 33, 34, 35, 36], zeros(9))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:476 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], [37, 38, 39, 40, 41, 42, 43, 44, 45], zeros(9))
                    end]
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:498 =#
            D = SparseVector(n, Int64[], zeros(0))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:502 =#
            return (A, B, C, D)
        end
    end
end