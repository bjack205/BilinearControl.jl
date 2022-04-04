begin
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:485 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:377 =#
        function se3_angvel_expand!(y, x)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:377 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:378 =#
            _x = x
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:379 =#
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
            y[11] = (^)((getindex)(_x, 4), 2)
            y[12] = (*)((getindex)(_x, 4), (getindex)(_x, 5))
            y[13] = (*)((getindex)(_x, 4), (getindex)(_x, 6))
            y[14] = (*)((getindex)(_x, 4), (getindex)(_x, 7))
            y[15] = (^)((getindex)(_x, 5), 2)
            y[16] = (*)((getindex)(_x, 5), (getindex)(_x, 6))
            y[17] = (*)((getindex)(_x, 5), (getindex)(_x, 7))
            y[18] = (^)((getindex)(_x, 6), 2)
            y[19] = (*)((getindex)(_x, 6), (getindex)(_x, 7))
            y[20] = (^)((getindex)(_x, 7), 2)
            y[21] = (*)((^)((getindex)(_x, 4), 2), (getindex)(_x, 8))
            y[22] = (*)((^)((getindex)(_x, 4), 2), (getindex)(_x, 9))
            y[23] = (*)((^)((getindex)(_x, 4), 2), (getindex)(_x, 10))
            y[24] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 5)), (getindex)(_x, 8))
            y[25] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 5)), (getindex)(_x, 9))
            y[26] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 5)), (getindex)(_x, 10))
            y[27] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 6)), (getindex)(_x, 8))
            y[28] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 6)), (getindex)(_x, 9))
            y[29] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 6)), (getindex)(_x, 10))
            y[30] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 7)), (getindex)(_x, 8))
            y[31] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 7)), (getindex)(_x, 9))
            y[32] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 7)), (getindex)(_x, 10))
            y[33] = (*)((^)((getindex)(_x, 5), 2), (getindex)(_x, 8))
            y[34] = (*)((^)((getindex)(_x, 5), 2), (getindex)(_x, 9))
            y[35] = (*)((^)((getindex)(_x, 5), 2), (getindex)(_x, 10))
            y[36] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 6)), (getindex)(_x, 8))
            y[37] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 6)), (getindex)(_x, 9))
            y[38] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 6)), (getindex)(_x, 10))
            y[39] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 7)), (getindex)(_x, 8))
            y[40] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 7)), (getindex)(_x, 9))
            y[41] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 7)), (getindex)(_x, 10))
            y[42] = (*)((^)((getindex)(_x, 6), 2), (getindex)(_x, 8))
            y[43] = (*)((^)((getindex)(_x, 6), 2), (getindex)(_x, 9))
            y[44] = (*)((^)((getindex)(_x, 6), 2), (getindex)(_x, 10))
            y[45] = (*)((*)((getindex)(_x, 6), (getindex)(_x, 7)), (getindex)(_x, 8))
            y[46] = (*)((*)((getindex)(_x, 6), (getindex)(_x, 7)), (getindex)(_x, 9))
            y[47] = (*)((*)((getindex)(_x, 6), (getindex)(_x, 7)), (getindex)(_x, 10))
            y[48] = (*)((^)((getindex)(_x, 7), 2), (getindex)(_x, 8))
            y[49] = (*)((^)((getindex)(_x, 7), 2), (getindex)(_x, 9))
            y[50] = (*)((^)((getindex)(_x, 7), 2), (getindex)(_x, 10))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:380 =#
            return y
        end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:486 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:390 =#
        function se3_angvel_dynamics!(xdot, x, u, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:390 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:391 =#
            (_x, _u, _c) = (x, u, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:392 =#
            xdot[1] = (+)((+)((+)((+)((+)((+)((+)((*)(2, (getindex)(_x, 29)), (*)(-1, (getindex)(_x, 42))), (*)(-2, (getindex)(_x, 31))), (*)(2, (getindex)(_x, 37))), (*)(2, (getindex)(_x, 41))), (*)(-1, (getindex)(_x, 48))), (getindex)(_x, 21)), (getindex)(_x, 33))
            xdot[2] = (+)((+)((+)((+)((+)((+)((+)((*)(-2, (getindex)(_x, 26)), (*)(-1, (getindex)(_x, 34))), (*)(-1, (getindex)(_x, 49))), (*)(2, (getindex)(_x, 30))), (*)(2, (getindex)(_x, 36))), (*)(2, (getindex)(_x, 47))), (getindex)(_x, 22)), (getindex)(_x, 43))
            xdot[3] = (+)((+)((+)((+)((+)((+)((+)((*)(-1, (getindex)(_x, 35)), (*)(-1, (getindex)(_x, 44))), (*)(2, (getindex)(_x, 25))), (*)(-2, (getindex)(_x, 27))), (*)(2, (getindex)(_x, 39))), (*)(2, (getindex)(_x, 46))), (getindex)(_x, 23)), (getindex)(_x, 50))
            xdot[4] = (+)((+)((*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 5)), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 6))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 7)))
            xdot[5] = (+)((+)((*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 4)), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 6))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 7)))
            xdot[6] = (+)((+)((*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 7)), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 4))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 5)))
            xdot[7] = (+)((+)((*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 6)), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 5))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 4)))
            xdot[8] = (+)((+)((/)((getindex)(_u, 1), (getindex)(_c, 1)), (*)((getindex)(_u, 6), (getindex)(_x, 9))), (*)((*)(-1, (getindex)(_u, 5)), (getindex)(_x, 10)))
            xdot[9] = (+)((+)((/)((getindex)(_u, 2), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 10))), (*)((*)(-1, (getindex)(_u, 6)), (getindex)(_x, 8)))
            xdot[10] = (+)((+)((/)((getindex)(_u, 3), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 8))), (*)((*)(-1, (getindex)(_u, 4)), (getindex)(_x, 9)))
            xdot[11] = (+)((+)((*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 12)), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 13))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 14)))
            xdot[12] = (+)((+)((+)((+)((+)((*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 11)), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 15))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 14))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 16))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 13))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 17)))
            xdot[13] = (+)((+)((+)((+)((+)((*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 14)), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 16))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 11))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 12))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 18))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 19)))
            xdot[14] = (+)((+)((+)((+)((+)((*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 12)), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 11))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 13))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 17))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 19))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 20)))
            xdot[15] = (+)((+)((*)((getindex)(_u, 4), (getindex)(_x, 12)), (*)((getindex)(_u, 6), (getindex)(_x, 16))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 17)))
            xdot[16] = (+)((+)((+)((+)((+)((*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 12)), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 13))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 17))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 15))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 18))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 19)))
            xdot[17] = (+)((+)((+)((+)((+)((*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 14)), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 16))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 12))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 15))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 19))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 20)))
            xdot[18] = (+)((+)((*)((getindex)(_u, 5), (getindex)(_x, 13)), (*)((getindex)(_u, 4), (getindex)(_x, 19))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 16)))
            xdot[19] = (+)((+)((+)((+)((+)((*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 14)), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 16))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 13))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 17))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 18))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 20)))
            xdot[20] = (+)((+)((*)((getindex)(_u, 5), (getindex)(_x, 17)), (*)((getindex)(_u, 6), (getindex)(_x, 14))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 19)))
            xdot[21] = (+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 11)), (getindex)(_c, 1)), (*)((getindex)(_u, 6), (getindex)(_x, 22))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 23))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 24))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 27))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 30)))
            xdot[22] = (+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 11)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 23))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 21))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 25))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 28))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 31)))
            xdot[23] = (+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 11)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 21))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 22))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 26))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 29))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 32)))
            xdot[24] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 12)), (getindex)(_c, 1)), (*)((getindex)(_u, 6), (getindex)(_x, 25))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 21))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 26))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 30))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 27))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 33))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 36))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 39)))
            xdot[25] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 12)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 26))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 22))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 24))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 31))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 34))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 37))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 28))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 40)))
            xdot[26] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 12)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 24))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 23))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 32))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 29))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 25))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 35))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 38))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 41)))
            xdot[27] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 13)), (getindex)(_c, 1)), (*)((getindex)(_u, 6), (getindex)(_x, 28))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 21))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 24))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 29))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 30))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 36))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 42))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 45)))
            xdot[28] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 13)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 29))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 22))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 25))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 27))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 31))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 37))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 43))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 46)))
            xdot[29] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 13)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 27))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 23))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 32))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 38))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 44))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 28))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 26))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 47)))
            xdot[30] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 14)), (getindex)(_c, 1)), (*)((getindex)(_u, 6), (getindex)(_x, 31))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 24))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 21))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 27))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 32))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 39))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 45))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 48)))
            xdot[31] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 14)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 32))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 22))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 28))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 25))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 30))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 40))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 46))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 49)))
            xdot[32] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 14)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 30))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 23))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 29))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 26))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 31))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 41))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 47))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 50)))
            xdot[33] = (+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 15)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 24))), (*)((getindex)(_u, 6), (getindex)(_x, 34))), (*)((getindex)(_u, 6), (getindex)(_x, 36))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 35))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 39)))
            xdot[34] = (+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 15)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 25))), (*)((getindex)(_u, 4), (getindex)(_x, 35))), (*)((getindex)(_u, 6), (getindex)(_x, 37))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 33))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 40)))
            xdot[35] = (+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 15)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 26))), (*)((getindex)(_u, 5), (getindex)(_x, 33))), (*)((getindex)(_u, 6), (getindex)(_x, 38))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 34))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 41)))
            xdot[36] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 16)), (getindex)(_c, 1)), (*)((getindex)(_u, 6), (getindex)(_x, 37))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 24))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 27))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 33))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 38))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 39))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 45))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 42)))
            xdot[37] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 16)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 38))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 28))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 25))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 34))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 36))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 40))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 46))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 43)))
            xdot[38] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 16)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 36))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 29))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 26))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 35))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 37))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 41))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 47))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 44)))
            xdot[39] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 17)), (getindex)(_c, 1)), (*)((getindex)(_u, 6), (getindex)(_x, 40))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 30))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 36))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 33))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 48))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 41))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 24))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 45)))
            xdot[40] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 17)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 41))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 25))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 31))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 34))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 37))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 46))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 39))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 49)))
            xdot[41] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 17)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 39))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 32))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 38))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 40))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 26))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 35))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 47))), (*)((*)(-1//2, (getindex)(_u, 5)), (getindex)(_x, 50)))
            xdot[42] = (+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 18)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 45))), (*)((getindex)(_u, 5), (getindex)(_x, 27))), (*)((getindex)(_u, 6), (getindex)(_x, 43))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 36))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 44)))
            xdot[43] = (+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 18)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 28))), (*)((getindex)(_u, 4), (getindex)(_x, 44))), (*)((getindex)(_u, 4), (getindex)(_x, 46))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 37))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 42)))
            xdot[44] = (+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 18)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 29))), (*)((getindex)(_u, 5), (getindex)(_x, 42))), (*)((getindex)(_u, 4), (getindex)(_x, 47))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 43))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 38)))
            xdot[45] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 19)), (getindex)(_c, 1)), (*)((getindex)(_u, 6), (getindex)(_x, 46))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 42))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 30))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 36))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 47))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 27))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 39))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 48)))
            xdot[46] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 19)), (getindex)(_c, 1)), (*)((getindex)(_u, 4), (getindex)(_x, 47))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 31))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 37))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 28))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 40))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 43))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 45))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 49)))
            xdot[47] = (+)((+)((+)((+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 19)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 45))), (*)((*)(1//2, (getindex)(_u, 6)), (getindex)(_x, 29))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 32))), (*)((*)(1//2, (getindex)(_u, 5)), (getindex)(_x, 38))), (*)((*)(-1//2, (getindex)(_u, 6)), (getindex)(_x, 41))), (*)((*)(-1//2, (getindex)(_u, 4)), (getindex)(_x, 44))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 46))), (*)((*)(1//2, (getindex)(_u, 4)), (getindex)(_x, 50)))
            xdot[48] = (+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 20)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 39))), (*)((getindex)(_u, 6), (getindex)(_x, 30))), (*)((getindex)(_u, 6), (getindex)(_x, 49))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 45))), (*)((*)(-1//1, (getindex)(_u, 5)), (getindex)(_x, 50)))
            xdot[49] = (+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 20)), (getindex)(_c, 1)), (*)((getindex)(_u, 5), (getindex)(_x, 40))), (*)((getindex)(_u, 4), (getindex)(_x, 50))), (*)((getindex)(_u, 6), (getindex)(_x, 31))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 46))), (*)((*)(-1//1, (getindex)(_u, 6)), (getindex)(_x, 48)))
            xdot[50] = (+)((+)((+)((+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 20)), (getindex)(_c, 1)), (*)((getindex)(_u, 6), (getindex)(_x, 32))), (*)((getindex)(_u, 5), (getindex)(_x, 41))), (*)((getindex)(_u, 5), (getindex)(_x, 48))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 47))), (*)((*)(-1//1, (getindex)(_u, 4)), (getindex)(_x, 49)))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:393 =#
            return
        end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:487 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:424 =#
        function se3_angvel_updateA!(A, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:424 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:425 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:426 =#
            nzval = A.nzval
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:427 =#
            nzval[1] = 1
            nzval[2] = 1
            nzval[3] = 1
            nzval[4] = 2
            nzval[5] = -2
            nzval[6] = -2
            nzval[7] = 2
            nzval[8] = 2
            nzval[9] = -2
            nzval[10] = 1
            nzval[11] = -1
            nzval[12] = -1
            nzval[13] = 2
            nzval[14] = 2
            nzval[15] = 2
            nzval[16] = 2
            nzval[17] = -1
            nzval[18] = 1
            nzval[19] = -1
            nzval[20] = 2
            nzval[21] = 2
            nzval[22] = -1
            nzval[23] = -1
            nzval[24] = 1
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:428 =#
            return A
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:430 =#
        function se3_angvel_updateB!(B, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:430 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:431 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:432 =#
            nzval = B.nzval
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:433 =#
            nzval[1] = (/)(1, (getindex)(_c, 1))
            nzval[2] = (/)(1, (getindex)(_c, 1))
            nzval[3] = (/)(1, (getindex)(_c, 1))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:434 =#
            return B
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:436 =#
        function se3_angvel_updateC!(C, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:436 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:437 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:438 =#
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:416 =#
                nzval = (C[1]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:417 =#
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
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:416 =#
                nzval = (C[2]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:417 =#
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
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:416 =#
                nzval = (C[3]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:417 =#
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
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:416 =#
                nzval = (C[4]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:417 =#
                nzval[1] = 1//2
                nzval[2] = -1//2
                nzval[3] = -1//2
                nzval[4] = 1//2
                nzval[5] = -1
                nzval[6] = 1
                nzval[7] = 1//2
                nzval[8] = -1//1
                nzval[9] = 1
                nzval[10] = -1//2
                nzval[11] = 1//2
                nzval[12] = 1//2
                nzval[13] = 1//2
                nzval[14] = -1//2
                nzval[15] = -1//2
                nzval[16] = -1//2
                nzval[17] = -1//2
                nzval[18] = 1//2
                nzval[19] = -1//2
                nzval[20] = 1
                nzval[21] = -1//1
                nzval[22] = 1//2
                nzval[23] = 1//2
                nzval[24] = -1//1
                nzval[25] = 1//2
                nzval[26] = 1
                nzval[27] = 1//2
                nzval[28] = -1//1
                nzval[29] = 1
                nzval[30] = -1//1
                nzval[31] = -1//1
                nzval[32] = 1
                nzval[33] = -1//1
                nzval[34] = 1
                nzval[35] = 1
                nzval[36] = -1//2
                nzval[37] = 1//2
                nzval[38] = -1//1
                nzval[39] = -1//2
                nzval[40] = 1//2
                nzval[41] = 1
                nzval[42] = -1//2
                nzval[43] = 1//2
                nzval[44] = 1//2
                nzval[45] = 1//2
                nzval[46] = 1//2
                nzval[47] = -1//1
                nzval[48] = 1//2
                nzval[49] = 1//2
                nzval[50] = 1
                nzval[51] = 1//2
                nzval[52] = -1//2
                nzval[53] = -1//2
                nzval[54] = -1//1
                nzval[55] = -1//2
                nzval[56] = 1
                nzval[57] = -1//2
                nzval[58] = -1//2
                nzval[59] = -1//2
                nzval[60] = -1//1
                nzval[61] = -1//2
                nzval[62] = -1//2
                nzval[63] = 1
                nzval[64] = -1//2
                nzval[65] = -1//2
                nzval[66] = 1//2
                nzval[67] = -1//2
                nzval[68] = 1//2
                nzval[69] = -1//1
                nzval[70] = -1//2
                nzval[71] = 1//2
                nzval[72] = 1
                nzval[73] = -1//2
                nzval[74] = -1//1
                nzval[75] = -1//2
                nzval[76] = 1
                nzval[77] = -1//2
                nzval[78] = 1
                nzval[79] = -1//1
                nzval[80] = 1
                nzval[81] = -1//1
                nzval[82] = -1//1
                nzval[83] = 1
                nzval[84] = 1
                nzval[85] = -1//1
                nzval[86] = 1//2
                nzval[87] = 1//2
                nzval[88] = -1//1
                nzval[89] = 1//2
                nzval[90] = 1
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:416 =#
                nzval = (C[5]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:417 =#
                nzval[1] = 1//2
                nzval[2] = 1//2
                nzval[3] = -1//2
                nzval[4] = -1//2
                nzval[5] = 1
                nzval[6] = -1
                nzval[7] = 1//2
                nzval[8] = 1//2
                nzval[9] = 1//2
                nzval[10] = -1//1
                nzval[11] = 1
                nzval[12] = -1//2
                nzval[13] = 1//2
                nzval[14] = 1//2
                nzval[15] = -1//2
                nzval[16] = 1//2
                nzval[17] = -1//1
                nzval[18] = 1
                nzval[19] = -1//2
                nzval[20] = -1//2
                nzval[21] = -1//2
                nzval[22] = -1//2
                nzval[23] = 1
                nzval[24] = 1//2
                nzval[25] = 1//2
                nzval[26] = -1//1
                nzval[27] = 1//2
                nzval[28] = 1
                nzval[29] = 1//2
                nzval[30] = 1//2
                nzval[31] = 1//2
                nzval[32] = 1//2
                nzval[33] = -1//1
                nzval[34] = 1//2
                nzval[35] = 1//2
                nzval[36] = -1//1
                nzval[37] = 1
                nzval[38] = 1
                nzval[39] = -1//1
                nzval[40] = 1
                nzval[41] = -1//1
                nzval[42] = -1//1
                nzval[43] = 1
                nzval[44] = -1//2
                nzval[45] = 1
                nzval[46] = 1//2
                nzval[47] = -1//2
                nzval[48] = 1//2
                nzval[49] = -1//2
                nzval[50] = -1//1
                nzval[51] = 1//2
                nzval[52] = 1
                nzval[53] = 1//2
                nzval[54] = 1//2
                nzval[55] = -1//1
                nzval[56] = 1//2
                nzval[57] = -1//2
                nzval[58] = 1
                nzval[59] = 1//2
                nzval[60] = -1//2
                nzval[61] = 1//2
                nzval[62] = -1//2
                nzval[63] = -1//1
                nzval[64] = 1//2
                nzval[65] = -1//1
                nzval[66] = 1
                nzval[67] = 1
                nzval[68] = -1//1
                nzval[69] = 1
                nzval[70] = -1//1
                nzval[71] = -1//1
                nzval[72] = 1
                nzval[73] = -1//2
                nzval[74] = 1
                nzval[75] = -1//2
                nzval[76] = -1//2
                nzval[77] = -1//1
                nzval[78] = -1//2
                nzval[79] = -1//2
                nzval[80] = 1
                nzval[81] = -1//2
                nzval[82] = -1//2
                nzval[83] = -1//2
                nzval[84] = -1//2
                nzval[85] = -1//1
                nzval[86] = -1//2
                nzval[87] = 1
                nzval[88] = -1//2
                nzval[89] = -1//2
                nzval[90] = -1//1
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:416 =#
                nzval = (C[6]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:417 =#
                nzval[1] = 1//2
                nzval[2] = -1//2
                nzval[3] = 1//2
                nzval[4] = -1//2
                nzval[5] = -1
                nzval[6] = 1
                nzval[7] = 1//2
                nzval[8] = -1//2
                nzval[9] = 1//2
                nzval[10] = 1//2
                nzval[11] = 1//2
                nzval[12] = -1//1
                nzval[13] = 1
                nzval[14] = -1//2
                nzval[15] = 1
                nzval[16] = -1//1
                nzval[17] = -1//2
                nzval[18] = -1//2
                nzval[19] = 1//2
                nzval[20] = -1//2
                nzval[21] = 1//2
                nzval[22] = -1//2
                nzval[23] = -1//1
                nzval[24] = 1//2
                nzval[25] = 1
                nzval[26] = 1//2
                nzval[27] = 1//2
                nzval[28] = -1//1
                nzval[29] = -1//2
                nzval[30] = 1//2
                nzval[31] = 1
                nzval[32] = -1//2
                nzval[33] = 1//2
                nzval[34] = -1//2
                nzval[35] = 1//2
                nzval[36] = 1//2
                nzval[37] = -1//1
                nzval[38] = 1//2
                nzval[39] = 1//2
                nzval[40] = 1
                nzval[41] = 1//2
                nzval[42] = 1//2
                nzval[43] = 1//2
                nzval[44] = -1//1
                nzval[45] = -1//1
                nzval[46] = 1
                nzval[47] = -1//1
                nzval[48] = 1
                nzval[49] = 1
                nzval[50] = -1//1
                nzval[51] = 1
                nzval[52] = -1//1
                nzval[53] = -1//2
                nzval[54] = 1
                nzval[55] = -1//2
                nzval[56] = -1//2
                nzval[57] = 1
                nzval[58] = -1//1
                nzval[59] = -1//1
                nzval[60] = 1
                nzval[61] = 1
                nzval[62] = -1//1
                nzval[63] = 1
                nzval[64] = -1//1
                nzval[65] = -1//2
                nzval[66] = -1//1
                nzval[67] = -1//2
                nzval[68] = -1//2
                nzval[69] = 1
                nzval[70] = -1//2
                nzval[71] = -1//2
                nzval[72] = -1//2
                nzval[73] = 1//2
                nzval[74] = -1//1
                nzval[75] = 1//2
                nzval[76] = 1
                nzval[77] = 1//2
                nzval[78] = -1//2
                nzval[79] = 1//2
                nzval[80] = -1//1
                nzval[81] = -1//2
                nzval[82] = 1//2
                nzval[83] = 1
                nzval[84] = -1//2
                nzval[85] = 1//2
                nzval[86] = -1//2
                nzval[87] = -1//1
                nzval[88] = -1//2
                nzval[89] = 1
                nzval[90] = -1//2
            end
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:439 =#
            return C
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:441 =#
        function se3_angvel_updateD!(D, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:441 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:442 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:443 =#
            nzval = D.nzval
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:444 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:445 =#
            return D
        end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:488 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:460 =#
        function se3_angvel_genarrays()
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:460 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:461 =#
            n = 50
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:462 =#
            m = 6
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:463 =#
            A = SparseMatrixCSC(n, n, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25], [1, 2, 3, 3, 2, 3, 1, 2, 1, 1, 2, 3, 2, 1, 3, 1, 1, 2, 3, 3, 2, 1, 2, 3], zeros(24))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:468 =#
            B = SparseMatrixCSC(n, m, [1, 2, 3, 4, 4, 4, 4], [8, 9, 10], zeros(3))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:473 =#
            C = [begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:452 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11], [21, 24, 27, 30, 33, 36, 39, 42, 45, 48], zeros(10))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:452 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11], [22, 25, 28, 31, 34, 37, 40, 43, 46, 49], zeros(10))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:452 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11], [23, 26, 29, 32, 35, 38, 41, 44, 47, 50], zeros(10))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:452 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 23, 24, 26, 28, 30, 33, 36, 38, 41, 44, 46, 49, 52, 53, 55, 57, 59, 62, 65, 67, 70, 73, 74, 76, 78, 80, 83, 86, 87, 89, 91], [5, 4, 7, 6, 10, 9, 12, 11, 15, 14, 16, 13, 17, 12, 13, 17, 14, 16, 19, 18, 20, 19, 24, 23, 25, 22, 26, 21, 33, 22, 26, 34, 23, 25, 35, 30, 36, 29, 31, 37, 28, 32, 38, 27, 39, 28, 32, 40, 29, 31, 41, 24, 25, 35, 26, 34, 27, 39, 28, 38, 40, 29, 37, 41, 30, 36, 31, 37, 41, 32, 38, 40, 45, 44, 46, 43, 47, 42, 48, 43, 47, 49, 44, 46, 50, 45, 46, 50, 47, 49], zeros(90))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:452 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 23, 25, 26, 28, 31, 33, 36, 39, 41, 44, 47, 49, 52, 54, 55, 57, 60, 62, 65, 68, 70, 73, 75, 76, 78, 81, 83, 86, 88, 89, 91], [6, 7, 4, 5, 10, 8, 13, 14, 16, 11, 18, 12, 19, 17, 12, 19, 15, 20, 13, 14, 16, 17, 23, 27, 28, 21, 29, 26, 30, 36, 31, 37, 24, 32, 38, 21, 29, 42, 22, 43, 23, 27, 44, 24, 32, 45, 25, 46, 26, 30, 47, 35, 39, 40, 33, 41, 24, 38, 45, 25, 46, 26, 36, 47, 33, 41, 48, 34, 49, 35, 39, 50, 27, 44, 28, 29, 42, 30, 36, 47, 31, 37, 32, 38, 45, 39, 50, 40, 41, 48], zeros(90))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:452 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 23, 25, 27, 28, 31, 34, 36, 39, 42, 44, 47, 50, 52, 54, 56, 57, 60, 63, 65, 68, 71, 73, 75, 77, 78, 81, 84, 86, 88, 90, 91], [7, 6, 5, 4, 9, 8, 14, 13, 17, 12, 19, 11, 20, 16, 15, 18, 12, 19, 16, 13, 17, 14, 22, 30, 21, 31, 32, 25, 27, 39, 24, 28, 40, 29, 41, 24, 28, 45, 25, 27, 46, 26, 47, 21, 31, 48, 22, 30, 49, 23, 50, 34, 36, 33, 37, 38, 33, 37, 42, 34, 36, 43, 35, 44, 24, 40, 45, 25, 39, 46, 26, 47, 36, 43, 37, 42, 38, 27, 39, 46, 28, 40, 45, 29, 41, 30, 49, 31, 48, 32], zeros(90))
                    end]
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:474 =#
            D = SparseVector(n, Int64[], zeros(0))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:478 =#
            return (A, B, C, D)
        end
    end
end