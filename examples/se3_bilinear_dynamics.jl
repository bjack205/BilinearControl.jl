begin
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:415 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:307 =#
        function se3_expand!(y, x)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:307 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:308 =#
            _x = x
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:309 =#
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
            y[14] = (^)((getindex)(_x, 4), 2)
            y[15] = (*)((getindex)(_x, 4), (getindex)(_x, 5))
            y[16] = (*)((getindex)(_x, 4), (getindex)(_x, 6))
            y[17] = (*)((getindex)(_x, 4), (getindex)(_x, 7))
            y[18] = (^)((getindex)(_x, 5), 2)
            y[19] = (*)((getindex)(_x, 5), (getindex)(_x, 6))
            y[20] = (*)((getindex)(_x, 5), (getindex)(_x, 7))
            y[21] = (^)((getindex)(_x, 6), 2)
            y[22] = (*)((getindex)(_x, 6), (getindex)(_x, 7))
            y[23] = (^)((getindex)(_x, 7), 2)
            y[24] = (^)((getindex)(_x, 11), 2)
            y[25] = (*)((getindex)(_x, 11), (getindex)(_x, 12))
            y[26] = (*)((getindex)(_x, 11), (getindex)(_x, 13))
            y[27] = (^)((getindex)(_x, 12), 2)
            y[28] = (*)((getindex)(_x, 12), (getindex)(_x, 13))
            y[29] = (^)((getindex)(_x, 13), 2)
            y[30] = (*)((getindex)(_x, 4), (getindex)(_x, 11))
            y[31] = (*)((getindex)(_x, 5), (getindex)(_x, 11))
            y[32] = (*)((getindex)(_x, 6), (getindex)(_x, 11))
            y[33] = (*)((getindex)(_x, 7), (getindex)(_x, 11))
            y[34] = (*)((getindex)(_x, 4), (getindex)(_x, 12))
            y[35] = (*)((getindex)(_x, 5), (getindex)(_x, 12))
            y[36] = (*)((getindex)(_x, 6), (getindex)(_x, 12))
            y[37] = (*)((getindex)(_x, 7), (getindex)(_x, 12))
            y[38] = (*)((getindex)(_x, 4), (getindex)(_x, 13))
            y[39] = (*)((getindex)(_x, 5), (getindex)(_x, 13))
            y[40] = (*)((getindex)(_x, 6), (getindex)(_x, 13))
            y[41] = (*)((getindex)(_x, 7), (getindex)(_x, 13))
            y[42] = (*)((getindex)(_x, 8), (getindex)(_x, 11))
            y[43] = (*)((getindex)(_x, 9), (getindex)(_x, 11))
            y[44] = (*)((getindex)(_x, 10), (getindex)(_x, 11))
            y[45] = (*)((getindex)(_x, 8), (getindex)(_x, 12))
            y[46] = (*)((getindex)(_x, 9), (getindex)(_x, 12))
            y[47] = (*)((getindex)(_x, 10), (getindex)(_x, 12))
            y[48] = (*)((getindex)(_x, 8), (getindex)(_x, 13))
            y[49] = (*)((getindex)(_x, 9), (getindex)(_x, 13))
            y[50] = (*)((getindex)(_x, 10), (getindex)(_x, 13))
            y[51] = (*)((^)((getindex)(_x, 11), 2), (getindex)(_x, 4))
            y[52] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 11)), (getindex)(_x, 12))
            y[53] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 11)), (getindex)(_x, 13))
            y[54] = (*)((^)((getindex)(_x, 12), 2), (getindex)(_x, 4))
            y[55] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 12)), (getindex)(_x, 13))
            y[56] = (*)((^)((getindex)(_x, 13), 2), (getindex)(_x, 4))
            y[57] = (*)((^)((getindex)(_x, 11), 2), (getindex)(_x, 5))
            y[58] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 11)), (getindex)(_x, 12))
            y[59] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 11)), (getindex)(_x, 13))
            y[60] = (*)((^)((getindex)(_x, 12), 2), (getindex)(_x, 5))
            y[61] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 12)), (getindex)(_x, 13))
            y[62] = (*)((^)((getindex)(_x, 13), 2), (getindex)(_x, 5))
            y[63] = (*)((^)((getindex)(_x, 11), 2), (getindex)(_x, 6))
            y[64] = (*)((*)((getindex)(_x, 6), (getindex)(_x, 11)), (getindex)(_x, 12))
            y[65] = (*)((*)((getindex)(_x, 6), (getindex)(_x, 11)), (getindex)(_x, 13))
            y[66] = (*)((^)((getindex)(_x, 12), 2), (getindex)(_x, 6))
            y[67] = (*)((*)((getindex)(_x, 6), (getindex)(_x, 12)), (getindex)(_x, 13))
            y[68] = (*)((^)((getindex)(_x, 13), 2), (getindex)(_x, 6))
            y[69] = (*)((^)((getindex)(_x, 11), 2), (getindex)(_x, 7))
            y[70] = (*)((*)((getindex)(_x, 7), (getindex)(_x, 11)), (getindex)(_x, 12))
            y[71] = (*)((*)((getindex)(_x, 7), (getindex)(_x, 11)), (getindex)(_x, 13))
            y[72] = (*)((^)((getindex)(_x, 12), 2), (getindex)(_x, 7))
            y[73] = (*)((*)((getindex)(_x, 7), (getindex)(_x, 12)), (getindex)(_x, 13))
            y[74] = (*)((^)((getindex)(_x, 13), 2), (getindex)(_x, 7))
            y[75] = (*)((^)((getindex)(_x, 11), 2), (getindex)(_x, 8))
            y[76] = (*)((*)((getindex)(_x, 8), (getindex)(_x, 11)), (getindex)(_x, 12))
            y[77] = (*)((*)((getindex)(_x, 8), (getindex)(_x, 11)), (getindex)(_x, 13))
            y[78] = (*)((^)((getindex)(_x, 12), 2), (getindex)(_x, 8))
            y[79] = (*)((*)((getindex)(_x, 8), (getindex)(_x, 12)), (getindex)(_x, 13))
            y[80] = (*)((^)((getindex)(_x, 13), 2), (getindex)(_x, 8))
            y[81] = (*)((^)((getindex)(_x, 11), 2), (getindex)(_x, 9))
            y[82] = (*)((*)((getindex)(_x, 9), (getindex)(_x, 11)), (getindex)(_x, 12))
            y[83] = (*)((*)((getindex)(_x, 9), (getindex)(_x, 11)), (getindex)(_x, 13))
            y[84] = (*)((^)((getindex)(_x, 12), 2), (getindex)(_x, 9))
            y[85] = (*)((*)((getindex)(_x, 9), (getindex)(_x, 12)), (getindex)(_x, 13))
            y[86] = (*)((^)((getindex)(_x, 13), 2), (getindex)(_x, 9))
            y[87] = (*)((^)((getindex)(_x, 11), 2), (getindex)(_x, 10))
            y[88] = (*)((*)((getindex)(_x, 10), (getindex)(_x, 11)), (getindex)(_x, 12))
            y[89] = (*)((*)((getindex)(_x, 10), (getindex)(_x, 11)), (getindex)(_x, 13))
            y[90] = (*)((^)((getindex)(_x, 12), 2), (getindex)(_x, 10))
            y[91] = (*)((*)((getindex)(_x, 10), (getindex)(_x, 12)), (getindex)(_x, 13))
            y[92] = (*)((^)((getindex)(_x, 13), 2), (getindex)(_x, 10))
            y[93] = (*)((^)((getindex)(_x, 4), 2), (getindex)(_x, 11))
            y[94] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 5)), (getindex)(_x, 11))
            y[95] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 6)), (getindex)(_x, 11))
            y[96] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 7)), (getindex)(_x, 11))
            y[97] = (*)((^)((getindex)(_x, 5), 2), (getindex)(_x, 11))
            y[98] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 6)), (getindex)(_x, 11))
            y[99] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 7)), (getindex)(_x, 11))
            y[100] = (*)((^)((getindex)(_x, 6), 2), (getindex)(_x, 11))
            y[101] = (*)((*)((getindex)(_x, 6), (getindex)(_x, 7)), (getindex)(_x, 11))
            y[102] = (*)((^)((getindex)(_x, 7), 2), (getindex)(_x, 11))
            y[103] = (*)((^)((getindex)(_x, 4), 2), (getindex)(_x, 12))
            y[104] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 5)), (getindex)(_x, 12))
            y[105] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 6)), (getindex)(_x, 12))
            y[106] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 7)), (getindex)(_x, 12))
            y[107] = (*)((^)((getindex)(_x, 5), 2), (getindex)(_x, 12))
            y[108] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 6)), (getindex)(_x, 12))
            y[109] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 7)), (getindex)(_x, 12))
            y[110] = (*)((^)((getindex)(_x, 6), 2), (getindex)(_x, 12))
            y[111] = (*)((*)((getindex)(_x, 6), (getindex)(_x, 7)), (getindex)(_x, 12))
            y[112] = (*)((^)((getindex)(_x, 7), 2), (getindex)(_x, 12))
            y[113] = (*)((^)((getindex)(_x, 4), 2), (getindex)(_x, 13))
            y[114] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 5)), (getindex)(_x, 13))
            y[115] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 6)), (getindex)(_x, 13))
            y[116] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 7)), (getindex)(_x, 13))
            y[117] = (*)((^)((getindex)(_x, 5), 2), (getindex)(_x, 13))
            y[118] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 6)), (getindex)(_x, 13))
            y[119] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 7)), (getindex)(_x, 13))
            y[120] = (*)((^)((getindex)(_x, 6), 2), (getindex)(_x, 13))
            y[121] = (*)((*)((getindex)(_x, 6), (getindex)(_x, 7)), (getindex)(_x, 13))
            y[122] = (*)((^)((getindex)(_x, 7), 2), (getindex)(_x, 13))
            y[123] = (*)((^)((getindex)(_x, 4), 2), (getindex)(_x, 8))
            y[124] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 5)), (getindex)(_x, 8))
            y[125] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 6)), (getindex)(_x, 8))
            y[126] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 7)), (getindex)(_x, 8))
            y[127] = (*)((^)((getindex)(_x, 5), 2), (getindex)(_x, 8))
            y[128] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 6)), (getindex)(_x, 8))
            y[129] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 7)), (getindex)(_x, 8))
            y[130] = (*)((^)((getindex)(_x, 6), 2), (getindex)(_x, 8))
            y[131] = (*)((*)((getindex)(_x, 6), (getindex)(_x, 7)), (getindex)(_x, 8))
            y[132] = (*)((^)((getindex)(_x, 7), 2), (getindex)(_x, 8))
            y[133] = (*)((^)((getindex)(_x, 4), 2), (getindex)(_x, 9))
            y[134] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 5)), (getindex)(_x, 9))
            y[135] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 6)), (getindex)(_x, 9))
            y[136] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 7)), (getindex)(_x, 9))
            y[137] = (*)((^)((getindex)(_x, 5), 2), (getindex)(_x, 9))
            y[138] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 6)), (getindex)(_x, 9))
            y[139] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 7)), (getindex)(_x, 9))
            y[140] = (*)((^)((getindex)(_x, 6), 2), (getindex)(_x, 9))
            y[141] = (*)((*)((getindex)(_x, 6), (getindex)(_x, 7)), (getindex)(_x, 9))
            y[142] = (*)((^)((getindex)(_x, 7), 2), (getindex)(_x, 9))
            y[143] = (*)((^)((getindex)(_x, 4), 2), (getindex)(_x, 10))
            y[144] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 5)), (getindex)(_x, 10))
            y[145] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 6)), (getindex)(_x, 10))
            y[146] = (*)((*)((getindex)(_x, 4), (getindex)(_x, 7)), (getindex)(_x, 10))
            y[147] = (*)((^)((getindex)(_x, 5), 2), (getindex)(_x, 10))
            y[148] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 6)), (getindex)(_x, 10))
            y[149] = (*)((*)((getindex)(_x, 5), (getindex)(_x, 7)), (getindex)(_x, 10))
            y[150] = (*)((^)((getindex)(_x, 6), 2), (getindex)(_x, 10))
            y[151] = (*)((*)((getindex)(_x, 6), (getindex)(_x, 7)), (getindex)(_x, 10))
            y[152] = (*)((^)((getindex)(_x, 7), 2), (getindex)(_x, 10))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:310 =#
            return y
        end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:416 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:320 =#
        function se3_dynamics!(xdot, x, u, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:320 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:321 =#
            (_x, _u, _c) = (x, u, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:322 =#
            xdot[1] = (+)((+)((+)((+)((+)((+)((+)((*)(-1, (getindex)(_x, 130)), (*)(-1, (getindex)(_x, 132))), (*)(-2, (getindex)(_x, 136))), (*)(2, (getindex)(_x, 138))), (*)(2, (getindex)(_x, 145))), (*)(2, (getindex)(_x, 149))), (getindex)(_x, 123)), (getindex)(_x, 127))
            xdot[2] = (+)((+)((+)((+)((+)((+)((+)((*)(2, (getindex)(_x, 126)), (*)(-1, (getindex)(_x, 137))), (*)(2, (getindex)(_x, 128))), (*)(-1, (getindex)(_x, 142))), (*)(-2, (getindex)(_x, 144))), (*)(2, (getindex)(_x, 151))), (getindex)(_x, 133)), (getindex)(_x, 140))
            xdot[3] = (+)((+)((+)((+)((+)((+)((+)((*)(-2, (getindex)(_x, 125)), (*)(-1, (getindex)(_x, 147))), (*)(-1, (getindex)(_x, 150))), (*)(2, (getindex)(_x, 129))), (*)(2, (getindex)(_x, 134))), (*)(2, (getindex)(_x, 141))), (getindex)(_x, 143)), (getindex)(_x, 152))
            xdot[4] = (+)((+)((*)(-1//2, (getindex)(_x, 31)), (*)(-1//2, (getindex)(_x, 36))), (*)(-1//2, (getindex)(_x, 41)))
            xdot[5] = (+)((+)((*)(1//2, (getindex)(_x, 30)), (*)(-1//2, (getindex)(_x, 37))), (*)(1//2, (getindex)(_x, 40)))
            xdot[6] = (+)((+)((*)(1//2, (getindex)(_x, 33)), (*)(1//2, (getindex)(_x, 34))), (*)(-1//2, (getindex)(_x, 39)))
            xdot[7] = (+)((+)((*)(-1//2, (getindex)(_x, 32)), (*)(1//2, (getindex)(_x, 35))), (*)(1//2, (getindex)(_x, 38)))
            xdot[8] = (+)((+)((/)((getindex)(_u, 1), (getindex)(_c, 1)), (*)(-1, (getindex)(_x, 47))), (getindex)(_x, 49))
            xdot[9] = (+)((+)((/)((getindex)(_u, 2), (getindex)(_c, 1)), (*)(-1, (getindex)(_x, 48))), (getindex)(_x, 44))
            xdot[10] = (+)((+)((/)((getindex)(_u, 3), (getindex)(_c, 1)), (*)(-1, (getindex)(_x, 43))), (getindex)(_x, 45))
            xdot[11] = (/)((+)((+)((*)((getindex)(_c, 3), (getindex)(_x, 28)), (*)((*)(-1, (getindex)(_c, 4)), (getindex)(_x, 28))), (getindex)(_u, 4)), (getindex)(_c, 2))
            xdot[12] = (/)((+)((+)((*)((getindex)(_c, 4), (getindex)(_x, 26)), (*)((*)(-1, (getindex)(_c, 2)), (getindex)(_x, 26))), (getindex)(_u, 5)), (getindex)(_c, 3))
            xdot[13] = (/)((+)((+)((*)((getindex)(_c, 2), (getindex)(_x, 25)), (*)((*)(-1, (getindex)(_c, 3)), (getindex)(_x, 25))), (getindex)(_u, 6)), (getindex)(_c, 4))
            xdot[14] = (+)((+)((*)(-1//1, (getindex)(_x, 94)), (*)(-1//1, (getindex)(_x, 105))), (*)(-1//1, (getindex)(_x, 116)))
            xdot[15] = (+)((+)((+)((+)((+)((*)(1//2, (getindex)(_x, 93)), (*)(-1//2, (getindex)(_x, 97))), (*)(-1//2, (getindex)(_x, 106))), (*)(-1//2, (getindex)(_x, 108))), (*)(1//2, (getindex)(_x, 115))), (*)(-1//2, (getindex)(_x, 119)))
            xdot[16] = (+)((+)((+)((+)((+)((*)(1//2, (getindex)(_x, 96)), (*)(-1//2, (getindex)(_x, 98))), (*)(1//2, (getindex)(_x, 103))), (*)(-1//2, (getindex)(_x, 110))), (*)(-1//2, (getindex)(_x, 114))), (*)(-1//2, (getindex)(_x, 121)))
            xdot[17] = (+)((+)((+)((+)((+)((*)(-1//2, (getindex)(_x, 95)), (*)(-1//2, (getindex)(_x, 99))), (*)(1//2, (getindex)(_x, 104))), (*)(-1//2, (getindex)(_x, 111))), (*)(1//2, (getindex)(_x, 113))), (*)(-1//2, (getindex)(_x, 122)))
            xdot[18] = (+)((+)((*)(-1//1, (getindex)(_x, 109)), (getindex)(_x, 94)), (getindex)(_x, 118))
            xdot[19] = (+)((+)((+)((+)((+)((*)(1//2, (getindex)(_x, 95)), (*)(1//2, (getindex)(_x, 99))), (*)(1//2, (getindex)(_x, 104))), (*)(-1//2, (getindex)(_x, 111))), (*)(-1//2, (getindex)(_x, 117))), (*)(1//2, (getindex)(_x, 120)))
            xdot[20] = (+)((+)((+)((+)((+)((*)(1//2, (getindex)(_x, 96)), (*)(-1//2, (getindex)(_x, 98))), (*)(1//2, (getindex)(_x, 107))), (*)(-1//2, (getindex)(_x, 112))), (*)(1//2, (getindex)(_x, 114))), (*)(1//2, (getindex)(_x, 121)))
            xdot[21] = (+)((+)((*)(-1//1, (getindex)(_x, 118)), (getindex)(_x, 101)), (getindex)(_x, 105))
            xdot[22] = (+)((+)((+)((+)((+)((*)(-1//2, (getindex)(_x, 100)), (*)(1//2, (getindex)(_x, 102))), (*)(1//2, (getindex)(_x, 106))), (*)(1//2, (getindex)(_x, 108))), (*)(1//2, (getindex)(_x, 115))), (*)(-1//2, (getindex)(_x, 119)))
            xdot[23] = (+)((+)((*)(-1//1, (getindex)(_x, 101)), (getindex)(_x, 109)), (getindex)(_x, 116))
            xdot[24] = (/)((*)((*)(2, (getindex)(_u, 4)), (getindex)(_x, 11)), (getindex)(_c, 2))
            xdot[25] = (+)((/)((*)((getindex)(_u, 5), (getindex)(_x, 11)), (getindex)(_c, 3)), (/)((*)((getindex)(_u, 4), (getindex)(_x, 12)), (getindex)(_c, 2)))
            xdot[26] = (+)((/)((*)((getindex)(_u, 6), (getindex)(_x, 11)), (getindex)(_c, 4)), (/)((*)((getindex)(_u, 4), (getindex)(_x, 13)), (getindex)(_c, 2)))
            xdot[27] = (/)((*)((*)(2, (getindex)(_u, 5)), (getindex)(_x, 12)), (getindex)(_c, 3))
            xdot[28] = (+)((/)((*)((getindex)(_u, 5), (getindex)(_x, 13)), (getindex)(_c, 3)), (/)((*)((getindex)(_u, 6), (getindex)(_x, 12)), (getindex)(_c, 4)))
            xdot[29] = (/)((*)((*)(2, (getindex)(_u, 6)), (getindex)(_x, 13)), (getindex)(_c, 4))
            xdot[30] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_c, 3), (getindex)(_x, 55)), (*)((getindex)(_u, 4), (getindex)(_x, 4))), (*)((*)(-1, (getindex)(_c, 4)), (getindex)(_x, 55))), (getindex)(_c, 2)), (*)(-1//2, (getindex)(_x, 57))), (*)(-1//2, (getindex)(_x, 64))), (*)(-1//2, (getindex)(_x, 71)))
            xdot[31] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_c, 3), (getindex)(_x, 61)), (*)((getindex)(_u, 4), (getindex)(_x, 5))), (*)((*)(-1, (getindex)(_c, 4)), (getindex)(_x, 61))), (getindex)(_c, 2)), (*)(1//2, (getindex)(_x, 51))), (*)(1//2, (getindex)(_x, 65))), (*)(-1//2, (getindex)(_x, 70)))
            xdot[32] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_u, 4), (getindex)(_x, 6)), (*)((getindex)(_c, 3), (getindex)(_x, 67))), (*)((*)(-1, (getindex)(_c, 4)), (getindex)(_x, 67))), (getindex)(_c, 2)), (*)(1//2, (getindex)(_x, 52))), (*)(-1//2, (getindex)(_x, 59))), (*)(1//2, (getindex)(_x, 69)))
            xdot[33] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_u, 4), (getindex)(_x, 7)), (*)((getindex)(_c, 3), (getindex)(_x, 73))), (*)((*)(-1, (getindex)(_c, 4)), (getindex)(_x, 73))), (getindex)(_c, 2)), (*)(1//2, (getindex)(_x, 53))), (*)(1//2, (getindex)(_x, 58))), (*)(-1//2, (getindex)(_x, 63)))
            xdot[34] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_c, 4), (getindex)(_x, 53)), (*)((getindex)(_u, 5), (getindex)(_x, 4))), (*)((*)(-1, (getindex)(_c, 2)), (getindex)(_x, 53))), (getindex)(_c, 3)), (*)(-1//2, (getindex)(_x, 58))), (*)(-1//2, (getindex)(_x, 66))), (*)(-1//2, (getindex)(_x, 73)))
            xdot[35] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_c, 4), (getindex)(_x, 59)), (*)((getindex)(_u, 5), (getindex)(_x, 5))), (*)((*)(-1, (getindex)(_c, 2)), (getindex)(_x, 59))), (getindex)(_c, 3)), (*)(1//2, (getindex)(_x, 52))), (*)(1//2, (getindex)(_x, 67))), (*)(-1//2, (getindex)(_x, 72)))
            xdot[36] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_c, 4), (getindex)(_x, 65)), (*)((getindex)(_u, 5), (getindex)(_x, 6))), (*)((*)(-1, (getindex)(_c, 2)), (getindex)(_x, 65))), (getindex)(_c, 3)), (*)(1//2, (getindex)(_x, 54))), (*)(-1//2, (getindex)(_x, 61))), (*)(1//2, (getindex)(_x, 70)))
            xdot[37] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_c, 4), (getindex)(_x, 71)), (*)((getindex)(_u, 5), (getindex)(_x, 7))), (*)((*)(-1, (getindex)(_c, 2)), (getindex)(_x, 71))), (getindex)(_c, 3)), (*)(1//2, (getindex)(_x, 55))), (*)(1//2, (getindex)(_x, 60))), (*)(-1//2, (getindex)(_x, 64)))
            xdot[38] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_u, 6), (getindex)(_x, 4)), (*)((getindex)(_c, 2), (getindex)(_x, 52))), (*)((*)(-1, (getindex)(_c, 3)), (getindex)(_x, 52))), (getindex)(_c, 4)), (*)(-1//2, (getindex)(_x, 59))), (*)(-1//2, (getindex)(_x, 67))), (*)(-1//2, (getindex)(_x, 74)))
            xdot[39] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_u, 6), (getindex)(_x, 5)), (*)((getindex)(_c, 2), (getindex)(_x, 58))), (*)((*)(-1, (getindex)(_c, 3)), (getindex)(_x, 58))), (getindex)(_c, 4)), (*)(1//2, (getindex)(_x, 53))), (*)(1//2, (getindex)(_x, 68))), (*)(-1//2, (getindex)(_x, 73)))
            xdot[40] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_u, 6), (getindex)(_x, 6)), (*)((getindex)(_c, 2), (getindex)(_x, 64))), (*)((*)(-1, (getindex)(_c, 3)), (getindex)(_x, 64))), (getindex)(_c, 4)), (*)(1//2, (getindex)(_x, 55))), (*)(-1//2, (getindex)(_x, 62))), (*)(1//2, (getindex)(_x, 71)))
            xdot[41] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_u, 6), (getindex)(_x, 7)), (*)((getindex)(_c, 2), (getindex)(_x, 70))), (*)((*)(-1, (getindex)(_c, 3)), (getindex)(_x, 70))), (getindex)(_c, 4)), (*)(1//2, (getindex)(_x, 56))), (*)(1//2, (getindex)(_x, 61))), (*)(-1//2, (getindex)(_x, 65)))
            xdot[42] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_c, 3), (getindex)(_x, 79)), (*)((getindex)(_u, 4), (getindex)(_x, 8))), (*)((*)(-1, (getindex)(_c, 4)), (getindex)(_x, 79))), (getindex)(_c, 2)), (/)((*)((getindex)(_u, 1), (getindex)(_x, 11)), (getindex)(_c, 1))), (*)(-1, (getindex)(_x, 88))), (getindex)(_x, 83))
            xdot[43] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_u, 4), (getindex)(_x, 9)), (*)((getindex)(_c, 3), (getindex)(_x, 85))), (*)((*)(-1, (getindex)(_c, 4)), (getindex)(_x, 85))), (getindex)(_c, 2)), (/)((*)((getindex)(_u, 2), (getindex)(_x, 11)), (getindex)(_c, 1))), (*)(-1, (getindex)(_x, 77))), (getindex)(_x, 87))
            xdot[44] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_c, 3), (getindex)(_x, 91)), (*)((getindex)(_u, 4), (getindex)(_x, 10))), (*)((*)(-1, (getindex)(_c, 4)), (getindex)(_x, 91))), (getindex)(_c, 2)), (/)((*)((getindex)(_u, 3), (getindex)(_x, 11)), (getindex)(_c, 1))), (*)(-1, (getindex)(_x, 81))), (getindex)(_x, 76))
            xdot[45] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_u, 5), (getindex)(_x, 8)), (*)((getindex)(_c, 4), (getindex)(_x, 77))), (*)((*)(-1, (getindex)(_c, 2)), (getindex)(_x, 77))), (getindex)(_c, 3)), (/)((*)((getindex)(_u, 1), (getindex)(_x, 12)), (getindex)(_c, 1))), (*)(-1, (getindex)(_x, 90))), (getindex)(_x, 85))
            xdot[46] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_u, 5), (getindex)(_x, 9)), (*)((getindex)(_c, 4), (getindex)(_x, 83))), (*)((*)(-1, (getindex)(_c, 2)), (getindex)(_x, 83))), (getindex)(_c, 3)), (/)((*)((getindex)(_u, 2), (getindex)(_x, 12)), (getindex)(_c, 1))), (*)(-1, (getindex)(_x, 79))), (getindex)(_x, 88))
            xdot[47] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_c, 4), (getindex)(_x, 89)), (*)((getindex)(_u, 5), (getindex)(_x, 10))), (*)((*)(-1, (getindex)(_c, 2)), (getindex)(_x, 89))), (getindex)(_c, 3)), (/)((*)((getindex)(_u, 3), (getindex)(_x, 12)), (getindex)(_c, 1))), (*)(-1, (getindex)(_x, 82))), (getindex)(_x, 78))
            xdot[48] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_u, 6), (getindex)(_x, 8)), (*)((getindex)(_c, 2), (getindex)(_x, 76))), (*)((*)(-1, (getindex)(_c, 3)), (getindex)(_x, 76))), (getindex)(_c, 4)), (/)((*)((getindex)(_u, 1), (getindex)(_x, 13)), (getindex)(_c, 1))), (*)(-1, (getindex)(_x, 91))), (getindex)(_x, 86))
            xdot[49] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_u, 6), (getindex)(_x, 9)), (*)((getindex)(_c, 2), (getindex)(_x, 82))), (*)((*)(-1, (getindex)(_c, 3)), (getindex)(_x, 82))), (getindex)(_c, 4)), (/)((*)((getindex)(_u, 2), (getindex)(_x, 13)), (getindex)(_c, 1))), (*)(-1, (getindex)(_x, 80))), (getindex)(_x, 89))
            xdot[50] = (+)((+)((+)((/)((+)((+)((*)((getindex)(_u, 6), (getindex)(_x, 10)), (*)((getindex)(_c, 2), (getindex)(_x, 88))), (*)((*)(-1, (getindex)(_c, 3)), (getindex)(_x, 88))), (getindex)(_c, 4)), (/)((*)((getindex)(_u, 3), (getindex)(_x, 13)), (getindex)(_c, 1))), (*)(-1, (getindex)(_x, 83))), (getindex)(_x, 79))
            xdot[51] = (/)((*)((*)(2, (getindex)(_u, 4)), (getindex)(_x, 30)), (getindex)(_c, 2))
            xdot[52] = (+)((/)((*)((getindex)(_u, 4), (getindex)(_x, 34)), (getindex)(_c, 2)), (/)((*)((getindex)(_u, 5), (getindex)(_x, 30)), (getindex)(_c, 3)))
            xdot[53] = (+)((/)((*)((getindex)(_u, 6), (getindex)(_x, 30)), (getindex)(_c, 4)), (/)((*)((getindex)(_u, 4), (getindex)(_x, 38)), (getindex)(_c, 2)))
            xdot[54] = (/)((*)((*)(2, (getindex)(_u, 5)), (getindex)(_x, 34)), (getindex)(_c, 3))
            xdot[55] = (+)((/)((*)((getindex)(_u, 5), (getindex)(_x, 38)), (getindex)(_c, 3)), (/)((*)((getindex)(_u, 6), (getindex)(_x, 34)), (getindex)(_c, 4)))
            xdot[56] = (/)((*)((*)(2, (getindex)(_u, 6)), (getindex)(_x, 38)), (getindex)(_c, 4))
            xdot[57] = (/)((*)((*)(2, (getindex)(_u, 4)), (getindex)(_x, 31)), (getindex)(_c, 2))
            xdot[58] = (+)((/)((*)((getindex)(_u, 4), (getindex)(_x, 35)), (getindex)(_c, 2)), (/)((*)((getindex)(_u, 5), (getindex)(_x, 31)), (getindex)(_c, 3)))
            xdot[59] = (+)((/)((*)((getindex)(_u, 6), (getindex)(_x, 31)), (getindex)(_c, 4)), (/)((*)((getindex)(_u, 4), (getindex)(_x, 39)), (getindex)(_c, 2)))
            xdot[60] = (/)((*)((*)(2, (getindex)(_u, 5)), (getindex)(_x, 35)), (getindex)(_c, 3))
            xdot[61] = (+)((/)((*)((getindex)(_u, 6), (getindex)(_x, 35)), (getindex)(_c, 4)), (/)((*)((getindex)(_u, 5), (getindex)(_x, 39)), (getindex)(_c, 3)))
            xdot[62] = (/)((*)((*)(2, (getindex)(_u, 6)), (getindex)(_x, 39)), (getindex)(_c, 4))
            xdot[63] = (/)((*)((*)(2, (getindex)(_u, 4)), (getindex)(_x, 32)), (getindex)(_c, 2))
            xdot[64] = (+)((/)((*)((getindex)(_u, 4), (getindex)(_x, 36)), (getindex)(_c, 2)), (/)((*)((getindex)(_u, 5), (getindex)(_x, 32)), (getindex)(_c, 3)))
            xdot[65] = (+)((/)((*)((getindex)(_u, 6), (getindex)(_x, 32)), (getindex)(_c, 4)), (/)((*)((getindex)(_u, 4), (getindex)(_x, 40)), (getindex)(_c, 2)))
            xdot[66] = (/)((*)((*)(2, (getindex)(_u, 5)), (getindex)(_x, 36)), (getindex)(_c, 3))
            xdot[67] = (+)((/)((*)((getindex)(_u, 5), (getindex)(_x, 40)), (getindex)(_c, 3)), (/)((*)((getindex)(_u, 6), (getindex)(_x, 36)), (getindex)(_c, 4)))
            xdot[68] = (/)((*)((*)(2, (getindex)(_u, 6)), (getindex)(_x, 40)), (getindex)(_c, 4))
            xdot[69] = (/)((*)((*)(2, (getindex)(_u, 4)), (getindex)(_x, 33)), (getindex)(_c, 2))
            xdot[70] = (+)((/)((*)((getindex)(_u, 5), (getindex)(_x, 33)), (getindex)(_c, 3)), (/)((*)((getindex)(_u, 4), (getindex)(_x, 37)), (getindex)(_c, 2)))
            xdot[71] = (+)((/)((*)((getindex)(_u, 6), (getindex)(_x, 33)), (getindex)(_c, 4)), (/)((*)((getindex)(_u, 4), (getindex)(_x, 41)), (getindex)(_c, 2)))
            xdot[72] = (/)((*)((*)(2, (getindex)(_u, 5)), (getindex)(_x, 37)), (getindex)(_c, 3))
            xdot[73] = (+)((/)((*)((getindex)(_u, 5), (getindex)(_x, 41)), (getindex)(_c, 3)), (/)((*)((getindex)(_u, 6), (getindex)(_x, 37)), (getindex)(_c, 4)))
            xdot[74] = (/)((*)((*)(2, (getindex)(_u, 6)), (getindex)(_x, 41)), (getindex)(_c, 4))
            xdot[75] = (+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 24)), (getindex)(_c, 1)), (/)((*)((*)(2, (getindex)(_u, 4)), (getindex)(_x, 42)), (getindex)(_c, 2)))
            xdot[76] = (+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 25)), (getindex)(_c, 1)), (/)((*)((getindex)(_u, 4), (getindex)(_x, 45)), (getindex)(_c, 2))), (/)((*)((getindex)(_u, 5), (getindex)(_x, 42)), (getindex)(_c, 3)))
            xdot[77] = (+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 26)), (getindex)(_c, 1)), (/)((*)((getindex)(_u, 4), (getindex)(_x, 48)), (getindex)(_c, 2))), (/)((*)((getindex)(_u, 6), (getindex)(_x, 42)), (getindex)(_c, 4)))
            xdot[78] = (+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 27)), (getindex)(_c, 1)), (/)((*)((*)(2, (getindex)(_u, 5)), (getindex)(_x, 45)), (getindex)(_c, 3)))
            xdot[79] = (+)((+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 28)), (getindex)(_c, 1)), (/)((*)((getindex)(_u, 5), (getindex)(_x, 48)), (getindex)(_c, 3))), (/)((*)((getindex)(_u, 6), (getindex)(_x, 45)), (getindex)(_c, 4)))
            xdot[80] = (+)((/)((*)((getindex)(_u, 1), (getindex)(_x, 29)), (getindex)(_c, 1)), (/)((*)((*)(2, (getindex)(_u, 6)), (getindex)(_x, 48)), (getindex)(_c, 4)))
            xdot[81] = (+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 24)), (getindex)(_c, 1)), (/)((*)((*)(2, (getindex)(_u, 4)), (getindex)(_x, 43)), (getindex)(_c, 2)))
            xdot[82] = (+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 25)), (getindex)(_c, 1)), (/)((*)((getindex)(_u, 5), (getindex)(_x, 43)), (getindex)(_c, 3))), (/)((*)((getindex)(_u, 4), (getindex)(_x, 46)), (getindex)(_c, 2)))
            xdot[83] = (+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 26)), (getindex)(_c, 1)), (/)((*)((getindex)(_u, 6), (getindex)(_x, 43)), (getindex)(_c, 4))), (/)((*)((getindex)(_u, 4), (getindex)(_x, 49)), (getindex)(_c, 2)))
            xdot[84] = (+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 27)), (getindex)(_c, 1)), (/)((*)((*)(2, (getindex)(_u, 5)), (getindex)(_x, 46)), (getindex)(_c, 3)))
            xdot[85] = (+)((+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 28)), (getindex)(_c, 1)), (/)((*)((getindex)(_u, 5), (getindex)(_x, 49)), (getindex)(_c, 3))), (/)((*)((getindex)(_u, 6), (getindex)(_x, 46)), (getindex)(_c, 4)))
            xdot[86] = (+)((/)((*)((getindex)(_u, 2), (getindex)(_x, 29)), (getindex)(_c, 1)), (/)((*)((*)(2, (getindex)(_u, 6)), (getindex)(_x, 49)), (getindex)(_c, 4)))
            xdot[87] = (+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 24)), (getindex)(_c, 1)), (/)((*)((*)(2, (getindex)(_u, 4)), (getindex)(_x, 44)), (getindex)(_c, 2)))
            xdot[88] = (+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 25)), (getindex)(_c, 1)), (/)((*)((getindex)(_u, 4), (getindex)(_x, 47)), (getindex)(_c, 2))), (/)((*)((getindex)(_u, 5), (getindex)(_x, 44)), (getindex)(_c, 3)))
            xdot[89] = (+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 26)), (getindex)(_c, 1)), (/)((*)((getindex)(_u, 4), (getindex)(_x, 50)), (getindex)(_c, 2))), (/)((*)((getindex)(_u, 6), (getindex)(_x, 44)), (getindex)(_c, 4)))
            xdot[90] = (+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 27)), (getindex)(_c, 1)), (/)((*)((*)(2, (getindex)(_u, 5)), (getindex)(_x, 47)), (getindex)(_c, 3)))
            xdot[91] = (+)((+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 28)), (getindex)(_c, 1)), (/)((*)((getindex)(_u, 6), (getindex)(_x, 47)), (getindex)(_c, 4))), (/)((*)((getindex)(_u, 5), (getindex)(_x, 50)), (getindex)(_c, 3)))
            xdot[92] = (+)((/)((*)((getindex)(_u, 3), (getindex)(_x, 29)), (getindex)(_c, 1)), (/)((*)((*)(2, (getindex)(_u, 6)), (getindex)(_x, 50)), (getindex)(_c, 4)))
            xdot[93] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 14)), (getindex)(_c, 2))
            xdot[94] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 15)), (getindex)(_c, 2))
            xdot[95] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 16)), (getindex)(_c, 2))
            xdot[96] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 17)), (getindex)(_c, 2))
            xdot[97] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 18)), (getindex)(_c, 2))
            xdot[98] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 19)), (getindex)(_c, 2))
            xdot[99] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 20)), (getindex)(_c, 2))
            xdot[100] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 21)), (getindex)(_c, 2))
            xdot[101] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 22)), (getindex)(_c, 2))
            xdot[102] = (/)((*)((getindex)(_u, 4), (getindex)(_x, 23)), (getindex)(_c, 2))
            xdot[103] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 14)), (getindex)(_c, 3))
            xdot[104] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 15)), (getindex)(_c, 3))
            xdot[105] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 16)), (getindex)(_c, 3))
            xdot[106] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 17)), (getindex)(_c, 3))
            xdot[107] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 18)), (getindex)(_c, 3))
            xdot[108] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 19)), (getindex)(_c, 3))
            xdot[109] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 20)), (getindex)(_c, 3))
            xdot[110] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 21)), (getindex)(_c, 3))
            xdot[111] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 22)), (getindex)(_c, 3))
            xdot[112] = (/)((*)((getindex)(_u, 5), (getindex)(_x, 23)), (getindex)(_c, 3))
            xdot[113] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 14)), (getindex)(_c, 4))
            xdot[114] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 15)), (getindex)(_c, 4))
            xdot[115] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 16)), (getindex)(_c, 4))
            xdot[116] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 17)), (getindex)(_c, 4))
            xdot[117] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 18)), (getindex)(_c, 4))
            xdot[118] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 19)), (getindex)(_c, 4))
            xdot[119] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 20)), (getindex)(_c, 4))
            xdot[120] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 21)), (getindex)(_c, 4))
            xdot[121] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 22)), (getindex)(_c, 4))
            xdot[122] = (/)((*)((getindex)(_u, 6), (getindex)(_x, 23)), (getindex)(_c, 4))
            xdot[123] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 14)), (getindex)(_c, 1))
            xdot[124] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 15)), (getindex)(_c, 1))
            xdot[125] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 16)), (getindex)(_c, 1))
            xdot[126] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 17)), (getindex)(_c, 1))
            xdot[127] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 18)), (getindex)(_c, 1))
            xdot[128] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 19)), (getindex)(_c, 1))
            xdot[129] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 20)), (getindex)(_c, 1))
            xdot[130] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 21)), (getindex)(_c, 1))
            xdot[131] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 22)), (getindex)(_c, 1))
            xdot[132] = (/)((*)((getindex)(_u, 1), (getindex)(_x, 23)), (getindex)(_c, 1))
            xdot[133] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 14)), (getindex)(_c, 1))
            xdot[134] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 15)), (getindex)(_c, 1))
            xdot[135] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 16)), (getindex)(_c, 1))
            xdot[136] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 17)), (getindex)(_c, 1))
            xdot[137] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 18)), (getindex)(_c, 1))
            xdot[138] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 19)), (getindex)(_c, 1))
            xdot[139] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 20)), (getindex)(_c, 1))
            xdot[140] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 21)), (getindex)(_c, 1))
            xdot[141] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 22)), (getindex)(_c, 1))
            xdot[142] = (/)((*)((getindex)(_u, 2), (getindex)(_x, 23)), (getindex)(_c, 1))
            xdot[143] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 14)), (getindex)(_c, 1))
            xdot[144] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 15)), (getindex)(_c, 1))
            xdot[145] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 16)), (getindex)(_c, 1))
            xdot[146] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 17)), (getindex)(_c, 1))
            xdot[147] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 18)), (getindex)(_c, 1))
            xdot[148] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 19)), (getindex)(_c, 1))
            xdot[149] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 20)), (getindex)(_c, 1))
            xdot[150] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 21)), (getindex)(_c, 1))
            xdot[151] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 22)), (getindex)(_c, 1))
            xdot[152] = (/)((*)((getindex)(_u, 3), (getindex)(_x, 23)), (getindex)(_c, 1))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:323 =#
            return
        end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:417 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:354 =#
        function se3_updateA!(A, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:354 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:355 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:356 =#
            nzval = A.nzval
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:357 =#
            nzval[1] = (+)((/)((*)(-1, (getindex)(_c, 3)), (getindex)(_c, 4)), (/)((getindex)(_c, 2), (getindex)(_c, 4)))
            nzval[2] = (+)((/)((*)(-1, (getindex)(_c, 2)), (getindex)(_c, 3)), (/)((getindex)(_c, 4), (getindex)(_c, 3)))
            nzval[3] = (+)((/)((*)(-1, (getindex)(_c, 4)), (getindex)(_c, 2)), (/)((getindex)(_c, 3), (getindex)(_c, 2)))
            nzval[4] = 1//2
            nzval[5] = -1//2
            nzval[6] = -1//2
            nzval[7] = 1//2
            nzval[8] = 1//2
            nzval[9] = 1//2
            nzval[10] = -1//2
            nzval[11] = -1//2
            nzval[12] = 1//2
            nzval[13] = -1//2
            nzval[14] = 1//2
            nzval[15] = -1//2
            nzval[16] = -1
            nzval[17] = 1
            nzval[18] = 1
            nzval[19] = -1
            nzval[20] = -1
            nzval[21] = 1
            nzval[22] = 1//2
            nzval[23] = 1//2
            nzval[24] = 1//2
            nzval[25] = (+)((/)((*)(-1, (getindex)(_c, 3)), (getindex)(_c, 4)), (/)((getindex)(_c, 2), (getindex)(_c, 4)))
            nzval[26] = 1//2
            nzval[27] = (+)((/)((*)(-1, (getindex)(_c, 2)), (getindex)(_c, 3)), (/)((getindex)(_c, 4), (getindex)(_c, 3)))
            nzval[28] = 1//2
            nzval[29] = 1//2
            nzval[30] = (+)((/)((*)(-1, (getindex)(_c, 4)), (getindex)(_c, 2)), (/)((getindex)(_c, 3), (getindex)(_c, 2)))
            nzval[31] = 1//2
            nzval[32] = 1//2
            nzval[33] = 1//2
            nzval[34] = -1//2
            nzval[35] = 1//2
            nzval[36] = -1//2
            nzval[37] = (+)((/)((*)(-1, (getindex)(_c, 3)), (getindex)(_c, 4)), (/)((getindex)(_c, 2), (getindex)(_c, 4)))
            nzval[38] = -1//2
            nzval[39] = (+)((/)((*)(-1, (getindex)(_c, 2)), (getindex)(_c, 3)), (/)((getindex)(_c, 4), (getindex)(_c, 3)))
            nzval[40] = -1//2
            nzval[41] = 1//2
            nzval[42] = (+)((/)((*)(-1, (getindex)(_c, 4)), (getindex)(_c, 2)), (/)((getindex)(_c, 3), (getindex)(_c, 2)))
            nzval[43] = -1//2
            nzval[44] = 1//2
            nzval[45] = -1//2
            nzval[46] = -1//2
            nzval[47] = -1//2
            nzval[48] = -1//2
            nzval[49] = (+)((/)((*)(-1, (getindex)(_c, 3)), (getindex)(_c, 4)), (/)((getindex)(_c, 2), (getindex)(_c, 4)))
            nzval[50] = 1//2
            nzval[51] = (+)((/)((*)(-1, (getindex)(_c, 2)), (getindex)(_c, 3)), (/)((getindex)(_c, 4), (getindex)(_c, 3)))
            nzval[52] = -1//2
            nzval[53] = -1//2
            nzval[54] = (+)((/)((*)(-1, (getindex)(_c, 4)), (getindex)(_c, 2)), (/)((getindex)(_c, 3), (getindex)(_c, 2)))
            nzval[55] = 1//2
            nzval[56] = -1//2
            nzval[57] = 1//2
            nzval[58] = 1//2
            nzval[59] = -1//2
            nzval[60] = 1//2
            nzval[61] = (+)((/)((*)(-1, (getindex)(_c, 3)), (getindex)(_c, 4)), (/)((getindex)(_c, 2), (getindex)(_c, 4)))
            nzval[62] = -1//2
            nzval[63] = (+)((/)((*)(-1, (getindex)(_c, 2)), (getindex)(_c, 3)), (/)((getindex)(_c, 4), (getindex)(_c, 3)))
            nzval[64] = 1//2
            nzval[65] = -1//2
            nzval[66] = (+)((/)((*)(-1, (getindex)(_c, 4)), (getindex)(_c, 2)), (/)((getindex)(_c, 3), (getindex)(_c, 2)))
            nzval[67] = -1//2
            nzval[68] = -1//2
            nzval[69] = -1//2
            nzval[70] = 1
            nzval[71] = (+)((/)((*)(-1, (getindex)(_c, 3)), (getindex)(_c, 4)), (/)((getindex)(_c, 2), (getindex)(_c, 4)))
            nzval[72] = -1
            nzval[73] = (+)((/)((*)(-1, (getindex)(_c, 2)), (getindex)(_c, 3)), (/)((getindex)(_c, 4), (getindex)(_c, 3)))
            nzval[74] = 1
            nzval[75] = (+)((/)((*)(-1, (getindex)(_c, 4)), (getindex)(_c, 2)), (/)((getindex)(_c, 3), (getindex)(_c, 2)))
            nzval[76] = -1
            nzval[77] = 1
            nzval[78] = -1
            nzval[79] = -1
            nzval[80] = -1
            nzval[81] = (+)((/)((*)(-1, (getindex)(_c, 3)), (getindex)(_c, 4)), (/)((getindex)(_c, 2), (getindex)(_c, 4)))
            nzval[82] = 1
            nzval[83] = (+)((/)((*)(-1, (getindex)(_c, 2)), (getindex)(_c, 3)), (/)((getindex)(_c, 4), (getindex)(_c, 3)))
            nzval[84] = -1
            nzval[85] = (+)((/)((*)(-1, (getindex)(_c, 4)), (getindex)(_c, 2)), (/)((getindex)(_c, 3), (getindex)(_c, 2)))
            nzval[86] = 1
            nzval[87] = 1
            nzval[88] = 1
            nzval[89] = -1
            nzval[90] = 1
            nzval[91] = (+)((/)((*)(-1, (getindex)(_c, 3)), (getindex)(_c, 4)), (/)((getindex)(_c, 2), (getindex)(_c, 4)))
            nzval[92] = (+)((/)((*)(-1, (getindex)(_c, 2)), (getindex)(_c, 3)), (/)((getindex)(_c, 4), (getindex)(_c, 3)))
            nzval[93] = 1
            nzval[94] = -1
            nzval[95] = (+)((/)((*)(-1, (getindex)(_c, 4)), (getindex)(_c, 2)), (/)((getindex)(_c, 3), (getindex)(_c, 2)))
            nzval[96] = -1
            nzval[97] = 1//2
            nzval[98] = -1//1
            nzval[99] = 1
            nzval[100] = -1//2
            nzval[101] = 1//2
            nzval[102] = 1//2
            nzval[103] = 1//2
            nzval[104] = -1//2
            nzval[105] = -1//2
            nzval[106] = -1//2
            nzval[107] = -1//2
            nzval[108] = 1//2
            nzval[109] = -1//2
            nzval[110] = 1
            nzval[111] = -1//1
            nzval[112] = 1//2
            nzval[113] = 1//2
            nzval[114] = 1//2
            nzval[115] = 1//2
            nzval[116] = -1//1
            nzval[117] = 1
            nzval[118] = -1//2
            nzval[119] = 1//2
            nzval[120] = 1//2
            nzval[121] = -1//2
            nzval[122] = 1//2
            nzval[123] = -1//1
            nzval[124] = 1
            nzval[125] = -1//2
            nzval[126] = -1//2
            nzval[127] = -1//2
            nzval[128] = -1//2
            nzval[129] = 1//2
            nzval[130] = -1//2
            nzval[131] = 1//2
            nzval[132] = 1//2
            nzval[133] = 1//2
            nzval[134] = -1//1
            nzval[135] = 1
            nzval[136] = -1//2
            nzval[137] = 1
            nzval[138] = -1//1
            nzval[139] = -1//2
            nzval[140] = -1//2
            nzval[141] = 1//2
            nzval[142] = -1//2
            nzval[143] = 1//2
            nzval[144] = -1//2
            nzval[145] = 1
            nzval[146] = -2
            nzval[147] = 2
            nzval[148] = 1
            nzval[149] = 2
            nzval[150] = 2
            nzval[151] = -1
            nzval[152] = -1
            nzval[153] = 1
            nzval[154] = 2
            nzval[155] = -2
            nzval[156] = -1
            nzval[157] = 2
            nzval[158] = 1
            nzval[159] = 2
            nzval[160] = -1
            nzval[161] = 1
            nzval[162] = -2
            nzval[163] = 2
            nzval[164] = -1
            nzval[165] = 2
            nzval[166] = -1
            nzval[167] = 2
            nzval[168] = 1
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:358 =#
            return A
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:360 =#
        function se3_updateB!(B, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:360 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:361 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:362 =#
            nzval = B.nzval
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:363 =#
            nzval[1] = (/)(1, (getindex)(_c, 1))
            nzval[2] = (/)(1, (getindex)(_c, 1))
            nzval[3] = (/)(1, (getindex)(_c, 1))
            nzval[4] = (/)(1, (getindex)(_c, 2))
            nzval[5] = (/)(1, (getindex)(_c, 3))
            nzval[6] = (/)(1, (getindex)(_c, 4))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:364 =#
            return B
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:366 =#
        function se3_updateC!(C, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:366 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:367 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:368 =#
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:346 =#
                nzval = (C[1]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:347 =#
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
                nzval[13] = (/)(1, (getindex)(_c, 1))
                nzval[14] = (/)(1, (getindex)(_c, 1))
                nzval[15] = (/)(1, (getindex)(_c, 1))
                nzval[16] = (/)(1, (getindex)(_c, 1))
                nzval[17] = (/)(1, (getindex)(_c, 1))
                nzval[18] = (/)(1, (getindex)(_c, 1))
                nzval[19] = (/)(1, (getindex)(_c, 1))
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:346 =#
                nzval = (C[2]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:347 =#
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
                nzval[13] = (/)(1, (getindex)(_c, 1))
                nzval[14] = (/)(1, (getindex)(_c, 1))
                nzval[15] = (/)(1, (getindex)(_c, 1))
                nzval[16] = (/)(1, (getindex)(_c, 1))
                nzval[17] = (/)(1, (getindex)(_c, 1))
                nzval[18] = (/)(1, (getindex)(_c, 1))
                nzval[19] = (/)(1, (getindex)(_c, 1))
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:346 =#
                nzval = (C[3]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:347 =#
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
                nzval[13] = (/)(1, (getindex)(_c, 1))
                nzval[14] = (/)(1, (getindex)(_c, 1))
                nzval[15] = (/)(1, (getindex)(_c, 1))
                nzval[16] = (/)(1, (getindex)(_c, 1))
                nzval[17] = (/)(1, (getindex)(_c, 1))
                nzval[18] = (/)(1, (getindex)(_c, 1))
                nzval[19] = (/)(1, (getindex)(_c, 1))
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:346 =#
                nzval = (C[4]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:347 =#
                nzval[1] = (/)(1, (getindex)(_c, 2))
                nzval[2] = (/)(1, (getindex)(_c, 2))
                nzval[3] = (/)(1, (getindex)(_c, 2))
                nzval[4] = (/)(1, (getindex)(_c, 2))
                nzval[5] = (/)(1, (getindex)(_c, 2))
                nzval[6] = (/)(1, (getindex)(_c, 2))
                nzval[7] = (/)(1, (getindex)(_c, 2))
                nzval[8] = (/)(2, (getindex)(_c, 2))
                nzval[9] = (/)(1, (getindex)(_c, 2))
                nzval[10] = (/)(1, (getindex)(_c, 2))
                nzval[11] = (/)(1, (getindex)(_c, 2))
                nzval[12] = (/)(1, (getindex)(_c, 2))
                nzval[13] = (/)(1, (getindex)(_c, 2))
                nzval[14] = (/)(1, (getindex)(_c, 2))
                nzval[15] = (/)(1, (getindex)(_c, 2))
                nzval[16] = (/)(1, (getindex)(_c, 2))
                nzval[17] = (/)(1, (getindex)(_c, 2))
                nzval[18] = (/)(1, (getindex)(_c, 2))
                nzval[19] = (/)(1, (getindex)(_c, 2))
                nzval[20] = (/)(1, (getindex)(_c, 2))
                nzval[21] = (/)(2, (getindex)(_c, 2))
                nzval[22] = (/)(2, (getindex)(_c, 2))
                nzval[23] = (/)(2, (getindex)(_c, 2))
                nzval[24] = (/)(2, (getindex)(_c, 2))
                nzval[25] = (/)(1, (getindex)(_c, 2))
                nzval[26] = (/)(1, (getindex)(_c, 2))
                nzval[27] = (/)(1, (getindex)(_c, 2))
                nzval[28] = (/)(1, (getindex)(_c, 2))
                nzval[29] = (/)(1, (getindex)(_c, 2))
                nzval[30] = (/)(1, (getindex)(_c, 2))
                nzval[31] = (/)(1, (getindex)(_c, 2))
                nzval[32] = (/)(1, (getindex)(_c, 2))
                nzval[33] = (/)(2, (getindex)(_c, 2))
                nzval[34] = (/)(2, (getindex)(_c, 2))
                nzval[35] = (/)(2, (getindex)(_c, 2))
                nzval[36] = (/)(1, (getindex)(_c, 2))
                nzval[37] = (/)(1, (getindex)(_c, 2))
                nzval[38] = (/)(1, (getindex)(_c, 2))
                nzval[39] = (/)(1, (getindex)(_c, 2))
                nzval[40] = (/)(1, (getindex)(_c, 2))
                nzval[41] = (/)(1, (getindex)(_c, 2))
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:346 =#
                nzval = (C[5]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:347 =#
                nzval[1] = (/)(1, (getindex)(_c, 3))
                nzval[2] = (/)(1, (getindex)(_c, 3))
                nzval[3] = (/)(1, (getindex)(_c, 3))
                nzval[4] = (/)(1, (getindex)(_c, 3))
                nzval[5] = (/)(1, (getindex)(_c, 3))
                nzval[6] = (/)(1, (getindex)(_c, 3))
                nzval[7] = (/)(1, (getindex)(_c, 3))
                nzval[8] = (/)(1, (getindex)(_c, 3))
                nzval[9] = (/)(2, (getindex)(_c, 3))
                nzval[10] = (/)(1, (getindex)(_c, 3))
                nzval[11] = (/)(1, (getindex)(_c, 3))
                nzval[12] = (/)(1, (getindex)(_c, 3))
                nzval[13] = (/)(1, (getindex)(_c, 3))
                nzval[14] = (/)(1, (getindex)(_c, 3))
                nzval[15] = (/)(1, (getindex)(_c, 3))
                nzval[16] = (/)(1, (getindex)(_c, 3))
                nzval[17] = (/)(1, (getindex)(_c, 3))
                nzval[18] = (/)(1, (getindex)(_c, 3))
                nzval[19] = (/)(1, (getindex)(_c, 3))
                nzval[20] = (/)(1, (getindex)(_c, 3))
                nzval[21] = (/)(1, (getindex)(_c, 3))
                nzval[22] = (/)(1, (getindex)(_c, 3))
                nzval[23] = (/)(1, (getindex)(_c, 3))
                nzval[24] = (/)(1, (getindex)(_c, 3))
                nzval[25] = (/)(2, (getindex)(_c, 3))
                nzval[26] = (/)(2, (getindex)(_c, 3))
                nzval[27] = (/)(2, (getindex)(_c, 3))
                nzval[28] = (/)(2, (getindex)(_c, 3))
                nzval[29] = (/)(1, (getindex)(_c, 3))
                nzval[30] = (/)(1, (getindex)(_c, 3))
                nzval[31] = (/)(1, (getindex)(_c, 3))
                nzval[32] = (/)(1, (getindex)(_c, 3))
                nzval[33] = (/)(1, (getindex)(_c, 3))
                nzval[34] = (/)(1, (getindex)(_c, 3))
                nzval[35] = (/)(1, (getindex)(_c, 3))
                nzval[36] = (/)(2, (getindex)(_c, 3))
                nzval[37] = (/)(2, (getindex)(_c, 3))
                nzval[38] = (/)(2, (getindex)(_c, 3))
                nzval[39] = (/)(1, (getindex)(_c, 3))
                nzval[40] = (/)(1, (getindex)(_c, 3))
                nzval[41] = (/)(1, (getindex)(_c, 3))
            end
            begin
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:346 =#
                nzval = (C[6]).nzval
                #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:347 =#
                nzval[1] = (/)(1, (getindex)(_c, 4))
                nzval[2] = (/)(1, (getindex)(_c, 4))
                nzval[3] = (/)(1, (getindex)(_c, 4))
                nzval[4] = (/)(1, (getindex)(_c, 4))
                nzval[5] = (/)(1, (getindex)(_c, 4))
                nzval[6] = (/)(1, (getindex)(_c, 4))
                nzval[7] = (/)(1, (getindex)(_c, 4))
                nzval[8] = (/)(1, (getindex)(_c, 4))
                nzval[9] = (/)(1, (getindex)(_c, 4))
                nzval[10] = (/)(2, (getindex)(_c, 4))
                nzval[11] = (/)(1, (getindex)(_c, 4))
                nzval[12] = (/)(1, (getindex)(_c, 4))
                nzval[13] = (/)(1, (getindex)(_c, 4))
                nzval[14] = (/)(1, (getindex)(_c, 4))
                nzval[15] = (/)(1, (getindex)(_c, 4))
                nzval[16] = (/)(1, (getindex)(_c, 4))
                nzval[17] = (/)(1, (getindex)(_c, 4))
                nzval[18] = (/)(1, (getindex)(_c, 4))
                nzval[19] = (/)(1, (getindex)(_c, 4))
                nzval[20] = (/)(1, (getindex)(_c, 4))
                nzval[21] = (/)(1, (getindex)(_c, 4))
                nzval[22] = (/)(1, (getindex)(_c, 4))
                nzval[23] = (/)(1, (getindex)(_c, 4))
                nzval[24] = (/)(1, (getindex)(_c, 4))
                nzval[25] = (/)(1, (getindex)(_c, 4))
                nzval[26] = (/)(1, (getindex)(_c, 4))
                nzval[27] = (/)(1, (getindex)(_c, 4))
                nzval[28] = (/)(1, (getindex)(_c, 4))
                nzval[29] = (/)(2, (getindex)(_c, 4))
                nzval[30] = (/)(2, (getindex)(_c, 4))
                nzval[31] = (/)(2, (getindex)(_c, 4))
                nzval[32] = (/)(2, (getindex)(_c, 4))
                nzval[33] = (/)(1, (getindex)(_c, 4))
                nzval[34] = (/)(1, (getindex)(_c, 4))
                nzval[35] = (/)(1, (getindex)(_c, 4))
                nzval[36] = (/)(1, (getindex)(_c, 4))
                nzval[37] = (/)(1, (getindex)(_c, 4))
                nzval[38] = (/)(1, (getindex)(_c, 4))
                nzval[39] = (/)(2, (getindex)(_c, 4))
                nzval[40] = (/)(2, (getindex)(_c, 4))
                nzval[41] = (/)(2, (getindex)(_c, 4))
            end
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:369 =#
            return C
        end
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:371 =#
        function se3_updateD!(D, constants)
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:371 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:372 =#
            _c = constants
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:373 =#
            nzval = D.nzval
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:374 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:375 =#
            return D
        end
    end
    #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:418 =#
    begin
        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:390 =#
        function se3_genarrays()
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:390 =#
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:391 =#
            n = 152
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:392 =#
            m = 6
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:393 =#
            A = SparseMatrixCSC(n, n, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 18, 19, 19, 20, 21, 22, 22, 23, 26, 29, 30, 33, 34, 35, 38, 41, 42, 45, 46, 47, 50, 53, 54, 57, 58, 59, 62, 65, 66, 69, 70, 70, 72, 74, 75, 78, 79, 80, 82, 85, 85, 87, 88, 89, 92, 94, 95, 97, 97, 98, 100, 102, 104, 105, 107, 109, 110, 112, 113, 114, 116, 118, 120, 121, 123, 125, 126, 128, 129, 130, 132, 134, 136, 137, 139, 141, 142, 144, 145, 146, 146, 147, 148, 149, 150, 151, 152, 152, 153, 154, 155, 155, 156, 157, 158, 158, 159, 160, 161, 162, 163, 164, 164, 165, 165, 166, 167, 168, 169], [13, 12, 11, 5, 4, 7, 6, 6, 7, 4, 5, 7, 6, 5, 4, 10, 9, 10, 8, 9, 8, 31, 32, 35, 38, 33, 34, 39, 36, 30, 37, 40, 41, 30, 33, 34, 39, 32, 35, 38, 37, 31, 36, 41, 40, 33, 30, 37, 40, 31, 36, 41, 34, 32, 35, 38, 39, 32, 31, 36, 41, 30, 37, 40, 35, 33, 34, 39, 38, 44, 48, 43, 45, 47, 42, 46, 50, 49, 44, 47, 49, 42, 46, 50, 43, 45, 48, 43, 42, 46, 50, 47, 49, 45, 44, 48, 15, 14, 18, 17, 19, 16, 20, 15, 16, 20, 17, 19, 22, 21, 23, 22, 16, 17, 19, 14, 21, 15, 22, 20, 15, 22, 18, 23, 16, 17, 19, 20, 17, 16, 20, 15, 22, 14, 23, 19, 18, 21, 15, 22, 19, 16, 20, 17, 1, 3, 2, 1, 2, 3, 1, 1, 2, 3, 1, 2, 1, 2, 3, 2, 3, 2, 1, 3, 1, 3, 2, 3], zeros(168))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:398 =#
            B = SparseMatrixCSC(n, m, [1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13], zeros(6))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:403 =#
            C = [begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:382 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], [42, 45, 48, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 75, 76, 77, 78, 79, 80], zeros(19))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:382 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], [43, 46, 49, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 81, 82, 83, 84, 85, 86], zeros(19))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:382 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], [44, 47, 50, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 87, 88, 89, 90, 91, 92], zeros(19))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:382 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 21, 21, 21, 21, 21, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42], [30, 31, 32, 33, 42, 43, 44, 24, 25, 26, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 51, 57, 63, 69, 52, 58, 64, 70, 53, 59, 65, 71, 75, 81, 87, 76, 82, 88, 77, 83, 89], zeros(41))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:382 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 21, 21, 21, 21, 21, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42], [34, 35, 36, 37, 45, 46, 47, 25, 27, 28, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 52, 58, 64, 70, 54, 60, 66, 72, 55, 61, 67, 73, 76, 82, 88, 78, 84, 90, 79, 85, 91], zeros(41))
                    end, begin
                        #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:382 =#
                        SparseMatrixCSC(n, n, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 21, 21, 21, 21, 21, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42], [38, 39, 40, 41, 48, 49, 50, 26, 28, 29, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 53, 59, 65, 71, 55, 61, 67, 73, 56, 62, 68, 74, 77, 83, 89, 79, 85, 91, 80, 86, 92], zeros(41))
                    end]
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:404 =#
            D = SparseVector(n, Int64[], zeros(0))
            #= /home/brian/.julia/dev/BilinearControl/examples/se3_dynamics.jl:408 =#
            return (A, B, C, D)
        end
    end
end