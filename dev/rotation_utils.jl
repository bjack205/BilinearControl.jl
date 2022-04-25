
function lmult(q)
    w, x, y, z = q
    SA[
        w -x -y -z;
        x  w -z  y;
        y  z  w -x;
        z -y  x  w;
    ]
end

function rmult(q)
    w, x, y, z = q
    SA[
        w -x -y -z;
        x  w  z -y;
        y -z  w  x;
        z  y -x  w;
    ]
end

function qrot(q)
    w,x,y,z = q
    ww = (w * w)
    xx = (x * x)
    yy = (y * y)
    zz = (z * z)

    xw = (w * x)
    xy = (x * y)
    xz = (x * z)
    yw = (y * w)
    yz = (y * z)
    zw = (w * z)

    
    A11 = ww + xx - yy - zz
    A21 = 2 * (xy + zw)
    A31 = 2 * (xz - yw)
    A12 = 2 * (xy - zw)
    A22 = ww - xx + yy - zz
    A32 = 2 * (yz + xw)
    A13 = 2 * (xz + yw)
    A23 = 2 * (yz - xw)
    A33 = ww - xx - yy + zz

    SA[
        A11 A12 A13
        A21 A22 A23
        A31 A32 A33
    ]
end

function skew(v::AbstractVector)
    @SMatrix [0   -v[3]  v[2];
              v[3] 0    -v[1];
             -v[2] v[1]  0]
end
