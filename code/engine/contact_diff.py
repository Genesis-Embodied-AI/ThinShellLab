import taichi as ti
import taichi.math as tm

@ti.func
def det(a:tm.vec3, b:tm.vec3, c:tm.vec3, diff, H, H_idx, G, G_idx):
    d = a[0]*b[1]*c[2] + a[1]*b[2]*c[0] + a[2]*b[0]*c[1] \
        - a[2]*b[1]*c[0] - a[1]*b[0]*c[2] - a[0]*b[2]*c[1]
    if diff:
        grad_a = tm.vec3(b[1]*c[2]-b[2]*c[1], b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0])
        grad_b = tm.vec3(c[1]*a[2]-c[2]*a[1], c[2]*a[0]-c[0]*a[2], c[0]*a[1]-c[1]*a[0])
        grad_c = tm.vec3(a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
        for i in range(3):
            G[G_idx, 0*3+i] = grad_a[i]
            G[G_idx, 1*3+i] = grad_b[i]
            G[G_idx, 2*3+i] = grad_c[i]
        for i in range(3):
            j = i+1 if i<2 else 0
            k = i-1 if i>0 else 2
            H[H_idx, 0*3+i, 1*3+j] = H[H_idx, 1*3+j, 0*3+i] = c[k]
            H[H_idx, 1*3+i, 2*3+j] = H[H_idx, 2*3+j, 1*3+i] = a[k]
            H[H_idx, 2*3+i, 0*3+j] = H[H_idx, 0*3+j, 2*3+i] = b[k]
            H[H_idx, 1*3+i, 0*3+j] = H[H_idx, 0*3+j, 1*3+i] = -c[k]
            H[H_idx, 2*3+i, 1*3+j] = H[H_idx, 1*3+j, 2*3+i] = -a[k]
            H[H_idx, 0*3+i, 2*3+j] = H[H_idx, 2*3+j, 0*3+i] = -b[k]
    return d

@ti.func
def cross(a:tm.vec3, b:tm.vec3, diff, H, H_idx, G, G_idx):
    a0, a1, a2 = a[0], a[1], a[2]
    b0, b1, b2 = b[0], b[1], b[2]
    c1 = a[0]*b[1]-b[0]*a[1]
    c2 = a[0]*b[2]-b[0]*a[2]
    c3 = a[1]*b[2]-b[1]*a[2]
    c0 = c1**2 + c2**2 + c3**2
    c = c0**0.5

    if diff:
        grad_a = tm.vec3( c1*b[1]+c2*b[2], -c1*b[0]+c3*b[2], -c2*b[0]-c3*b[1]) / c
        grad_b = tm.vec3(-c1*a[1]-c2*a[2],  c1*a[0]-c3*a[2],  c2*a[0]+c3*a[1]) / c
        for i in range(3):
            G[G_idx, 0*3+i] = grad_a[i]
            G[G_idx, 1*3+i] = grad_b[i]

        # Hessian from SymPy
        x0 = b1**2
        x1 = b2**2
        x2 = a0*b1
        x3 = a1*b0
        x4 = x2 - x3
        x5 = a0*b2
        x6 = a2*b0
        x7 = x5 - x6
        x8 = a1*b2
        x9 = a2*b1
        x10 = x8 - x9
        x11 = x10**2 + x4**2 + x7**2
        x12 = x11**(-0.5)
        x13 = b1*x4 + b2*x7
        x14 = x11**(-1.5)
        x15 = b0*x12
        x16 = b0*x4 - b2*x10
        x17 = -b1*x15 + x13*x14*x16
        x18 = b0*x7 + b1*x10
        x19 = -b2*x15 + x13*x14*x18
        x20 = a1*b1
        x21 = a2*b2
        x22 = a1*x4 + a2*x7
        x23 = -x12*(x20 + x21) + x13*x14*x22
        x24 = a0*x4 - a2*x10
        x25 = x13*x14
        x26 = x12*(2.0*x2 - x3) - x24*x25
        x27 = a0*x7 + a1*x10
        x28 = x12*(2.0*x5 - x6) - x25*x27
        x29 = b0**2
        x30 = x14*x16
        x31 = -b1*b2*x12 - x18*x30
        x32 = x30
        x33 = -x12*(x2 - 2.0*x3) - x22*x32
        x34 = a0*b0
        x35 = -x12*(x21 + x34) + x14*x16*x24
        x36 = x12*(2.0*x8 - x9) + x27*x32
        x37 = x18
        x38 = -x12*(x5 - 2.0*x6) - x14*x22*x37
        x39 = -x12*(x8 - 2.0*x9) + x14*x24*x37
        x40 = -x12*(x20 + x34) + x14*x18*x27
        x41 = a1**2
        x42 = a2**2
        x43 = a0*x12
        x44 = -a1*x43 + x14*x22*x24
        x45 = -a2*x43 + x14*x22*x27
        x46 = a0**2
        x47 = -a1*a2*x12 - x14*x24*x27

        H[H_idx, 0, 0] = x12*(x0 + x1) - x13**2*x14
        H[H_idx, 0, 1] = x17
        H[H_idx, 0, 2] = x19
        H[H_idx, 0, 3] = x23
        H[H_idx, 0, 4] = x26
        H[H_idx, 0, 5] = x28
        H[H_idx, 1, 0] = x17
        H[H_idx, 1, 1] = x12*(x1 + x29) - x14*x16**2
        H[H_idx, 1, 2] = x31
        H[H_idx, 1, 3] = x33
        H[H_idx, 1, 4] = x35
        H[H_idx, 1, 5] = x36
        H[H_idx, 2, 0] = x19
        H[H_idx, 2, 1] = x31
        H[H_idx, 2, 2] = x12*(x0 + x29) - x14*x18**2
        H[H_idx, 2, 3] = x38
        H[H_idx, 2, 4] = x39
        H[H_idx, 2, 5] = x40
        H[H_idx, 3, 0] = x23
        H[H_idx, 3, 1] = x33
        H[H_idx, 3, 2] = x38
        H[H_idx, 3, 3] = x12*(x41 + x42) - x14*x22**2
        H[H_idx, 3, 4] = x44
        H[H_idx, 3, 5] = x45
        H[H_idx, 4, 0] = x26
        H[H_idx, 4, 1] = x35
        H[H_idx, 4, 2] = x39
        H[H_idx, 4, 3] = x44
        H[H_idx, 4, 4] = x12*(x42 + x46) - x14*x24**2
        H[H_idx, 4, 5] = x47
        H[H_idx, 5, 0] = x28
        H[H_idx, 5, 1] = x36
        H[H_idx, 5, 2] = x40
        H[H_idx, 5, 3] = x45
        H[H_idx, 5, 4] = x47
        H[H_idx, 5, 5] = x12*(x41 + x46) - x14*x27**2
    return c