import taichi as ti
import numpy as np
import random

@ti.func
def SPD_project_2d(A):
    U, S, V = ti.svd(A)
    if U[0, 0]*V[0, 0] + U[1, 0]*V[1, 0] < 0:
        S[0, 0] = 0
    if U[0, 1]*V[0, 1] + U[1, 1]*V[1, 1] < 0:
        S[1, 1] = 0
    return U @ S @ V.transpose()

@ti.data_oriented
class SPD_Projector:
    def __init__(self, N, D, K) -> None:
        self.N, self.D, self.K = N, D, K
        self.T = ti.field(ti.f64, (N, D, D))
        self.Q = ti.field(ti.f64, (N, D, D))

    @ti.func
    def clear(self, T, Q, t, n):
        for i in range(n):
            for j in range(n):
                T[t, i, j] = 0
                Q[t, i, j] = i==j

    @ti.func
    def Householder(self, A, T, Q, t, n): # transform A[t, :, :] in-place
        for i in range(n-2):
            b = 0.0
            for j in range(i+1, n):
                b += A[t, j, i]**2
            b = b**0.5
            if b<1e-6:
                T[t, i, i] = -1
                for j in range(i+1, n):
                    A[t, i, j] = 0
            else:
                T[t, i, i] = 1
                if A[t, i+1, i]<0:
                    b *= -1
                T[t, i+1, i] = A[t, i+1, i] + b
                c = T[t, i+1, i]**2
                for j in range(i+2, n):
                    T[t, j, i] = A[t, j, i]
                    c += A[t, j, i]**2
                c = (2/c)**0.5
                for j in range(i+1, n):
                    T[t, j, i] *= c
                for j in range(i+1, n):
                    T[t, i, j] = 0
                for j in range(i+1, n):
                    for k in range(i+1, j+1):
                        T[t, i, j] += A[t, j, k] * T[t, k, i]
                    for k in range(j+1, n):
                        T[t, i, j] += A[t, k, j] * T[t, k, i]
                d = 0.0
                for j in range(i+1, n):
                    d += T[t, i, j] * T[t, j, i]
                d *= 0.5
                for j in range(i+1, n):
                    T[t, i, j] -= T[t, j, i] * d
                    A[t, i, j] = A[t, j, i] = 0
                A[t, i+1, i] = A[t, i, i+1] = -b
                for j in range(i+1, n):
                    for k in range(i+1, j+1):
                        A[t, j, k] -= T[t, i, j] * T[t, k, i] + T[t, i, k] * T[t, j, i]
                for k in range(n):
                    s = 0.0
                    for j in range(i+1, n):
                        s += Q[t, k, j] * T[t, j, i]
                    for j in range(i+1, n):
                        Q[t, k, j] -= s*T[t, j, i]
        A[t, n-2, n-1] = A[t, n-1, n-2]

    @ti.func
    def QR(self, A, T, Q, t, n, k):
        # subd = 0.0
        for j in range(k):
            m = 0
            for i in range(n-1):
                if ti.abs(A[t, i+1, i])>1e-5:
                    m = i+2
            if m==0:
                break
            a = A[t, m-2, m-2]
            b = A[t, m-2, m-1]
            c = A[t, m-1, m-1]
            d = (a-c)/2
            sd = 1 if d>0 else -1
            mu = c
            if ti.abs(b)>1e-6:
                mu -= (sd * b**2)/(ti.abs(d)+(d**2+b**2)**0.5)
            for i in range(n):
                A[t, i, i] -= mu
            for i in range(m-1):
                a = A[t, i, i]
                b = A[t, i, i+1]
                e = A[t, i+1, i]
                d = A[t, i+1, i+1]
                s = ti.abs(e/(a**2+e**2)**0.5) if ti.abs(e)>1e-5 else 0
                if a*e<0:
                    s *= -1
                c = ti.max(1-s**2, 0)**0.5
                T[t, 0, i] = s
                A[t, i, i] = a*c+e*s
                A[t, i, i+1] = b*c+d*s
                A[t, i+1, i+1] = d*c-b*s
                if i<n-2:
                    A[t, i+1, i+2] *= c
            for i in range(m-1):
                a = A[t, i, i]
                b = A[t, i, i+1]
                d = A[t, i+1, i+1]
                s = T[t, 0, i]
                c = ti.max(1-s**2, 0)**0.5
                A[t, i, i] = a*c+b*s
                A[t, i+1, i] = s*d
                A[t, i+1, i+1] = c*d
                for r in range(n):
                    a, b = Q[t, r, i], Q[t, r, i+1]
                    Q[t, r, i], Q[t, r, i+1] = a*c+b*s, -a*s+b*c
            # subd = 0.0
            for i in range(n-1):
                A[t, i, i+1] = A[t, i+1, i]
                # ti.atomic_max(subd, ti.abs(A[t, i, i+1]))
            for i in range(n):
                A[t, i, i] += mu
        # print(subd)

    @ti.func
    def project(self, A, t, n, debug=False):
        self.clear(self.T, self.Q, t, n)
        self.Householder(A, self.T, self.Q, t, n)
        self.QR(A, self.T, self.Q, t, n, self.K)
        for i in range(n):
            self.T[t, 0, i] = A[t, i, i]
        for i in range(n):
            for j in range(n):
                A[t, i, j] = 0
        for i in range(n):
            v = self.T[t, 0, i]
            if v>0:
                for j in range(n):
                    v2 = v*self.Q[t, j, i]
                    for k in range(n):
                        A[t, j, k] += v2*self.Q[t, k, i]

@ti.kernel
def test_kernel(A:ti.template(), p:ti.template()):
    for i in range(1):
        p.project(A, 0, 9)

if __name__ == '__main__':
    ti.init(ti.gpu, random_seed=5, debug=True, default_fp=ti.f64)
    random.seed(1)
    n = 9
    p = SPD_Projector(1, n, n)
    A = ti.field(ti.f64, (1, n, n))
    for i in range(n):
        for j in range(n):
            A[0, i, j] = random.random()
    for i in range(n):
        for j in range(i):
            A[0, i, j] = A[0, j, i]
    A0 = A.to_numpy(dtype='float64')[0].copy()
    test_kernel(A, p)
    A = A.to_numpy(dtype='float64')[0]
    print(np.linalg.eigh(A0)[0])
    print(np.linalg.eigh(A)[0])