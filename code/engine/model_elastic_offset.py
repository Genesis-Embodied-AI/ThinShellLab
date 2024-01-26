import argparse

import numpy as np

import taichi as ti
from sparse_solver import SparseMatrix
import os
import readfile

@ti.data_oriented
class Elastic:
    def __init__(self, dt, Len, offset, Nx, Ny, Nz, density=2000.0, load=False):
        self.E = 5e5
        self.nu = 0.0
        mu, lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))  # lambda = 0
        self.mu = ti.field(ti.f64, ())
        self.mu[None] = mu
        self.lam = ti.field(ti.f64, ())
        self.lam[None] = lam
        self.density = density
        self.dt = dt
        self.offset = offset
        self.gravity = ti.Vector.field(3, ti.f64, ())
        self.gravity[None] = ti.Vector([0, 0, -9.8])
        n_cube = np.array([int(Nx), int(Ny), int(Nz)])
        n_verts = int(n_cube[0] * n_cube[1] * n_cube[2])
        n_cells = 5 * np.product(n_cube - 1)
        dx = Len / (n_cube.max() - 1)

        self.n_verts = n_verts
        self.n_cells = n_cells
        self.n_cube = n_cube
        su = 0
        for i in range(3):
            su += (self.n_cube[i] - 1) * (self.n_cube[(i + 1) % 3] - 1)
        self.n_surfaces = int(2 * su * 2)

        self.load = False
        if load:
            self.n_verts, self.vertex = readfile.read_node("../data/ball.node")
            self.n_cells, self.tet_mesh = readfile.read_ele("../data/ball.ele")
            self.n_surfaces, self.surface_mesh = readfile.read_smesh("../data/ball.face")
            self.vertex = np.array(self.vertex)
            self.tet_mesh = np.array(self.tet_mesh)
            self.surface_mesh = np.array(self.surface_mesh)
            n_verts = self.n_verts
            n_cells = self.n_cells
            self.load = True


        self.F_vertices = ti.Vector.field(4, dtype=ti.i32, shape=n_cells)

        self.F_x = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.F_x_prev = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.F_ox = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.F_v = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.F_f = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        # self.F_mul_ans = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.F_m = ti.field(dtype=ti.f64, shape=n_verts)
        self.F_b = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)

        n_cells = (n_cube - 1).prod() * 5
        self.damping = 14.5

        self.F_B = ti.Matrix.field(3, 3, dtype=ti.f64, shape=n_cells)
        self.F_W = ti.field(dtype=ti.f64, shape=n_cells)

        self.dx = dx

        self.U = ti.field(float, ())


        self.f2v = ti.Vector.field(3, int, self.n_surfaces)
        self.offset_faces = 0
        self.body_idx = 0

        self.ext_force = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)

        self.d_mu = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.d_lam = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)

        # derivatives
        self.dim = 3
        self.dD = ti.Matrix.field(self.dim, self.dim, dtype=ti.f64, shape=(n_cells, 4, 3))
        self.dF = ti.Matrix.field(self.dim, self.dim, dtype=ti.f64, shape=(n_cells, 4, 3))
        self.dP = ti.Matrix.field(self.dim, self.dim, dtype=ti.f64, shape=(n_cells, 4, 3))
        self.dH = ti.Matrix.field(self.dim, self.dim, dtype=ti.f64, shape=(n_cells, 4, 3))

        self.uv = ti.Vector.field(2, dtype=ti.f64, shape=n_verts)
        self.uv1 = ti.Vector.field(2, dtype=ti.f64, shape=n_verts)
        self.uv2 = ti.Vector.field(2, dtype=ti.f64, shape=n_verts)

        
    @ti.kernel
    def compute_Hessian(self, A: ti.template(), spd: ti.i32):
        # A = M - dt * dt * K
        for i in range(self.n_verts):
            for j in range(3):
                A.H.add(3 * (i + self.offset) + j, 3 * (i + self.offset) + j, self.F_m[i] / (self.dt ** 2))

        for e in self.F_vertices:
            for n in range(4):
                for dim in range(self.dim):
                    for i in ti.static(range(self.dim)):
                        for j in ti.static(range(self.dim)):
                            self.dD[e, n, dim][i, j] = 0
                            self.dF[e, n, dim][i, j] = 0
                            self.dP[e, n, dim][i, j] = 0

            for n in ti.static(range(3)):
                for dim in ti.static(range(self.dim)):
                    self.dD[e, n, dim][dim, n] = 1
            for dim in ti.static(range(self.dim)):
                self.dD[e, 3, dim] = - (self.dD[e, 0, dim] + self.dD[e, 1, dim] + self.dD[e, 2, dim])

            for n in range(4):
                for dim in range(self.dim):
                    self.dF[e, n, dim] = self.dD[e, n, dim] @ self.F_B[e]  # !!! matrix multiplication

            F = self.Ds(self.F_vertices[e]) @ self.F_B[e]
            F_1 = F.inverse()
            F_1_T = F_1.transpose()
            J = ti.max(F.determinant(), 0.01)

            for n in range(4):
                for dim in range(self.dim):
                    for i in ti.static(range(self.dim)):
                        for j in ti.static(range(self.dim)):
                            # dF/dF_{ij}
                            dF = ti.Matrix([[0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0]])
                            dF[i, j] = 1

                            # dF^T/dF_{ij}
                            dF_T = dF.transpose()

                            # Tr( F^{-1} dF/dF_{ij} )
                            dTr = F_1_T[i, j]

                            dP_dFij = self.mu[None] * dF + (
                                        self.mu[None] - self.lam[None] * ti.log(J)) * F_1_T @ dF_T @ F_1_T + self.lam[None] * dTr * F_1_T
                            dFij_ndim = self.dF[e, n, dim][i, j]

                            self.dP[e, n, dim] += dP_dFij * dFij_ndim

            for n in range(4):
                for dim in range(self.dim):
                    self.dH[e, n, dim] = - self.F_W[e] * self.dP[e, n, dim] @ self.F_B[e].transpose()

            for n in ti.static(range(4)):
                i = self.F_vertices[e][n]
                for dim in ti.static(range(self.dim)):
                    ind = (i + self.offset) * self.dim + dim
                    for j in ti.static(range(3)):
                        idx = self.F_vertices[e][j] + self.offset
                        A.add_H(idx * 3 + 0, ind, -self.dH[e, n, dim][0, j])  # df_{jx}/dx_{ndim}
                        A.add_H(idx * 3 + 1, ind, -self.dH[e, n, dim][1, j])  # df_{jy}/dx_{ndim}
                        A.add_H(idx * 3 + 2, ind, -self.dH[e, n, dim][2, j])  # df_{jz}/dx_{ndim}

                    # df_{3x}/dx_{ndim}
                    idx = self.F_vertices[e][3] + self.offset
                    A.add_H(idx * 3 + 0, ind, self.dH[e, n, dim][0, 0] + self.dH[e, n, dim][0, 1] + self.dH[e, n, dim][0, 2])
                    # df_{3y}/dx_{ndim}
                    A.add_H(idx * 3 + 1, ind, self.dH[e, n, dim][1, 0] + self.dH[e, n, dim][1, 1] + self.dH[e, n, dim][1, 2])
                    # df_{3x}/dx_{ndim}
                    A.add_H(idx * 3 + 2, ind, self.dH[e, n, dim][2, 0] + self.dH[e, n, dim][2, 1] + self.dH[e, n, dim][2, 2])


    @ti.func
    def Ds(self, verts):
        return ti.Matrix.cols([self.F_x[verts[i]] - self.F_x[verts[3]] for i in range(3)])
    
    @ti.func
    def ssvd(self, F):
        U, sig, V = ti.svd(F)
        if U.determinant() < 0:
            for i in ti.static(range(3)):
                U[i, 2] *= -1
            sig[2, 2] = -sig[2, 2]
        if V.determinant() < 0:
            for i in ti.static(range(3)):
                V[i, 2] *= -1
            sig[2, 2] = -sig[2, 2]
        return U, sig, V


    @ti.func
    def get_force_func(self, c, verts):
        F = self.Ds(verts) @ self.F_B[c]
        F_T = F.inverse().transpose()
        J = ti.max(F.determinant(), 0.01)
        P = self.mu[None] * (F - F_T) + self.lam[None] * ti.log(J) * F_T
        H = -self.F_W[c] * P @ self.F_B[c].transpose()
        for i in ti.static(range(3)):
            force = ti.Vector([H[j, i] for j in range(3)])
            self.F_f[verts[i]] += force
            self.F_f[verts[3]] -= force


    @ti.kernel
    def get_force(self):
        self.F_f.fill(0)
        for c in self.F_vertices:
            self.get_force_func(c, self.F_vertices[c])
        for u in self.F_f:
            self.F_f[u] += self.gravity[None] * self.F_m[u]
            self.F_f[u] += self.ext_force[u]

    @ti.kernel
    def compute_residual(self):
        for i in self.F_b:
            self.F_b[i] = self.F_m[i] * (self.F_x[i] - self.F_x_prev[i] - self.F_v[i] * self.dt) / (self.dt ** 2) - self.F_f[i]

    @ti.kernel
    def update_vel(self):
        for i in self.F_v:
            self.F_v[i] = (self.F_x[i] - self.F_x_prev[i]) / self.dt * ti.exp(-self.dt * self.damping)

    @ti.kernel
    def get_prev_pos(self):
        for i in self.F_x_prev:
            self.F_x_prev[i] = self.F_x[i]

    @ti.kernel
    def compute_residual_norm(self) -> ti.f64:
        result = 0.0
        for i in self.F_b:
            result += self.F_b[i].dot(self.F_b[i])
        return ti.sqrt(result)

    @ti.kernel
    def init_pos(self, offsetx: ti.f64, offsety: ti.f64, offsetz: ti.f64):
        for u in self.F_x:
            self.F_x[u] = self.F_ox[u]
            self.F_v[u] = [0.0] * 3
            self.F_f[u] = [0.0] * 3
            self.F_m[u] = 0.0

        for c in self.F_vertices:
            F = self.Ds(self.F_vertices[c])
            self.F_B[c] = F.inverse()
            self.F_W[c] = ti.abs(F.determinant()) / 6
            for i in range(4):
                self.F_m[self.F_vertices[c][i]] += self.F_W[c] / 4 * self.density

        for u in self.F_x:
            self.F_x[u].x += offsetx
            self.F_x[u].y += offsety
            self.F_x[u].z += offsetz

    @ti.kernel
    def init_pos_arch(self, offsetx: ti.f64, offsety: ti.f64, offsetz: ti.f64, arch: ti.f64):
        for u in self.F_x:
            self.F_x[u] = self.F_ox[u]
            self.F_v[u] = [0.0] * 3
            self.F_f[u] = [0.0] * 3
            self.F_m[u] = 0.0
        for I in ti.grouped(ti.ndrange(*(self.n_cube))):
            self.F_x[self.i2p(I)].z += arch * ti.sin(ti.cast(I.x, ti.f64) / ti.cast(self.n_cube[0] - 1, ti.f64) * 3.1415926)
        for c in self.F_vertices:
            F = self.Ds(self.F_vertices[c])
            self.F_B[c] = F.inverse()
            self.F_W[c] = ti.abs(F.determinant()) / 6
            for i in range(4):
                self.F_m[self.F_vertices[c][i]] += self.F_W[c] / 4 * self.density
        for u in self.F_x:
            self.F_x[u].x += offsetx
            self.F_x[u].y += offsety
            self.F_x[u].z += offsetz

    @ti.kernel
    def floor_bound(self):
        for u in self.F_x:
            if self.F_x[u].y < 0:
                self.F_x[u].y = 0
                if self.F_v[u].y < 0:
                    self.F_v[u].y = 0

    @ti.func
    def i2p(self, I):
        return (I.x * self.n_cube[1] + I.y) * self.n_cube[2] + I.z


    @ti.func
    def set_element(self, e, I, verts):
        for i in ti.static(range(3 + 1)):
            self.F_vertices[e][i] = self.i2p(I + (([verts[i] >> k
                                        for k in range(3)] ^ I) & 1))


    @ti.kernel
    def get_vertices(self):
        '''
        This kernel partitions the cube into tetrahedrons.
        Each unit cube is divided into 5 tetrahedrons.
        '''
        for I in ti.grouped(ti.ndrange(*(self.n_cube - 1))):
            e = ((I.x * (self.n_cube[1] - 1) + I.y) * (self.n_cube[2] - 1) + I.z) * 5
            for i, j in ti.static(enumerate([0, 3, 5, 6])):
                self.set_element(e + i, I, (j, j ^ 1, j ^ 2, j ^ 4))
            self.set_element(e + 4, I, (1, 2, 4, 7))
        for I in ti.grouped(ti.ndrange(*(self.n_cube))):
            self.F_ox[self.i2p(I)] = I * self.dx

        for I in ti.grouped(ti.ndrange(*(self.n_cube))):
            if I.z == 0 or I.z == self.n_cube[2] - 1:
                self.uv[self.i2p(I)] = ti.Vector([I.x / (self.n_cube[0] - 1), I.y / (self.n_cube[1] - 1)])
            if I.x == 0 or I.x == self.n_cube[0] - 1:
                self.uv1[self.i2p(I)] = ti.Vector([I.y / (self.n_cube[1] - 1), I.z / (self.n_cube[2] - 1)])
            if I.y == 0 or I.y == self.n_cube[1] - 1:
                self.uv2[self.i2p(I)] = ti.Vector([I.x / (self.n_cube[0] - 1), I.z / (self.n_cube[2] - 1)])
            
    @ti.kernel
    def compute_energy(self):
        self.U[None] = 0
        for c in self.F_x:
            self.U[None] += -self.F_m[c] * self.gravity[None].dot(self.F_x[c])
            self.U[None] += -self.ext_force[c].dot(self.F_x[c])

        for c in self.F_x:
            X = self.F_x[c] - self.F_x_prev[c] - self.F_v[c] * self.dt
            self.U[None] += 0.5 * self.F_m[c] * X.dot(X) / (self.dt ** 2)

        for c in self.F_vertices:
            F_i = self.Ds(self.F_vertices[c]) @ self.F_B[c]            
            log_J_i = ti.log(ti.max(0.01, F_i.determinant()))
            phi_i = self.mu[None] / 2 * ((F_i.transpose() @ F_i).trace() - 3)
            phi_i -= self.mu[None] * log_J_i
            phi_i += self.lam[None] / 2 * log_J_i**2
            self.U[None] += self.F_W[c] * phi_i

    @ti.func
    def check(self, u):
        ans = 0
        rest = u
        for i in ti.static(range(3)):
            k = rest % self.n_cube[2 - i]
            rest = rest // self.n_cube[2 - i]
            if k == 0:
                ans |= (1 << (i * 2))
            if k == self.n_cube[2 - i] - 1:
                ans |= (1 << (i * 2 + 1))
        return ans

    @ti.kernel
    def get_surface_indices(self):
        # calculate all the meshes on surface
        cnt = 0
        for c in self.F_vertices:
            if c % 5 != 4:
                for i in ti.static([0, 2, 3]):
                    verts = [self.F_vertices[c][(i + j) % 4] for j in range(3)]
                    sum_ = self.check(verts[0]) & self.check(verts[1]) & self.check(verts[2])
                    if sum_:
                        m = ti.atomic_add(cnt, 1)
                        # det = ti.Matrix.rows([
                        #     self.F_x[verts[i]] - [0.5, 1.5, 0.5] for i in range(3)
                        # ]).determinant()
                        # if det < 0:
                        #     tmp = verts[1]
                        #     verts[1] = verts[2]
                        #     verts[2] = tmp
                        verts3 = self.F_vertices[c][(i + 3) % 4]
                        normal = (self.F_x[verts[1]] - self.F_x[verts[0]]).cross(self.F_x[verts[2]] - self.F_x[verts[0]])
                        if normal.dot(self.F_x[verts3] - self.F_x[verts[0]]) > 0:
                            tmp = verts[1]
                            verts[1] = verts[2]
                            verts[2] = tmp
                        # normal = (self.F_x[verts[1]] - self.F_x[verts[0]]).cross(
                        #     self.F_x[verts[2]] - self.F_x[verts[0]])
                        # if normal.dot(self.F_x[verts3] - self.F_x[verts[0]]) > 0:
                        #     print("????????")
                        self.f2v[m][0] = verts[0]
                        self.f2v[m][1] = verts[1]
                        self.f2v[m][2] = verts[2]

    @ti.kernel
    def init_normal(self, offset_x: ti.f64, offset_y: ti.f64, offset_z: ti.f64):
        for i in self.f2v:

            p1 = self.F_x[self.f2v[i][0]]
            p2 = self.F_x[self.f2v[i][1]]
            p3 = self.F_x[self.f2v[i][2]]
            n = (p2 - p1).cross(p3 - p1).normalized()

            # pointing to outside
            inner_point = ti.Vector([offset_x, offset_y, offset_z])

            if (n.dot(inner_point - p1) > 0):
                tmp = self.f2v[i][1]
                self.f2v[i][1] = self.f2v[i][2]
                self.f2v[i][2] = tmp

    def init(self, offsetx, offsety, offsetz):
        if not self.load:
            self.get_vertices()
            self.init_pos(offsetx, offsety, offsetz)
            self.get_surface_indices()
        else:
            self.F_vertices.from_numpy(self.tet_mesh)
            self.f2v.from_numpy(self.surface_mesh)
            self.F_ox.from_numpy(self.vertex)
            self.init_pos(offsetx, offsety, offsetz)
            self.init_normal(offsetx, offsety, offsetz)


    def init_arch(self, offsetx, offsety, offsetz, arch):
        self.get_vertices()
        self.init_pos_arch(offsetx, offsety, offsetz, arch)
        self.get_surface_indices()

    @ti.kernel
    def compute_deri(self):
        self.F_f.fill(0)
        for c in self.F_vertices:
            verts = self.F_vertices[c]
            F = self.Ds(verts) @ self.F_B[c]
            F_T = F.inverse().transpose()
            J = ti.max(F.determinant(), 0.01)
            P1 = self.mu[None] * (F - F_T)
            P2 = self.lam[None] * ti.log(J) * F_T
            H1 = -self.F_W[c] * P1 @ self.F_B[c].transpose()
            H2 = -self.F_W[c] * P2 @ self.F_B[c].transpose()
            for i in ti.static(range(3)):
                force = ti.Vector([H1[j, i] for j in range(3)])
                self.d_mu[verts[i]] += force / self.mu[None]
                self.d_mu[verts[3]] -= force / self.mu[None]
                force = ti.Vector([H2[j, i] for j in range(3)])
                self.d_lam[verts[i]] += force / self.lam[None]
                self.d_lam[verts[3]] -= force / self.lam[None]

    @ti.kernel
    def check_reverse(self, ii: ti.i64):
        for i in self.F_x:
            if i % self.n_cube[2] == 0:
                i1 = i + 1
                if self.F_x[i].z > self.F_x[i1].z:
                    print("reverse lowest layer!!!!", ii)

            if i % self.n_cube[2] == self.n_cube[2] - 1:
                i1 = i - 1
                if self.F_x[i].z < self.F_x[i1].z:
                    print("reverse top layer!!!!", ii)
