import argparse

import numpy as np

import taichi as ti
from .sparse_solver import SparseMatrix
from . import readfile
from .linalg import SPD_Projector
import matplotlib.pyplot as plt

@ti.data_oriented
class Elastic:
    def __init__(self, dt, offset, ratio):
        self.E = 300000
        self.nu = 0.2
        mu, lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))  # lambda = 0
        self.mu = ti.field(ti.f64, ())
        self.mu[None] = mu
        self.lam = ti.field(ti.f64, ())
        self.lam[None] = lam
        self.alpha = ti.field(ti.f64, ())
        self.alpha[None] = 1 + mu / lam
        self.density = 2000.0
        self.dt = dt
        self.offset = offset
        self.gravity = ti.Vector.field(3, ti.f64, ())
        self.gravity[None] = ti.Vector([0, 0, -9.8])
        self.ratio = ratio
        self.frozen_cnt = 0
        self.surf_point = 0

        n_verts, self.F_ox_array = readfile.read_node()
        n_cells, self.F_vertices_array = readfile.read_ele()
        self.n_surfaces, self.f2v_array = readfile.read_smesh()
        self.F_ox_array = np.array(self.F_ox_array)
        self.F_vertices_array = np.array(self.F_vertices_array)
        self.f2v_array = np.array(self.f2v_array)
        self.n_verts = n_verts
        self.n_cells = n_cells
        self.is_surface = ti.field(dtype=ti.i32, shape=n_verts)
        self.count()

        self.F_vertices = ti.Vector.field(4, dtype=ti.i32, shape=n_cells)

        self.F_x = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.F_x_prev = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.F_ox = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.F_v = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.F_f = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        # self.F_mul_ans = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.F_m = ti.field(dtype=ti.f64, shape=n_verts)
        self.F_b = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)

        self.damping = 14.5

        self.F_B = ti.Matrix.field(3, 3, dtype=ti.f64, shape=n_cells)
        self.F_W = ti.field(dtype=ti.f64, shape=n_cells)

        self.n_verts = n_verts
        self.n_cells = n_cells

        self.U = ti.field(float, ())

        self.f2v = ti.Vector.field(3, int, self.n_surfaces)
        self.offset_faces = 0
        self.body_idx = 0

        self.ext_force = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)

        self.d_mu = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.d_lam = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)

        # derivatives
        self.dim = 3
        self.H_e = ti.field(dtype=ti.f64, shape=(n_cells, 9, 9))
        self.H_proj = SPD_Projector(n_cells, 9, 20)

        self.normals = ti.Vector.field(3, ti.f64, self.n_surfaces)
        self.normal_points = ti.Vector.field(3, ti.f64, self.n_surfaces)

    @ti.kernel
    def compute_Hessian(self, A: ti.template(), spd: ti.i32):
        # A = M - dt * dt * K
        for i in range(self.n_verts):
            for j in range(3):
                A.H.add(3 * (i + self.offset) + j, 3 * (i + self.offset) + j, self.F_m[i] / (self.dt ** 2))

        for e in self.F_vertices:
            F = self.Ds(self.F_vertices[e]) @ self.F_B[e]
            F_inv = F.inverse()
            F_inv_T = F_inv.transpose()
            J = F.determinant()

            for n in range(3):
                for dim in range(self.dim):
                    dD = ti.Matrix.zero(ti.f64, 3, 3)
                    dD[dim, n] = 1
                    dF = dD @ self.F_B[e]  # !!! matrix multiplication
                    dF_T = dF.transpose()

                    # Tr(F^{-1}dF)
                    dTr = (F_inv @ dF).trace()

                    dP = -self.mu[None] * dF
                    dP -= self.lam[None] * 2 * J ** 2 * dTr * F_inv_T
                    dP += self.lam[None] * self.alpha[None] * J * dTr * F_inv_T
                    dP += self.lam[None] * (J - self.alpha[None]) * J * F_inv_T @ dF_T @ F_inv_T

                    dH = - self.F_W[e] * dP @ self.F_B[e].transpose()
                    for i, j in ti.static(ti.ndrange(3, 3)):
                        self.H_e[e, n * 3 + dim, i * 3 + j] = dH[j, i]

            # f = self.check_symmetry(e)
            if spd:
                self.H_proj.project(self.H_e, e, 9)

            idx = self.F_vertices[e] + self.offset
            for j, k in ti.ndrange(3, 3):
                for j2, k2 in ti.ndrange(3, 3):
                    h = self.H_e[e, k * 3 + j, k2 * 3 + j2]
                    A.add_H(idx[k] * 3 + j, idx[k2] * 3 + j2, h)
                    A.add_H(idx[k] * 3 + j, idx[3] * 3 + j2, -h)
                    A.add_H(idx[3] * 3 + j, idx[k2] * 3 + j2, -h)
                    A.add_H(idx[3] * 3 + j, idx[3] * 3 + j2, h)


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
        J = F.determinant()
        P = self.mu[None] * F + self.lam[None] * (J - self.alpha[None]) * J * F_T
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
            J_i = F_i.determinant()
            I_i = (F_i.transpose() @ F_i).trace()
            phi_i = self.mu[None] / 2 * (I_i - 3)
            phi_i += self.lam[None] / 2 * (J_i - self.alpha[None]) ** 2
            # phi_i += self.mu[None] / 2 * ti.log(I_i + 1)
            self.U[None] += self.F_W[c] * phi_i

    @ti.kernel
    def get_vertices(self, p: ti.types.ndarray(), p1: ti.types.ndarray()):
        for i in range(self.n_cells):
            self.F_vertices[i][0] = p[i, 0]
            self.F_vertices[i][1] = p[i, 1]
            self.F_vertices[i][2] = p[i, 2]
            self.F_vertices[i][3] = p[i, 3]

        for i in self.F_ox:
            self.F_ox[i] = ti.Vector([p1[i, 0], p1[i, 1], p1[i, 2]])

    @ti.kernel
    def init_pos(self, offsetx: ti.f64, offsety: ti.f64, offsetz: ti.f64, filp: ti.i32):
        self.F_v.fill(0)
        self.F_f.fill(0)
        self.F_m.fill(0)
        for i in self.F_x:
            self.F_x[i] = self.ratio * self.F_ox[i]
            if filp:
                self.F_x[i] = -self.F_x[i]
            self.F_x[i] += ti.Vector([offsetx, offsety, offsetz])

        for c in self.F_vertices:
            F = self.Ds(self.F_vertices[c])
            self.F_B[c] = F.inverse()
            self.F_W[c] = ti.abs(F.determinant()) / 6
            for i in range(4):
                self.F_m[self.F_vertices[c][i]] += self.F_W[c] / 4 * self.density

    @ti.kernel
    def init_pos_6d(self, rot: ti.types.matrix(3, 3, float), pos: ti.types.vector(3, float)):
        self.F_v.fill(0)
        self.F_f.fill(0)
        self.F_m.fill(0)
        for i in self.F_x:
            self.F_x[i] = rot @ (self.ratio * self.F_ox[i]) + pos

        for c in self.F_vertices:
            F = self.Ds(self.F_vertices[c])
            self.F_B[c] = F.inverse()
            self.F_W[c] = ti.abs(F.determinant()) / 6
            for i in range(4):
                self.F_m[self.F_vertices[c][i]] += self.F_W[c] / 4 * self.density

    @ti.kernel
    def update_bottom(self, rot: ti.types.matrix(3, 3, float), pos: ti.types.vector(3, float)):
        for i in self.F_x:
            if self.is_bottom(i):
                self.F_x[i] = rot @ (self.ratio * self.F_ox[i]) + pos

    @ti.func
    def is_bottom(self, i):
        return self.F_ox[i][2] < 0.001 and self.is_surface[i]

    @ti.func
    def is_inner_circle(self, i):
        return (self.F_ox[i] - ti.Vector([0., 0., 0.])).norm() < 0.0076 and self.is_surface[i]

    @ti.func
    def is_surf(self, i):
        return (self.F_ox[i] - ti.Vector([0., 0., 0.])).norm() > 0.0148 and self.is_surface[i]

    @ti.kernel
    def init_surface_indices(self, p: ti.types.ndarray(), flip: ti.i32, offset_x: ti.f64, offset_y: ti.f64, offset_z: ti.f64):
        for i in self.f2v:
            self.f2v[i][0] = p[i, 0]
            self.f2v[i][1] = p[i, 1]
            self.f2v[i][2] = p[i, 2]

            p1 = self.F_x[self.f2v[i][0]]
            p2 = self.F_x[self.f2v[i][1]]
            p3 = self.F_x[self.f2v[i][2]]
            n = (p2 - p1).cross(p3 - p1).normalized()

            # pointing to outside
            inner_point = ti.Vector([offset_x, offset_y, offset_z + 0.002 * self.ratio])
            if flip:
                inner_point = ti.Vector([offset_x, offset_y, offset_z - 0.002 * self.ratio])

            if (n.dot(inner_point - p1) > 0):
                if not (self.is_inner_circle(self.f2v[i][0]) and self.is_inner_circle(self.f2v[i][1]) and self.is_inner_circle(self.f2v[i][2])):
                    tmp = self.f2v[i][1]
                    self.f2v[i][1] = self.f2v[i][2]
                    self.f2v[i][2] = tmp
            else:
                if (self.is_inner_circle(self.f2v[i][0]) and self.is_inner_circle(self.f2v[i][1]) and self.is_inner_circle(self.f2v[i][2])):
                    tmp = self.f2v[i][1]
                    self.f2v[i][1] = self.f2v[i][2]
                    self.f2v[i][2] = tmp

    def is_bottom_func(self, i):
        return self.F_ox_array[i][2] < 0.001

    def is_inner_circle_func(self, i):
        return np.linalg.norm(self.F_ox_array[i] - np.array([0., 0., 0.])) < 0.0076

    def is_surf_func(self, i):
        return np.linalg.norm(self.F_ox_array[i] - np.array([0., 0., 0.])) > 0.0148

    def count(self):
        self.frozen_cnt = 0
        self.surf_point = 0
        for i in range(self.n_verts):
            self.is_surface[i] = False

        for i in range(self.n_surfaces):
            self.is_surface[self.f2v_array[i][0]] = True
            self.is_surface[self.f2v_array[i][1]] = True
            self.is_surface[self.f2v_array[i][2]] = True

        for i in range(self.n_verts):
            if self.is_surface[i]:
                if self.is_bottom_func(i) or self.is_inner_circle_func(i):
                    self.frozen_cnt += 1
                else:
                    if self.is_surf_func(i):
                        self.surf_point += 1
        # print("frozen cnt", self.frozen_cnt)
        # print("surf cnt", self.surf_point)

    def init(self, offsetx, offsety, offsetz, flip):
        self.get_vertices(self.F_vertices_array, self.F_ox_array)
        self.init_pos(offsetx, offsety, offsetz, flip)
        self.init_surface_indices(self.f2v_array, flip, offsetx, offsety, offsetz)

    @ti.kernel
    def compute_deri(self):
        self.d_mu.fill(0)
        self.d_lam.fill(0)
        for c in self.F_vertices:
            verts = self.F_vertices[c]
            F = self.Ds(verts) @ self.F_B[c]
            F_T = F.inverse().transpose()
            J = F.determinant()
            P1 = self.mu[None] * (F - J * F_T)
            P2 = self.lam[None] * (J - 1) * J * F_T
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
    def check_determinant(self) -> ti.i32:
        ret = True
        for c in self.F_vertices:
            verts = self.F_vertices[c]
            F = self.Ds(verts) @ self.F_B[c]
            Deter = F.determinant()
            if Deter < 0:
                ret = False
        return ret

    # def get_frozen(self):

    @ti.func
    def has_frozen(self, e):
        ret = False
        for j in range(4):
            k = self.F_vertices[e][j]
            if self.F_ox[k][2] < 0.001 or (self.F_ox[k] - ti.Vector([0., 0., 0.])).norm() < 0.0076:
                ret = True
        return ret


    @ti.kernel
    def compute_force_deri(self, analy_grad: ti.template(), step: int, grad_force: ti.types.vector(3, ti.f64)):
        # A = M - dt * dt * K
        for e in self.F_vertices:
            if self.has_frozen(e):
                F = self.Ds(self.F_vertices[e]) @ self.F_B[e]
                F_inv = F.inverse()
                F_inv_T = F_inv.transpose()
                J = F.determinant()

                for n in range(3):
                    for dim in range(self.dim):
                        dD = ti.Matrix.zero(ti.f64, 3, 3)
                        dD[dim, n] = 1
                        dF = dD @ self.F_B[e]  # !!! matrix multiplication
                        dF_T = dF.transpose()

                        # Tr(F^{-1}dF)
                        dTr = (F_inv @ dF).trace()

                        dP = -self.mu[None] * dF
                        dP -= self.lam[None] * 2 * J ** 2 * dTr * F_inv_T
                        dP += self.lam[None] * self.alpha[None] * J * dTr * F_inv_T
                        dP += self.lam[None] * (J - self.alpha[None]) * J * F_inv_T @ dF_T @ F_inv_T

                        dH = - self.F_W[e] * dP @ self.F_B[e].transpose()
                        for i, j in ti.static(ti.ndrange(3, 3)):
                            self.H_e[e, n * 3 + dim, i * 3 + j] = dH[j, i]

                # f = self.check_symmetry(e)

                self.H_proj.project(self.H_e, e, 9)
                # #
                idx = self.F_vertices[e] + self.offset
                for j, k in ti.static(ti.ndrange(3, 3)):
                    for j2, k2 in ti.static(ti.ndrange(3, 3)):
                        h = -self.H_e[e, k * 3 + j, k2 * 3 + j2]
                        if self.is_bottom(self.F_vertices[e][k2]) or self.is_inner_circle(self.F_vertices[e][k2]):
                            analy_grad.pos_grad[step, idx[k], j] += h * grad_force[j2]
                            analy_grad.pos_grad[step, idx[3], j] += -h * grad_force[j2]
                        if self.is_bottom(self.F_vertices[e][3]) or self.is_inner_circle(self.F_vertices[e][3]):
                            analy_grad.pos_grad[step, idx[k], j] += -h * grad_force[j2]
                            analy_grad.pos_grad[step, idx[3], j] += h * grad_force[j2]

    @ti.kernel
    def get_surf_normal(self):
        for i in self.f2v:
            p1 = self.F_x[self.f2v[i][0]]
            p2 = self.F_x[self.f2v[i][1]]
            p3 = self.F_x[self.f2v[i][2]]
            self.normals[i] = (p2 - p1).cross(p3 - p1).normalized() * 0.001
            self.normal_points[i] = (p1 + p2 + p3) / 3.0


    def plot_normal(self):
        self.get_surf_normal()
        x = self.normal_points.to_torch('cpu')
        norm = self.normals.to_torch('cpu')
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2])

        ax.quiver(x[:, 0], x[:, 1], x[:, 2], norm[:, 0], norm[:, 1], norm[:, 2], length=1, color='red')
        plt.draw()
        plt.savefig(f'../imgs/plot_normal.png')
        plt.close()

    def dmu_dnu(self):
        return -self.E / (2 * ((1 + self.nu) ** 2))

    def dmu_dE(self):
        return 1.0 / (2 * (1 + self.nu))

    def dlam_dnu(self):
        return (2 * self.E * (self.nu ** 2) + self.E) / ((2 * (self.nu ** 2) + self.nu - 1) ** 2)

    def dlam_dE(self):
        return self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    def update_paramters(self):
        mu, lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu[None] = mu
        self.lam[None] = lam
        self.alpha[None] = 1 + mu / lam