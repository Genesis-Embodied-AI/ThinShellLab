import argparse

import numpy as np

import taichi as ti
from sparse_solver import SparseMatrix
import linalg

@ti.data_oriented
class Cloth:
    def __init__(self, N, dt, Len, tot_NV, rho, offset, is_square=True, M=0):
        self.is_square = is_square
        if not is_square:
            self.N = N
            self.M = M
            self.dt = dt
            self.dx = Len / N
            self.NF = 2 * N * M  # number of faces
            self.NV = (N + 1) * (M + 1)  # number of vertices
        else:
            self.N = N
            self.M = N
            self.dt = dt
            self.dx = Len / N
            self.NF = 2 * N ** 2  # number of faces
            self.NV = (N + 1) ** 2  # number of vertices
        self.offset = offset
        self.rho = rho
        self.base_area = self.dx ** 2 * 0.5
        self.grid_len = self.dx
        self.h = 0.001  # thickness not used
        self.mass = rho * (self.dx ** 2)

        self.Kl = ti.field(ti.f64, ())
        self.Ka = ti.field(ti.f64, ())
        self.Kb = ti.field(ti.f64, ())
        self.Kl[None] = 1000.0
        self.Ka[None] = 1000.0
        self.Kb[None] = 100.0
        self.damping = 14.5

        self.pos = ti.Vector.field(3, ti.f64, self.NV)
        self.prev_pos = ti.Vector.field(3, ti.f64, self.NV)
        self.vel = ti.Vector.field(3, ti.f64, self.NV)
        self.f2v = ti.Vector.field(3, int, self.NF)  # ids of three vertices of each face
        self.f_deri = ti.Vector.field(3, ti.f64, self.NF * 3)

        self.V = ti.field(ti.f64, self.NF)

        self.l_i = ti.field(ti.f64, (self.NF, 3))
        self.t_i = ti.Vector.field(3, ti.f64, (self.NF, 3))
        self.norm_dir = ti.Vector.field(3, ti.f64, self.NF)

        self.gravity = ti.Vector.field(3, ti.f64, ())
        self.gravity[None] = ti.Vector([0, 0, -9.8])

        self.F_b = ti.Vector.field(3, dtype=ti.f64, shape=self.NV)

        # Hessian matrix
        self.U = ti.field(ti.f64, ())
        self.manipulate_force = ti.Vector.field(3, ti.f64, shape=self.NV)
        self.colors = ti.Vector.field(3, dtype=ti.f64, shape=self.NV)
        self.indices = ti.field(int, shape=self.NF * 3)

        # bending
        # Discrete bending forces and their Jacobians
        self.heights = ti.Vector.field(3, ti.f64, self.NF)
        self.counter_face = ti.Vector.field(3, ti.i32, self.NF)
        self.counter_point = ti.Vector.field(3, ti.i32, self.NF)
        self.mat_M = ti.Matrix.field(3, 3, ti.f64, self.NF * 3)
        self.mat_N = ti.Matrix.field(3, 3, ti.f64, self.NF * 3)
        self.angle = ti.Vector.field(3, ti.f64, self.NF)
        self.c_i = ti.Vector.field(3, ti.f64, self.NF)
        self.d_i = ti.Vector.field(3, ti.f64, self.NF)

        # plastic bend
        self.ref_angle = ti.Vector.field(3, ti.f64, self.NF)
        self.k_angle = ti.field(float, ())
        self.k_angle[None] = 3.14
        self.stiff_loss = ti.Vector.field(3, ti.f64, self.NF)
        self.weaken = 1
        self.offset_faces = 0
        self.body_idx = 0

        #spd project
        self.H_me = ti.field(ti.f64, shape=(self.NF * 3, 3, 3))
        self.p_me = linalg.SPD_Projector(self.NF * 3, 3, 10)

        # self.H_ma = ti.field(ti.f64, shape=(self.NF, 9, 9))
        # self.p_ma = linalg.SPD_Projector(self.NF, 9, 10)

        # self.H_bending_9 = ti.field(ti.f64, shape=(self.NF, 9, 9))
        # self.p_bending_9 = linalg.SPD_Projector(self.NF, 9, 10)
        # self.H_bending_12 = ti.field(ti.f64, shape=(self.NF * 3, 12, 12))
        # self.p_bending_12 = linalg.SPD_Projector(self.NF * 3, 12, 10)

        # visualize
        self.x32 = ti.Vector.field(3, ti.f32, shape=(self.NV,))
        self.f_vis = ti.field(ti.i32, shape=(self.NF * 3,))

        # derivative
        self.d_kl = ti.Vector.field(3, ti.f64, self.NV)
        self.d_ka = ti.Vector.field(3, ti.f64, self.NV)
        self.d_kb = ti.Vector.field(3, ti.f64, self.NV)

        self.uv = ti.Vector.field(2, dtype=ti.f64, shape=self.NV)

    @ti.func
    def compute_bending_energy(self, i1, i2, ref_angle, l):
        cos_theta = self.norm_dir[i1].dot(self.norm_dir[i2])
        theta = 0.0
        if cos_theta < 0.999999:
            theta = ti.acos(cos_theta)
        else:
            theta = 2 * ti.sqrt(ti.abs(1.0 - cos_theta)) / ti.sqrt(1 + cos_theta)
        if self.norm_dir[i2].dot(self.pos[self.f2v[i1][(l + 1) % 2]] - self.pos[self.f2v[i1][l]]) < 0:
            theta = -theta
        # if ti.math.isnan(theta):
        #     print("???", cos_theta)
        self.U[None] += self.Kb[None] * (theta - ref_angle) ** 2 * self.dx ** 2 * 1.0 / 3.0

        # (theta - theta_ori)**2 * |e_ori| / ((h_ori_1 + h_ori_2) * 1/2 * 1/3)
        # = (theta - theta_ori)**2 * 3 or = (theta - theta_ori)**2 * 6 depends on the edge!!!!
        # need to change this!!!

    @ti.func
    def compute_angle(self, i1, i2, l):
        theta = 0.0
        if i2 != -1:
            cos_theta = self.norm_dir[i1].dot(self.norm_dir[i2])
            if cos_theta < 0.999999:
                theta = ti.acos(cos_theta)
            else:
                theta = 2 * ti.sqrt(ti.abs(1.0 - cos_theta)) / ti.sqrt(1 + cos_theta)
            if self.norm_dir[i2].dot(self.pos[self.f2v[i1][(l + 1) % 2]] - self.pos[self.f2v[i1][l]]) < 0:
                theta = -theta

        return theta

    @ti.func
    def judge_angle(self, i1, i2, l):
        ret = True
        if i2 != -1:
            if self.norm_dir[i2].dot(self.pos[self.f2v[i1][(l + 1) % 2]] - self.pos[self.f2v[i1][l]]) < 0:
                ret = False

        return ret

    @ti.func
    def compute_membrane_energy(self, i):
        ia, ib, ic = self.f2v[i]
        a, b, c = self.pos[ia], self.pos[ib], self.pos[ic]
        l0 = b - a
        l1 = c - a
        area = l0.cross(l1).norm() * 0.5
        self.U[None] += self.Ka[None] * (1 - area / self.V[i]) ** 2 * self.V[i]

    @ti.func
    def compute_membrane_energy_edge(self, p1, p2, edge_idx):
        a, b = self.pos[p1], self.pos[p2]
        l = (b - a).norm()
        if edge_idx == 2:
            base_len = self.dx * ti.math.sqrt(2.0)
            self.U[None] += self.Kl[None] * (1 - l / base_len) ** 2 * base_len
        else:
            base_len = self.dx
            self.U[None] += self.Kl[None] * (1 - l / base_len) ** 2 * base_len

    @ti.kernel
    def compute_normal_dir(self):
        for i in range(self.NF):
            ia, ib, ic = self.f2v[i]
            a, b, c = self.pos[ia], self.pos[ib], self.pos[ic]
            self.norm_dir[i] = (b - a).cross(c - b).normalized()

    @ti.kernel
    def update_ref_angle(self):
        for i in range(self.NF):
            for l in range(3):
                if self.counter_face[i][l] > i:
                    theta = self.compute_angle(i, self.counter_face[i][l], l)
                    theta_dis = theta - self.ref_angle[i][l]
                    abs_dis = ti.abs(theta_dis)
                    if abs_dis > self.k_angle[None]:
                        self.ref_angle[i][l] += (abs_dis - self.k_angle[None]) * theta_dis / abs_dis
                        # self.stiff_loss[i][l] += (abs_dis - self.k_angle[None]) / self.k_angle[None] * self.weaken



    @ti.kernel
    def compute_energy(self):
        self.U[None] = 0
        for i in range(self.NV):
            self.U[None] += -self.manipulate_force[i].dot(self.pos[i])
        for i in range(self.NV):
            self.U[None] += -self.pos[i].dot(self.gravity[None]) * self.mass

        for i in range(self.NV):
            X = self.pos[i] - self.prev_pos[i] - self.vel[i] * self.dt
            self.U[None] += 0.5 * self.mass * X.dot(X) / (self.dt ** 2)

        for i, j in ti.ndrange(self.N, self.M):
            k = (i * self.M + j) * 2
            self.compute_membrane_energy(k)
            self.compute_membrane_energy(k + 1)
            self.compute_membrane_energy_edge(self.f2v[k][1], self.f2v[k][2], 1)
            self.compute_membrane_energy_edge(self.f2v[k + 1][0], self.f2v[k + 1][1], 0)
            self.compute_membrane_energy_edge(self.f2v[k][0], self.f2v[k][2], 2)
            self.compute_membrane_energy_edge(self.f2v[k][0], self.f2v[k][2], 2)
            # else:
            self.compute_membrane_energy_edge(self.f2v[k + 1][1], self.f2v[k + 1][2], 1)
            # else:
            self.compute_membrane_energy_edge(self.f2v[k][0], self.f2v[k][1], 0)

        for i in range(self.NF):
            for l in range(3):
                if self.counter_face[i][l] > i:
                    self.compute_bending_energy(i, self.counter_face[i][l], self.ref_angle[i][l], l)

    @ti.kernel
    def compute_energy_me(self):
        self.U[None] = 0
        for i in range(self.NV):
            self.U[None] += -self.manipulate_force[i].dot(self.pos[i])
        for i in range(self.NV):
            self.U[None] += -self.pos[i].dot(self.gravity[None]) * self.mass

        for i in range(self.NV):
            X = self.pos[i] - self.prev_pos[i] - self.vel[i] * self.dt
            self.U[None] += 0.5 * self.mass * X.dot(X) / (self.dt ** 2)

        for i, j in ti.ndrange(self.N, self.M):
            k = (i * self.M + j) * 2

            self.compute_membrane_energy_edge(self.f2v[k][1], self.f2v[k][2], 1)
            self.compute_membrane_energy_edge(self.f2v[k + 1][0], self.f2v[k + 1][1], 0)
            self.compute_membrane_energy_edge(self.f2v[k][0], self.f2v[k][2], 2)
            self.compute_membrane_energy_edge(self.f2v[k][0], self.f2v[k][2], 2)
            self.compute_membrane_energy_edge(self.f2v[k + 1][1], self.f2v[k + 1][2], 1)
            self.compute_membrane_energy_edge(self.f2v[k][0], self.f2v[k][1], 0)

    @ti.kernel
    def compute_energy_ma(self):
        self.U[None] = 0

        for i, j in ti.ndrange(self.N, self.M):
            k = (i * self.M + j) * 2
            self.compute_membrane_energy(k)
            self.compute_membrane_energy(k + 1)

    @ti.kernel
    def compute_energy_bending(self):
        self.U[None] = 0

        for i in range(self.NF):
            for l in range(3):
                if self.counter_face[i][l] > i:
                    self.compute_bending_energy(i, self.counter_face[i][l], self.ref_angle[i][l], l)

    @ti.func
    def compute_membrane_dl(self, l_tau, l_base):
        return -self.Kl[None] * 2.0 * (1.0 - l_tau / l_base)

    @ti.func
    def compute_membrane_dl2(self, l_base):
        return self.Kl[None] * 2.0 / l_base

    @ti.func
    def compute_membrane_darea(self, area, base):
        return -self.Ka[None] * 2.0 * (1.0 - area / base)

    @ti.func
    def compute_membrane_darea2(self, base):
        return self.Ka[None] * 2.0 / base

    @ti.func
    def compute_bending_dtheta(self, theta):
        return 2.0 * self.Kb[None] * (theta - 0) * self.dx ** 2 * 1.0 / 3.0

    @ti.func
    def compute_bending_dtheta_ref(self, theta, ref_theta):
        return 2.0 * self.Kb[None] * (theta - ref_theta) * self.dx ** 2 * 1.0 / 3.0

    @ti.func
    def compute_bending_dtheta2(self, theta):
        return 2.0 * self.Kb[None] * self.dx ** 2 * 1.0 / 3.0

    @ti.func
    def compute_l_dx2(self, p1, p2, l_tau, dim):
        return (l_tau ** 2 - (p1[dim] - p2[dim]) ** 2) / (l_tau ** 3)

    @ti.func
    def compute_l_dxy(self, p1, p2, l_tau, dim, d1):
        return (p1[dim] - p2[dim]) * (p1[d1] - p2[d1]) / (l_tau ** 3)

    @ti.func
    def compute_area_dx2(self, area, p1, p2, p3, dim):
        area = area * 2.0
        d1 = 0
        if dim == 0:
            d1 = 1
        d2 = 3 - d1 - dim
        deri = ((p2[d1] - p3[d1]) ** 2 + (p2[d2] - p3[d2]) ** 2) / area - \
               ((p2[d1] - p3[d1]) * (
                           (p2[dim] - p1[dim]) * (p3[d1] - p1[d1]) - (p3[dim] - p1[dim]) * (p2[d1] - p1[d1])) +
                (p2[d2] - p3[d2]) * (
                            (p2[dim] - p1[dim]) * (p3[d2] - p1[d2]) - (p3[dim] - p1[dim]) * (p2[d2] - p1[d2]))) ** 2 / (
                           area ** 3)

        return deri * 0.5

    @ti.func
    def compute_area_dx(self, area, p1, p2, p3, dim):
        area = area * 2.0
        d1 = 0
        if dim == 0:
            d1 = 1
        d2 = 3 - d1 - dim
        deri = 0.5 * (p1[dim] * ((p2[d1] - p3[d1]) ** 2 + (p2[d2] - p3[d2]) ** 2) -
                      p2[dim] * (p1[d1] * (p2[d1] - p3[d1]) - p2[d1] * p3[d1] + p3[d1] ** 2 + p1[d2] * p2[d2] - p1[d2] *
                                 p3[d2] - p2[d2] * p3[d2] + p3[d2] ** 2) +
                      p3[dim] * (p1[d1] * (p2[d1] - p3[d1]) - p2[d1] ** 2 + p2[d1] * p3[d1] + (p1[d2] - p2[d2]) * (
                            p2[d2] - p3[d2]))) / area

        return deri

    @ti.func
    def compute_area_dxy_p1(self, area, p1, p2, p3, dim, d1):
        area = area * 2.0
        d2 = 3 - d1 - dim
        deri = ((p3[dim] - p2[dim]) * (p2[d1] - p3[d1])) / area - \
               (((p3[dim] - p2[dim]) * (
                           (p2[dim] - p1[dim]) * (p3[d1] - p1[d1]) - (p3[dim] - p1[dim]) * (p2[d1] - p1[d1])) +
                 (p2[d2] - p3[d2]) * ((p2[d1] - p1[d1]) * (p3[d2] - p1[d2]) - (p3[d1] - p1[d1]) * (p2[d2] - p1[d2]))) *
                ((p2[d1] - p3[d1]) * (
                            (p2[dim] - p1[dim]) * (p3[d1] - p1[d1]) - (p3[dim] - p1[dim]) * (p2[d1] - p1[d1])) +
                 (p2[d2] - p3[d2]) * (
                             (p2[dim] - p1[dim]) * (p3[d2] - p1[d2]) - (p3[dim] - p1[dim]) * (p2[d2] - p1[d2])))) / (
                           area ** 3)

        return deri * 0.5

    @ti.func
    def compute_area_dx2_p12(self, area, p1, p2, p3, dim):
        area = area * 2.0
        d1 = 0
        if dim == 0:
            d1 = 1
        d2 = 3 - d1 - dim
        deri = ((p3[d1] - p1[d1]) * (p2[d1] - p3[d1]) + (p3[d2] - p1[d2]) * (p2[d2] - p3[d2])) / area - \
               (((p2[d1] - p3[d1]) * (
                           (p2[dim] - p1[dim]) * (p3[d1] - p1[d1]) - (p3[dim] - p1[dim]) * (p2[d1] - p1[d1])) +
                 (p2[d2] - p3[d2]) * (
                             (p2[dim] - p1[dim]) * (p3[d2] - p1[d2]) - (p3[dim] - p1[dim]) * (p2[d2] - p1[d2]))) *
                ((p3[d1] - p1[d1]) * (
                            (p2[dim] - p1[dim]) * (p3[d1] - p1[d1]) - (p3[dim] - p1[dim]) * (p2[d1] - p1[d1])) +
                 (p3[d2] - p1[d2]) * (
                             (p2[dim] - p1[dim]) * (p3[d2] - p1[d2]) - (p3[dim] - p1[dim]) * (p2[d2] - p1[d2])))) / (
                           area ** 3)

        return deri * 0.5

    @ti.func
    def compute_area_dxy_p12(self, area, p1, p2, p3, dim, d1):
        area = area * 2.0
        d2 = 3 - d1 - dim
        deri = (((p2[dim] - p1[dim]) * (p3[d1] - p1[d1]) - (p3[dim] - p1[dim]) * (p2[d1] - p1[d1])) + (
                    p1[dim] - p3[dim]) * (p2[d1] - p3[d1])) / area - \
               ((2 * (p1[dim] - p3[dim]) * (
                           (p2[dim] - p1[dim]) * (p3[d1] - p1[d1]) - (p3[dim] - p1[dim]) * (p2[d1] - p1[d1])) +
                 (p3[d2] - p1[d2]) * ((p2[d1] - p1[d1]) * (p3[d2] - p1[d2]) - (p3[d1] - p1[d1]) * (p2[d2] - p1[d2]))) *
                ((p2[d1] - p3[d1]) * (
                            (p2[dim] - p1[dim]) * (p3[d1] - p1[d1]) - (p3[dim] - p1[dim]) * (p2[d1] - p1[d1])) +
                 (p2[d2] - p3[d2]) * (
                             (p2[dim] - p1[dim]) * (p3[d2] - p1[d2]) - (p3[dim] - p1[dim]) * (p2[d2] - p1[d2])))) / (
                           area ** 3)
        return deri * 0.5

    @ti.func
    def compute_bending_grad(self, i1, l):
        i2 = self.counter_face[i1][l]
        p0 = l
        p11 = (l + 1) % 3
        p12 = (l + 2) % 3
        p4 = self.counter_point[i1][l]
        p21 = (p4 + 1) % 3
        if self.f2v[i1][p11] != self.f2v[i2][p21]:
            p21 = (p4 + 2) % 3
        p22 = 3 - p21 - p4

        # print(i1, i2, self.f2v[i1][p0], self.f2v[i1][p11], self.f2v[i1][p12])
        # print(self.f2v[i2][p4], self.f2v[i2][p21], self.f2v[i2][p22])
        # print(i1, i2, self.angle[i1], self.angle[i2], self.heights[i1], self.heights[i2])

        a = - 1.0 / self.heights[i1][l] * self.norm_dir[i1]
        d = - 1.0 / self.heights[i2][p4] * self.norm_dir[i2]
        b = self.angle[i1][p12] / self.heights[i1][p11] * self.norm_dir[i1] + self.angle[i2][p22] / self.heights[i2][
            p21] * self.norm_dir[i2]
        c = self.angle[i1][p11] / self.heights[i1][p12] * self.norm_dir[i1] + self.angle[i2][p21] / self.heights[i2][
            p22] * self.norm_dir[i2]

        return a, b, c, d

    @ti.func
    def choice(self, a, b, c, d, id):
        ret = a
        if id == 1:
            ret = b
        if id == 2:
            ret = c
        if id == 3:
            ret = d
        return ret

    @ti.kernel
    def prepare_bending(self):
        for i in range(self.NF):
            for l in range(3):
                pi = self.f2v[i][l]
                ai = self.f2v[i][(l + 1) % 3]
                bi = self.f2v[i][(l + 2) % 3]
                p = self.pos[pi]
                a = self.pos[ai]
                b = self.pos[bi]
                edge = b - a
                norm_dir = self.norm_dir[i]
                if self.judge_angle(i, self.counter_face[i][l], l):
                    norm_dir = -norm_dir
                edge_norm = norm_dir.cross(edge)
                edge1 = a - p
                angle = edge_norm.dot(edge1)
                if angle > 0:
                    edge_norm = -edge_norm
                self.mat_M[i * 3 + l] = norm_dir.outer_product(edge_norm)
                self.mat_N[i * 3 + l] = self.mat_M[i * 3 + l] / edge.norm()
                self.angle[i][l] = (a - p).normalized().dot((b - p).normalized())
                self.heights[i][l] = ti.abs((p - a).dot(edge_norm)) / edge_norm.norm()
                # print(self.angle[i][l], self.heights[i][l], self.dx)
                if self.counter_face[i][l] != -1:
                    theta = self.compute_angle(i, self.counter_face[i][l], l)
                    delta_phi = self.compute_bending_dtheta_ref(theta, self.ref_angle[i][l])
                    self.c_i[i][l] = delta_phi
                else:
                    self.c_i[i][l] = 0

            for l in range(3):
                self.d_i[i][l] = self.c_i[i][(l + 1) % 3] * self.angle[i][(l + 2) % 3] + self.c_i[i][(l + 2) % 3] * \
                                 self.angle[i][(l + 1) % 3] - self.c_i[i][l]

    @ti.func
    def set_H_me(self, j, k, i, value):
        self.H_me[i, j, k] = value

    @ti.func
    def add_H_ma(self, j, k, i, value):
        self.H_ma[i, j, k] += value

    @ti.func
    def add_H_bending_9(self, j, k, i, value):
        self.H_bending_9[i, j, k] += value

    @ti.func
    def add_H_bending_12(self, j, k, i, value):
        self.H_bending_12[i, j, k] += value

    @ti.kernel
    def compute_Hessian_me(self, H: ti.template(), spd: ti.i32):
        for i in range(self.NV):
            for j in range(3):
                H.H.add(3 * (i + self.offset) + j, 3 * (i + self.offset) + j, self.mass / (self.dt ** 2))
        dt = self.dt
        for i in range(self.NF):
            ia, ib, ic = self.f2v[i]
            # membrane_energy_edge
            # print("compute edge")
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    for l in ti.static(range(3)):
                        xx = self.f2v[i][l]
                        yy = self.f2v[i][(l + 1) % 3]

                        delta = self.pos[xx] - self.pos[yy]
                        a, b = self.pos[xx], self.pos[yy]
                        l_tau = delta.norm()
                        dldx = delta / l_tau
                        dldy = -dldx
                        base_len = self.l_i[i, l]

                        if (j == k):
                            self.set_H_me(j, k, i * 3 + l, (
                                    self.compute_membrane_dl(l_tau, base_len) * self.compute_l_dx2(a, b, l_tau,
                                                                                                   j) + self.compute_membrane_dl2(
                                base_len) * dldx[j] * dldx[k]))

                        else:
                            self.set_H_me(j, k, i * 3 + l, (
                                    self.compute_membrane_dl(l_tau, base_len) * self.compute_l_dxy(a, b, l_tau, j,
                                                                                                   k) + self.compute_membrane_dl2(
                                base_len) * dldx[j] * dldx[k]))

            for l in ti.static(range(3)):
                idx = i * 3 + l
                # for j in ti.static(range(3)):
                #     for k in ti.static(range(3)):
                #         if idx == 0:
                #             print(self.H_me[idx, j, k], end=' ')
                # if idx == 0:
                #     print('')
                if spd:
                    self.p_me.project(self.H_me, idx, 3)
                xx = self.f2v[i][l]
                yy = self.f2v[i][(l + 1) % 3]
                xx = xx + self.offset
                yy = yy + self.offset
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        # if idx == 0:
                        #     print(self.H_me[idx, j, k], end=' ')
                        H.add_H(xx * 3 + j, xx * 3 + k, self.H_me[idx, j, k])
                        H.add_H(xx * 3 + j, yy * 3 + k, -self.H_me[idx, j, k])
                        H.add_H(yy * 3 + j, xx * 3 + k, -self.H_me[idx, j, k])
                        H.add_H(yy * 3 + j, yy * 3 + k, self.H_me[idx, j, k])
                # if idx == 0:
                #     print('')

    @ti.kernel
    def compute_Hessian_ma(self, H: ti.template()):
        for i in range(self.NF):
            ia, ib, ic = self.f2v[i]
            a, b, c = self.pos[ia], self.pos[ib], self.pos[ic]
            base_area = self.V[i]
            darea2 = self.compute_membrane_darea2(base_area)
            v1 = b - a
            v2 = c - a
            area = 0.5 * v1.cross(v2).norm()
            for l in range(3):
                for j in range(3):
                    self.f_deri[i * 3 + l][j] = self.compute_area_dx(area, self.pos[self.f2v[i][l]],
                                                                     self.pos[self.f2v[i][(l + 1) % 3]],
                                                                     self.pos[self.f2v[i][(l + 2) % 3]], j)

            # print(area**3)
            dt = self.dt
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            xx = self.f2v[i][l] + self.offset
                            yy = self.f2v[i][m] + self.offset

                            H.add_H(xx * 3 + j, yy * 3 + k, self.f_deri[i * 3 + l][j] * \
                                    self.f_deri[i * 3 + m][k] * darea2)
                            if j == k:
                                if l == m:
                                    H.add_H(xx * 3 + j, yy * 3 + k,
                                            self.compute_membrane_darea(area, base_area) * \
                                            self.compute_area_dx2(area, self.pos[self.f2v[i][l]],
                                                                  self.pos[self.f2v[i][(l + 1) % 3]],
                                                                  self.pos[self.f2v[i][(l + 2) % 3]], j))
                                else:
                                    H.add_H(xx * 3 + j, yy * 3 + k,
                                            self.compute_membrane_darea(area, base_area) * \
                                            self.compute_area_dx2_p12(area, self.pos[self.f2v[i][l]],
                                                                      self.pos[self.f2v[i][m]],
                                                                      self.pos[self.f2v[i][3 - l - m]], j))
                            else:
                                if l == m:
                                    H.add_H(xx * 3 + j, yy * 3 + k,
                                            self.compute_membrane_darea(area, base_area) * \
                                            self.compute_area_dxy_p1(area, self.pos[self.f2v[i][l]],
                                                                     self.pos[self.f2v[i][(l + 1) % 3]],
                                                                     self.pos[self.f2v[i][(l + 2) % 3]],
                                                                     j, k))
                                else:
                                    H.add_H(xx * 3 + j, yy * 3 + k,
                                            self.compute_membrane_darea(area, base_area) * \
                                            self.compute_area_dxy_p12(area, self.pos[self.f2v[i][l]],
                                                                      self.pos[self.f2v[i][m]],
                                                                      self.pos[self.f2v[i][3 - l - m]],
                                                                      j, k))

    @ti.kernel
    def compute_Hessian_bending(self, H: ti.template()):
        dt = self.dt
        for i in range(self.NF):
            for l in range(3):
                for lm in range(l, l + 2):
                    m = lm % 3

                    H_lm = ti.Matrix([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                    if (l == m):
                        i1 = (l + 1) % 3
                        i2 = (l + 2) % 3
                        H_lm = 1.0 / (self.heights[i][l] * self.heights[i][m]) * (
                                self.d_i[i][l] * self.mat_M[i * 3 + m].transpose() + self.d_i[i][m] * self.mat_M[
                            i * 3 + l]) - \
                               self.c_i[l][i1] * self.mat_N[l * 3 + i1] - self.c_i[l][i2] * self.mat_N[l * 3 + i2]
                    else:
                        i3 = 3 - l - m
                        H_lm = 1.0 / (self.heights[i][l] * self.heights[i][m]) * (
                                self.d_i[i][l] * self.mat_M[i * 3 + m].transpose() + self.d_i[i][m] * self.mat_M[
                            i * 3 + l])
                        # if self.f2v[i][l] > self.f2v[i][m]:
                        #     H_lm += self.c_i[l][i3] * self.mat_N[l * 3 + i3].transpose()
                        # else:
                        H_lm += self.c_i[l][i3] * self.mat_N[l * 3 + i3]

                    xx = self.f2v[i][l] + self.offset
                    yy = self.f2v[i][m] + self.offset
                    for j in range(3):
                        for k in range(3):
                            H.add_H(xx * 3 + j, yy * 3 + k, H_lm[j, k])
                            if l != m:
                                H.add_H(yy * 3 + j, xx * 3 + k, H_lm[k, j])

            for l in range(3):
                if self.counter_face[i][l] > i:
                    a, b, c, d = self.compute_bending_grad(i, l)
                    theta = self.compute_angle(i, self.counter_face[i][l], l)
                    list_grad = [a, b, c, d]
                    list_point = [self.f2v[i][l], self.f2v[i][(l + 1) % 3], self.f2v[i][(l + 2) % 3],
                                  self.f2v[self.counter_face[i][l]][self.counter_point[i][l]]]
                    d2_theta = self.compute_bending_dtheta2(theta)
                    for j in range(4):
                        for k in range(4):
                            gradj = self.choice(a, b, c, d, j)
                            gradk = self.choice(a, b, c, d, k)
                            pj = self.choice(self.f2v[i][l], self.f2v[i][(l + 1) % 3], self.f2v[i][(l + 2) % 3],
                                             self.f2v[self.counter_face[i][l]][self.counter_point[i][l]],
                                             j) + self.offset
                            pk = self.choice(self.f2v[i][l], self.f2v[i][(l + 1) % 3], self.f2v[i][(l + 2) % 3],
                                             self.f2v[self.counter_face[i][l]][self.counter_point[i][l]],
                                             k) + self.offset
                            gridmat = d2_theta * gradj.outer_product(gradk)
                            for jj in range(3):
                                for kk in range(3):
                                    H.add_H(pj * 3 + jj, pk * 3 + kk, gridmat[jj, kk])

    @ti.kernel
    def compute_residual(self):
        for i in self.F_b:
            self.F_b[i] = -self.mass * self.gravity[None]
        # self.F_b.fill(0)
        dt = self.dt
        for i in range(self.NV):
            self.F_b[i] -= self.manipulate_force[i]
        for i in range(self.NV):
            self.F_b[i] += self.mass * (self.pos[i] - self.prev_pos[i] - self.vel[i] * dt) / (dt ** 2)
        # print("mass", rho * dx ** 2 * (pos[1] - prev_pos[1] - vel[1]*dt))

        # print("res 35 before bend:", self.F_b[35])

        for i in range(self.NF):
            ia, ib, ic = self.f2v[i]
            a, b, c = self.pos[ia], self.pos[ib], self.pos[ic]

            # membrane_energy_edge_grad
            for l in ti.static(range(3)):
                xx = self.f2v[i][l]
                yy = self.f2v[i][(l + 1) % 3]
                base_len = self.l_i[i, l]
                delta = self.pos[xx] - self.pos[yy]
                l_tau = delta.norm()
                self.F_b[xx] += delta * self.compute_membrane_dl(l_tau, base_len) / l_tau
                self.F_b[yy] += -delta * self.compute_membrane_dl(l_tau, base_len) / l_tau

            # membrane_energy_area_grad
            base_area = self.V[i]
            v1 = b - a
            v2 = c - a
            area = 0.5 * v1.cross(v2).norm()
            for l in range(3):
                for j in range(3):
                    self.F_b[self.f2v[i][l]][j] += self.compute_membrane_darea(area,
                                                                                         base_area) * self.compute_area_dx(
                        area, self.pos[self.f2v[i][l]], self.pos[self.f2v[i][(l + 1) % 3]],
                        self.pos[self.f2v[i][(l + 2) % 3]], j)

            for l in range(3):
                if self.counter_face[i][l] > i:
                    a, b, c, d = self.compute_bending_grad(i, l)
                    theta = self.compute_angle(i, self.counter_face[i][l], l)
                    d_theta = self.compute_bending_dtheta_ref(theta, self.ref_angle[i][l])
                    self.F_b[self.f2v[i][l]] += d_theta * a
                    self.F_b[self.f2v[i][(l + 1) % 3]] += d_theta * b
                    self.F_b[self.f2v[i][(l + 2) % 3]] += d_theta * c
                    self.F_b[self.f2v[self.counter_face[i][l]][self.counter_point[i][l]]] += d_theta * d

        # print("res 35 after bend:", self.F_b[35])

    @ti.kernel
    def compute_residual_ma(self):
        self.F_b.fill(0)

        for i in range(self.NF):
            ia, ib, ic = self.f2v[i]
            a, b, c = self.pos[ia], self.pos[ib], self.pos[ic]

            # membrane_energy_area_grad
            base_area = self.V[i]
            v1 = b - a
            v2 = c - a
            area = 0.5 * v1.cross(v2).norm()
            for l in range(3):
                for j in range(3):
                    self.F_b[self.f2v[i][l]][j] += self.compute_membrane_darea(area,
                                                                               base_area) * self.compute_area_dx(
                        area, self.pos[self.f2v[i][l]], self.pos[self.f2v[i][(l + 1) % 3]],
                        self.pos[self.f2v[i][(l + 2) % 3]], j)

    @ti.kernel
    def compute_residual_me(self):
        for i in self.F_b:
            self.F_b[i] = -self.mass * self.gravity[None]
        # self.F_b.fill(0)
        dt = self.dt
        for i in range(self.NV):
            self.F_b[i] -= self.manipulate_force[i]
        for i in range(self.NV):
            self.F_b[i] += self.mass * (self.pos[i] - self.prev_pos[i] - self.vel[i] * dt) / (dt ** 2)
        # print("mass", rho * dx ** 2 * (pos[1] - prev_pos[1] - vel[1]*dt))

        # print("res 35 before bend:", self.F_b[35])

        for i in range(self.NF):
            ia, ib, ic = self.f2v[i]
            a, b, c = self.pos[ia], self.pos[ib], self.pos[ic]

            # membrane_energy_edge_grad
            for l in ti.static(range(3)):
                xx = self.f2v[i][l]
                yy = self.f2v[i][(l + 1) % 3]
                base_len = self.l_i[i, l]
                delta = self.pos[xx] - self.pos[yy]
                l_tau = delta.norm()
                self.F_b[xx] += delta * self.compute_membrane_dl(l_tau, base_len) / l_tau
                self.F_b[yy] += -delta * self.compute_membrane_dl(l_tau, base_len) / l_tau

    @ti.kernel
    def compute_residual_bending(self):

        self.F_b.fill(0)
        dt = self.dt

        for i in range(self.NF):
            ia, ib, ic = self.f2v[i]
            a, b, c = self.pos[ia], self.pos[ib], self.pos[ic]
            # membrane_energy_edge_grad

            for l in range(3):
                if self.counter_face[i][l] > i:
                    a, b, c, d = self.compute_bending_grad(i, l)
                    theta = self.compute_angle(i, self.counter_face[i][l], l)
                    d_theta = self.compute_bending_dtheta_ref(theta, self.ref_angle[i][l])
                    self.F_b[self.f2v[i][l]] += d_theta * a
                    self.F_b[self.f2v[i][(l + 1) % 3]] += d_theta * b
                    self.F_b[self.f2v[i][(l + 2) % 3]] += d_theta * c
                    self.F_b[self.f2v[self.counter_face[i][l]][self.counter_point[i][l]]] += d_theta * d

    @ti.kernel
    def update_vel(self):
        for i in range(self.NV):
            for j in range(3):
                self.vel[i][j] = (self.pos[i][j] - self.prev_pos[i][j]) / self.dt

    @ti.kernel
    def get_prev_pos(self):
        for i in range(self.NV):
            for j in range(3):
                self.prev_pos[i][j] = self.pos[i][j]

    @ti.kernel
    def init_pos(self):
        self.ref_angle.fill(0)
        for i, j in ti.ndrange(self.N + 1, self.M + 1):
            k = i * (self.M + 1) + j
            self.pos[k] = ti.Vector([i * self.grid_len, j * self.grid_len, 0.01])
            self.vel[k] = ti.Vector([0, 0, 0])

        for i in range(self.NF):
            ia, ib, ic = self.f2v[i]
            self.V[i] = self.grid_len ** 2 * 0.5
            self.l_i[i, 0] = self.grid_len
            self.l_i[i, 1] = self.grid_len
            self.l_i[i, 2] = self.grid_len * ti.math.sqrt(2.0)

    @ti.kernel
    def init_ref_angle(self):
        for i in range(self.NF):
            for l in range(3):
                if self.counter_face[i][l] > i:
                    theta = self.compute_angle(i, self.counter_face[i][l], l)
                    theta_dis = theta - self.ref_angle[i][l]
                    abs_dis = ti.abs(theta_dis)
                    if abs_dis > self.k_angle[None]:
                        # print(i, self.counter_face[i][l], theta)
                        self.ref_angle[i][l] += (abs_dis - self.k_angle[None]) * theta_dis / abs_dis

    @ti.kernel
    def init_ref_angle_real(self):
        for i in range(self.NF):
            for l in range(3):
                if self.counter_face[i][l] > i:
                    theta = self.compute_angle(i, self.counter_face[i][l], l)
                    theta_dis = theta - self.ref_angle[i][l]
                    abs_dis = ti.abs(theta_dis)
                    if abs_dis > self.k_angle[None]:
                        # print(i, self.counter_face[i][l], theta)
                        self.ref_angle[i][l] += (abs_dis - self.k_angle[None] + 0.3) * theta_dis / abs_dis

    @ti.kernel
    def init_ref_angle_bridge(self):
        for i in range(self.NF):
            for l in range(3):
                if self.counter_face[i][l] > i:
                    p = self.f2v[self.counter_face[i][l]][self.counter_point[i][l]]
                    if ti.cast(self.f2v[i][l] / (self.M + 1), ti.i32) == 4 and ti.cast(
                            p / (self.M + 1), ti.i32) == 6:
                        self.ref_angle[i][l] = 1.7
                    if ti.cast(self.f2v[i][l] / (self.M + 1), ti.i32) == 9 and ti.cast(
                            p / (self.M + 1), ti.i32) == 11:
                        self.ref_angle[i][l] = 1.7


    @ti.kernel
    def init_pos_offset(self, offsetx: ti.f64, offsety: ti.f64, offsetz: ti.f64):
        self.ref_angle.fill(0)
        for i, j in ti.ndrange(self.N + 1, self.M + 1):
            k = i * (self.M + 1) + j
            self.pos[k] = ti.Vector([i * self.grid_len + offsetx, j * self.grid_len + offsety, offsetz])
            self.vel[k] = ti.Vector([0, 0, 0])

        for i in range(self.NF):
            ia, ib, ic = self.f2v[i]
            self.V[i] = self.grid_len ** 2 * 0.5
            self.l_i[i, 0] = self.grid_len
            self.l_i[i, 1] = self.grid_len
            self.l_i[i, 2] = self.grid_len * ti.math.sqrt(2.0)

    @ti.kernel
    def init_pos_offset_fold(self, offsetx: ti.f64, offsety: ti.f64, offsetz: ti.f64, half_curv_num: ti.i32):
        self.ref_angle.fill(0)
        r = self.grid_len
        # print("fold init radius:", r)
        if half_curv_num != 2:
            r = self.grid_len * (half_curv_num * 2 - 1) / 3.1415
        L = 7 - half_curv_num + 1
        R = 7 + half_curv_num
        for i, j in ti.ndrange(self.N + 1, self.M + 1):
            k = i * (self.M + 1) + j
            if i <= L:
                self.pos[k] = ti.Vector([(15 - i) * self.grid_len + offsetx, j * self.grid_len + offsety, offsetz + 2 * r])
                self.vel[k] = ti.Vector([0, 0, 0])
            if i >= L + 1 and i <= R - 1:
                x = (15 - L) * self.grid_len
                angle = (i - L) / (half_curv_num * 2 - 1) * 3.1415
                self.pos[k] = ti.Vector([x - r * ti.sin(angle) + offsetx, j * self.grid_len + offsety, offsetz + r * (1 + ti.cos(angle))])
                self.vel[k] = ti.Vector([0, 0, 0])
            if i >= R:
                self.pos[k] = ti.Vector([i * self.grid_len + offsetx, j * self.grid_len + offsety, offsetz])
                self.vel[k] = ti.Vector([0, 0, 0])

        for i in range(self.NF):
            ia, ib, ic = self.f2v[i]
            self.V[i] = self.grid_len ** 2 * 0.5
            self.l_i[i, 0] = self.grid_len
            self.l_i[i, 1] = self.grid_len
            self.l_i[i, 2] = self.grid_len * ti.math.sqrt(2.0)

    @ti.kernel
    def init_pos_offset_fold_real(self, offsetx: ti.f64, offsety: ti.f64, offsetz: ti.f64, half_curv_num: ti.i32):
        self.ref_angle.fill(0)
        r = self.grid_len * 2

        L = 7
        R = 13
        for i, j in ti.ndrange(self.N + 1, self.M + 1):
            k = i * (self.M + 1) + j
            if i <= L:
                self.pos[k] = ti.Vector(
                    [(20 - i) * self.grid_len + offsetx, j * self.grid_len + offsety, offsetz + 2 * r])
                self.vel[k] = ti.Vector([0, 0, 0])
            if i >= R:
                self.pos[k] = ti.Vector([i * self.grid_len + offsetx, j * self.grid_len + offsety, offsetz])
                self.vel[k] = ti.Vector([0, 0, 0])

        for j in range(self.M + 1):
            x = (20 - L) * self.grid_len
            angle = 3.1415 / 3
            k = 9 * (self.M + 1) + j
            self.pos[k] = ti.Vector(
                        [x - r * ti.sin(angle) + offsetx, j * self.grid_len + offsety, offsetz + r * (1 + ti.cos(angle))])
            angle = 3.1415 / 3 * 2
            k = 11 * (self.M + 1) + j
            self.pos[k] = ti.Vector(
                [x - r * ti.sin(angle) + offsetx, j * self.grid_len + offsety, offsetz + r * (1 + ti.cos(angle))])
            k = 8 * (self.M + 1) + j
            self.pos[k] = (self.pos[k + self.M + 1] + self.pos[k - (self.M + 1)]) * 0.5
            k = 10 * (self.M + 1) + j
            self.pos[k] = (self.pos[k + self.M + 1] + self.pos[k - (self.M + 1)]) * 0.5
            k = 12 * (self.M + 1) + j
            self.pos[k] = (self.pos[k + self.M + 1] + self.pos[k - (self.M + 1)]) * 0.5

        for i in range(self.NF):
            ia, ib, ic = self.f2v[i]
            self.V[i] = self.grid_len ** 2 * 0.5
            self.l_i[i, 0] = self.grid_len
            self.l_i[i, 1] = self.grid_len
            self.l_i[i, 2] = self.grid_len * ti.math.sqrt(2.0)

    @ti.kernel
    def init_pos_load(self, ref_pos: ti.types.ndarray()):
        self.ref_angle.fill(0)
        for i, j in ti.ndrange(self.N + 1, self.M + 1):
            k = i * (self.M + 1) + j
            self.pos[k][0] = ref_pos[k, 0]
            self.pos[k][1] = ref_pos[k, 1]
            self.pos[k][2] = ref_pos[k, 2]
            self.vel[k] = ti.Vector([0, 0, 0])

        for i in range(self.NF):
            ia, ib, ic = self.f2v[i]
            self.V[i] = self.grid_len ** 2 * 0.5
            self.l_i[i, 0] = self.grid_len
            self.l_i[i, 1] = self.grid_len
            self.l_i[i, 2] = self.grid_len * ti.math.sqrt(2.0)

    @ti.kernel
    def init_mesh(self):
        for i, j in ti.ndrange(self.N, self.M):
            k = (i * self.M + j) * 2
            a = i * (self.M + 1) + j
            b = a + 1
            c = a + self.M + 2
            d = a + self.M + 1
            if (i + j) % 2 == 0:
                self.f2v[k + 0] = [c, b, a]
                self.f2v[k + 1] = [a, d, c]  # counterclockwise
            else:
                self.f2v[k + 0] = [b, a, d]
                self.f2v[k + 1] = [d, c, b]

            if (i + j) % 2 == 0:
                if i > 0:
                    self.counter_face[k][0] = ((i - 1) * self.M + j) * 2 + 1
                    self.counter_point[k][0] = 2
                else:
                    self.counter_face[k][0] = -1

                if j < self.M - 1:
                    self.counter_face[k][2] = k + 2
                    self.counter_point[k][2] = 0
                else:
                    self.counter_face[k][2] = -1

                if i < self.N - 1:
                    self.counter_face[k + 1][0] = ((i + 1) * self.M + j) * 2
                    self.counter_point[k + 1][0] = 2
                else:
                    self.counter_face[k + 1][0] = -1

                if j > 0:
                    self.counter_face[k + 1][2] = k - 2
                    self.counter_point[k + 1][2] = 0
                else:
                    self.counter_face[k + 1][2] = -1

                self.counter_face[k][1] = k + 1
                self.counter_point[k][1] = 1
                self.counter_face[k + 1][1] = k
                self.counter_point[k + 1][1] = 1

                quad_id = i * self.M + j
                self.indices[quad_id * 6 + 0] = a
                self.indices[quad_id * 6 + 1] = c
                self.indices[quad_id * 6 + 2] = b
                # 2nd triangle of the square
                self.indices[quad_id * 6 + 3] = d
                self.indices[quad_id * 6 + 4] = c
                self.indices[quad_id * 6 + 5] = a
            else:
                if i > 0:
                    self.counter_face[k][2] = ((i - 1) * self.M + j) * 2 + 1
                    self.counter_point[k][2] = 0
                else:
                    self.counter_face[k][2] = -1

                if j < self.M - 1:
                    self.counter_face[k + 1][0] = k + 3
                    self.counter_point[k + 1][0] = 2
                else:
                    self.counter_face[k + 1][0] = -1

                if i < self.N - 1:
                    self.counter_face[k + 1][2] = ((i + 1) * self.M + j) * 2
                    self.counter_point[k + 1][2] = 0
                else:
                    self.counter_face[k + 1][2] = -1

                if j > 0:
                    self.counter_face[k][2] = k - 2
                    self.counter_point[k][2] = 2
                else:
                    self.counter_face[k][2] = -1

                self.counter_face[k][1] = k + 1
                self.counter_point[k][1] = 1
                self.counter_face[k + 1][1] = k
                self.counter_point[k + 1][1] = 1

                quad_id = i * self.M + j
                self.indices[quad_id * 6 + 0] = b
                self.indices[quad_id * 6 + 1] = a
                self.indices[quad_id * 6 + 2] = d
                # 2nd triangle of the square
                self.indices[quad_id * 6 + 3] = d
                self.indices[quad_id * 6 + 4] = c
                self.indices[quad_id * 6 + 5] = b

        for i, j in ti.ndrange(self.N + 1, self.M + 1):
            if (i // 4 + j // 4) % 2 == 0:
                self.colors[i * (self.M + 1) + j] = (0.22, 0.72, 0.52)
            else:
                self.colors[i * (self.M + 1) + j] = (1, 0.334, 0.52)
            self.uv[i * (self.M + 1) + j] = ti.Vector([1 - j / self.M, i / self.N])

    @ti.kernel
    def clear_manipulation(self):
        for i in range(self.NV):
            self.manipulate_force[i] = ti.Vector([0, 0, 0])

    @ti.kernel
    def set_manipulation(self):
        # for i in range(self.NV):
        #     if i < self.N * 2:
        #         self.manipulate_acc[i] = ti.Vector([0, 0, 400])
        # ((i % self.N) >= 13) and ((i % self.N) <= 18) and
        # for i in range(self.NV):
        #     if ((i % (self.N + 1)) >= 13) and ((i % (self.N + 1)) <= 18) and (i / self.N >= 13) and (i / self.N <= 18):
        #         self.manipulate_acc[i] = ti.Vector([0, 0, 4000])
        for i in range(self.NV):
            self.manipulate_force[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def get_vert_mass(self, i):
        return self.mass

    def init(self, offsetx, offsety, offsetz):
        self.init_mesh()
        self.init_pos_offset(offsetx, offsety, offsetz)
        self.ref_angle.fill(0)

    def init_fold(self, offsetx, offsety, offsetz, curv_num):
        self.init_mesh()
        self.init_pos_offset_fold(offsetx, offsety, offsetz, curv_num)
        self.compute_normal_dir()
        self.init_ref_angle()

    def init_fold_real(self, offsetx, offsety, offsetz, curv_num):
        self.init_mesh()
        self.init_pos_offset_fold_real(offsetx, offsety, offsetz, curv_num)
        self.compute_normal_dir()
        self.init_ref_angle_real()

    def init_load(self, ref_pos):
        self.init_mesh()
        self.init_pos_load(ref_pos)
        self.ref_angle.fill(0)

    @ti.kernel
    def update_visual(self):
        for i in range(self.NV):
            self.x32[i] = ti.cast(self.pos[i], ti.f32)

    @ti.kernel
    def build_f_vis(self):
        for i in range(self.NF):
            self.f_vis[i * 3 + 0] = self.f2v[i][0]
            self.f_vis[i * 3 + 1] = self.f2v[i][1]
            self.f_vis[i * 3 + 2] = self.f2v[i][2]

    @ti.kernel
    def compute_deri(self):
        self.d_ka.fill(0)
        self.d_kl.fill(0)
        self.d_kb.fill(0)

        for i in range(self.NF):
            ia, ib, ic = self.f2v[i]
            a, b, c = self.pos[ia], self.pos[ib], self.pos[ic]

            # membrane_energy_edge_grad
            for l in ti.static(range(3)):
                xx = self.f2v[i][l]
                yy = self.f2v[i][(l + 1) % 3]
                base_len = self.l_i[i, l]
                delta = self.pos[xx] - self.pos[yy]
                l_tau = delta.norm()
                self.d_kl[xx] += -delta * self.compute_membrane_dl(l_tau, base_len) / l_tau
                self.d_kl[yy] += delta * self.compute_membrane_dl(l_tau, base_len) / l_tau

            # membrane_energy_area_grad
            base_area = self.V[i]
            v1 = b - a
            v2 = c - a
            area = 0.5 * v1.cross(v2).norm()
            for l in range(3):
                for j in range(3):
                    self.d_ka[self.f2v[i][l]][j] -= self.compute_membrane_darea(area,
                                                                               base_area) * self.compute_area_dx(
                        area, self.pos[self.f2v[i][l]], self.pos[self.f2v[i][(l + 1) % 3]],
                        self.pos[self.f2v[i][(l + 2) % 3]], j)

            for l in range(3):
                if self.counter_face[i][l] > i:
                    a, b, c, d = self.compute_bending_grad(i, l)
                    theta = self.compute_angle(i, self.counter_face[i][l], l)
                    d_theta = self.compute_bending_dtheta_ref(theta, self.ref_angle[i][l])
                    self.d_kb[self.f2v[i][l]] -= d_theta * a
                    self.d_kb[self.f2v[i][(l + 1) % 3]] -= d_theta * b
                    self.d_kb[self.f2v[i][(l + 2) % 3]] -= d_theta * c
                    self.d_kb[self.f2v[self.counter_face[i][l]][self.counter_point[i][l]]] -= d_theta * d

        for i in self.d_ka:
            self.d_ka[i] /= self.Ka[None]
            self.d_kl[i] /= self.Kl[None]
            self.d_kb[i] /= self.Kb[None]

    @ti.kernel
    def compute_deri_Kb(self):
        self.d_kb.fill(0)

        for i in range(self.NF):
            ia, ib, ic = self.f2v[i]
            a, b, c = self.pos[ia], self.pos[ib], self.pos[ic]

            for l in range(3):
                if self.counter_face[i][l] > i:
                    a, b, c, d = self.compute_bending_grad(i, l)
                    theta = self.compute_angle(i, self.counter_face[i][l], l)
                    d_theta = self.compute_bending_dtheta_ref(theta, self.ref_angle[i][l])
                    self.d_kb[self.f2v[i][l]] -= d_theta * a
                    self.d_kb[self.f2v[i][(l + 1) % 3]] -= d_theta * b
                    self.d_kb[self.f2v[i][(l + 2) % 3]] -= d_theta * c
                    self.d_kb[self.f2v[self.counter_face[i][l]][self.counter_point[i][l]]] -= d_theta * d

        for i in self.d_kb:
            self.d_kb[i] /= self.Kb[None]

    @ti.func
    def dtheta_ref(self):
        return -2.0 * self.Kb[None] * self.dx ** 2 * 1.0 / 3.0

    @ti.kernel
    def ref_angle_backprop_x2a(self, analy_grad: ti.template(), step: int, p: ti.types.ndarray(), cnt: int):
        for i in range(self.NF):
            for l in range(3):
                if self.counter_face[i][l] > i:
                    a, b, c, d = self.compute_bending_grad(i, l)
                    d_ref = self.dtheta_ref()
                    for j in ti.static(range(3)):
                        analy_grad.angleref_grad[step - 1, cnt, i][l] += -p[(self.f2v[i][l] + self.offset) * 3 + j] * d_ref * a[j]
                        analy_grad.angleref_grad[step - 1, cnt, i][l] += -p[
                            (self.f2v[i][(l + 1) % 3] + self.offset) * 3 + j] * d_ref * b[j]
                        analy_grad.angleref_grad[step - 1, cnt, i][l] += -p[
                            (self.f2v[i][(l + 2) % 3] + self.offset) * 3 + j] * d_ref * c[j]
                        analy_grad.angleref_grad[step - 1, cnt, i][l] += -p[
                            (self.f2v[self.counter_face[i][l]][self.counter_point[i][l]] + self.offset) * 3 + j] * d_ref * d[j]

    @ti.func
    def sign(self, x):
        ret = 0.0
        if x > 0.0:
            ret = 1.0
        else:
            ret = -1.0
        return ret

    @ti.kernel
    def ref_angle_backprop_a2ax(self, analy_grad: ti.template(), step: int, cnt: int):
        has_nan = False
        for i in range(self.NF):
            for l in range(3):
                if self.counter_face[i][l] > i:
                    a, b, c, d = self.compute_bending_grad(i, l)
                    for j in ti.static(range(3)):
                        if ti.math.isnan(a[j]) or ti.math.isnan(b[j]) or ti.math.isnan(c[j]) or ti.math.isnan(d[j]):
                            has_nan = True
                    theta = self.compute_angle(i, self.counter_face[i][l], l)
                    analy_grad.angleref_grad[step - 1, cnt, i][l] += analy_grad.angleref_grad[step, cnt, i][l]
                    theta_dis = theta - self.ref_angle[i][l]
                    abs_dis = ti.abs(theta_dis)
                    if abs_dis > self.k_angle[None]:
                        sign = analy_grad.angleref_grad[step, cnt, i][l]
                        for j in ti.static(range(3)):
                            analy_grad.pos_grad[step, self.f2v[i][l] + self.offset, j] += sign * a[j]
                            analy_grad.pos_grad[step, self.f2v[i][(l + 1) % 3] + self.offset, j] += sign * b[j]
                            analy_grad.pos_grad[step, self.f2v[i][(l + 2) % 3] + self.offset, j] += sign * c[j]
                            analy_grad.pos_grad[step, self.f2v[self.counter_face[i][l]][self.counter_point[i][l]] + self.offset, j] += sign * d[j]
                    else:
                        sign = analy_grad.angleref_grad[step, cnt, i][l] * 0.1
                        for j in ti.static(range(3)):
                            analy_grad.pos_grad[step, self.f2v[i][l] + self.offset, j] += sign * a[j]
                            analy_grad.pos_grad[step, self.f2v[i][(l + 1) % 3] + self.offset, j] += sign * b[j]
                            analy_grad.pos_grad[step, self.f2v[i][(l + 2) % 3] + self.offset, j] += sign * c[j]
                            analy_grad.pos_grad[step, self.f2v[self.counter_face[i][l]][self.counter_point[i][l]] + self.offset, j] += sign * d[j]

        if has_nan:
            print("nan in backprop a2ax!")

    @ti.kernel
    def copy_refangle(self, ref_angle: ti.template(), step: int, cnt: int):
        for i in range(self.NF):
            for l in range(3):
                self.ref_angle[i][l] = ref_angle[step, cnt, i][l]




