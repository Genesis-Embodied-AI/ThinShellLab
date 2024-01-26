import taichi as ti

import taichi as ti
from dataclasses import dataclass
import numpy as np
from BaseScene import BaseScene

vec3 = ti.types.vector(3, ti.f64)
vec3i = ti.types.vector(3, ti.i32)

@dataclass
class Body:
    v_start: ti.i32
    v_end: ti.i32
    f_start: ti.i32
    f_end: ti.i32

@ti.data_oriented
class Scene(BaseScene):

    def __init__(self, cloth_size=0.06):
        super(Scene, self).__init__(cloth_size=cloth_size, enable_gripper=False)
        self.gravity[None] = ti.Vector([0., 0., 0.])
        self.cloths[0].k_angle[None] = 3.14
        self.mu_cloth_cloth = ti.field(ti.f64, ())
        self.nc1 = ti.field(ti.i32, ())
        self.elastics[1].E = 500000
        self.elastics[1].nu = 0.2
        mu, lam = self.elastics[1].E / (2 * (1 + self.elastics[1].nu)), self.elastics[1].E * self.elastics[1].nu / ((1 + self.elastics[1].nu) * (1 - 2 * self.elastics[1].nu))  # lambda = 0
        self.elastics[1].mu[None] = mu
        self.elastics[1].lam[None] = lam
        self.elastics[1].alpha[None] = 1 + mu / lam

    def init_scene_parameters(self):

        self.dt = 5e-3
        self.h = self.dt
        self.cloth_cnt = 3
        self.elastic_cnt = 2
        self.elastic_size = [0.1, 0.015]
        self.elastic_Nx = int(16)
        self.elastic_Ny = int(16)
        self.elastic_Nz = int(2)
        self.cloth_N = 15

        self.k_contact = 10000
        self.eps_contact = 0.0004
        self.eps_v = 0.01
        self.max_n_constraints = 10000
        self.damping = 1.0

    def init_all(self):
        self.init()
        self.init_property()
        self.set_frozen()
        self.set_ext_force()
        self.update_visual()
        # print(self.cloths[0].mass)
        # print(self.elastics[0].F_m[0])

    def init(self):
        self.cloths[0].init(-0.03, -0.03, 0.0004)
        self.cloths[1].init(-0.03, -0.03, 0.0008)
        self.cloths[2].init(-0.03, -0.03, 0.0012)
        self.elastics[0].init(-0.05, -0.05, -0.00666)
        self.elastics[1].init(0.0, 0., 0.0105, True)
        pos = np.array([[0.0, 0., 0.0105]])
        self.gripper.init(self, pos)

    def reset_pos(self):
        self.cloths[0].init(-0.03, -0.03, 0.0004)
        self.cloths[1].init(-0.03, -0.03, 0.0008)
        self.cloths[2].init(-0.03, -0.03, 0.0012)
        self.elastics[0].init(-0.05, -0.05, -0.00666)
        self.elastics[1].init(0.0, 0., 0.0105, True)
        pos = np.array([[0.0, 0., 0.0105]])
        self.gripper.init(self, pos)

    def contact_analysis(self):
        self.nc[None] = 0
        for i in range(self.cloth_cnt):
            for j in range(self.cloth_cnt):
                mu = self.mu_cloth_cloth[None]
                if abs(i - j) == 1:  # TODO: contact relationship
                    self.contact_pair_analysis(self.cloths[i].body_idx, self.cloths[j].offset,
                                               self.cloths[j].offset + self.cloths[j].NV, mu)
                    self.contact_pair_analysis(self.cloths[j].body_idx, self.cloths[i].offset,
                                               self.cloths[i].offset + self.cloths[i].NV, mu)
        self.nc1[None] = self.nc[None]

        for i in range(self.cloth_cnt):
            for j in range(self.elastic_cnt):
                mu = self.mu_cloth_elastic[None]
                if j == 0:
                    mu = 0.4
                self.contact_pair_analysis(self.cloths[i].body_idx, self.elastics[j].offset,
                                           self.elastics[j].offset + self.elastics[j].n_verts, mu)
                self.contact_pair_analysis(self.elastics[j].body_idx, self.cloths[i].offset,
                                               self.cloths[i].offset + self.cloths[i].NV, mu)

    @ti.kernel
    def set_frozen_kernel(self):
        for i in range(self.elastics[0].n_verts):
            xx = self.elastics[0].offset + i
            self.frozen[xx * 3] = 1
            self.frozen[xx * 3 + 1] = 1
            self.frozen[xx * 3 + 2] = 1
        for i in range(self.elastics[1].n_verts):
            if self.elastics[1].is_bottom(i) or self.elastics[1].is_inner_circle(i):
                xx = self.elastics[1].offset + i
                self.frozen[xx * 3] = 1
                self.frozen[xx * 3 + 1] = 1
                self.frozen[xx * 3 + 2] = 1

    @ti.kernel
    def compute_reward(self) -> ti.f64:
        ret = 0.0
        for i in self.cloths[0].pos:
            ret = ret - self.cloths[0].pos[i].x
        return ret

    def action(self, step, delta_pos, delta_rot):
        # self.gripper.print_plot()
        self.gripper.step_simple(delta_pos, delta_rot)
        # self.gripper.print_plot()
        self.gripper.update_bound(self)
        a1 = self.elastics[1].check_determinant()
        if not a1:
            print("penetrate!!!!")
        # self.gripper.print_plot()
        self.pushup_property(self.pos, self.elastics[1].F_x, self.elastics[1].offset)

    def timestep_finish(self):
        self.update_vel()
        self.push_down_vel()
        self.update_ref_angle()
        self.update_visual()

    @ti.kernel
    def contact_energy_backprop_friction(self, diff: ti.i32, analy_grad: ti.template(), step: int,
                                         p_array: ti.types.ndarray()):

        max_k = -100.0

        # friction
        for i in range(self.nc1[None]):
            idx = self.const_idx[i]
            w = self.const_w[i]
            # k = 0
            k = self.const_k[i]
            if k > max_k:
                max_k = k

            x_c = self.pos[idx[0]] * w[0] + self.pos[idx[1]] * w[1] + self.pos[idx[2]] * w[2]
            dx = self.pos[idx[3]] - x_c - self.const_dx0[i]
            u = self.const_T[i] @ dx
            r = u.norm()

            # if r > self.dt * self.eps_v * 0.9:
            pressure = k / self.const_mu[i]
            g = u * k * self.f1(r)
            g1 = g @ self.const_T[i]
            n_c = self.const_n[i]
            w1 = ti.Vector([w[0], w[1], w[2], -1], ti.f64)
            for i1 in range(4):
                for j1 in ti.static(range(3)):
                    dfdmu = w1[i1] * g1[j1] / self.mu_cloth_cloth[None]
                    zT = p_array[idx[i1] * 3 + j1]
                    # if j1 == 0 and idx[i1] >= self.cloths[2].offset:
                    #     print("j", j1, "dfdmu:", dfdmu, "zT:", zT)
                    # if ti.abs(zT) > 1.0 / self.k_contact and (self.bel(idx[3]) == 3 or self.bel(idx[0]) == 3):
                    #     print("zT:", zT, "dfdp:", dfdp, end=" ")
                    #     print("obj:", self.bel(idx[i1]), "dim:", j1)
                    if not self.frozen[idx[i1] * 3 + j1]:
                        # print("zT:", zT, "dfdmu:", dfdmu, end=" ")
                        analy_grad.grad_friction_coef[None] += zT * dfdmu

    @ti.kernel
    def contact_energy_backprop(self, diff: ti.i32, analy_grad: ti.template(), step: int, p_array: ti.types.ndarray()):

        max_k = -100.0

        # friction
        for i in range(self.nc[None]):
            idx = self.const_idx[i]
            w = self.const_w[i]
            # k = 0
            k = self.const_k[i]
            if k > max_k:
                max_k = k

            x_c = self.pos[idx[0]] * w[0] + self.pos[idx[1]] * w[1] + self.pos[idx[2]] * w[2]
            dx = self.pos[idx[3]] - x_c - self.const_dx0[i]
            u = self.const_T[i] @ dx
            r = u.norm()

            # if r > self.dt * self.eps_v * 0.8:
            pressure = k / self.const_mu[i]
            g = u * k * self.f1(r)
            g1 = g @ self.const_T[i]
            n_c = self.const_n[i]
            w1 = ti.Vector([w[0], w[1], w[2], -1], ti.f64)
            for i1 in range(4):
                for j1 in ti.static(range(3)):
                    dfdp = w1[i1] * g1[j1] / pressure
                    zT = p_array[idx[i1] * 3 + j1]
                    # if ti.abs(zT) > 1.0 / self.k_contact and (self.bel(idx[3]) == 3 or self.bel(idx[0]) == 3):
                    #     print("zT:", zT, "dfdp:", dfdp, end=" ")
                    #     print("obj:", self.bel(idx[i1]), "dim:", j1)
                    for i2, j2 in ti.static(ti.ndrange(4, 3)):
                        analy_grad.pos_grad[step, idx[i2], j2] += zT * dfdp * w1[i2] * n_c[j2] * self.k_contact
                        # ret = self.bel(idx[i2])
                        # if ret == 3 and j2 == 2 and ti.abs(zT) > 1.0 / self.k_contact:
                        #     print("pos grad", zT * dfdp * w1[i2] * n_c[j2] * self.k_contact)

            h = self.f1(r) * ti.Matrix.identity(ti.f64, 2)
            if r > 1e-9:
                h += self.f2(r) * (u / r).outer_product(u)

            h1 = k * self.const_T[i].transpose() @ h @ self.const_T[i]

            w1 = ti.Vector([-w[0], -w[1], -w[2], 1], ti.f64)
            for i1, i2 in ti.ndrange(4, 4):
                for j1, j2 in ti.static(ti.ndrange(3, 3)):
                    zT = p_array[idx[i1] * 3 + j1]
                    analy_grad.pos_grad[step, idx[i2], j2] += zT * w1[i1] * w1[i2] * h1[j1, j2]

    @ti.kernel
    def get_colors(self, colors: ti.template()):
        colors.fill(0)
        for i in range(self.cloths[0].NV):
            colors[i + self.cloths[0].offset] = ti.Vector([1, 1, 1])
        for i in range(self.cloths[1].NV):
            colors[i + self.cloths[1].offset] = ti.Vector([0.23, 0.66, 0.9])
        for i in range(self.cloths[2].NV):
            colors[i + self.cloths[2].offset] = ti.Vector([0.33, 0.33, 0.33])
        for i in range(self.elastics[1].n_verts):
            colors[i + self.elastics[1].offset] = ti.Vector([0.22, 0.72, 0.52])  # Agent1 Color

    def time_step(self, f_contact, frame_idx, force_stick=True):
        self.timestep_init()

        self.calc_vn()
        f_contact(self)
        self.contact_analysis()

        iter = 0
        delta = 1e5
        temp_delta = 1e6
        PD = True
        while iter < 50:
            iter += 1

            self.newton_step_init()

            self.compute_energy()
            PD = self.compute_residual_and_Hessian(False, iter, spd=True)
            temp_delta = delta
            if not PD:
                break

            delta, alpha = self.newton_step(iter)

            if delta < 1e-7:
                break

        self.timestep_finish()


