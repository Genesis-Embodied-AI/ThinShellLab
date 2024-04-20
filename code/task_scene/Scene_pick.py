import taichi as ti

import taichi as ti
from dataclasses import dataclass
import numpy as np
from ..engine.BaseScene import BaseScene

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

    def __init__(self, cloth_size=0.06, device="cuda:0"):
        super(Scene, self).__init__(cloth_size=cloth_size, enable_gripper=False, device=device)
        self.gravity[None] = ti.Vector([0., 0., -9.8])
        self.cloths[0].k_angle[None] = 0.5
        self.nc1 = ti.field(ti.i32, shape=())
        self.nc2 = ti.field(ti.i32, shape=())
        self.tot_force = ti.field(ti.f64, shape=(2, 3))

    def init_scene_parameters(self):

        self.dt = 5e-3
        self.h = self.dt
        self.cloth_cnt = 1
        self.elastic_cnt = 3
        self.elastic_size = [0.06, 0.015, 0.015]
        self.elastic_Nx = int(16)
        self.elastic_Ny = int(16)
        self.elastic_Nz = int(2)
        self.cloth_N = 16

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
        self.elastics[0].init_arch(-0.03, -0.03, -0.008, 0.004)
        self.elastics[1].init(-0.025, 0., 0.0079, True)
        self.elastics[2].init(0.025, 0., 0.0079, True)
        pos = np.array([[-0.025, 0., 0.0079], [0.025, 0., 0.0079]])
        self.gripper.init(self, pos)

    def reset_pos(self):
        self.cloths[0].init(-0.03, -0.03, 0.0004)
        self.elastics[0].init_arch(-0.03, -0.03, -0.008, 0.004)
        self.elastics[1].init(-0.025, 0., 0.0079, True)
        self.elastics[2].init(0.025, 0., 0.0079, True)
        pos = np.array([[-0.025, 0., 0.0079], [0.025, 0., 0.0079]])
        self.gripper.init(self, pos)

    def contact_analysis(self):
        self.nc[None] = 0
        for i in range(self.cloth_cnt):
            for j in range(self.elastic_cnt):

                mu = self.mu_cloth_elastic[None]
                if j == 0:
                    mu = 0.1
                self.contact_pair_analysis(self.cloths[i].body_idx, self.elastics[j].offset,
                                           self.elastics[j].offset + self.elastics[j].n_verts, mu)
                self.contact_pair_analysis(self.elastics[j].body_idx, self.cloths[i].offset,
                                       self.cloths[i].offset + self.cloths[i].NV, mu)

                if j == 0:
                    self.nc1[None] = self.nc[None]
                if j == 1:
                    self.nc2[None] = self.nc[None]

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
        for i in range(self.elastics[2].n_verts):
            if self.elastics[2].is_bottom(i) or self.elastics[2].is_inner_circle(i):
                xx = self.elastics[2].offset + i
                self.frozen[xx * 3] = 1
                self.frozen[xx * 3 + 1] = 1
                self.frozen[xx * 3 + 2] = 1


    @ti.kernel
    def set_frozen_kernel_2(self):
        for i in range(self.elastics[0].n_verts):
            xx = self.elastics[0].offset + i
            self.frozen[xx * 3] = 1
            self.frozen[xx * 3 + 1] = 1
            self.frozen[xx * 3 + 2] = 1

    @ti.kernel
    def compute_reward(self) -> ti.f64:
        ret = 0.0
        for i in range(self.cloths[0].NV):
            if ti.cast(i / (self.cloths[0].M + 1), ti.i32) == 8:
                ret += self.cloths[0].pos[i].z

        return ret

    @ti.kernel
    def compute_reward_deliver(self, analy_grad: ti.template()) -> ti.f64:
        ret = 0.0
        for i in range(self.cloths[0].NV):
            ret -= (self.cloths[0].pos[i][0] - analy_grad.pos_buffer[69, i + self.cloths[0].offset, 0] - 0.01) ** 2
            ret -= (self.cloths[0].pos[i][1] - analy_grad.pos_buffer[69, i + self.cloths[0].offset, 1] - 0.01) ** 2
            ret -= (self.cloths[0].pos[i][2] - analy_grad.pos_buffer[69, i + self.cloths[0].offset, 2] - 0.01) ** 2

        return ret

    @ti.kernel
    def compute_reward_pick_fold(self) -> ti.f64:
        ret = 0.0
        for i in range(self.cloths[0].NF):
            for l in range(3):
                if self.cloths[0].counter_face[i][l] > i:
                    p = self.cloths[0].f2v[self.cloths[0].counter_face[i][l]][self.cloths[0].counter_point[i][l]]
                    if ti.cast(self.cloths[0].f2v[i][l] / (self.cloths[0].M + 1), ti.i32) == 7 and ti.cast(
                            p / (self.cloths[0].M + 1), ti.i32) == 9:
                        ret += self.cloths[0].ref_angle[i][l]

                        theta = self.cloths[0].compute_angle(i, self.cloths[0].counter_face[i][l], l)
                        ret += 0.01 * theta

        return ret

    @ti.kernel
    def compute_reward_pick_and_fold(self) -> ti.f64:
        ret = 0.0
        for i in range(self.cloths[0].NF):
            for l in range(3):
                if self.cloths[0].counter_face[i][l] > i:
                    p = self.cloths[0].f2v[self.cloths[0].counter_face[i][l]][self.cloths[0].counter_point[i][l]]
                    if ti.cast(self.cloths[0].f2v[i][l] / (self.cloths[0].M + 1), ti.i32) == 7 and ti.cast(
                            p / (self.cloths[0].M + 1), ti.i32) == 9:
                        ret += self.cloths[0].ref_angle[i][l]

                        theta = self.cloths[0].compute_angle(i, self.cloths[0].counter_face[i][l], l)
                        ret += 0.01 * theta

        for i in range(self.cloths[0].NV):
            if ti.cast(i / (self.cloths[0].M + 1), ti.i32) == 8:
                ret += self.cloths[0].pos[i].z

        return ret

    def action(self, step, delta_pos, delta_rot):
        # self.gripper.print_plot()
        self.gripper.step_simple(delta_pos, delta_rot)
        # self.gripper.print_plot()
        self.gripper.update_bound(self)
        # a1 = self.elastics[1].check_determinant()
        # a2 = self.elastics[2].check_determinant()
        # if not (a1 and a2):
        #     print("penetrate!!!!")
        # self.gripper.print_plot()
        self.pushup_property(self.pos, self.elastics[1].F_x, self.elastics[1].offset)
        self.pushup_property(self.pos, self.elastics[2].F_x, self.elastics[2].offset)

    def timestep_finish(self):
        self.update_vel()
        self.push_down_vel()
        self.update_ref_angle()
        self.update_visual()

    @ti.kernel
    def static_friction_loss(self, analy_grad: ti.template(), step: int):

        max_k = -100.0

        # friction
        for i in range(self.nc1[None], self.nc[None]):
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

            if r > self.dt * self.eps_v * 0.9:
                pressure = k / self.const_mu[i]
                g = u * k * self.f1(r)
                g1 = g @ self.const_T[i]
                n_c = self.const_n[i]
                w1 = ti.Vector([w[0], w[1], w[2], -1], ti.f64)
                for i1 in range(4):
                    for j1 in ti.static(range(3)):
                        dfdp = w1[i1] * g1[j1] / pressure
                        # if ti.abs(zT) > 1.0 / self.k_contact and (self.bel(idx[3]) == 3 or self.bel(idx[0]) == 3):
                        #     print("zT:", zT, "dfdp:", dfdp, end=" ")
                        #     print("obj:", self.bel(idx[i1]), "dim:", j1)
                        for i2, j2 in ti.static(ti.ndrange(4, 3)):
                            analy_grad.pos_grad[step - 1, idx[i2], j2] += -dfdp * w1[i2] * n_c[j2] * self.k_contact * analy_grad.f_loss_ratio
                            # ret = self.bel(idx[i2])
                            # if ret == 3 and j2 == 2 and ti.abs(zT) > 1.0 / self.k_contact:
                            #     print("pos grad", zT * dfdp * w1[i2] * n_c[j2] * self.k_contact)

                # w1 = ti.Vector([-w[0], -w[1], -w[2], 1], ti.f64)
                # u_3d = ti.Vector([u[0] * self.const_T[i][0, 0] + u[1] * self.const_T[i][1, 0],
                #                   u[0] * self.const_T[i][0, 1] + u[1] * self.const_T[i][1, 1],
                #                   u[0] * self.const_T[i][0, 2] + u[1] * self.const_T[i][1, 2]])
                # for i1 in range(4):
                #     for j1 in ti.static(range(3)):
                #         analy_grad.pos_grad[step, idx[i1], j1] += u_3d[j1] * w1[i1] * analy_grad.f_loss_ratio * k

    def time_step(self, f_contact, frame_idx, force_stick=True):
        self.timestep_init()

        self.calc_vn()
        f_contact(self)
        self.contact_analysis()

        # self.save_constraints('../debug/const_%d.pt' % frame_idx)
        # print("contact_cnt", self.nc[None])
        # self.pos.copy_from(self.x_hat)
        # self.push_down_pos()
        iter = 0
        delta = 1e5
        temp_delta = 1e6
        PD = True
        while iter < 50:
            iter += 1
            # if frame_idx >= 5:
            #     self.check_differential()
            # if not self.elastics[1].check_determinant():
            #     print("not deter 1!!!")
            # if not self.elastics[2].check_determinant():
            #     print("not deter 2!!!")
            self.newton_step_init()
            # self.calc_vn()
            # self.contact_analysis()
            # split energy and residual!!!!
            self.compute_energy()
            PD = self.compute_residual_and_Hessian(False, iter, spd=True)
            temp_delta = delta
            if not PD:
                break
            # if iter > 20:
            #     alpha = "GD"
            #     delta = self.calc_f_norm(self.F) * 0.01
            #     self.gradient_step(delta)
            # else:
            delta, alpha = self.newton_step(iter)
            # if iter > 70:
            #     print(f"iter: {iter}, delta: {delta}, step size: {alpha}, energy {self.E[None]}")
            if delta < 1e-7:
                break
        # print(f'pass {frame_idx}, iter:', iter, "delta:", delta)

        self.timestep_finish()

    @ti.kernel
    def gather_force(self):

        for i in range(self.elastics[1].n_verts):
            if self.elastics[1].is_bottom(i) or self.elastics[1].is_inner_circle(i):
                self.tot_force[0, 0] += self.elastics[1].F_f[i][0]
                self.tot_force[0, 1] += self.elastics[1].F_f[i][1]
                self.tot_force[0, 2] += self.elastics[1].F_f[i][2]

        for i in range(self.elastics[2].n_verts):
            if self.elastics[2].is_bottom(i) or self.elastics[2].is_inner_circle(i):
                self.tot_force[1, 0] += self.elastics[2].F_f[i][0]
                self.tot_force[1, 1] += self.elastics[2].F_f[i][1]
                self.tot_force[1, 2] += self.elastics[2].F_f[i][2]

    def print_force(self):
        self.elastics[1].get_force()
        self.elastics[2].get_force()
        self.tot_force.fill(0)
        self.gather_force()
        for i in range(2):
            for j in range(3):
                print(self.tot_force[i, j], end=" ")
        print("")

    @ti.kernel
    def check_pos_nan(self) -> ti.i32:
        has_nan = False
        for i in range(self.tot_NV):
            if ti.math.isnan(self.pos[i][0]) or ti.math.isnan(self.pos[i][1]) or ti.math.isnan(self.pos[i][2]):
                has_nan = True
        return has_nan

    def check_early_stop(self, frame, ifprint=False, RL=False):

        if self.check_pos_nan():
            if ifprint:
                print("exist nan")
            return True

        self.elastics[1].get_force()
        self.elastics[2].get_force()
        self.tot_force.fill(0)
        self.gather_force()
        for i in range(2):
            for j in range(3):
                if ti.abs(self.tot_force[i, j]) > 10:
                    if ifprint:
                        print("too much force")
                    return True

        if ((self.nc1[None] == self.nc2[None]) or (self.nc[None] == self.nc2[None])) and (frame > 10) and (not RL):
            if ifprint:
                print("no contact")
            return True

        return False
