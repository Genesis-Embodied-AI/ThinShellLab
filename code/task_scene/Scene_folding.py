import taichi as ti

from ..engine.model_fold_offset import Cloth
from ..engine.model_elastic_offset import Elastic
from ..engine.model_elastic_tactile import Elastic as tactile

from typing import List
import taichi as ti
import torch
from dataclasses import dataclass
import os
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
        self.gravity[None] = ti.Vector([0., 0., 0.])
        self.cloths[0].k_angle[None] = 0.5


    def init_scene_parameters(self):
        self.dt = 5e-3
        self.h = self.dt
        self.cloth_cnt = 1
        self.elastic_cnt = 2
        self.elastic_size = [0.07, 0.015]
        self.elastic_Nx = int(9)
        self.elastic_Ny = int(9)
        self.elastic_Nz = int(2)
        self.cloth_N = 15
        self.cloth_M = 3

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

    def init_objects(self):
        rho = 4e1

        self.cloths.append(Cloth(self.cloth_N, self.dt, self.cloth_size, self.tot_NV, rho, 0, False, 3))

        self.elastic_offset = (self.cloth_N + 1) * (self.cloth_M + 1)
        tmp_tot = self.elastic_offset

        self.elastics.append(Elastic(self.dt, self.elastic_size[0], tmp_tot, self.elastic_Nx, self.elastic_Ny, self.elastic_Nz))
        tmp_tot += self.elastic_Nx * self.elastic_Ny * self.elastic_Nz

        for i in range(1, self.elastic_cnt):
            self.elastics.append(tactile(self.dt, tmp_tot, self.elastic_size[i] / 0.03))
            tmp_tot += self.elastics[i].n_verts

        self.tot_NV = tmp_tot

    def init(self):
        half_curve_num = 2
        self.cloths[0].init_fold(-0.07, -0.01, 0.0004, half_curve_num)
        self.elastics[0].init(-0.035, -0.035, -0.00875)
        r = self.cloths[0].grid_len * (half_curve_num * 2 - 1) / 3.1415
        x = -0.07 + (7 + half_curve_num) / 16 * 0.1 - r * 0.86 + 0.005
        self.elastics[1].init(x, 0.0, 2 * r + 0.0079, True)
        print("pos:", x, 2 * r + 0.0079)
        pos = np.array([[x, 0.0, 2 * r + 0.0079]])
        self.gripper.init(self, pos)

    def reset_pos(self):
        half_curve_num = 2
        self.cloths[0].init_fold(-0.07, -0.01, 0.0004, half_curve_num)
        self.elastics[0].init(-0.035, -0.035, -0.00875)
        r = self.cloths[0].grid_len * (half_curve_num * 2 - 1) / 3.1415
        x = -0.07 + (7 + half_curve_num) / 16 * 0.1 - r * 0.86 + 0.005
        self.elastics[1].init(x, 0.0, 2 * r + 0.0079, True)
        pos = np.array([[x, 0.0, 2 * r + 0.0079]])
        self.gripper.init(self, pos)

    def contact_analysis(self):
        self.nc[None] = 0
        for i in range(self.cloth_cnt):
            for j in range(self.elastic_cnt):

                mu = self.mu_cloth_elastic[None]
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
        for i in range(self.cloths[0].M + 1):
            xx = self.cloths[0].offset + self.cloths[0].N * (self.cloths[0].M + 1) + i
            self.frozen[xx * 3] = 1
            self.frozen[xx * 3 + 1] = 1
            self.frozen[xx * 3 + 2] = 1

    @ti.kernel
    def compute_reward(self, curve7: ti.f64, curve8: ti.f64) -> ti.f64:
        ret = 0.0
        ret1 = 0.0
        ret2 = 0.0
        for i in range(self.cloths[0].NF):
            for l in range(3):
                if self.cloths[0].counter_face[i][l] > i:
                    p = self.cloths[0].f2v[self.cloths[0].counter_face[i][l]][self.cloths[0].counter_point[i][l]]
                    if ti.cast(self.cloths[0].f2v[i][l] / (self.cloths[0].M + 1), ti.i32) == 6 and ti.cast(
                            p / (self.cloths[0].M + 1), ti.i32) == 8:
                        ret += - self.cloths[0].ref_angle[i][l] * curve7
                    elif ti.cast(self.cloths[0].f2v[i][l] / (self.cloths[0].M + 1), ti.i32) == 7 and ti.cast(
                            p / (self.cloths[0].M + 1), ti.i32) == 9:
                        ret += - self.cloths[0].ref_angle[i][l] * curve8

                    # else:
                    #     ret -= 2 * self.cloths[0].ref_angle[i][l] ** 2
        return ret

    @ti.kernel
    def compute_reward_8(self) -> ti.f64:
        ret = 0.0
        ret1 = 0.0
        ret2 = 0.0
        curve8 = 1
        curve7 = -1
        for i in range(self.cloths[0].NF):
            for l in range(3):
                if self.cloths[0].counter_face[i][l] > i:
                    p = self.cloths[0].f2v[self.cloths[0].counter_face[i][l]][self.cloths[0].counter_point[i][l]]
                    if ti.cast(self.cloths[0].f2v[i][l] / (self.cloths[0].M + 1), ti.i32) == 6 and ti.cast(
                            p / (self.cloths[0].M + 1), ti.i32) == 8:
                        ret += - self.cloths[0].ref_angle[i][l] * curve7
                    elif ti.cast(self.cloths[0].f2v[i][l] / (self.cloths[0].M + 1), ti.i32) == 7 and ti.cast(
                            p / (self.cloths[0].M + 1), ti.i32) == 9:
                        ret += - self.cloths[0].ref_angle[i][l] * curve8

                    # else:
                    #     ret -= 2 * self.cloths[0].ref_angle[i][l] ** 2
        return ret

    @ti.kernel
    def compute_reward_7(self) -> ti.f64:
        ret = 0.0
        ret1 = 0.0
        ret2 = 0.0
        curve8 = -1
        curve7 = 1
        for i in range(self.cloths[0].NF):
            for l in range(3):
                if self.cloths[0].counter_face[i][l] > i:
                    p = self.cloths[0].f2v[self.cloths[0].counter_face[i][l]][self.cloths[0].counter_point[i][l]]
                    if ti.cast(self.cloths[0].f2v[i][l] / (self.cloths[0].M + 1), ti.i32) == 6 and ti.cast(
                            p / (self.cloths[0].M + 1), ti.i32) == 8:
                        ret += - self.cloths[0].ref_angle[i][l] * curve7
                    elif ti.cast(self.cloths[0].f2v[i][l] / (self.cloths[0].M + 1), ti.i32) == 7 and ti.cast(
                            p / (self.cloths[0].M + 1), ti.i32) == 9:
                        ret += - self.cloths[0].ref_angle[i][l] * curve8

                    # else:
                    #     ret -= 2 * self.cloths[0].ref_angle[i][l] ** 2
        return ret

    @ti.func
    def is_upper_curve(self, x):
        if ti.cast(x / (self.cloths[0].M + 1), ti.i32) == 7:
            return True
        return False

    @ti.func
    def is_lower_curve(self, x):
        if ti.cast(x / (self.cloths[0].M + 1), ti.i32) == 8:
            return True
        return False
    
    def is_upper_curve_py(self, x):
        if x // (self.cloths[0].M + 1) == 7:
            return True
        return False

    def is_lower_curve_py(self, x):
        if x // (self.cloths[0].M + 1) == 8:
            return True
        return False
    
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

    def timestep_finish(self):
        self.update_vel()
        self.push_down_vel()
        self.update_ref_angle()
        self.update_visual()

    def check_early_stop(self, frame, ifprint=False, RL=False):

        if self.check_pos_nan():
            if ifprint:
                print("exist nan")
            return True

        for j in range(1, self.effector_cnt):
            self.elastics[j].get_force()

        self.tot_force.fill(0)
        self.gather_force()
        for i in range(self.effector_cnt - 1):
            for j in range(3):
                if ti.abs(self.tot_force[i, j]) > 10:
                    if ifprint:
                        print("too much force")
                    return True

            force = ti.sqrt(self.tot_force[i, 0]**2 + self.tot_force[i, 1]**2 + self.tot_force[i, 2]**2)
            if (force < 0.2) and (frame > 10) and (not RL):
                if ifprint:
                    print("no contact")
                return True

        return False

    def print_force(self):
        self.elastics[1].get_force()
        self.tot_force.fill(0)
        self.gather_force()
        for i in range(1):
            for j in range(3):
                print(self.tot_force[i, j], end=" ")
        print("")

    @ti.kernel
    def get_colors(self, colors: ti.template()):
        colors.fill(0)
        for i in range(self.cloths[0].NV):
            colors[i + self.cloths[0].offset] = ti.Vector([1, 1, 1])
        # for i in range(self.cloths[1].NV):
        #     colors[i + self.cloths[1].offset] = ti.Vector([0.5, 0.5, 0.5])
        for i in range(self.elastics[1].n_verts):
            colors[i + self.elastics[1].offset] = ti.Vector([0.22, 0.72, 0.52])  # Agent1 Color

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
            # print(f"iter: {iter}, delta: {delta}, step size: {alpha}, energy {self.E[None]}")
            if delta < 1e-7:
                break
        # print(f'pass {frame_idx}, iter:', iter, "delta:", delta)

        self.timestep_finish()