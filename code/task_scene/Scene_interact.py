import taichi as ti

from model_fold_offset import Cloth
from model_elastic_offset import Elastic
from model_elastic_tactile import Elastic as tactile

from typing import List
import taichi as ti
import torch
from dataclasses import dataclass
import os
import numpy as np
import matplotlib.pyplot as plt
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

    def __init__(self, cloth_size=0.06, device="cuda:0", soft=False, dense=10000.0):
        self.dense = dense
        self.soft = soft
        super(Scene, self).__init__(cloth_size=cloth_size, enable_gripper=True, device=device)
        self.gravity[None] = ti.Vector([0., 0., -9.8])
        self.cloths[0].k_angle[None] = 3.14


    def init_scene_parameters(self):

        self.dt = 5e-3
        self.h = self.dt
        self.cloth_cnt = 1
        self.elastic_cnt = 4
        self.elastic_size = [0.06, 0.015, 0.015, 0.012]
        self.elastic_Nx = int(16)
        self.elastic_Ny = int(16)
        self.elastic_Nz = int(2)
        self.cloth_N = 15
        self.extra_obj = True
        self.effector_cnt = 3

        self.k_contact = 30000
        self.eps_contact = 0.0004
        self.eps_v = 0.01
        self.max_n_constraints = 10000
        self.damping = 1.0

    def init_objects(self):
        rho = 4e1
        for i in range(self.cloth_cnt):
            self.cloths.append(Cloth(self.cloth_N, self.dt, self.cloth_size, self.tot_NV, rho, i * ((self.cloth_N + 1)**2)))

        self.elastic_offset = ((self.cloth_N + 1)**2) * self.cloth_cnt
        tmp_tot = self.elastic_offset

        self.elastics.append(Elastic(self.dt, self.elastic_size[0], tmp_tot, self.elastic_Nx, self.elastic_Ny, self.elastic_Nz))
        tmp_tot += self.elastic_Nx * self.elastic_Ny * self.elastic_Nz

        for i in range(1, self.elastic_cnt - 1):
            self.elastics.append(tactile(self.dt, tmp_tot, self.elastic_size[i] / 0.03))
            tmp_tot += self.elastics[i].n_verts

        if self.soft:
            self.elastics.append(
                Elastic(self.dt, self.elastic_size[3], tmp_tot, 6, 6, 4, density=self.dense))
        else:
            self.elastics.append(
                Elastic(self.dt, self.elastic_size[3], tmp_tot, 6, 6, 4, density=self.dense))
        tmp_tot += 6 * 6 * 4

        self.tot_NV = tmp_tot

    def init_all(self):
        self.init()
        self.init_property()
        self.set_frozen()
        self.set_ext_force()
        self.update_visual()
        # print(self.cloths[0].mass)
        # print(self.elastics[0].F_m[0])

    def init(self):
        self.cloths[0].init(-0.045, -0.03, 0.0004)
        self.elastics[0].init(-0.03, -0.03, -0.004)
        self.elastics[1].init(-0.04, 0., 0.0083, True)
        self.elastics[2].init(-0.04, 0., -0.0075, False)
        self.elastics[3].init(0.001, -0.006, 0.0008)
        pos = np.array([[-0.04, 0., 0.0004]])
        self.gripper.init(self, pos)

    def reset_pos(self):
        self.cloths[0].init(-0.045, -0.03, 0.0004)
        self.elastics[0].init(-0.03, -0.03, -0.004)
        self.elastics[1].init(-0.04, 0., 0.0083, True)
        self.elastics[2].init(-0.04, 0., -0.0075, False)
        self.elastics[3].init(0.001, -0.006, 0.0008)
        pos = np.array([[-0.04, 0., 0.0004]])
        self.gripper.init(self, pos)

    def contact_analysis(self):
        self.nc[None] = 0

        for i in range(self.cloth_cnt):
            for j in range(self.elastic_cnt):
                mu = self.mu_cloth_elastic[None]
                if j == 0 or j == 3:
                    mu = 0.2
                self.contact_pair_analysis(self.cloths[i].body_idx, self.elastics[j].offset,
                                           self.elastics[j].offset + self.elastics[j].n_verts, mu)
                self.contact_pair_analysis(self.elastics[j].body_idx, self.cloths[i].offset,
                                               self.cloths[i].offset + self.cloths[i].NV, mu)

        self.contact_pair_analysis(self.elastics[0].body_idx, self.elastics[3].offset,
                                   self.elastics[3].offset + self.elastics[3].n_verts, 0.1)
        self.contact_pair_analysis(self.elastics[3].body_idx, self.elastics[0].offset,
                                   self.elastics[0].offset + self.elastics[0].n_verts, 0.1)


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
    def compute_reward(self) -> ti.f64:
        ret = 0.0
        for i in self.cloths[0].pos:
            ret = ret - self.cloths[0].pos[i].x
        for i in self.elastics[3].F_x:
            ret = ret + self.elastics[3].F_x[i].x * 256.0 / 144.0
        return ret

    @ti.kernel
    def compute_reward_1(self) -> ti.f64:
        ret = 0.0
        for i in self.elastics[3].F_x:
            ret = ret - self.elastics[3].F_x[i].x
        return ret

    def action(self, step, delta_pos, delta_rot):
        # self.gripper.print_plot()
        if step < 5:
            self.gripper.step(delta_pos, delta_rot, ti.Vector([-0.0006]))
        else:
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
    def get_colors(self, colors: ti.template()):
        colors.fill(0)
        for i in range(self.cloths[0].NV):
            colors[i + self.cloths[0].offset] = ti.Vector([1, 1, 1])
        for i in range(self.elastics[1].n_verts):
            colors[i + self.elastics[1].offset] = ti.Vector([0.22, 0.72, 0.52])  # Agent1 Color
        for i in range(self.elastics[2].n_verts):
            colors[i + self.elastics[2].offset] = ti.Vector([1, 0.334, 0.52])    # Agent1 Color

        for i in range(self.elastics[3].n_verts):
            colors[i + self.elastics[3].offset] = ti.Vector([1, 0.334, 0.52])


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

