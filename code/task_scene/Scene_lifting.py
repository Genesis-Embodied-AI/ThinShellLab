import taichi as ti

from model_fold_offset import Cloth
from model_elastic_offset import Elastic

from typing import List
import taichi as ti
import torch
from dataclasses import dataclass
from model_elastic_tactile import Elastic as tactile
import os
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

    def __init__(self, cloth_size=0.06, device="cuda:0"):
        super(Scene, self).__init__(cloth_size=cloth_size, enable_gripper=False, device=device)
        self.cloths[0].k_angle[None] = 3.14
        print("tot node:", self.tot_NV)

    def init_scene_parameters(self):
        self.dt = 5e-3
        self.h = self.dt
        self.cloth_cnt = 1
        self.elastic_cnt = 4
        self.elastic_size = [0.007, 0.015, 0.015, 0.015]
        self.elastic_Nx = int(5)
        self.elastic_Ny = int(5)
        self.elastic_Nz = int(5)
        self.cloth_N = 15

        self.k_contact = 500
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

        self.elastics.append(Elastic(self.dt, self.elastic_size[0], tmp_tot, self.elastic_Nx, self.elastic_Ny, self.elastic_Nz, 20000.0))
        tmp_tot += self.elastic_Nx * self.elastic_Ny * self.elastic_Nz

        for i in range(1, self.elastic_cnt):
            self.elastics.append(tactile(self.dt, tmp_tot, self.elastic_size[i] / 0.03))
            tmp_tot += self.elastics[i].n_verts

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

        self.cloths[0].init(-0.03, -0.03, 0.)
        self.elastics[0].init(-0.025, -0.005, 0.0003)
        self.elastics[1].init(0.01, 0., 0.0079, True)
        self.elastics[2].init(0., -0.015, -0.0079, False)
        self.elastics[3].init(0., 0.015, -0.0079, False)
        pos = np.array([[0.01, 0., 0.0079], [0., -0.015, -0.0079], [0., 0.015, -0.0079]])
        self.gripper.init(self, pos)

    def init_property(self):
        for i in range(self.cloth_cnt):
            self.pushup_property(self.pos, self.cloths[i].pos, self.cloths[i].offset)
            self.pushup_property(self.vel, self.cloths[i].vel, self.cloths[i].offset)
            self.cloths[i].gravity[None] = ti.Vector([0., 0., 0.])

        self.pushup_property(self.pos, self.elastics[0].F_x, self.elastics[0].offset)
        self.pushup_property(self.vel, self.elastics[0].F_v, self.elastics[0].offset)
        self.elastics[0].gravity[None] = self.gravity[None]

        for i in range(1, self.elastic_cnt):
            self.pushup_property(self.pos, self.elastics[i].F_x, self.elastics[i].offset)
            self.pushup_property(self.vel, self.elastics[i].F_v, self.elastics[i].offset)
            self.elastics[i].gravity[None] = ti.Vector([0., 0., 0.])
        self.init_mass()
        self.init_faces()
        self.build_f_vis()

    def reset_pos(self):
        self.cloths[0].init(-0.03, -0.03, 0.)
        self.elastics[0].init(-0.025, -0.005, 0.0003)
        self.elastics[1].init(0.01, 0., 0.0079, True)
        self.elastics[2].init(0., -0.015, -0.0079, False)
        self.elastics[3].init(0., 0.015, -0.0079, False)
        pos = np.array([[0.01, 0., 0.0079], [0., -0.015, -0.0079], [0., 0.015, -0.0079]])
        self.gripper.init(self, pos)

    def contact_analysis(self):
        self.nc[None] = 0
        for i in range(self.cloth_cnt):
            for j in range(self.cloth_cnt):
                if abs(i - j) == 1:  # TODO: contact relationship
                    self.contact_pair_analysis(self.cloths[i].body_idx, self.cloths[j].offset, self.cloths[j].offset + self.cloths[j].NV, 0.05)
                    self.contact_pair_analysis(self.cloths[j].body_idx, self.cloths[i].offset,
                                               self.cloths[i].offset + self.cloths[i].NV, 0.05)
        for i in range(self.cloth_cnt):
            for j in range(self.elastic_cnt):

                mu = self.mu_cloth_elastic[None]
                self.contact_pair_analysis(self.cloths[i].body_idx, self.elastics[j].offset,
                                           self.elastics[j].offset + self.elastics[j].n_verts, mu)
                self.contact_pair_analysis(self.elastics[j].body_idx, self.cloths[i].offset,
                                       self.cloths[i].offset + self.cloths[i].NV, mu)
        print("tot contact:", self.nc[None])

    @ti.kernel
    def set_frozen_kernel(self):
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
        for i in range(self.elastics[3].n_verts):
            if self.elastics[3].is_bottom(i) or self.elastics[3].is_inner_circle(i):
                xx = self.elastics[3].offset + i
                self.frozen[xx * 3] = 1
                self.frozen[xx * 3 + 1] = 1
                self.frozen[xx * 3 + 2] = 1

    @ti.kernel
    def compute_reward(self) -> ti.f64:
        ret = 0.0
        for i in range(self.elastics[0].n_verts):
            ret -= (self.elastics[0].F_x[i].x - self.elastics[0].F_ox[i].x + 0.025 + 0.012) ** 2
            ret -= (self.elastics[0].F_x[i].y - self.elastics[0].F_ox[i].y + 0.005 + 0.012) ** 2
            ret -= (self.elastics[0].F_x[i].z - self.elastics[0].F_ox[i].z - 0.0003) ** 2
        return ret

    def action(self, step, delta_pos, delta_rot):
        # self.gripper.print_plot()
        self.gripper.step_simple(delta_pos, delta_rot)
        # self.gripper.print_plot()
        self.gripper.update_bound(self)
        # a1 = self.elastics[1].check_determinant()
        # a2 = self.elastics[2].check_determinant()
        # a3 = self.elastics[3].check_determinant()
        # if not (a1 and a2 and a3):
        #     print("penetrate!!!!")
        # self.gripper.print_plot()
        self.pushup_property(self.pos, self.elastics[1].F_x, self.elastics[1].offset)
        self.pushup_property(self.pos, self.elastics[2].F_x, self.elastics[2].offset)
        self.pushup_property(self.pos, self.elastics[3].F_x, self.elastics[3].offset)

    def time_step(self, f_contact, frame_idx, force_stick=True):
        import time
        start_time = time.time()
        self.timestep_init()
        self.calc_vn()
        end_time = time.time()
        print("init time:", end_time - start_time)
        start_time = end_time

        f_contact(self)

        end_time = time.time()
        print("contact time:", end_time - start_time)
        start_time = end_time
        self.contact_analysis()
        end_time = time.time()
        print("contact analysis time:", end_time - start_time)
        start_time = end_time

        # self.save_constraints('../debug/const_%d.pt' % frame_idx)
        # print("contact_cnt", self.nc[None])
        # self.pos.copy_from(self.x_hat)
        # self.push_down_pos()
        iter = 0
        delta = 1e5
        temp_delta = 1e6
        PD = True
        while iter < 15:
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
        end_time = time.time()
        print("solving time:", end_time - start_time, "iter:", iter)
        start_time = end_time
        self.timestep_finish()