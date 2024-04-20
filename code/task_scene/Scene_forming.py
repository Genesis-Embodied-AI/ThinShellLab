import taichi as ti

from ..engine.model_fold_offset import Cloth
from ..engine.model_elastic_offset import Elastic
from ..engine.model_elastic_tactile import Elastic as tactile

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

    def __init__(self, cloth_size=0.06):
        super(Scene, self).__init__(cloth_size=cloth_size, enable_gripper=False)
        self.gravity[None] = ti.Vector([0., 0., 0.])
        print("tot nv", self.tot_NV)

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
        self.cloth_M = 7

        self.k_contact = 20000
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

        self.cloths.append(Cloth(self.cloth_N, self.dt, self.cloth_size, self.tot_NV, rho, 0, False, self.cloth_M))

        self.elastic_offset = (self.cloth_N + 1) * (self.cloth_M + 1)
        tmp_tot = self.elastic_offset

        self.elastics.append(Elastic(self.dt, self.elastic_size[0], tmp_tot, self.elastic_Nx, self.elastic_Ny, self.elastic_Nz))
        tmp_tot += self.elastic_Nx * self.elastic_Ny * self.elastic_Nz

        for i in range(1, self.elastic_cnt):
            self.elastics.append(tactile(self.dt, tmp_tot, self.elastic_size[i] / 0.03))
            tmp_tot += self.elastics[i].n_verts

        self.tot_NV = tmp_tot

    def init(self):
        half_curve_num = 3
        self.cloths[0].init_fold(-0.07, -0.02, 0.00035, half_curve_num)
        self.elastics[0].init(-0.035, -0.035, -0.00875)
        r = self.cloths[0].grid_len * (half_curve_num * 2 - 1) / 3.1415
        x = -0.07 + (7 + half_curve_num) / 16 * 0.1 - r * 0.86 + 0.01
        self.elastics[1].init(x, 0.0, 2 * r + 0.00785, True)
        print("pos:", x, 2 * r + 0.00785)
        pos = np.array([[x, 0.0, 2 * r + 0.00785]])
        self.gripper.init(self, pos)

    def reset_pos(self):
        half_curve_num = 3
        self.cloths[0].init_fold(-0.07, -0.02, 0.00035, half_curve_num)
        self.elastics[0].init(-0.035, -0.035, -0.00875)
        r = self.cloths[0].grid_len * (half_curve_num * 2 - 1) / 3.1415
        x = -0.07 + (7 + half_curve_num) / 16 * 0.1 - r * 0.86 + 0.01
        self.elastics[1].init(x, 0.0, 2 * r + 0.00785, True)
        pos = np.array([[x, 0.0, 2 * r + 0.00785]])
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
    def compute_reward(self, target_pos: ti.types.ndarray()) -> ti.f64:
        ret = 0.0
        for j, k in ti.ndrange(self.cloths[0].NV, 3):
            ret -= (self.cloths[0].pos[j][k] - target_pos[j, k]) ** 2
            # ret -= (analy_grad.pos_buffer[analy_grad.tot_timestep - 2, i + self.cloths[0].offset, j] - target_pos[
            #     i, j]) ** 2
        return ret

    def action(self, step, delta_pos, delta_rot):
        # self.gripper.print_plot()
        self.gripper.step_simple(delta_pos, delta_rot)
        # self.gripper.print_plot()
        self.gripper.update_bound(self)
        self.pushup_property(self.pos, self.elastics[1].F_x, self.elastics[1].offset)

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
        # for i in range(self.cloths[1].NV):
        #     colors[i + self.cloths[1].offset] = ti.Vector([0.5, 0.5, 0.5])
        for i in range(self.elastics[1].n_verts):
            colors[i + self.elastics[1].offset] = ti.Vector([0.22, 0.72, 0.52])  # Agent1 Color