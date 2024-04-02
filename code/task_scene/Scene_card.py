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
import linalg

from sparse_solver import SparseMatrix
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

    def init_scene_parameters(self):
        self.dt = 5e-3
        self.h = self.dt
        self.cloth_cnt = 3
        self.elastic_cnt = 4
        self.elastic_size = [0.07, 0.015, 0.015, 0.015]
        self.elastic_Nx = int(9)
        self.elastic_Ny = int(9)
        self.elastic_Nz = int(2)
        self.cloth_N = 12
        self.cloth_M = 8

        self.k_contact = 20000
        self.eps_contact = 0.0004
        self.eps_v = 0.01
        self.max_n_constraints = 10000
        self.damping = 0.95

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
        for i in range(self.cloth_cnt):
            self.cloths.append(Cloth(self.cloth_N, self.dt, self.cloth_size, self.tot_NV, rho, i * (self.cloth_N + 1) * (self.cloth_M + 1), False, self.cloth_M))

        self.elastic_offset = (self.cloth_N + 1) * (self.cloth_M + 1) * self.cloth_cnt
        tmp_tot = self.elastic_offset

        self.elastics.append(Elastic(self.dt, self.elastic_size[0], tmp_tot, self.elastic_Nx, self.elastic_Ny, self.elastic_Nz))
        tmp_tot += self.elastic_Nx * self.elastic_Ny * self.elastic_Nz

        for i in range(1, self.elastic_cnt):
            self.elastics.append(tactile(self.dt, tmp_tot, self.elastic_size[i] / 0.03))
            tmp_tot += self.elastics[i].n_verts

        self.tot_NV = tmp_tot

    def init(self):
        self.cloths[0].init(-0.02, -0.02, 0.01)
        self.cloths[1].init(-0.02, -0.02, 0.0104)
        self.cloths[2].init(-0.02, -0.02, 0.0108)
        self.elastics[0].init(-0.025, -0.025, -0.00875)
        self.elastics[1].init(-0.0285, 0.0, 0.01, False)
        self.elastics[2].init(0.0485, 0.0, 0.01, False)
        self.elastics[3].init(0.01, 0.0, 0.0185, True)
        pos = np.array([[-0.0285, 0.0, 0.01], [0.0485, 0.0, 0.01], [0.01, 0.0, 0.0185]])
        self.gripper.init(self, pos)
        self.gripper.rot[0] = ti.Vector([ti.sqrt(2) * 0.5, 0, ti.sqrt(2) * 0.5, 0])
        self.gripper.rot[1] = ti.Vector([ti.sqrt(2) * 0.5, 0, -ti.sqrt(2) * 0.5, 0])
        self.gripper.get_rotmat()
        self.gripper.get_vert_pos()
        self.gripper.update_all(self)

    def reset_pos(self):
        self.cloths[0].init(-0.02, -0.02, 0.01)
        self.cloths[1].init(-0.02, -0.02, 0.0104)
        self.cloths[2].init(-0.02, -0.02, 0.0108)
        self.elastics[0].init(-0.025, -0.025, -0.00875)
        self.elastics[1].init(-0.0285, 0.0, 0.01, False)
        self.elastics[2].init(0.0485, 0.0, 0.01, False)
        self.elastics[3].init(0.01, 0.0, 0.0185, True)
        pos = np.array([[-0.0285, 0.0, 0.01], [0.0485, 0.0, 0.01], [0.01, 0.0, 0.0185]])
        self.gripper.init(self, pos)
        self.gripper.rot[0] = ti.Vector([ti.sqrt(2) * 0.5, 0, ti.sqrt(2) * 0.5, 0])
        self.gripper.rot[1] = ti.Vector([ti.sqrt(2) * 0.5, 0, -ti.sqrt(2) * 0.5, 0])
        self.gripper.get_rotmat()
        self.gripper.get_vert_pos()
        self.gripper.update_all(self)

    def contact_analysis(self):
        self.nc[None] = 0
        for i in range(self.cloth_cnt):
            for j in range(self.cloth_cnt):
                mu = 0.1
                if abs(i - j) == 1:  # TODO: contact relationship
                    self.contact_pair_analysis(self.cloths[i].body_idx, self.cloths[j].offset,
                                               self.cloths[j].offset + self.cloths[j].NV, mu)
                    self.contact_pair_analysis(self.cloths[j].body_idx, self.cloths[i].offset,
                                               self.cloths[i].offset + self.cloths[i].NV, mu)
        for i in range(self.cloth_cnt):
            for j in range(self.elastic_cnt):

                mu = self.mu_cloth_elastic[None]
                if i != 0:
                    mu *= 10
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
        for i in range(self.cloths[0].NV):
            ret -= self.cloths[0].pos[i].x
        return ret

    def action(self, step, delta_pos, delta_rot):
        # self.gripper.print_plot()
        self.gripper.step_simple(delta_pos, delta_rot)
        # self.gripper.print_plot()
        self.gripper.update_bound(self)
        a1 = self.elastics[1].check_determinant()
        a2 = self.elastics[2].check_determinant()
        if not (a1 and a2):
            print("penetrate!!!!")
        # self.gripper.print_plot()
        self.pushup_property(self.pos, self.elastics[1].F_x, self.elastics[1].offset)
        self.pushup_property(self.pos, self.elastics[2].F_x, self.elastics[2].offset)

    def timestep_finish(self):
        self.update_vel()
        self.push_down_vel()
        self.update_ref_angle()
        self.update_visual()

    def get_paramters_grad(self):
        for i in range(self.cloth_cnt):
            self.cloths[i].compute_deri_Kb()
            self.pushup_property(self.d_kb, self.cloths[i].d_kb, self.cloths[i].offset)

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
        for i in range(self.elastics[2].n_verts):
            colors[i + self.elastics[2].offset] = ti.Vector([1, 0.334, 0.52])  # Agent2 Color

