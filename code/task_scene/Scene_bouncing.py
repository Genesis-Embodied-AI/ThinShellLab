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
from gripper import gripper

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
        self.gravity[None] = ti.Vector([0., 0., -9.8])
        self.cloths[0].k_angle[None] = 3.14

    def init_scene_parameters(self):
        self.dt = 2e-3
        self.h = self.dt
        self.cloth_cnt = 1
        self.elastic_cnt = 1
        self.elastic_size = [0.07]
        self.elastic_Nx = int(9)
        self.elastic_Ny = int(9)
        self.elastic_Nz = int(2)
        self.cloth_N = 15
        self.cloth_M = 15

        self.k_contact = 40000
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

        self.cloths.append(Cloth(self.cloth_N, self.dt, self.cloth_size, self.tot_NV, rho, 0))

        self.elastic_offset = (self.cloth_N + 1) * (self.cloth_M + 1)
        tmp_tot = self.elastic_offset

        self.elastics.append(Elastic(self.dt, self.elastic_size[0], tmp_tot, self.elastic_Nx, self.elastic_Ny, self.elastic_Nz))
        tmp_tot += self.elastic_Nx * self.elastic_Ny * self.elastic_Nz

        for i in range(1, self.elastic_cnt):
            self.elastics.append(tactile(self.dt, tmp_tot, self.elastic_size[i] / 0.03))
            tmp_tot += self.elastics[i].n_verts

        self.tot_NV = tmp_tot

    def init(self):
        self.cloths[0].init(-0.03, -0.03, 0.00039)
        self.elastics[0].init(-0.035, -0.035, -0.00875)
        self.cloths[0].init_ref_angle_bridge()

    def reset_pos(self):
        self.cloths[0].init(-0.03, -0.03, 0.0039)
        self.elastics[0].init(-0.035, -0.035, -0.00875)
        self.cloths[0].init_ref_angle_bridge()

    def contact_analysis(self):
        self.nc[None] = 0
        for i in range(self.cloth_cnt):
            for j in range(self.elastic_cnt):
                mu = self.mu_cloth_elastic[None]
                self.contact_pair_analysis(self.elastics[j].body_idx, self.cloths[i].offset,
                                       self.cloths[i].offset + self.cloths[i].NV, mu)

    @ti.kernel
    def set_frozen_kernel(self):
        for i in range(self.elastics[0].n_verts):
            xx = self.elastics[0].offset + i
            self.frozen[xx * 3] = 1
            self.frozen[xx * 3 + 1] = 1
            self.frozen[xx * 3 + 2] = 1

    @ti.kernel
    def compute_reward(self) -> ti.f64:
        ret = 0.0
        for i in range(self.cloths[0].NV):
            if ti.cast(i / (self.cloths[0].M + 1), ti.i32) == 5 or ti.cast(i / (self.cloths[0].M + 1), ti.i32) == 10:
                  ret += self.cloths[0].pos[i].z

        return ret

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