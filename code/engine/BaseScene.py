from .model_fold_offset import Cloth
from .model_elastic_tactile import Elastic as tactile
from .model_elastic_offset import Elastic

from typing import List
import taichi as ti
import torch
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from . import linalg
from . import contact_diff
from .gripper_tactile import gripper
from . import gripper_single
import random

from .sparse_solver import SparseMatrix

vec3 = ti.types.vector(3, ti.f64)
vec3i = ti.types.vector(3, ti.i32)

@dataclass
class Body:
    v_start: ti.i32
    v_end: ti.i32
    f_start: ti.i32
    f_end: ti.i32

@ti.data_oriented
class BaseScene:
    def __init__(self, cloth_size=0.1, dt=5e-3, enable_gripper=True, device="cuda:0"):
        self.dt = dt
        self.h = self.dt
        self.cloth_cnt = 2
        self.elastic_cnt = 3
        self.cloth_size = cloth_size
        self.elastic_size = [0.06, 0.015, 0.015]
        self.cloth_N = 31
        self.elastic_Nx = int(16)
        self.elastic_Ny = int(16)
        self.elastic_Nz = int(2)
        self.enable_gripper = enable_gripper

        self.k_contact = 1000
        self.eps_contact = 0.001
        self.eps_v = 0.01
        self.max_n_constraints = 100000
        self.damping = 1.0
        self.extra_obj = False
        self.effector_cnt = -1
        self.device = device

        self.init_scene_parameters()
        if self.effector_cnt == -1:
            self.effector_cnt = self.elastic_cnt
        self.gravity = ti.Vector.field(3, ti.f64, ())
        self.gravity[None] = ti.Vector([0, 0, -9.8])
        self.mu_cloth_elastic = ti.field(ti.f64, ())
        self.mu_cloth_elastic[None] = 1.0

        self.cloths = []
        self.elastics = []

        self.tot_NV = ((self.cloth_N + 1)**2) * self.cloth_cnt

        self.init_objects()

        #properties
        self.pos = ti.Vector.field(3, ti.f64, self.tot_NV) # x
        self.vel = ti.Vector.field(3, ti.f64, self.tot_NV) # velocity
        self.mass = ti.field(ti.f64, self.tot_NV)
        self.tot_NF = 0
        for i in range(self.cloth_cnt):
            self.cloths[i].offset_faces = self.tot_NF
            self.tot_NF += self.cloths[i].NF
        for i in range(self.elastic_cnt):
            self.elastics[i].offset_faces = self.tot_NF
            self.tot_NF += self.elastics[i].n_surfaces

        self.frozen = ti.field(ti.i32, shape=(self.tot_NV * 3,))
        self.faces = ti.Vector.field(3, ti.i32, shape=self.tot_NF)
        self.border_flag = ti.field(ti.i32, shape=(self.tot_NV,))
        self.ext_force = ti.Vector.field(3, ti.f64, self.tot_NV) # ext force on each point

        # visualize
        self.x32 = ti.Vector.field(3, ti.f32, shape=(self.tot_NV,))
        self.f_vis = ti.field(ti.i32, shape=(self.tot_NF * 3,))
        self.tmp_f = ti.field(ti.f64, shape=(3, self.tot_NV * 3))

        # contact
        self.body_list = []
        for i in range(self.cloth_cnt):
            self.body_list.append(
                Body(self.cloths[i].offset, self.cloths[i].offset + self.cloths[i].NV, self.cloths[i].offset_faces,
                     self.cloths[i].offset_faces + self.cloths[i].NF))
        for i in range(self.elastic_cnt):
            self.body_list.append(
                Body(self.elastics[i].offset, self.elastics[i].offset + self.elastics[i].n_verts, self.elastics[i].offset_faces,
                     self.elastics[i].offset_faces + self.elastics[i].n_surfaces))

        self.vn = ti.Vector.field(3, ti.f64, shape=(self.tot_NV,))
        self.proj_flag = ti.field(ti.i32)
        self.proj_dir = ti.field(ti.i32)
        self.proj_idx = ti.Vector.field(3, ti.i32)
        self.proj_w = ti.Vector.field(3, ti.f64)
        self.contact_force = ti.field(ti.f64)
        ti.root.dense(ti.ij, (len(self.body_list), self.tot_NV)).place(
            self.proj_flag, self.proj_dir, self.proj_idx, self.proj_w, self.contact_force
        )

        # constraints
        self.nc = ti.field(ti.i32, shape=())
        self.const_idx = ti.Vector.field(4, ti.i32)
        self.const_w = ti.Vector.field(3, ti.f64)
        self.const_n = ti.Vector.field(3, ti.f64)
        self.const_k = ti.field(ti.f64) # mu*T
        self.const_mu = ti.field(ti.f64)
        self.const_dx0 = ti.Vector.field(3, ti.f64)
        self.const_T = ti.Matrix.field(2, 3, ti.f64)
        ti.root.dense(ti.i, (self.max_n_constraints)).place(
            self.const_idx, self.const_w, self.const_k, self.const_dx0, self.const_T, self.const_mu, self.const_n
        )

        self.det_H = ti.field(ti.f64, shape=(self.max_n_constraints, 9, 9))
        self.det_G = ti.field(ti.f64, shape=(self.max_n_constraints, 9))
        self.cross_H = ti.field(ti.f64, shape=(self.max_n_constraints, 9, 9))
        self.cross_G = ti.field(ti.f64, shape=(self.max_n_constraints, 9))
        self.d_H = ti.field(ti.f64, shape=(self.max_n_constraints, 9, 9))
        self.d_G = ti.field(ti.f64, shape=(self.max_n_constraints, 9))
        self.force_T = ti.Vector.field(3, ti.f64, shape=(self.max_n_constraints,))
        self.grad_T = ti.field(ti.f64, shape=(self.max_n_constraints, 3))
        self.force_f = ti.Vector.field(3, ti.f64, shape=(self.max_n_constraints,))
        self.grad_f = ti.field(ti.f64, shape=(self.max_n_constraints, 3))
        self.proj_d = linalg.SPD_Projector(self.max_n_constraints, 9, 20)

        self.E = ti.field(ti.f64, shape=())
        self.F_b = ti.Vector.field(3, dtype=ti.f64, shape=self.tot_NV) # equals to self.F
        self.F = ti.field(ti.f64, shape=(self.tot_NV * 3,))
        self.H = SparseMatrix(self.tot_NV * 3, device=self.device)

        for i in range(self.cloth_cnt):
            self.cloths[i].body_idx = i
        for i in range(self.elastic_cnt):
            self.elastics[i].body_idx = i + self.cloth_cnt

        # # temp vars for simulation
        self.prev_pos = ti.Vector.field(3, ti.f64, self.tot_NV)  # x0
        self.x1 = ti.Vector.field(3, ti.f64, self.tot_NV)
        # self.x0 = ti.Vector.field(3, ti.f64, shape=(self.tot_NV,))  # x_t
        self.x_hat = ti.Vector.field(3, ti.f64, shape=(self.tot_NV,))  # x_t + h*v_t + h^2 * M^-1 * f_ext

        self.tmp_mat = ti.field(dtype=ti.f64, shape=(9, 9))
        self.lr = 1e-7

        self.tmp_z_not_frozen = ti.field(ti.f64, shape=(self.tot_NV * 3,))
        self.tmp_z_frozen = ti.field(ti.f64, shape=(self.tot_NV * 3,))
        self.counting_z_frozen = ti.field(ti.i32, shape=())
        self.counting_z_frozen[None] = False

        self.d_kb = ti.Vector.field(3, ti.f64, self.tot_NV)
        self.d_ka = ti.Vector.field(3, ti.f64, self.tot_NV)
        self.d_kl = ti.Vector.field(3, ti.f64, self.tot_NV)
        self.d_mu = ti.Vector.field(3, ti.f64, self.tot_NV)
        self.d_lam = ti.Vector.field(3, ti.f64, self.tot_NV)

        # gripper traj
        if enable_gripper:
            self.gripper = gripper(self.dt, self.elastics[1].n_verts, self.elastics[1].frozen_cnt,
                                   self.elastics[1].surf_point, int((self.effector_cnt - 1) // 2))
        elif self.elastic_cnt > 1:
            self.gripper = gripper_single.gripper(self.dt, self.elastics[1].n_verts, self.elastics[1].frozen_cnt,
                                                  self.elastics[1].surf_point, self.effector_cnt - 1)
        # self.gripper_pos = ti.Vector.field(3, ti.f64, self.tot_timestep)
        # self.gripper_rot = ti.Vector.field(3, ti.f64, self.tot_timestep)
        # self.gripper_dist = ti.field(ti.f64, self.tot_timestep)

        self.action_dim = 3 * (self.effector_cnt - 1)
        if not enable_gripper:
            self.action_dim = int(6 * (self.effector_cnt - 1))

        if self.effector_cnt - 1 > 0:
            self.tot_force = ti.field(ti.f64, shape=(self.effector_cnt - 1, 3))

            self.n_obs_cloth = 4
            self.n_obs_elastic = 16

            self.n_sample_cloth = self.cloths[0].N // 4
            self.m_sample_cloth = self.cloths[0].M // 4

            self.obs_dim = (self.n_obs_cloth * self.n_obs_cloth * self.cloth_cnt + self.n_obs_elastic * self.elastic_cnt) * 6 + 7 * self.gripper.n_part
            self.observation = ti.field(dtype=ti.f64, shape=self.obs_dim)

            self.delta_pos = ti.Vector.field(3, dtype=ti.f64, shape=self.gripper.n_part)
            self.delta_rot = ti.Vector.field(3, dtype=ti.f64, shape=self.gripper.n_part)

    def init_objects(self):
        rho = 4e1
        for i in range(self.cloth_cnt):
            self.cloths.append(Cloth(self.cloth_N, self.dt, self.cloth_size, self.tot_NV, rho, i * ((self.cloth_N + 1)**2)))

        self.elastic_offset = ((self.cloth_N + 1)**2) * self.cloth_cnt
        tmp_tot = self.elastic_offset

        self.elastics.append(Elastic(self.dt, self.elastic_size[0], tmp_tot, self.elastic_Nx, self.elastic_Ny, self.elastic_Nz))
        tmp_tot += self.elastic_Nx * self.elastic_Ny * self.elastic_Nz

        for i in range(1, self.elastic_cnt):
            self.elastics.append(tactile(self.dt, tmp_tot, self.elastic_size[i] / 0.03))
            tmp_tot += self.elastics[i].n_verts

        self.tot_NV = tmp_tot

    def init_scene_parameters(self):
        self.dt = 5e-3
        self.h = self.dt
        self.cloth_cnt = 1
        self.elastic_cnt = 3
        self.elastic_size = [0.06, 0.015, 0.015]
        self.cloth_N = 15

        self.k_contact = 500
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


    def init(self):
        self.cloths[0].init(-0.03, -0.03, 0.000399)
        # self.cloths[1].init(-0.03, -0.03, 0.000)
        self.elastics[0].init(-0.03, -0.03, -0.004)
        self.elastics[1].init(-0.02, 0., 0.0105, True)
        # self.elastics[1].plot_normal()
        self.elastics[2].init(-0.02, 0., -0.0105, False)
        self.gripper.init(self.elastics[1], self.elastics[2], -0.02, 0., 0.0105, -0.02, 0., -0.0105)

    def reset_pos(self):
        self.cloths[0].init(-0.03, -0.03, 0.000399)
        # self.cloths[1].init(-0.03, -0.03, 0.000)
        self.elastics[0].init(-0.03, -0.03, -0.004)
        self.elastics[1].init(-0.02, 0., 0.0105, True)
        self.elastics[2].init(-0.02, 0., -0.0105, False)
        self.gripper.init(self.elastics[1], self.elastics[2], -0.02, 0., 0.0105, -0.02, 0., -0.0105)

    def reset(self):
        self.reset_pos()
        # self.cloths[0].init_pos_offset(-0.05, -0.05, 0.0)
        # self.cloths[0].ref_angle.fill(0)
        # self.elastics[0].init_pos(-0.03, -0.03, -0.016)
        # self.elastics[1].init_pos(-0.05, -0.007, 0.005)
        # self.elastics[2].init_pos(-0.05, -0.007, -0.00875)
        self.set_ext_force()
        self.set_frozen()
        for i in range(self.cloth_cnt):
            self.pushup_property(self.pos, self.cloths[i].pos, self.cloths[i].offset)
            self.pushup_property(self.vel, self.cloths[i].vel, self.cloths[i].offset)
        for i in range(self.elastic_cnt):
            self.pushup_property(self.pos, self.elastics[i].F_x, self.elastics[i].offset)
            self.pushup_property(self.vel, self.elastics[i].F_v, self.elastics[i].offset)
        self.update_visual()
        self.proj_flag.fill(0)

    @ti.kernel
    def copy_pos_kernel(self, target_pos: ti.template(), step: int):
        for i in self.pos:
            self.pos[i][0] = target_pos[step, i, 0]
            self.pos[i][1] = target_pos[step, i, 1]
            self.pos[i][2] = target_pos[step, i, 2]

    @ti.kernel
    def copy_prev_pos_kernel(self, target_pos: ti.template(), step: int):
        for i in self.prev_pos:
            self.prev_pos[i][0] = target_pos[step - 1, i, 0]
            self.prev_pos[i][1] = target_pos[step - 1, i, 1]
            self.prev_pos[i][2] = target_pos[step - 1, i, 2]

    def copy_pos_and_refangle(self, analy_grad, step):
        self.copy_pos_kernel(analy_grad.pos_buffer, step)
        self.copy_prev_pos_kernel(analy_grad.pos_buffer, step)
        self.push_down_pos()
        for i in range(self.cloth_cnt):
            self.pushdown_property(self.cloths[i].prev_pos, self.prev_pos, self.cloths[i].offset)
            self.cloths[i].copy_refangle(analy_grad.ref_angle_buffer, step - 1, i)
        for i in range(self.elastic_cnt):
            self.pushdown_property(self.elastics[i].F_x_prev, self.prev_pos, self.elastics[i].offset)

    def copy_refangle(self, analy_grad, step):
        for i in range(self.cloth_cnt):
            self.cloths[i].copy_refangle(analy_grad.ref_angle_buffer, step, i)

    def copy_pos(self, target_pos, step):
        self.copy_pos_kernel(target_pos, step)
        self.copy_prev_pos_kernel(target_pos, step)
        self.push_down_pos()
        for i in range(self.cloth_cnt):
            self.pushdown_property(self.cloths[i].prev_pos, self.prev_pos, self.cloths[i].offset)
        for i in range(self.elastic_cnt):
            self.pushdown_property(self.elastics[i].F_x_prev, self.prev_pos, self.elastics[i].offset)
        # copy ref_angle if needed

    def copy_pos_only(self, target_pos, step):
        self.copy_pos_kernel(target_pos, step)
        self.copy_prev_pos_kernel(target_pos, step + 1)
        self.push_down_pos()
        for i in range(self.cloth_cnt):
            self.pushdown_property(self.cloths[i].prev_pos, self.prev_pos, self.cloths[i].offset)
        for i in range(self.elastic_cnt):
            self.pushdown_property(self.elastics[i].F_x_prev, self.prev_pos, self.elastics[i].offset)

    @ti.kernel
    def pushup_property(self, dest: ti.template(), src: ti.template(), offset: ti.i32):
        for i in src:
            dest[i + offset] = src[i]

    @ti.kernel
    def pushup_property_add(self, dest: ti.template(), src: ti.template(), offset: ti.i32):
        for i in src:
            dest[i + offset] += src[i]

    @ti.kernel
    def pushdown_property(self, dest: ti.template(), src: ti.template(), offset: ti.i32):
        for i in dest:
            dest[i] = src[i + offset]

    @ti.kernel
    def init_mass_kernel_cloth(self, cloth: ti.template()):
        for i in range(cloth.NV):
            self.mass[cloth.offset + i] = cloth.mass

    @ti.kernel
    def init_mass_kernel_elastic(self, elastic: ti.template()):
        for i in range(elastic.n_verts):
            self.mass[elastic.offset + i] = elastic.F_m[i]

    def init_mass(self):
        for i in range(self.cloth_cnt):
            self.init_mass_kernel_cloth(self.cloths[i])
        for i in range(self.elastic_cnt):
            self.init_mass_kernel_elastic(self.elastics[i])

    @ti.kernel
    def init_faces_kernel(self, obj: ti.template(), NF: int):
        for i in range(NF):
            self.faces[obj.offset_faces + i][0] = obj.f2v[i][0] + obj.offset
            self.faces[obj.offset_faces + i][1] = obj.f2v[i][1] + obj.offset
            self.faces[obj.offset_faces + i][2] = obj.f2v[i][2] + obj.offset

    def init_faces(self):
        for i in range(self.cloth_cnt):
            self.init_faces_kernel(self.cloths[i], self.cloths[i].NF)
        for i in range(self.elastic_cnt):
            self.init_faces_kernel(self.elastics[i], self.elastics[i].n_surfaces)

    def init_property(self):
        for i in range(self.cloth_cnt):
            self.pushup_property(self.pos, self.cloths[i].pos, self.cloths[i].offset)
            self.pushup_property(self.vel, self.cloths[i].vel, self.cloths[i].offset)
            self.cloths[i].gravity[None] = self.gravity[None]

        self.pushup_property(self.pos, self.elastics[0].F_x, self.elastics[0].offset)
        self.pushup_property(self.vel, self.elastics[0].F_v, self.elastics[0].offset)
        self.elastics[0].gravity[None] = self.gravity[None]

        for i in range(1, self.effector_cnt):
            self.pushup_property(self.pos, self.elastics[i].F_x, self.elastics[i].offset)
            self.pushup_property(self.vel, self.elastics[i].F_v, self.elastics[i].offset)
            self.elastics[i].gravity[None] = ti.Vector([0., 0., 0.])

        for i in range(self.effector_cnt, self.elastic_cnt):
            self.pushup_property(self.pos, self.elastics[i].F_x, self.elastics[i].offset)
            self.pushup_property(self.vel, self.elastics[i].F_v, self.elastics[i].offset)
            self.elastics[i].gravity[None] = self.gravity[None]

        self.init_mass()
        self.init_faces()
        self.build_f_vis()

    @ti.kernel
    def build_f_vis(self):
        for i in range(self.tot_NF):
            self.f_vis[i * 3 + 0] = self.faces[i][0]
            self.f_vis[i * 3 + 1] = self.faces[i][1]
            self.f_vis[i * 3 + 2] = self.faces[i][2]

    @ti.func
    def add_F(self, i, v, idx=-1):
        if idx >= 0:
            self.tmp_f[idx, i] -= v
        if not self.frozen[i]:
            self.F[i] += v

    @ti.func
    def add_H(self, i, j, v):
        if not self.frozen[i] and not self.frozen[j]:
            self.H.add(i, j, v)
        elif self.counting_z_frozen[None] and self.frozen[j] and (not self.frozen[i]):
            self.tmp_z_frozen[j] -= v * self.tmp_z_not_frozen[i]
            # here df/dx is negative to the hessian matrix

    @ti.kernel
    def check_energy_nan(self) -> ti.i32:
        ret = False
        if ti.math.isnan(self.E[None]):
            ret = True
        return ret

    @ti.kernel
    def check_energy_pos(self) -> ti.i32:
        ret = False
        for i in range(self.tot_NV):
            if ti.math.isnan(self.pos[i][0]):
                ret = True
            if ti.math.isnan(self.pos[i][1]):
                ret = True
            if ti.math.isnan(self.pos[i][2]):
                ret = True

        return ret

    def compute_energy(self):
        self.E[None] = 0.0
        # if self.check_energy_pos():
        #     print("already nan!!!")
        self.contact_energy(False, False)
        # print(f"contact", self.E[None])
        # if self.check_energy_nan():
        #     print("nan energy in contact!!!")
        # if self.E[None] > 0:
        #     print("contact exist!")
        # print("contact energy:", self.E[None])
        for i in range(self.cloth_cnt):
            self.cloths[i].compute_normal_dir()
            self.cloths[i].compute_energy()
            self.E[None] += self.cloths[i].U[None]
            # print(f"cloth {i} energy", self.cloths[i].U[None])
        # if self.check_energy_nan():
        #     print("nan energy before cloths!!!")
        for i in range(self.elastic_cnt):
            self.elastics[i].compute_energy()
            self.E[None] += self.elastics[i].U[None]
        #     print(f"elastic {i} energy", self.elastics[i].U[None])
        # if self.check_energy_nan():
        #     exit(0)
        #     print("nan energy before elastic!!!")

    @ti.func
    def f0(self, x):
        ret = 0.0
        if x > self.eps_v * self.h:
            ret = x
        else :
            ret = -x/(3.0 * self.eps_v**2) * x / (self.h**2) * x + x/(self.eps_v*self.h)*x + self.eps_v * self.h / 3.0
        return ret

    @ti.func
    def f1(self, x): # f1(x) / x
        ret = 0.0
        if x > self.eps_v * self.h:
            ret = 1.0 / x
        else:
            ret = -x / (self.eps_v*self.h)**2 + 2.0 / (self.eps_v*self.h)
        return ret

    @ti.func
    def f2(self, x): # (f1'(x)x-f1(x)) / x**2
        ret = 0.0
        if x > self.eps_v * self.h:
            ret = -1.0 / x**2
        else:
            ret = -1.0 / (self.eps_v*self.h)**2
        return ret

    def project_to_psd_22(self, matrix):
        eigenvalues, eigenvectors = ti.sym_eig(matrix)
        corrected_eigenvalues = ti.max(eigenvalues, 0)
        projected_matrix = eigenvectors @ ti.Matrix([[corrected_eigenvalues[0], 0],
                                                     [0, corrected_eigenvalues[1]]]) @ eigenvectors.transpose()
        return projected_matrix

    @ti.kernel
    def contact_energy(self, diff:ti.i32, spd: ti.i32):
        has_nan = False
        for i in range(self.nc[None]):
            idx = self.const_idx[i]
            w = self.const_w[i]

            p1 = self.pos[idx[1]] - self.pos[idx[0]]
            p2 = self.pos[idx[2]] - self.pos[idx[0]]
            p = self.pos[idx[3]] - self.pos[idx[0]]

            d = contact_diff.det(p1, p2, p, diff, self.det_H, i, self.det_G, i)
            c = contact_diff.cross(p1, p2, diff, self.cross_H, i, self.cross_G, i)

            if d / c < self.eps_contact:
                if diff:
                    for j in range(9):
                        self.d_G[i, j] = self.det_G[i, j] / c - d*self.cross_G[i, j] / c**2
                    for j in range(9):
                        for k in range(9):
                            self.d_H[i, j, k] = self.det_H[i, j, k] / c \
                                                - self.det_G[i, j] * self.cross_G[i, k] / c**2 \
                                                - self.det_G[i, k] * self.cross_G[i, j] / c**2 \
                                                - d*self.cross_H[i, j, k] / c**2 \
                                                + 2*d*self.cross_G[i, j]*self.cross_G[i, k] / c**3
                d /= c
                e = 0.5*self.k_contact * (d-self.eps_contact)**2
                pe_pd = self.k_contact * (d-self.eps_contact)
                if diff:
                    for j in range(9):
                        for k in range(9):
                            self.d_H[i, j, k] = self.k_contact*self.d_G[i, j]*self.d_G[i, k] \
                                                + pe_pd*self.d_H[i, j, k]
                    for j in range(9):
                        self.d_G[i, j] *= pe_pd

                    if spd:
                        self.proj_d.project(self.d_H, i, 9)

                    self.force_T[i] = vec3(0.0, 0.0, 0.0)
                    for j, k in ti.ndrange(3, 3):
                        g = self.d_G[i, k*3+j]
                        # if ti.math.isnan(g):
                        #     print("!?!?!?")
                        self.add_F(idx[k+1]*3+j, g, 0)
                        self.add_F(idx[0]*3+j,  -g, 0)
                        for j2, k2 in ti.static(ti.ndrange(3, 3)):
                            h = self.d_H[i, k*3+j, k2*3+j2]
                            # if ti.math.isnan(h):
                            #     print("!!!!!!")
                            self.add_H(idx[k+1]*3+j, idx[k2+1]*3+j2,  h)
                            self.add_H(idx[k+1]*3+j, idx[0]*3+j2,    -h)
                            self.add_H(idx[0]*3+j,   idx[k2+1]*3+j2, -h)
                            self.add_H(idx[0]*3+j,   idx[0]*3+j2,     h)
                        self.force_T[i][j] += g
                else:
                    self.E[None] += e

        max_k = -100.0

        # friction
        for i in range(self.nc[None]):
            idx = self.const_idx[i]
            w = self.const_w[i]
            # k = 0
            k = self.const_k[i]
            if k > max_k:
                max_k = k

            x_c = self.pos[idx[0]]*w[0] + self.pos[idx[1]]*w[1] + self.pos[idx[2]]*w[2]
            dx = self.pos[idx[3]] - x_c - self.const_dx0[i]
            u = self.const_T[i] @ dx
            r = u.norm()

            if diff:
                g = u * k * self.f1(r)
                g1 = g @ self.const_T[i]
                h = self.f1(r) * ti.Matrix.identity(ti.f64, 2)
                if r>1e-9:
                    h += self.f2(r)*(u/r).outer_product(u)
                if spd:
                    h = linalg.SPD_project_2d(h)
                    check_pd = h[0, 0]>-1e-9 and h.determinant()>-1e-9
                # if not h[0, 0]>-1e-6:
                #     print('assert 1:', h[0, 0])
                # if not h.determinant()>-1e-6:
                #     print('assert 2:', h.determinant())
                h1 = k * self.const_T[i].transpose() @ h @ self.const_T[i]
                # eigenvalues, eigenvectors = ti.sym_eig(h1)
                # if eigenvalues[0] < -0.00001 or eigenvalues[1] < -0.00001 or eigenvalues[2] < -0.00001:
                #     print("wtf ???", eigenvalues[0], eigenvalues[1], eigenvalues[2])

                self.force_f[i][0] = g1[0]
                self.force_f[i][1] = g1[1]
                self.force_f[i][2] = g1[2]

                w1 = ti.Vector([-w[0], -w[1], -w[2], 1], ti.f64)
                for i1 in range(4):
                    for j1 in ti.static(range(3)):
                        # if ti.math.isnan(g1[j1]):
                        #     print("friction nan")
                        self.add_F(idx[i1]*3+j1, w1[i1]*g1[j1], 0)
                for i1, i2 in ti.ndrange(4, 4):
                    for j1, j2 in ti.static(ti.ndrange(3, 3)):
                        if ti.math.isnan(h1[j1, j2]):
                            has_nan = True
                        self.add_H(idx[i1]*3+j1, idx[i2]*3+j2, w1[i1]*w1[i2]*h1[j1, j2])
            else:
                self.E[None] += k * self.f0(r)

        if has_nan:
            print("friction hessian nan!!!!")

    @ti.kernel
    def contact_differential(self, grad_x:ti.template()):
        for i in range(self.nc[None]):
            idx = self.const_idx[i]
            w = self.const_w[i]

            p1 = self.pos[idx[1]] - self.pos[idx[0]]
            p2 = self.pos[idx[2]] - self.pos[idx[0]]
            p = self.pos[idx[3]] - self.pos[idx[0]]

            d = contact_diff.det(p1, p2, p, True, self.det_H, i, self.det_G, i)
            c = contact_diff.cross(p1, p2, True, self.cross_H, i, self.cross_G, i)

            if d / c < self.eps_contact:
                for j in range(9):
                    self.d_G[i, j] = self.det_G[i, j] / c - d*self.cross_G[i, j] / c**2
                for j in range(9):
                    for k in range(9):
                        self.d_H[i, j, k] = self.det_H[i, j, k] / c \
                                            - self.det_G[i, j] * self.cross_G[i, k] / c**2 \
                                            - self.det_G[i, k] * self.cross_G[i, j] / c**2 \
                                            - d*self.cross_H[i, j, k] / c**2 \
                                            + 2*d*self.cross_G[i, j]*self.cross_G[i, k] / c**3
                d /= c
                e = 0.5*self.k_contact * (d-self.eps_contact)**2
                pe_pd = self.k_contact * (d-self.eps_contact)
                for j in range(9):
                    for k in range(9):
                        self.d_H[i, j, k] = self.k_contact*self.d_G[i, j]*self.d_G[i, k] \
                                            + pe_pd*self.d_H[i, j, k]
                for j in range(9):
                    self.d_G[i, j] *= pe_pd

                for j, k in ti.ndrange(3, 3):
                    g = self.d_G[i, k*3+j]
                    for j2, k2 in ti.static(ti.ndrange(3, 3)):
                        h = self.d_H[i, k*3+j, k2*3+j2]
                        grad_x[idx[k+1]*3+j] -= h * self.grad_T[i, j2]
                        grad_x[idx[0]*3+j] += h * self.grad_T[i, j2]

        max_k = -100.0

        # friction
        for i in range(self.nc[None]):
            idx = self.const_idx[i]
            w = self.const_w[i]
            # k = 0
            k = self.const_k[i]
            if k > max_k:
                max_k = k

            x_c = self.pos[idx[0]]*w[0] + self.pos[idx[1]]*w[1] + self.pos[idx[2]]*w[2]
            dx = self.pos[idx[3]] - x_c - self.const_dx0[i]
            u = self.const_T[i] @ dx
            r = u.norm()

            g = u * k * self.f1(r)
            g1 = g @ self.const_T[i]
            h = self.f1(r) * ti.Matrix.identity(ti.f64, 2)
            if r>1e-9:
                h += self.f2(r)*(u/r).outer_product(u)
            h1 = k * self.const_T[i].transpose() @ h @ self.const_T[i]

            w1 = ti.Vector([-w[0], -w[1], -w[2], 1], ti.f64)
            for i1 in range(4):
                for j1, j2 in ti.static(ti.ndrange(3, 3)):
                    grad_x[idx[i1]*3+j1] += self.grad_f[i, j2] * w1[i1] * h1[j1, j2]

    @ti.func
    def bel(self, i):
        ret = 0
        if i < self.cloths[0].NV:
            # print("cloth", end=" ")
            ret = 1
        if i >= self.elastics[0].offset and i < self.elastics[0].offset + self.elastics[0].n_verts:
            # print("table", end=" ")
            ret = 2
        if i >= self.elastics[1].offset and i < self.elastics[1].offset + self.elastics[1].n_verts:
            # print("top gripper", end=" ")
            ret = 3
        return ret

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
    def static_friction_loss(self, analy_grad: ti.template(), step: int):

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

            if r > self.dt * self.eps_v * 0.9:
                # pressure = k / self.const_mu[i]
                # g = u * k * self.f1(r)
                # g1 = g @ self.const_T[i]
                # n_c = self.const_n[i]
                # w1 = ti.Vector([w[0], w[1], w[2], -1], ti.f64)
                # for i1 in range(4):
                #     for j1 in ti.static(range(3)):
                #         dfdp = w1[i1] * g1[j1] / pressure
                #         # if ti.abs(zT) > 1.0 / self.k_contact and (self.bel(idx[3]) == 3 or self.bel(idx[0]) == 3):
                #         #     print("zT:", zT, "dfdp:", dfdp, end=" ")
                #         #     print("obj:", self.bel(idx[i1]), "dim:", j1)
                #         for i2, j2 in ti.static(ti.ndrange(4, 3)):
                #             analy_grad.pos_grad[step - 1, idx[i2], j2] += -dfdp * w1[i2] * n_c[j2] * self.k_contact * analy_grad.f_loss_ratio
                #             # ret = self.bel(idx[i2])
                #             # if ret == 3 and j2 == 2 and ti.abs(zT) > 1.0 / self.k_contact:
                #             #     print("pos grad", zT * dfdp * w1[i2] * n_c[j2] * self.k_contact)

                w1 = ti.Vector([-w[0], -w[1], -w[2], 1], ti.f64)
                u_3d = ti.Vector([u[0] * self.const_T[i][0, 0] + u[1] * self.const_T[i][1, 0],
                                  u[0] * self.const_T[i][0, 1] + u[1] * self.const_T[i][1, 1],
                                  u[0] * self.const_T[i][0, 2] + u[1] * self.const_T[i][1, 2]])
                for i1 in range(4):
                    for j1 in ti.static(range(3)):

                        analy_grad.pos_grad[step, idx[i1], j1] += u_3d[j1] * w1[i1] * analy_grad.f_loss_ratio * k

    @ti.kernel
    def contact_pair_analysis(self, b_idx: int, v_start: ti.i32, v_end: ti.i32, mu: ti.f64):
        for i in range(v_start, v_end):
            if self.proj_flag[b_idx, i]:
                idx = self.proj_idx[b_idx, i]
                w = self.proj_w[b_idx, i]
                x_c = self.pos[idx[0]]*w[0] + self.pos[idx[1]]*w[1] + self.pos[idx[2]]*w[2]
                x0_c = self.prev_pos[idx[0]]*w[0] + self.prev_pos[idx[1]]*w[1] + self.prev_pos[idx[2]]*w[2]
                n_c = (self.pos[idx[1]]-self.pos[idx[0]]).cross(self.pos[idx[2]]-self.pos[idx[0]]).normalized()
                if self.proj_dir[b_idx, i] == 0:
                    n_c = -n_c
                    idx = (idx[0], idx[2], idx[1])
                    w = (w[0], w[2], w[1])
                if (self.pos[i] - x_c).dot(n_c) < self.eps_contact:
                    # if b_idx == 2:
                    #     print("?!?!?!?!?!?")
                        # for ii in range(3):
                        #     print(f"[{self.pos[idx[ii]][0]} {self.pos[idx[ii]][1]} {self.pos[idx[ii]][2]}]", end=",")
                        # print("")
                    c_idx = ti.atomic_add(self.nc[None], 1)
                    self.contact_force[b_idx, i] = self.k_contact*((self.pos[i]-x_c).dot(n_c)-self.eps_contact)
                    self.const_idx[c_idx] = (idx[0], idx[1], idx[2], i)
                    self.const_w[c_idx] = w
                    self.const_k[c_idx] = -mu*self.contact_force[b_idx, i]
                    self.const_mu[c_idx] = mu
                    self.const_dx0[c_idx] = self.prev_pos[i] - x0_c
                    t1 = vec3(0)
                    if ti.abs(n_c[0]) < 0.5:
                        t1 = (n_c[0], n_c[2], -n_c[1])
                    else:
                        t1 = (n_c[1], -n_c[0], n_c[2])
                    t2 = n_c.cross(t1)
                    t1 = n_c.cross(t2)
                    self.const_T[c_idx] = ti.Matrix([[t1[0], t1[1], t1[2]], [t2[0], t2[1], t2[2]]], ti.f64)
                    self.const_n[c_idx] = n_c
                else:
                    self.contact_force[b_idx, i] = 0
            else:
                self.contact_force[b_idx, i] = 0

    def contact_analysis(self):
        self.nc[None] = 0
        for i in range(self.cloth_cnt):
            for j in range(self.cloth_cnt):
                if abs(i - j) == 1:  # TODO: contact relationship
                    self.contact_pair_analysis(self.cloths[i].body_idx, self.cloths[j].offset,
                                               self.cloths[j].offset + self.cloths[j].NV, 0.1)
                    self.contact_pair_analysis(self.cloths[j].body_idx, self.cloths[i].offset,
                                               self.cloths[i].offset + self.cloths[i].NV, 0.1)
        for i in range(self.cloth_cnt):
            for j in range(self.elastic_cnt):
                mu = self.mu_cloth_elastic[None]
                if j == 0:
                    mu = 0.2
                self.contact_pair_analysis(self.cloths[i].body_idx, self.elastics[j].offset,
                                           self.elastics[j].offset + self.elastics[j].n_verts, mu)
                self.contact_pair_analysis(self.elastics[j].body_idx, self.cloths[i].offset,
                                               self.cloths[i].offset + self.cloths[i].NV, mu)

    @ti.kernel
    def calc_vn(self):
        for i in range(self.tot_NV):
            self.vn[i] = vec3(0)
        for i in range(self.tot_NF):
            v1 = self.pos[self.faces[i][0]]
            v2 = self.pos[self.faces[i][1]]
            v3 = self.pos[self.faces[i][2]]
            n = (v2 - v1).cross(v3 - v1)
            self.vn[self.faces[i][0]] += n
            self.vn[self.faces[i][1]] += n
            self.vn[self.faces[i][2]] += n
        for i in range(self.tot_NV):
            self.vn[i] = self.vn[i].normalized()

    def push_down_pos(self):
        for i in range(self.cloth_cnt):
            self.pushdown_property(self.cloths[i].pos, self.pos, self.cloths[i].offset)
        for i in range(self.elastic_cnt):
            self.pushdown_property(self.elastics[i].F_x, self.pos, self.elastics[i].offset)

    def push_down_vel(self):
        for i in range(self.cloth_cnt):
            self.pushdown_property(self.cloths[i].vel, self.vel, self.cloths[i].offset)
        for i in range(self.elastic_cnt):
            self.pushdown_property(self.elastics[i].F_v, self.vel, self.elastics[i].offset)

    def update_ref_angle(self):
        for i in range(self.cloth_cnt):
            self.cloths[i].update_ref_angle()

    @ti.kernel
    def update_vel(self):
        for i in range(self.tot_NV):
            for j in range(3):
                self.vel[i][j] = (self.pos[i][j] - self.prev_pos[i][j]) * self.damping / self.dt

    @ti.kernel
    def flatten(self, dest: ti.template(), src: ti.template()):
        for i in range(self.tot_NV):
            for j in range(3):
                dest[3 * i + j] = src[i][j]

    def check_differential(self):
        def energy_and_force():
            self.push_down_pos()
            for i in range(self.cloth_cnt):
                self.cloths[i].compute_normal_dir()
                self.cloths[i].prepare_bending()
                self.cloths[i].compute_energy_bending()
                self.E[None] += self.cloths[i].U[None]
                print(f"cloth {i} energy", self.cloths[i].U[None])
            # for i in range(self.elastic_cnt):
            #     self.elastics[i].compute_energy()
            #     self.E[None] += self.elastics[i].U[None]
                # print(f"elastic {i} energy", self.elastics[i].U[None])
            # self.contact_energy(False)

            for i in range(self.cloth_cnt):
                # self.cloths[i].compute_normal_dir()
                # self.cloths[i].prepare_bending()
                self.cloths[i].compute_residual_bending()
                self.pushup_property_add(self.F_b, self.cloths[i].F_b, self.cloths[i].offset)
            # for i in range(self.elastic_cnt):
            #     self.elastics[i].get_force()
            #     self.elastics[i].compute_residual()
            #     self.pushup_property_add(self.F_b, self.elastics[i].F_b, self.elastics[i].offset)
            self.flatten(self.F, self.F_b)
            self.apply_frozen()
            # self.contact_energy(True)

        def hessian():
            # for i in range(self.elastic_cnt):
            #     self.elastics[i].compute_Hessian(self)
            # print("hessian elastic")
            # self.H.check_PD()
            # for i in range(self.cloth_cnt):
            #     self.cloths[i].compute_Hessian_ma(self)
            # # # print("hessian with me")
            # # # self.H.check_PD()
            # for i in range(self.cloth_cnt):
            #     self.cloths[i].compute_Hessian_ma(self)
            # # # # # print("hessian with ma")
            # # # # # self.H.check_PD()
            for i in range(self.cloth_cnt):
                self.cloths[i].compute_Hessian_bending(self)
            return

        # import pdb; pdb.set_trace()
        self.newton_step_init()
        energy_and_force()
        hessian()
        E0 = self.E[None]
        F0 = self.F.to_torch(self.device)
        H0 = self.H.value.to_torch(self.device)
        dx = torch.randn_like(F0)
        # dx = torch.zeros_like(F0)
        # while True:
        #     idx = random.randint(0, self.tot_NV*3-1)
        #     if not self.frozen[idx]:
        #         dx[idx] = random.random()
        #         break
        for i in range(self.tot_NV*3):
            if self.frozen[i]:
                dx[i] = 0
        dx *= 0.000001 / (dx**2).sum()**0.5
        for i in range(self.tot_NV):
            self.pos[i][0] += dx[i*3+0]
            self.pos[i][1] += dx[i*3+1]
            self.pos[i][2] += dx[i*3+2]


        self.newton_step_init()
        energy_and_force()
        hessian()
        H1 = self.H.value.to_torch('cuda:0')
        E1 = self.E[None]
        F1 = self.F.to_torch('cuda:0')
        print('Gradient Check:', E1-E0, float(F0@dx), float(F1@dx))

        dF = torch.mm(H0, dx.view(-1, 1)).view(-1)
        dF_1 = torch.mm(H1, dx.view(-1, 1)).view(-1)
        dF_gt = F1 - F0

        print("hessian dF:", dF.norm().item(), dF_1.norm().item())
        print("gt dF:", dF_gt.norm().item())
        print("distance 0:", torch.abs(dF - dF_gt).norm().item())
        print("distance 1:", torch.abs(dF_1 - dF_gt).norm().item())
        # cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
        # print("angle", cos(dF.view(-1) * 1e8, dF_1.view(-1) * 1e8))

        for i in range(self.tot_NV):
            self.pos[i][0] -= dx[i*3+0]
            self.pos[i][1] -= dx[i*3+1]
            self.pos[i][2] -= dx[i*3+2]

        self.push_down_pos()
        self.newton_step_init()

    def compute_residual_and_Hessian(self, check_PD=True, iter=0, spd=True):
        # residual
        for i in range(self.cloth_cnt):
            self.cloths[i].compute_normal_dir()
            self.cloths[i].prepare_bending()
            self.cloths[i].compute_residual()
            self.pushup_property_add(self.F_b, self.cloths[i].F_b, self.cloths[i].offset)
        for i in range(self.elastic_cnt):
            self.elastics[i].get_force()
            self.elastics[i].compute_residual()
            self.pushup_property_add(self.F_b, self.elastics[i].F_b, self.elastics[i].offset)

        self.flatten(self.F, self.F_b)
        self.apply_frozen()

        # hessian
        PD = True

        self.contact_energy(True, spd)  # residual and hessian
        if check_PD:
            print("hessian with contact")
            PD = PD and self.H.check_PD()

        # print("hessian nan contact?")
        # self.H.check_nan()

        for i in range(self.elastic_cnt):
            self.elastics[i].compute_Hessian(self, spd)
        if check_PD:
            print("hessian elastic")
            PD = PD and self.H.check_PD()

        # print("hessian nan -1?")
        # self.H.check_nan()
        for i in range(self.cloth_cnt):
            self.cloths[i].compute_Hessian_me(self, spd)
        if check_PD:
            print("hessian with me")
            PD = PD and self.H.check_PD()

        for i in range(self.cloth_cnt):
            self.cloths[i].compute_Hessian_ma(self)
        if check_PD:
            print("hessian with ma")
            PD = PD and self.H.check_PD()

        # print("hessian nan 0?")
        # self.H.check_nan()

        for i in range(self.cloth_cnt):
            self.cloths[i].compute_Hessian_bending(self)
        if check_PD:
            print("hessian with bending")
            PD = self.H.check_PD()

        # print("hessian nan?")
        # self.H.check_nan()
        # if check_PD == True:
        #     # self.plot_contact_force(0, iter)
        #     self.debug_plot(0, iter)
        #     self.debug_plot(2, iter)
        # if not PD:
        #     self.check_differential()

        return PD

    def compute_Hessian(self, spd=True):

        for i in range(self.elastic_cnt):
            self.elastics[i].compute_Hessian(self, spd)
        for i in range(self.cloth_cnt):
            self.cloths[i].compute_normal_dir()
            self.cloths[i].prepare_bending()
            self.cloths[i].compute_Hessian_me(self, spd)
            self.cloths[i].compute_Hessian_ma(self)
            self.cloths[i].compute_Hessian_bending(self)
        self.contact_energy(True, spd)  # residual and hessian

    def compute_residual(self):
        # residual
        for i in range(self.cloth_cnt):
            self.cloths[i].compute_normal_dir()
            self.cloths[i].prepare_bending()
            self.cloths[i].compute_residual()
            self.pushup_property_add(self.F_b, self.cloths[i].F_b, self.cloths[i].offset)
        for i in range(self.elastic_cnt):
            self.elastics[i].get_force()
            self.elastics[i].compute_residual()
            self.pushup_property_add(self.F_b, self.elastics[i].F_b, self.elastics[i].offset)

        self.flatten(self.F, self.F_b)
        self.apply_frozen()
        # hessian
        self.contact_energy(True)

    @ti.kernel
    def apply_frozen(self):
        for i in self.F:
            if self.frozen[i]:
                self.F[i] = 0

    def newton_step_init(self):
        self.H.clear_all()
        self.E[None] = 0
        self.F.fill(0)
        self.tmp_f.fill(0)
        self.F_b.fill(0)
        # for i in range(self.cloth_cnt):
        #     self.cloths[i].H_me.fill(0)
        #     self.cloths[i].H_ma.fill(0)
        #     self.cloths[i].H_bending_9.fill(0)
        #     self.cloths[i].H_bending_12.fill(0)

    @ti.kernel
    def linesearch_step(self, p: ti.types.ndarray(), alpha: ti.f64):
        for i in range(self.tot_NV):
            self.pos[i][0] = self.x1[i][0] - p[i * 3 + 0] * alpha
            self.pos[i][1] = self.x1[i][1] - p[i * 3 + 1] * alpha
            self.pos[i][2] = self.x1[i][2] - p[i * 3 + 2] * alpha

    @ti.kernel
    def calc_p_norm(self, p: ti.types.ndarray()) -> ti.f64:
        p_norm = 0.0
        for i in range(self.tot_NV):
            ti.atomic_max(p_norm, ti.abs(p[i * 3 + 0]))
            ti.atomic_max(p_norm, ti.abs(p[i * 3 + 1]))
            ti.atomic_max(p_norm, ti.abs(p[i * 3 + 2]))
        return p_norm

    @ti.kernel
    def calc_f_norm(self, p: ti.template()) -> ti.f64:
        F_norm = 0.0
        for i in range(self.tot_NV):
            ti.atomic_max(F_norm, ti.abs(p[i * 3 + 0]))
            ti.atomic_max(F_norm, ti.abs(p[i * 3 + 1]))
            ti.atomic_max(F_norm, ti.abs(p[i * 3 + 2]))
        return F_norm

    @ti.kernel
    def norm_F(self, src: ti.template()) -> ti.f64:
        # print(step)
        ret = 0.0
        for i in range(self.tot_NV):
            ret += src[3 * i + 0] * src[3 * i + 0]
            ret += src[3 * i + 1] * src[3 * i + 1]
            ret += src[3 * i + 2] * src[3 * i + 2]
        return ti.math.sqrt(ret)

    @ti.kernel
    def norm_P(self, src: ti.types.ndarray()) -> ti.f64:
        # print(step)
        ret = 0.0
        for i in range(self.tot_NV):
            ret += src[3 * i + 0] * src[3 * i + 0]
            ret += src[3 * i + 1] * src[3 * i + 1]
            ret += src[3 * i + 2] * src[3 * i + 2]
        return ti.math.sqrt(ret)

    @ti.kernel
    def norm_debug(self, src: ti.template(), src1: ti.types.ndarray()) -> ti.f32:
        # print(step)
        ret = 0.0
        for i in range(self.tot_NV):
            for j in range(3):
                ret += src[3 * i + j] * src1[3 * i + j]
        return ret

    def export_matrix(self, _A):
        A = torch.zeros((self.tot_NV * 3, self.tot_NV * 3))
        for i in range(self.tot_NV * 3):
            for j in range(self.tot_NV * 3):
                A[i, j] = _A[i, j]
        torch.save(A, 'tmp.pt')
        # L, _ = torch.linalg.eigh(A.cuda())
        # print(L.min(), L.max())

    @ti.kernel
    def gradient_descent(self, norm: ti.f64):
        for i in range(self.tot_NV):
            self.pos[i][0] -= self.F[i * 3 + 0] * self.lr / norm
            self.pos[i][1] -= self.F[i * 3 + 1] * self.lr / norm
            self.pos[i][2] -= self.F[i * 3 + 2] * self.lr / norm

    def newton_step(self, iter):
        F_normal = self.norm_F(self.F)
        F_array = self.F.to_torch(device=self.device)
        p = self.H.solve(F_array)
        p_norm = self.calc_p_norm(p)
        # F_norm =self.calc_f_norm(self.F)
        # print("max dim", p_norm)
        p_normal = self.norm_P(p)
        # print("dx norm:", p_norm)
        # print("res norm:", F_normal)
        # # print("lr:", p_norm / F_normal)
        # print("angle:", self.norm_debug(self.F, p)/p_normal/F_normal)
        # _H = self.H.build()
        # try:
        #     solver.compute(_H)
        #     self.copy_F(self.F_array)
        #     p = solver.solve(self.F_array)
        # except RuntimeError:
        #     self.export_matrix(_H)
        #     raise RuntimeError()

        # if p_norm/self.h < 1e-4:
        #     # print('terminate', p_norm/self.h)
        #     return p_norm/self.h


        self.x1.copy_from(self.pos)
        E0 = self.E[None]
        # print("original energy:", E0)
        alpha = 1.0
        while alpha > 1e-8:
            self.linesearch_step(p, alpha)
            self.push_down_pos()
            # self.calc_vn()
            # self.contact_analysis(set_contact=True)
            self.compute_energy()
            # print("now energy:", self.E[None])
            if self.E[None] < E0:
                break
            alpha /= 2

        # if alpha < 1e-8:
        #     # todo: different point with different mass
        #     p = F_array * self.h ** 2 / self.mass[0]
        #     alpha = 1.0
        #     while alpha > 1e-8:
        #         self.linesearch_step(p, alpha)
        #         self.push_down_pos()
        #         self.calc_vn()
        #         self.compute_energy(contact_force_stick)
        #         if self.E[None] < E0:
        #             break
        #         alpha /= 2
        #     print('tiny toi,', alpha)
        # else:
        #     print(alpha)
        # print(alpha)
        # if alpha < 1e-4 and iter > 20:
        #     self.pos.copy_from(self.x1)
        #     self.gradient_descent(1)
        #     self.push_down_pos()
        #     self.calc_vn()
        #     self.compute_energy()
        #     print("gradient descent")

        # view delta separately
        # delta_cloth = self.compute_delta(self.cloths[0].offset, self.cloths[0].offset + self.cloths[0].NV, p)
        # delta_el1 = self.compute_delta(self.elastics[1].offset, self.elastics[1].offset + self.elastics[1].n_verts, p)
        # delta_el2 = self.compute_delta(self.elastics[2].offset, self.elastics[2].offset + self.elastics[2].n_verts, p)
        # print(f"delta c:{delta_cloth}, delta elastic 1:{delta_el1}, delta elastic 2:{delta_el2}")

        return p_norm / self.h, alpha

    @ti.kernel
    def compute_delta(self, l:ti.i32, r:ti.i32, p: ti.types.ndarray()) -> ti.f64:
        p_norm = 0.0
        for i in range(l, r):
            ti.atomic_max(p_norm, ti.abs(p[i * 3 + 0]))
            ti.atomic_max(p_norm, ti.abs(p[i * 3 + 1]))
            ti.atomic_max(p_norm, ti.abs(p[i * 3 + 2]))
        return p_norm / self.h

    def gradient_step(self, norm):
        self.gradient_descent(norm)
        self.push_down_pos()
        self.calc_vn()
        self.compute_energy()

    def save_constraints(self, file):
        torch.save({
            'pos': self.pos.to_torch('cpu'),
            'c': self.const_idx.to_torch('cpu'),
            'w': self.const_w.to_torch('cpu'),
        }, file)

    def debug_plot(self, idx2, iter):
        v_start = self.body_list[idx2].v_start
        v_end = self.body_list[idx2].v_end
        x = torch.tensor([(self.pos[i][0], self.pos[i][1], self.pos[i][2]) for i in range(v_start, v_end)])
        n = torch.tensor([(self.vn[i][0], self.vn[i][1], self.vn[i][2]) for i in range(v_start, v_end)])

        ax = plt.figure().add_subplot(projection='3d')
        ax.set_zlim(-0.001, 0.01)
        ax.scatter(x[:, 0], x[:, 1], x[:, 2])
        # ax.quiver(x[:, 0], x[:, 1], x[:, 2], n[:, 0], n[:, 1], n[:, 2], normalize=True, length=0.01)
        plt.draw()
        plt.savefig(f'../imgs/pic-debug_plot_{idx2}_{iter}.png')
        plt.close()

    def plot_contact_force(self, idx2, iter):
        v_start = self.body_list[idx2].v_start
        v_end = self.body_list[idx2].v_end
        x_array = self.pos.to_torch()[v_start:v_end]
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(-0.3, 0.3)
        # flag = self.contact_flag.to_torch()[idx1, v_start:v_end]
        f0 = self.tmp_f.to_torch()[0, v_start * 3:v_end * 3].view(-1, 3)
        # f1 = self.tmp_f.to_torch()[1, v_start * 3:v_end * 3].view(-1, 3)
        # f2 = self.tmp_f.to_torch()[2, v_start * 3:v_end * 3].view(-1, 3)
        # sc = ax.scatter(x_array[:, 0], x_array[:, 1], x_array[:, 2], c=flag, vmin=0, vmax=2)
        ax.quiver(x_array[:, 0], x_array[:, 1], x_array[:, 2], f0[:, 0], f0[:, 1], f0[:, 2], length=1, color='red')
        # ax.quiver(x_array[:, 0], x_array[:, 1], x_array[:, 2], f1[:, 0], f1[:, 1], f1[:, 2], length=10, color='green')
        # ax.quiver(x_array[:, 0], x_array[:, 1], x_array[:, 2], f2[:, 0], f2[:, 1], f2[:, 2], length=10, color='blue')
        # plt.colorbar(sc)
        # plt.show()
        plt.draw()
        plt.savefig(f'../imgs/pic-debug_contact_{iter}.png')
        plt.close()


    def timestep_init(self):
        self.get_prev_pos()
        self.get_x_hat()
        for i in range(self.cloth_cnt):
            self.pushdown_property(self.cloths[i].prev_pos, self.prev_pos, self.cloths[i].offset)
        for i in range(self.elastic_cnt):
            self.pushdown_property(self.elastics[i].F_x_prev, self.prev_pos, self.elastics[i].offset)

    @ti.kernel
    def get_prev_pos(self):
        for i in range(self.tot_NV):
            for j in range(3):
                self.prev_pos[i][j] = self.pos[i][j]

    @ti.kernel
    def get_x_hat(self):
        for i in range(self.tot_NV):
            ext_force = self.mass[i] * self.gravity[None] + self.ext_force[i]
            x_hat = self.pos[i] + self.dt * self.vel[i] + self.dt ** 2 / self.mass[i] * ext_force
            for j in ti.static(range(3)):
                if self.frozen[i * 3 + j]:
                    self.x_hat[i][j] = self.pos[i][j]
                else:
                    self.x_hat[i][j] = x_hat[j]

    @ti.kernel
    def update_visual(self):
        for i in range(self.tot_NV):
            self.x32[i] = ti.cast(self.pos[i], ti.f32)

    def timestep_finish(self):
        self.update_vel()
        self.push_down_vel()
        # self.update_ref_angle()
        self.update_visual()

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
        while iter < 1000:
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
        # if frame_idx%10==0:
        # self.plot_contact_force(0, frame_idx)
        # self.debug_plot()
        # self.save_state('../ckpt/frame_%d.pt' % frame_idx)

    def load_state(self, save_path):
        data = torch.load(save_path)
        for i in range(self.tot_NV):
            self.pos[i] = data['pos'][i]
            self.vel[i] = data['vel'][i]
        self.push_down_pos()
        self.push_down_vel()
        self.update_ref_angle()
        self.update_visual()

    def save_state(self, save_path):
        pos = self.pos.to_torch('cpu')
        vel = self.vel.to_torch('cpu')
        torch.save({
            'pos': pos,
            'vel': vel
        }, save_path)

    @ti.kernel
    def set_ext_force_kernel(self):
        # for i in range(self.elastics[0].n_verts):
        #
        for i in range(self.elastics[1].n_verts):
            if i % 5 == 1:
                self.elastics[1].ext_force[i] = ti.Vector([0, 0, -100]) * self.elastics[1].F_m[i]
        for i in range(self.elastics[2].n_verts):
            if i % 5 == 0:
                self.elastics[2].ext_force[i] = ti.Vector([0, 0, 100]) * self.elastics[2].F_m[i]
        # for i in range(self.cloths[0].NV):
        #     if i % (self.cloths[0].N + 1) == 0:
        #         self.cloths[0].manipulate_force[i] = self.cloths[0].mass * ti.Vector([0, -20, 0])
        #     if i % (self.cloths[0].N + 1) == self.cloths[0].N:
        #         self.cloths[0].manipulate_force[i] = self.cloths[0].mass * ti.Vector([0, 20, 0])

    def set_ext_force(self):
        self.ext_force.fill(0)
        # self.set_ext_force_kernel()
        # self.pushup_property(self.ext_force, self.cloths[0].manipulate_force, self.cloths[0].offset)
        for i in range(self.cloth_cnt):
            self.cloths[i].clear_manipulation()
        # self.pushup_property(self.ext_force, self.elastics[1].ext_force, self.elastics[1].offset)
        # self.pushup_property(self.ext_force, self.elastics[2].ext_force, self.elastics[2].offset)
        # for i in range(5):
        #     self.gripper_dist[i] = -0.001 * i
        # self.gripper_dist[5] = -0.005
        # for i in range(6, 20):
        #     self.gripper_dist[i] = -0.005
        #     self.gripper_pos[i].z = self.gripper_pos[i - 1].z + 0.0003

    @ti.kernel
    def set_ext_force_kernel2(self):
        # for i in range(self.elastics[0].n_verts):
        #
        for i in range(self.elastics[1].n_verts):
            if i % 5 == 1:
                self.elastics[1].ext_force[i] = ti.Vector([0, 0, -50]) * self.elastics[1].F_m[i]
        for i in range(self.elastics[2].n_verts):
            if i % 5 == 0:
                self.elastics[2].ext_force[i] = ti.Vector([0, 0, 300]) * self.elastics[2].F_m[i]

    def set_ext_force2(self):
        self.ext_force.fill(0)
        self.set_ext_force_kernel2()
        # self.pushup_property(self.ext_force, self.cloths[0].manipulate_force, self.cloths[0].offset)
        for i in range(self.cloth_cnt):
            self.cloths[i].clear_manipulation()
        self.pushup_property(self.ext_force, self.elastics[1].ext_force, self.elastics[1].offset)
        self.pushup_property(self.ext_force, self.elastics[2].ext_force, self.elastics[2].offset)

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

    def set_frozen(self):
        self.frozen.fill(0)
        self.set_frozen_kernel()

    @ti.kernel
    def get_colors(self, colors: ti.template()):
        colors.fill(0)
        for i in range(self.cloths[0].NV):
            colors[i + self.cloths[0].offset] = ti.Vector([1, 1, 1])
        # for i in range(self.cloths[1].NV):
        #     colors[i + self.cloths[1].offset] = ti.Vector([0.5, 0.5, 0.5])
        for i in range(self.elastics[1].n_verts):
            colors[i + self.elastics[1].offset] = ti.Vector([0.22, 0.72, 0.52])     # Agent1 Color 
        for i in range(self.elastics[2].n_verts):
            colors[i + self.elastics[2].offset] = ti.Vector([1, 0.334, 0.52])       # Agent2 Color 

    def action(self, step, delta_pos, delta_rot, delta_dis):
        # self.gripper.print_plot()
        self.gripper.step_simple(delta_pos, delta_rot, delta_dis)
        # self.gripper.print_plot()
        self.gripper.update_bound(self.elastics[1], self.elastics[2])
        # a1 = self.elastics[1].check_determinant()
        # a2 = self.elastics[2].check_determinant()
        # if not (a1 and a2):
        #     print("penetrate!!!!")
        # self.gripper.print_plot()
        self.pushup_property(self.pos, self.elastics[1].F_x, self.elastics[1].offset)
        self.pushup_property(self.pos, self.elastics[2].F_x, self.elastics[2].offset)

    @ti.kernel
    def compute_reward(self) -> ti.f64:
        ret = 0.0
        for i in self.cloths[0].pos:
            ret = ret + self.cloths[0].pos[i].z

        return ret

    def plot_tactile(self, frame):
        self.gripper.plot_tactile(frame, self.elastics[1], self.elastics[2])

    def get_paramters_grad(self):
        self.d_ka.fill(0)
        self.d_kl.fill(0)
        self.d_kb.fill(0)
        self.d_mu.fill(0)
        for i in range(self.cloth_cnt):
            self.cloths[i].compute_deri()
            self.pushup_property(self.d_ka, self.cloths[i].d_ka, self.cloths[i].offset)
            self.pushup_property(self.d_kb, self.cloths[i].d_kb, self.cloths[i].offset)
            self.pushup_property(self.d_kl, self.cloths[i].d_kl, self.cloths[i].offset)
        for i in range(self.elastic_cnt):
            self.elastics[i].compute_deri()
            self.pushup_property(self.d_mu, self.elastics[i].d_mu, self.elastics[i].offset)

    def init_folding(self):
        for i in range(self.cloth_cnt):
            self.cloths[i].compute_normal_dir()
            self.cloths[i].prepare_bending()

    def ref_angle_backprop_x2a(self, analy_grad, step, p):
        # calculate partial L partial ref_angle_i
        for i in range(self.cloth_cnt):
            self.cloths[i].ref_angle_backprop_x2a(analy_grad, step, p, i)

    def ref_angle_backprop_a2ax(self, analy_grad: ti.template(), step: int):
        for i in range(self.cloth_cnt):
            self.cloths[i].ref_angle_backprop_a2ax(analy_grad, step, i)

    @ti.kernel
    def gather_force(self):

        for i in range(self.elastics[1].n_verts):
            for j in ti.static(range(1, self.effector_cnt)):
                if self.elastics[j].is_bottom(i) or self.elastics[j].is_inner_circle(i):
                    self.tot_force[j - 1, 0] += self.elastics[j].F_f[i][0]
                    self.tot_force[j - 1, 1] += self.elastics[j].F_f[i][1]
                    self.tot_force[j - 1, 2] += self.elastics[j].F_f[i][2]

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

    @ti.kernel
    def get_observation_kernel(self):
        for j, k in ti.ndrange(self.n_obs_cloth, self.n_obs_cloth):
            for i in ti.static(range(self.cloth_cnt)):
                xx = i * (self.n_obs_cloth ** 2) + j * self.n_obs_cloth + k
                jj = self.n_sample_cloth // 2 + j * self.n_sample_cloth
                kk = self.m_sample_cloth // 2 + k * self.m_sample_cloth
                self.observation[xx * 6 + 0] = self.cloths[i].pos[jj * self.cloth_N + kk][0]
                self.observation[xx * 6 + 1] = self.cloths[i].pos[jj * self.cloth_N + kk][1]
                self.observation[xx * 6 + 2] = self.cloths[i].pos[jj * self.cloth_N + kk][2]
                self.observation[xx * 6 + 3] = self.cloths[i].vel[jj * self.cloth_N + kk][0]
                self.observation[xx * 6 + 4] = self.cloths[i].vel[jj * self.cloth_N + kk][1]
                self.observation[xx * 6 + 5] = self.cloths[i].vel[jj * self.cloth_N + kk][2]

        for j in range(self.n_obs_elastic):
            for i in ti.static(range(self.elastic_cnt)):
                xx = (self.n_obs_cloth ** 2) * self.cloth_cnt + i * self.n_obs_elastic + j
                ii = (self.elastics[i].n_verts // self.n_obs_elastic) * j - 1
                self.observation[xx * 6 + 0] = self.elastics[i].F_x[ii][0]
                self.observation[xx * 6 + 1] = self.elastics[i].F_x[ii][1]
                self.observation[xx * 6 + 2] = self.elastics[i].F_x[ii][2]
                self.observation[xx * 6 + 3] = self.elastics[i].F_v[ii][0]
                self.observation[xx * 6 + 4] = self.elastics[i].F_v[ii][1]
                self.observation[xx * 6 + 5] = self.elastics[i].F_v[ii][2]

        for j in range(self.gripper.n_part):
            xx = ((self.n_obs_cloth ** 2) * self.cloth_cnt + self.elastic_cnt * self.n_obs_elastic) * 6 + j * 7
            self.observation[xx + 0] = self.gripper.pos[j][0]
            self.observation[xx + 1] = self.gripper.pos[j][1]
            self.observation[xx + 2] = self.gripper.pos[j][2]
            self.observation[xx + 3] = self.gripper.rot[j][0]
            self.observation[xx + 4] = self.gripper.rot[j][1]
            self.observation[xx + 5] = self.gripper.rot[j][2]
            self.observation[xx + 6] = self.gripper.rot[j][3]
