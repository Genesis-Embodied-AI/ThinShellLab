import taichi as ti

from model_fold_offset_new import Cloth
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
from BaseScene import BaseScene as base_Scene

vec3 = ti.types.vector(3, ti.f64)
vec3i = ti.types.vector(3, ti.i32)

@dataclass
class Body:
    v_start: ti.i32
    v_end: ti.i32
    f_start: ti.i32
    f_end: ti.i32

@ti.data_oriented
class Scene(base_Scene):

    def __init__(self, cloth_size=0.06, target=0.015):
        super(Scene, self).__init__(cloth_size=cloth_size, enable_gripper=False)
        self.gravity[None] = ti.Vector([0., 0., 0])
        self.target = target
        self.cloths[0].k_angle[None] = 3.14

    def init_scene_parameters(self):
        self.dt = 0.0005
        self.h = self.dt
        self.cloth_cnt = 1
        self.elastic_cnt = 2
        self.elastic_size = [0.06, 0.015]
        self.elastic_Nx = int(9)
        self.elastic_Ny = int(9)
        self.elastic_Nz = int(5)
        self.cloth_N = 16

        self.k_contact = 10000
        self.eps_contact = 0.0004
        self.eps_v = 0.01
        self.max_n_constraints = 10000
        self.damping = 1.0

    def init_objects(self):
        rho = 2
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

    def init_all(self):
        self.init()
        self.init_property()
        self.set_frozen()
        self.set_ext_force()
        self.update_visual()
        # print(self.cloths[0].mass)
        # print(self.elastics[0].F_m[0])

    def init(self):

        self.cloths[0].init(-0.03, -0.03, 0.00039)
        self.elastics[0].init(0.0, -0.03, -0.03)
        self.elastics[1].init(-0.029, 0.0, 0.00828, True)
        pos = np.array([[-0.029, 0.0, 0.00828]])
        self.gripper.init(self, pos)

    def reset_pos(self):
        self.cloths[0].init(-0.03, -0.03, 0.00039)
        self.elastics[0].init(0.0, -0.03, -0.03)
        self.elastics[1].init(-0.029, 0.0, 0.00828, True)
        pos = np.array([[-0.029, 0.0, 0.00828]])
        self.gripper.init(self, pos)

    def contact_analysis(self):
        self.nc[None] = 0
        for i in range(self.cloth_cnt):
            for j in range(1, self.elastic_cnt):

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
        for i in range(self.cloths[0].M + 1):
            xx = self.cloths[0].offset + self.cloths[0].N * (self.cloths[0].M + 1) + i
            self.frozen[xx * 3] = 1
            self.frozen[xx * 3 + 1] = 1
            self.frozen[xx * 3 + 2] = 1
        for i in range(self.cloths[0].M + 1):
            xx = self.cloths[0].offset + 7 * (self.cloths[0].M + 1) + i
            self.frozen[xx * 3] = 1
            self.frozen[xx * 3 + 1] = 1
            self.frozen[xx * 3 + 2] = 1
        for i in range(self.cloths[0].M + 1):
            xx = self.cloths[0].offset + 8 * (self.cloths[0].M + 1) + i
            self.frozen[xx * 3] = 1
            self.frozen[xx * 3 + 1] = 1
            self.frozen[xx * 3 + 2] = 1

        for i in range(self.elastics[1].n_verts):
            if self.elastics[1].is_bottom(i) or self.elastics[1].is_inner_circle(i):
                xx = self.elastics[1].offset + i
                self.frozen[xx * 3] = 1
                self.frozen[xx * 3 + 1] = 1
                self.frozen[xx * 3 + 2] = 1


    def compute_reward(self, analy_grad: ti.template()) -> ti.f64:
        ret = 0.0
        tt = analy_grad.tot_timestep - 1
        max_z = -1.0
        for j in range(40, analy_grad.tot_timestep):
            now_z = 0
            for i in range(self.cloths[0].M + 1):
                now_z += analy_grad.pos_buffer[j, i + self.cloths[0].offset, 2]
            if now_z > max_z:
                max_z = now_z
                tt = j

        for i in range(self.cloths[0].M + 1):
            ret -= (analy_grad.pos_buffer[tt, i + self.cloths[0].offset, 2] - self.target)**2
        return ret

    def action(self, step, delta_pos, delta_rot):
        # self.gripper.print_plot()
        self.gripper.step_simple(delta_pos, delta_rot)
        # self.gripper.print_plot()
        self.gripper.update_bound(self)
        a1 = self.elastics[1].check_determinant()
        # a2 = self.elastics[2].check_determinant()
        if not a1:
            print("penetrate!!!!")
        # self.gripper.print_plot()
        self.pushup_property(self.pos, self.elastics[1].F_x, self.elastics[1].offset)
        # self.pushup_property(self.pos, self.elastics[2].F_x, self.elastics[2].offset)

    def timestep_finish(self):
        self.update_vel()
        self.push_down_vel()
        self.update_ref_angle()
        self.update_visual()

    def ref_angle_backprop_x2a(self, analy_grad, step, p):
        # calculate partial L partial ref_angle_i
        for i in range(self.cloth_cnt):
            self.cloths[i].ref_angle_backprop_x2a(analy_grad, step, p, i)

    def ref_angle_backprop_a2ax(self, analy_grad: ti.template(), step: int):
        for i in range(self.cloth_cnt):
            self.cloths[i].ref_angle_backprop_a2ax(analy_grad, step, i)

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
            # if iter > 70:
            #     print(f"iter: {iter}, delta: {delta}, step size: {alpha}, energy {self.E[None]}")
            if delta < 1e-7:
                break
        # print(f'pass {frame_idx}, iter:', iter, "delta:", delta)

        self.timestep_finish()
        # if frame_idx%10==0:
        # self.plot_contact_force(0, frame_idx)
        # self.debug_plot()
        # self.save_state('../ckpt/frame_%d.pt' % frame_idx)

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
                if ti.abs(self.tot_force[i, j]) > 15:
                    if ifprint:
                        print("too much force")
                    return True

            force = ti.sqrt(self.tot_force[i, 0]**2 + self.tot_force[i, 1]**2 + self.tot_force[i, 2]**2)
            if (force < 0.2) and (frame > 10) and (frame < 50):
                if ifprint:
                    print("no contact")
                return True

        return False