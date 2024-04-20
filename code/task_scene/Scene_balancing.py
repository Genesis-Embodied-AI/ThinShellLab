import taichi as ti

from ..engine.model_fold_offset import Cloth
from ..engine.model_elastic_offset import Elastic

import taichi as ti
from dataclasses import dataclass
from ..engine.model_elastic_tactile import Elastic as tactile
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
        super(Scene, self).__init__(cloth_size=cloth_size, enable_gripper=True, device=device)
        self.cloths[0].k_angle[None] = 3.14

    def init_scene_parameters(self):
        self.dt = 5e-3
        self.h = self.dt
        self.cloth_cnt = 1
        self.elastic_cnt = 5
        self.elastic_size = [0.007, 0.015, 0.015, 0.015, 0.015]
        self.elastic_Nx = int(5)
        self.elastic_Ny = int(5)
        self.elastic_Nz = int(5)
        self.cloth_N = 15
        self.cloth_M = 7

        self.k_contact = 10000
        self.eps_contact = 0.00041
        self.eps_v = 0.01
        self.max_n_constraints = 10000
        self.damping = 1.0

    def init_objects(self):
        rho = 4e1

        self.cloths.append(Cloth(self.cloth_N, self.dt, self.cloth_size, self.tot_NV, rho, 0, is_square=False, M=self.cloth_M))

        self.elastic_offset = (self.cloth_N + 1) * (self.cloth_M + 1)
        tmp_tot = self.elastic_offset

        self.elastics.append(Elastic(
            self.dt, self.elastic_size[0], tmp_tot,
            self.elastic_Nx, self.elastic_Ny, self.elastic_Nz, 10000.0, load=True
        ))
        tmp_tot += self.elastics[0].n_verts

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
        self.cloths[0].init(-0.03, -0.015, 0.)
        self.elastics[0].init(0., 0., 0.0039)
        self.elastics[1].init(0.023, 0., 0.0079, True)
        self.elastics[2].init(0.023, 0., -0.0079, False)
        self.elastics[3].init(-0.023, 0, 0.0079, True)
        self.elastics[4].init(-0.023, 0, -0.0079, False)
        pos = np.array([[0.023, 0., 0.0], [-0.023, 0., 0.0]])
        self.gripper.init(self, pos)

    def reset_pos(self):
        self.cloths[0].init(-0.03, -0.015, 0.)
        self.elastics[0].init(0., 0., 0.0039)
        self.elastics[1].init(0.023, 0., 0.0079, True)
        self.elastics[2].init(0.023, 0., -0.0079, False)
        self.elastics[3].init(-0.023, 0, 0.0079, True)
        self.elastics[4].init(-0.023, 0, -0.0079, False)
        pos = np.array([[0.023, 0., 0.0], [-0.023, 0., 0.0]])
        self.gripper.init(self, pos)

    def contact_analysis(self):
        self.nc[None] = 0
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
        for i in range(self.elastics[4].n_verts):
            if self.elastics[4].is_bottom(i) or self.elastics[4].is_inner_circle(i):
                xx = self.elastics[4].offset + i
                self.frozen[xx * 3] = 1
                self.frozen[xx * 3 + 1] = 1
                self.frozen[xx * 3 + 2] = 1

    @ti.kernel
    def compute_reward(self) -> ti.f64:
        ret = 0.0
        tt = (self.cloth_N + 1) // 2 * (self.cloth_M + 1) + (self.cloth_M + 1) // 2
        for i in range(self.elastics[0].n_verts):
            ret -= (self.elastics[0].F_x[i].x - self.cloths[0].pos[tt].x) ** 2
            ret -= (self.elastics[0].F_x[i].y - self.cloths[0].pos[tt].y) ** 2
        return ret

    @ti.kernel
    def compute_reward_all(self, analy_grad: ti.template()) -> ti.f64:
        ret = 0.0
        tt = (self.cloth_N + 1) // 2 * (self.cloth_M + 1) + (self.cloth_M + 1) // 2
        for i, j in ti.ndrange(self.elastics[0].n_verts, analy_grad.tot_timestep):
            ret -= (analy_grad.pos_buffer[j, self.elastics[0].offset + i, 0] - analy_grad.pos_buffer[j, self.cloths[0].offset + tt, 0]) ** 2
            ret -= (analy_grad.pos_buffer[j, self.elastics[0].offset + i, 1] - analy_grad.pos_buffer[j, self.cloths[0].offset + tt, 1]) ** 2
        return ret

    @ti.kernel
    def compute_reward_throwing(self, analy_grad: ti.template()) -> ti.f64:
        ret = 0.0
        tt = (self.cloth_N + 1) // 2 * (self.cloth_M + 1) + (self.cloth_M + 1) // 2
        for i in range(self.elastics[0].n_verts):
            ret += analy_grad.pos_buffer[analy_grad.tot_timestep - 1, self.elastics[0].offset + i, 2]

        for i in range(self.cloth_M + 1):
            ret -= 10 * (self.cloths[0].pos[i].z - 0.0) ** 2
            ret -= 10 * (self.cloths[0].pos[i + self.cloth_N * (self.cloth_M + 1)].z - 0.0) ** 2

        return ret

    @ti.kernel
    def compute_reward_throwing_RL(self) -> ti.f64:
        ret = 0.0
        tt = (self.cloth_N + 1) // 2 * (self.cloth_M + 1) + (self.cloth_M + 1) // 2
        for i in range(self.elastics[0].n_verts):
            ret += self.elastics[0].F_x[i].z

        for i in range(self.cloth_M + 1):
            ret -= 10 * (self.cloths[0].pos[i].z - 0.0) ** 2
            ret -= 10 * (self.cloths[0].pos[i + self.cloth_N * (self.cloth_M + 1)].z - 0.0) ** 2

        return ret

    def action(self, step, delta_pos, delta_rot):
        # self.gripper.print_plot()
        # if step < 5:
        #     self.gripper.step(delta_pos, delta_rot, ti.Vector([-0.0005, -0.0005]))
        # else:
        self.gripper.step_simple(delta_pos, delta_rot)
        # self.gripper.print_plot()
        self.gripper.update_bound(self)
        # a1 = self.elastics[1].check_determinant()
        # a2 = self.elastics[2].check_determinant()
        # a3 = self.elastics[3].check_determinant()
        # a4 = self.elastics[4].check_determinant()
        # if not (a1 and a2 and a3 and a4):
        #     print("penetrate!!!!")
        # self.gripper.print_plot()
        self.pushup_property(self.pos, self.elastics[1].F_x, self.elastics[1].offset)
        self.pushup_property(self.pos, self.elastics[2].F_x, self.elastics[2].offset)
        self.pushup_property(self.pos, self.elastics[3].F_x, self.elastics[3].offset)
        self.pushup_property(self.pos, self.elastics[4].F_x, self.elastics[4].offset)

    def save_all(self, path):
        self.gripper.save_all(path)
        state_save = os.path.join(path, "state")
        self.save_state(state_save)
        nps = self.proj_flag.to_numpy()
        np.save(os.path.join(path, "proj_flag.npy"), nps)
        nps = self.proj_dir.to_numpy()
        np.save(os.path.join(path, "proj_dir.npy"), nps)
        nps = self.border_flag.to_numpy()
        np.save(os.path.join(path, "border_flag.npy"), nps)

    def load_all(self, path):
        self.gripper.load_all(path)
        state_save = os.path.join(path, "state")
        self.load_state(state_save)
        npl = np.load(os.path.join(path, "proj_flag.npy"))
        self.proj_flag.from_numpy(npl)
        npl = np.load(os.path.join(path, "proj_dir.npy"))
        self.proj_dir.from_numpy(npl)
        npl = np.load(os.path.join(path, "border_flag.npy"))
        self.border_flag.from_numpy(npl)

    @ti.kernel
    def get_colors(self, colors: ti.template()):
        colors.fill(0)
        for i in range(self.cloths[0].NV):
            colors[i + self.cloths[0].offset] = ti.Vector([1, 1, 1])
        # for i in range(self.cloths[1].NV):
        #     colors[i + self.cloths[1].offset] = ti.Vector([0.5, 0.5, 0.5])
        for i in range(self.elastics[1].n_verts):
            colors[i + self.elastics[1].offset] = ti.Vector([0.22, 0.72, 0.52])  # Agent1 Color
        for i in range(self.elastics[2].n_verts):
            colors[i + self.elastics[2].offset] = ti.Vector([1, 0.334, 0.52])  # Agent2 Color
        for i in range(self.elastics[3].n_verts):
            colors[i + self.elastics[3].offset] = ti.Vector([0.22, 0.72, 0.52])  # Agent1 Color
        for i in range(self.elastics[4].n_verts):
            colors[i + self.elastics[4].offset] = ti.Vector([1, 0.334, 0.52])  # Agent2 Color

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




