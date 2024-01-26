import argparse

import numpy as np
import torch
import taichi as ti
import matplotlib.pyplot as plt

@ti.func
def quat_to_rotmat(quat):
    s = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    rotmat = ti.Matrix([[s * s + x * x - y * y - z * z, 2 * (x * y - s * z), 2 * (x * z + s * y)],
                                   [2 * (x * y + s * z), s * s - x * x + y * y - z * z, 2 * (y * z - s * x)],
                                   [2 * (x * z - s * y), 2 * (y * z + s * x), s * s - x * x - y * y + z * z]])
    return rotmat

@ti.func
def rotmat_to_quat(rotmat):
    s = ti.math.sqrt(1 + rotmat[0][0] + rotmat[1][1] + rotmat[2][2]) / 2
    x = (rotmat[2][1] - rotmat[1][2]) / (4 * s)
    y = (rotmat[0][2] - rotmat[2][0]) / (4 * s)
    z = (rotmat[1][0] - rotmat[0][1]) / (4 * s)
    return ti.Vector([s, x, y, z])

@ti.data_oriented
class gripper:
    def __init__(self, dt, n_verts, n_bound, n_surf, cnt):

        self.n_verts = n_verts
        self.dt = dt
        self.n_bound = n_bound
        self.n_surf = n_surf
        self.n_part = cnt

        self.F_x = ti.Vector.field(3, dtype=ti.f64, shape=(cnt, n_verts))
        self.F_x_world = ti.Vector.field(3, dtype=ti.f64, shape=(cnt, n_verts))

        self.bound_idx = ti.field(ti.i32, shape=n_bound) # it's the same in all tactile
        self.surface_idx = ti.field(ti.i32, shape=n_surf)

        self.pos = ti.Vector.field(3, dtype=ti.f64, shape=cnt)
        self.rot = ti.Vector.field(4, dtype=ti.f64, shape=cnt)
        self.d_pos = ti.Vector.field(3, dtype=ti.f64, shape=cnt)
        self.d_angle = ti.Vector.field(3, dtype=ti.f64, shape=cnt)

        self.rotmat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=cnt)

    @ti.kernel
    def init_kernel(self, sys: ti.template(), pos_array: ti.types.ndarray()):
        for i in ti.static(range(1, sys.effector_cnt)):
            self.pos[i - 1] = ti.Vector([pos_array[i - 1, 0], pos_array[i - 1, 1], pos_array[i - 1, 2]])
            self.rot[i - 1] = ti.Vector([1.0, 0., 0., 0.])
        for i in range(self.n_verts):
            for jj in ti.static(range(1, sys.effector_cnt)):
                j = jj - 1
                self.F_x[j, i] = sys.elastics[jj].F_x[i] - self.pos[j]

        cnt0 = 0
        cnt1 = 0
        for i in range(sys.elastics[1].n_verts):
            if sys.elastics[1].is_bottom(i) or sys.elastics[1].is_inner_circle(i):
                ii = ti.atomic_add(cnt0, 1)
                self.bound_idx[ii] = i
            else:
                if sys.elastics[1].is_surf(i):
                    ii = ti.atomic_add(cnt1, 1)
                    self.surface_idx[ii] = i
        # print(cnt0, cnt1)

    def init(self, sys, pos_array):
        self.init_kernel(sys, pos_array)
        self.get_rotmat()

    def set(self, pos: ti.template(), rot: ti.template(), step: int):
        for i in range(self.n_part):
            self.pos[i] = pos[step, i]
            self.rot[i] = rot[step, i]

    @ti.kernel
    def get_vert_pos(self):
        for i in range(self.n_verts):
            for j in ti.static(range(self.n_part)):
                self.F_x_world[j, i] = self.pos[j] + self.rotmat[j] @ self.F_x[j, i]

    def get_rotmat(self):
        for j in range(self.n_part):
            s = self.rot[j][0]
            x = self.rot[j][1]
            y = self.rot[j][2]
            z = self.rot[j][3]
            self.rotmat[j] = ti.Matrix([[s*s+x*x-y*y-z*z, 2*(x*y-s*z), 2*(x*z+s*y)],
                                     [2*(x*y+s*z), s*s-x*x+y*y-z*z, 2*(y*z-s*x)],
                                     [2*(x*z-s*y), 2*(y*z+s*x), s*s-x*x-y*y+z*z]])
    @ti.func
    def quat_to_rotmat(self, quat):
        s = quat[0]
        x = quat[1]
        y = quat[2]
        z = quat[3]
        rotmat = ti.Matrix([[s * s + x * x - y * y - z * z, 2 * (x * y - s * z), 2 * (x * z + s * y)],
                                       [2 * (x * y + s * z), s * s - x * x + y * y - z * z, 2 * (y * z - s * x)],
                                       [2 * (x * z - s * y), 2 * (y * z + s * x), s * s - x * x - y * y + z * z]])
        return rotmat

    @ti.func
    def rotmat_to_quat(self, rotmat):
        s = ti.math.sqrt(1 + rotmat[0][0] + rotmat[1][1] + rotmat[2][2]) / 2
        x = (rotmat[2][1] - rotmat[1][2]) / (4 * s)
        y = (rotmat[0][2] - rotmat[2][0]) / (4 * s)
        z = (rotmat[1][0] - rotmat[0][1]) / (4 * s)
        return ti.Vector([s, x, y, z])

    def step_simple(self, delta_pos: ti.template(), delta_rot: ti.template()):
        for j in range(self.n_part):
            self.pos[j] += delta_pos[j]
            self.get_rotmat()

            v2 = ti.Vector([self.rot[j][1], self.rot[j][2], self.rot[j][3]])
            real = -delta_rot[j].dot(v2)
            res = self.rot[j][0] * delta_rot[j] + delta_rot[j].cross(v2)

            self.rot[j][0] += real
            self.rot[j][1] += res[0]
            self.rot[j][2] += res[1]
            self.rot[j][3] += res[2]
            self.rot[j] = self.rot[j].normalized()

        self.get_rotmat()
        self.get_vert_pos()

    @ti.kernel
    def gather_grad(self, grad: ti.template(), sys: ti.template()):
        self.d_pos.fill(0)
        self.d_angle.fill(0)

        for i in range(self.n_bound):
            for j in ti.static(range(self.n_part)):
                xx = sys.elastics[j + 1].offset + self.bound_idx[i]
                gradient = ti.Vector([grad[xx * 3 + 0], grad[xx * 3 + 1], grad[xx * 3 + 2]])
                self.d_pos[j] += gradient
                self.d_angle[j] += (self.rotmat[j] @ self.F_x[j, self.bound_idx[i]]).cross(gradient)

        for j in ti.static(range(self.n_part)):
            self.d_pos[j] /= 1.0 * self.n_bound
            self.d_angle[j] /= 1.0 * self.n_bound
            for k in ti.static(range(3)):
                self.d_pos[j][k] = ti.math.clamp(self.d_pos[j][k], -10, 10)
                self.d_angle[j][k] = ti.math.clamp(self.d_angle[j][k], -100, 100)

    @ti.kernel
    def update_bound(self, sys: ti.template()):
        for i in range(self.n_bound):
            for j in ti.static(range(self.n_part)):
                sys.elastics[j + 1].F_x[self.bound_idx[i]] = self.F_x_world[j, self.bound_idx[i]]

    @ti.kernel
    def update_all(self, sys: ti.template()):
        for i in range(self.n_verts):
            for j in ti.static(range(self.n_part)):
                sys.elastics[j + 1].F_x[i] = self.F_x_world[j, i]