import argparse

import numpy as np
import torch
import taichi as ti
import matplotlib.pyplot as plt
import os

@ti.data_oriented
class gripper:
    def __init__(self, dt, n_verts, n_bound, n_surf, cnt):

        self.n_verts = n_verts
        self.dt = dt
        self.n_bound = n_bound
        self.n_surf = n_surf
        self.n_part = cnt

        self.F_x_upper = ti.Vector.field(3, dtype=ti.f64, shape=(cnt, n_verts))
        self.F_x_upper_world = ti.Vector.field(3, dtype=ti.f64, shape=(cnt, n_verts))
        self.F_x_lower = ti.Vector.field(3, dtype=ti.f64, shape=(cnt, n_verts))
        self.F_x_lower_world = ti.Vector.field(3, dtype=ti.f64, shape=(cnt, n_verts))
        self.bound_idx = ti.field(ti.i32, shape=n_bound) # it's the same in upper and lower
        self.surface_idx = ti.field(ti.i32, shape=n_surf)

        self.pos = ti.Vector.field(3, dtype=ti.f64, shape=cnt)
        self.rot = ti.Vector.field(4, dtype=ti.f64, shape=cnt)
        self.d_pos = ti.Vector.field(3, dtype=ti.f64, shape=cnt)
        self.d_angle = ti.Vector.field(3, dtype=ti.f64, shape=cnt)
        self.d_dist = ti.field(dtype=ti.f64, shape=cnt)

        self.rotmat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=cnt)

        self.half_gripper_dist = ti.field(ti.f64, shape=cnt)

        # self.upper_camera_pos = ti.Vector.field(3, dtype=ti.f64, shape=())
        # self.upper_camera_rot = ti.Vector.field(4, dtype=ti.f64, shape=())
        #
        # self.lower_camera_pos = ti.Vector.field(3, dtype=ti.f64, shape=())
        # self.lower_camera_rot = ti.Vector.field(4, dtype=ti.f64, shape=())
        #
        # self.upper_now_plot = ti.field(dtype=ti.f64, shape=(n_surf, 2))
        # self.lower_now_plot = ti.field(dtype=ti.f64, shape=(n_surf, 2))
        #
        # self.upper_ori_plot = ti.field(dtype=ti.f64, shape=(n_surf, 2))
        # self.lower_ori_plot = ti.field(dtype=ti.f64, shape=(n_surf, 2))
        #
        # self.temp_plot = ti.field(dtype=ti.f64, shape=(n_surf, 2))


    # @ti.kernel
    # def get_plot_ori(self, elastic1: ti.template(), elastic2: ti.template()):
    #     for ii in self.surface_idx:
    #         i = self.surface_idx[ii]
    #         pos = elastic1.F_x[i]
    #         pos = self.rotmat[None].transpose() @ (pos - self.pos[None])
    #         pos.z -= self.half_gripper_dist[None]
    #         cam_pos = self.upper_camera_pos[None]
    #         rot = self.upper_camera_rot[None]
    #         rotmat = self.quat_to_rotmat(rot)
    #         x, y = self.proj_3d_to_2d(pos, cam_pos, rot, rotmat)
    #         self.upper_ori_plot[ii, 0] = x
    #         self.upper_ori_plot[ii, 1] = y
    #
    #     for ii in self.surface_idx:
    #         i = self.surface_idx[ii]
    #         pos = elastic2.F_x[i]
    #         pos = self.rotmat[None].inverse() @ (pos - self.pos[None])
    #         pos.z += self.half_gripper_dist[None]
    #         cam_pos = self.lower_camera_pos[None]
    #         rot = self.lower_camera_rot[None]
    #         rotmat = self.quat_to_rotmat(rot)
    #         x, y = self.proj_3d_to_2d(pos, cam_pos, rot, rotmat)
    #         self.lower_ori_plot[ii, 0] = x
    #         self.lower_ori_plot[ii, 1] = y
    #
    # @ti.kernel
    # def get_plot_now(self, elastic1: ti.template(), elastic2: ti.template()):
    #     for ii in self.surface_idx:
    #         i = self.surface_idx[ii]
    #         pos = elastic1.F_x[i]
    #         pos = self.rotmat[None].transpose() @ (pos - self.pos[None])
    #         pos.z -= self.half_gripper_dist[None]
    #         cam_pos = self.upper_camera_pos[None]
    #         rot = self.upper_camera_rot[None]
    #         rotmat = self.quat_to_rotmat(rot)
    #         x, y = self.proj_3d_to_2d(pos, cam_pos, rot, rotmat)
    #         self.upper_now_plot[ii, 0] = x
    #         self.upper_now_plot[ii, 1] = y
    #
    #     for ii in self.surface_idx:
    #         i = self.surface_idx[ii]
    #         pos = elastic2.F_x[i]
    #         pos = self.rotmat[None].inverse() @ (pos - self.pos[None])
    #         pos.z += self.half_gripper_dist[None]
    #         cam_pos = self.lower_camera_pos[None]
    #         rot = self.lower_camera_rot[None]
    #         rotmat = self.quat_to_rotmat(rot)
    #         x, y = self.proj_3d_to_2d(pos, cam_pos, rot, rotmat)
    #         self.lower_now_plot[ii, 0] = x
    #         self.lower_now_plot[ii, 1] = y

    @ti.kernel
    def init_kernel(self, sys: ti.template(), pos_array: ti.types.ndarray()):
        for i in ti.static(range(1, self.n_part + 1)):
            self.pos[i - 1] = ti.Vector([pos_array[i - 1, 0], pos_array[i - 1, 1], pos_array[i - 1, 2]])
            self.rot[i - 1] = ti.Vector([1.0, 0., 0., 0.])
            self.half_gripper_dist[i - 1] = 0
        # self.upper_camera_pos[None] = ti.Vector([0, 0, z])
        # self.lower_camera_pos[None] = ti.Vector([0, 0, z1])
        # self.upper_camera_rot[None] = ti.Vector([1.0, 0., 0., 0.])
        # self.lower_camera_rot[None] = ti.Vector([1.0, 0., 0., 0.])

        for i in range(self.n_verts):
            for j in ti.static(range(self.n_part)):
                self.F_x_upper[j, i] = sys.elastics[j * 2 + 1].F_x[i] - self.pos[j]
                self.F_x_lower[j, i] = sys.elastics[j * 2 + 2].F_x[i] - self.pos[j]

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

    def init(self, sys, pos_array):
        self.init_kernel(sys, pos_array)
        self.get_rotmat()
        # self.get_plot_ori(elastic1, elastic2)

    @ti.kernel
    def set(self, pos: ti.template(), rot: ti.template(), step: int):
        for i in range(self.n_part):
            self.pos[i] = pos[step, i]
            self.rot[i] = rot[step, i]

    @ti.kernel
    def get_vert_pos(self):
        for i in range(self.n_verts):
            for j in ti.static(range(self.n_part)):
                self.F_x_upper_world[j, i] = self.pos[j] + self.rotmat[j] @ self.F_x_upper[j, i]
        for i in range(self.n_verts):
            for j in ti.static(range(self.n_part)):
                self.F_x_lower_world[j, i] = self.pos[j] + self.rotmat[j] @ self.F_x_lower[j, i]

    def get_rotmat(self):
        for j in range(self.n_part):
            s = self.rot[j][0]
            x = self.rot[j][1]
            y = self.rot[j][2]
            z = self.rot[j][3]
            self.rotmat[j] = ti.Matrix([[s * s + x * x - y * y - z * z, 2 * (x * y - s * z), 2 * (x * z + s * y)],
                                        [2 * (x * y + s * z), s * s - x * x + y * y - z * z, 2 * (y * z - s * x)],
                                        [2 * (x * z - s * y), 2 * (y * z + s * x), s * s - x * x - y * y + z * z]])
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

            v2 = ti.Vector([self.rot[j][1], self.rot[j][2], self.rot[j][3]])
            real = -delta_rot[j].dot(v2)
            res = self.rot[j][0] * delta_rot[j] + delta_rot[j].cross(v2)

            self.rot[j][0] += real
            self.rot[j][1] += res[0]
            self.rot[j][2] += res[1]
            self.rot[j][3] += res[2]
            self.rot[j] = self.rot[j].normalized()
            # self.half_gripper_dist[j] += delta_dis[j]
            # self.open_gripper(delta_dis[j], j)
        self.get_rotmat()
        self.get_vert_pos()

    def step(self, delta_pos: ti.template(), delta_rot: ti.template(), delta_dis: ti.template()):
        for j in range(self.n_part):
            self.pos[j] += delta_pos[j]

            v2 = ti.Vector([self.rot[j][1], self.rot[j][2], self.rot[j][3]])
            real = -delta_rot[j].dot(v2)
            res = self.rot[j][0] * delta_rot[j] + delta_rot[j].cross(v2)

            self.rot[j][0] += real
            self.rot[j][1] += res[0]
            self.rot[j][2] += res[1]
            self.rot[j][3] += res[2]
            self.rot[j] = self.rot[j].normalized()
            self.half_gripper_dist[j] += delta_dis[j]
            self.open_gripper(delta_dis[j], j)
        self.get_rotmat()
        self.get_vert_pos()

    @ti.kernel
    def open_gripper(self, delta_dis: ti.f64, j: ti.i32):
        for i in range(self.n_verts):
            self.F_x_upper[j, i].z += delta_dis
            self.F_x_lower[j, i].z -= delta_dis

    @ti.kernel
    def gather_grad(self, grad: ti.template(), sys: ti.template()):
        self.d_pos.fill(0)
        self.d_angle.fill(0)

        for i in range(self.n_bound):
            for j in ti.static(range(self.n_part)):
                xx = sys.elastics[j * 2 + 1].offset + self.bound_idx[i]
                gradient = ti.Vector([grad[xx * 3 + 0], grad[xx * 3 + 1], grad[xx * 3 + 2]])
                self.d_pos[j] += gradient
                self.d_angle[j] += (self.rotmat[j] @ self.F_x_upper[j, self.bound_idx[i]]).cross(gradient)

                xx = sys.elastics[j * 2 + 2].offset + self.bound_idx[i]
                gradient = ti.Vector([grad[xx * 3 + 0], grad[xx * 3 + 1], grad[xx * 3 + 2]])
                self.d_pos[j] += gradient
                self.d_angle[j] += (self.rotmat[j] @ self.F_x_lower[j, self.bound_idx[i]]).cross(gradient)

        for j in ti.static(range(self.n_part)):
            self.d_pos[j] /= 2.0 * self.n_bound
            self.d_angle[j] /= 2.0 * self.n_bound
            for k in ti.static(range(3)):
                self.d_pos[j][k] = ti.math.clamp(self.d_pos[j][k], -10, 10)
                self.d_angle[j][k] = ti.math.clamp(self.d_angle[j][k], -10, 10)

    @ti.kernel
    def update_bound(self, sys: ti.template()):
        for i in range(self.n_bound):
            for j in ti.static(range(self.n_part)):
                sys.elastics[j * 2 + 1].F_x[self.bound_idx[i]] = self.F_x_upper_world[j, self.bound_idx[i]]
                sys.elastics[j * 2 + 2].F_x[self.bound_idx[i]] = self.F_x_lower_world[j, self.bound_idx[i]]

    @ti.kernel
    def update_all(self, sys: ti.template()):
        for i in range(self.n_verts):
            for j in ti.static(range(self.n_part)):
                sys.elastics[j * 2 + 1].F_x[i] = self.F_x_upper_world[j, i]
                sys.elastics[j * 2 + 2].F_x[i] = self.F_x_lower_world[j, i]

    def save_all(self, path):
        nps = self.F_x_upper.to_numpy()
        np.save(os.path.join(path, 'F_x_upper.npy'), nps)
        nps = self.F_x_upper_world.to_numpy()
        np.save(os.path.join(path, 'F_x_upper_world.npy'), nps)
        nps = self.F_x_lower.to_numpy()
        np.save(os.path.join(path, 'F_x_lower.npy'), nps)
        nps = self.F_x_lower_world.to_numpy()
        np.save(os.path.join(path, 'F_x_lower_world.npy'), nps)
        nps = self.pos.to_numpy()
        np.save(os.path.join(path, 'pos.npy'), nps)
        nps = self.rot.to_numpy()
        np.save(os.path.join(path, 'rot.npy'), nps)
        nps = self.rotmat.to_numpy()
        np.save(os.path.join(path, 'rotmat.npy'), nps)
        nps = self.half_gripper_dist.to_numpy()
        np.save(os.path.join(path, 'half_gripper_dist.npy'), nps)

    def load_all(self, path):
        npl = np.load(os.path.join(path, 'F_x_upper.npy'))
        self.F_x_upper.from_numpy(npl)
        npl = np.load(os.path.join(path, 'F_x_upper_world.npy'))
        self.F_x_upper_world.from_numpy(npl)
        npl = np.load(os.path.join(path, 'F_x_lower.npy'))
        self.F_x_lower.from_numpy(npl)
        npl = np.load(os.path.join(path, 'F_x_lower_world.npy'))
        self.F_x_lower_world.from_numpy(npl)
        npl = np.load(os.path.join(path, 'pos.npy'))
        self.pos.from_numpy(npl)
        npl = np.load(os.path.join(path, 'rot.npy'))
        self.rot.from_numpy(npl)
        npl = np.load(os.path.join(path, 'rotmat.npy'))
        self.rotmat.from_numpy(npl)
        npl = np.load(os.path.join(path, 'half_gripper_dist.npy'))
        self.half_gripper_dist.from_numpy(npl)

    # @ti.func
    # def proj_3d_to_2d(self, pos, cam_pos, cam_rot, rotmat):
    #     # rotmat = self.quat_to_rotmat(cam_rot)
    #     d_pos = pos - cam_pos
    #     x_axis = ti.Vector([rotmat[0, 0], rotmat[0, 1], rotmat[0, 2]])
    #     y_axis = ti.Vector([rotmat[1, 0], rotmat[1, 1], rotmat[1, 2]])
    #     x = d_pos.dot(x_axis)
    #     y = d_pos.dot(y_axis)
    #     return x, y
    #
    # @ti.func
    # def grad_2d_to_3d(self, pos, cam_pos, cam_rot, rotmat, gradx, grady):
    #     # rotmat = self.quat_to_rotmat(cam_rot)
    #     d_pos = pos - cam_pos
    #     x_axis = ti.Vector([rotmat[0, 0], rotmat[0, 1], rotmat[0, 2]])
    #     y_axis = ti.Vector([rotmat[1, 0], rotmat[1, 1], rotmat[1, 2]])
    #
    #     grad_cam_pos = -gradx * x_axis - grady * y_axis
    #     grad_pos = gradx * x_axis + grady * y_axis
    #
    #     grad_x_axis = gradx * d_pos
    #     grad_y_axis = grady * d_pos
    #     s = cam_rot[0]
    #     x = cam_rot[1]
    #     y = cam_rot[2]
    #     z = cam_rot[3]
    #     # ti.Matrix([[s * s + x * x - y * y - z * z, 2 * (x * y - s * z), 2 * (x * z + s * y)],
    #     #            [2 * (x * y + s * z), s * s - x * x + y * y - z * z, 2 * (y * z - s * x)],
    #     #            [2 * (x * z - s * y), 2 * (y * z + s * x), s * s - x * x - y * y + z * z]])
    #     grad_s = grad_x_axis[0] * 2 * s - grad_x_axis[1] * 2 * z + grad_x_axis[2] * 2 * y + grad_y_axis[0] * 2 * z - grad_y_axis[1] * 2 * s - grad_y_axis[2] * 2 * x
    #     grad_x = grad_x_axis[0] * 2 * x - grad_x_axis[1] * 2 * y + grad_x_axis[2] * 2 * z + grad_y_axis[0] * 2 * y - \
    #              grad_y_axis[1] * 2 * x - grad_y_axis[2] * 2 * s
    #     grad_y = -grad_x_axis[0] * 2 * y + grad_x_axis[1] * 2 * x + grad_x_axis[2] * 2 * s + grad_y_axis[0] * 2 * x + \
    #              grad_y_axis[1] * 2 * y + grad_y_axis[2] * 2 * z
    #     grad_z = -grad_x_axis[0] * 2 * z - grad_x_axis[1] * 2 * s + grad_x_axis[2] * 2 * x + grad_y_axis[0] * 2 * s - \
    #              grad_y_axis[1] * 2 * z + grad_y_axis[2] * 2 * y
    #     grad_cam_rot = ti.Vector([grad_s, grad_x, grad_y, grad_z])
    #
    #     return grad_pos, grad_cam_pos, grad_cam_rot
    #
    # def plot_tactile(self, frame, elastic1, elastic2):
    #     self.get_rotmat()
    #     self.get_plot_now(elastic1, elastic2)
    #     # self.print_plot()
    #     self.temp_plot.copy_from(self.upper_ori_plot)
    #     ax = plt.figure().add_subplot()
    #     x = self.temp_plot.to_torch()
    #     now_x = self.upper_now_plot.to_torch()
    #     delta = now_x - x
    #     ax.scatter(x[:, 0], x[:, 1])
    #     ax.quiver(x[:, 0], x[:, 1], delta[:, 0], delta[:, 1], color='red')
    #     plt.draw()
    #     plt.savefig(f'../imgs/contact_upper_{frame}.png')
    #     plt.close()
    #     # self.print_plot()
    #
    # def print_plot(self):
    #     print(self.upper_ori_plot[0, 0])
    #     # for i in range(self.n_surf):
    #     #     if ti.abs(self.upper_ori_plot[i, 0] - self.upper_now_plot[i, 0]) > 0.001:
    #     #         print("broken", i, 0)
    #     #     if ti.abs(self.upper_ori_plot[i, 1] - self.upper_now_plot[i, 1]) > 0.001:
    #     #         print("broken", i, 1)