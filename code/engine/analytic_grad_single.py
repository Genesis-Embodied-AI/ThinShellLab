import taichi as ti

@ti.data_oriented
class Grad:
    def __init__(self, sys, tot_timestep, n_parts, friction_loss=False, f_loss_ratio=0.001, vertical_only=False):
        self.n_part = n_parts
        self.tot_NV = sys.tot_NV
        self.pos_buffer = ti.field(ti.f64, (tot_timestep, sys.tot_NV, 3))
        self.gripper_pos_buffer = ti.Vector.field(3, dtype=ti.f64, shape=(tot_timestep, n_parts))
        self.gripper_rot_buffer = ti.Vector.field(4, dtype=ti.f64, shape=(tot_timestep, n_parts))
        self.ref_angle_buffer = ti.Vector.field(3, ti.f64, shape=(tot_timestep, sys.cloth_cnt, sys.cloths[0].NF))
        self.dt = sys.dt
        self.pos_grad = ti.field(ti.f64, (tot_timestep, sys.tot_NV, 3))
        self.x_hat_grad = ti.field(ti.f64, sys.tot_NV * 3)
        self.gripper_grad = ti.field(dtype=ti.f64, shape=(tot_timestep, n_parts, 6))
        self.angleref_grad = ti.Vector.field(3, ti.f64, shape=(tot_timestep, sys.cloth_cnt, sys.cloths[0].NF))
        self.cloth_cnt = sys.cloth_cnt
        self.NF = sys.cloths[0].NF
        self.mass = ti.field(ti.f64, sys.tot_NV)
        self.F = ti.field(ti.f64, shape=(self.tot_NV * 3,))
        self.tot_timestep = tot_timestep
        self.damping = 1.0
        self.friction_loss = friction_loss
        self.f_loss_ratio = f_loss_ratio
        self.vertical_only = vertical_only

    def reset(self):
        self.pos_buffer.fill(0)
        self.pos_grad.fill(0)
        self.angleref_grad.fill(0)

    @ti.kernel
    def init_mass(self, sys: ti.template()):
        for i in range(self.tot_NV):
            self.mass[i] = sys.mass[i]

    @ti.kernel
    def copy_pos(self, sys: ti.template(), step: int):
        for i in range(self.tot_NV):
            self.pos_buffer[step, i, 0] = sys.pos[i][0]
            self.pos_buffer[step, i, 1] = sys.pos[i][1]
            self.pos_buffer[step, i, 2] = sys.pos[i][2]

        for i in range(sys.cloths[0].NF):
            for j in ti.static(range(sys.cloth_cnt)):
                for l in ti.static(range(3)):
                    self.ref_angle_buffer[step, j, i][l] = sys.cloths[j].ref_angle[i][l]

        for j in ti.static(range(self.n_part)):
            self.gripper_pos_buffer[step, j] = sys.gripper.pos[j]
            self.gripper_rot_buffer[step, j] = sys.gripper.rot[j]



    @ti.kernel
    def get_F(self, step: int, sys: ti.template()):
        for i in range(self.tot_NV):
            self.F[i * 3 + 0] = self.pos_grad[step, i, 0]
            self.F[i * 3 + 1] = self.pos_grad[step, i, 1]
            self.F[i * 3 + 2] = self.pos_grad[step, i, 2]

        # for i in range(sys.gripper.n_bound):
        #     for j in ti.static(range(self.n_part)):
        #         xx = sys.gripper.bound_idx[i] + sys.elastics[j].offset
        #         self.F[xx * 3 + 0] = 0
        #         self.F[xx * 3 + 1] = 0
        #         self.F[xx * 3 + 2] = 0

    @ti.kernel
    def check_nan_F(self):
        has_nan = False
        for i in range(self.tot_NV):
            for j in range(3):
                if ti.math.isnan(self.F[i * 3 + j]):
                    has_nan = True

        if has_nan:
            print("nan in F!!!!")


    @ti.kernel
    def get_grad(self, p: ti.types.ndarray()):
        has_nan = False
        for i in range(self.tot_NV):
            self.x_hat_grad[i * 3 + 0] = p[i * 3 + 0] * self.mass[i] / (self.dt ** 2)
            self.x_hat_grad[i * 3 + 1] = p[i * 3 + 1] * self.mass[i] / (self.dt ** 2)
            self.x_hat_grad[i * 3 + 2] = p[i * 3 + 2] * self.mass[i] / (self.dt ** 2)
            for j in range(3):
                if ti.math.isnan(self.x_hat_grad[i * 3 + j]):
                    has_nan = True
        if has_nan:
            print("nan in get grad func!!!")

    @ti.kernel
    def get_prev_grad(self, sys: ti.template(), step: int):
        for i in range(self.tot_NV):
            for j in ti.static(range(3)):
                if not sys.frozen[i * 3 + j]:
                    self.pos_grad[step - 1, i, j] += self.x_hat_grad[i * 3 + j] * (1 + self.damping)

    @ti.kernel
    def get_prev_prev_grad(self, sys: ti.template(), step: int):
        for i in range(self.tot_NV):
            for j in ti.static(range(3)):
                if not sys.frozen[i * 3 + j]:
                    self.pos_grad[step - 2, i, j] -= self.x_hat_grad[i * 3 + j] * self.damping

    @ti.kernel
    def check_nan_gripper(self, step: int):
        has_nan = False
        for j in range(self.n_part):
            for k in range(6):
                if ti.math.isnan(self.gripper_grad[step, j, k]):
                    has_nan = True
        if has_nan:
            print("nan after computing gripper grade!!!")

    def get_gripper_grad(self, step: int, sys: ti.template()):
        sys.gripper.get_rotmat()
        sys.gripper.gather_grad(sys.tmp_z_frozen, sys)
        ifprint = False
        if ifprint:
            print("gripper grad:", end=" ")
        for j in range(self.n_part):
            # print(j, end=" ")
            if self.vertical_only:
                self.gripper_grad[step, j, 2] = sys.gripper.d_pos[j][2]
            else:
                self.gripper_grad[step, j, 0] = sys.gripper.d_pos[j][0]
                self.gripper_grad[step, j, 1] = sys.gripper.d_pos[j][1]
                self.gripper_grad[step, j, 2] = sys.gripper.d_pos[j][2]
                self.gripper_grad[step, j, 3] = sys.gripper.d_angle[j][0]
                self.gripper_grad[step, j, 4] = sys.gripper.d_angle[j][1]
                self.gripper_grad[step, j, 5] = sys.gripper.d_angle[j][2]
            for k in range(6):
                if ifprint and j == 1:
                    print(self.gripper_grad[step, j, k], end=" ")
        if ifprint:
            print("")

        # self.check_nan_gripper(step)
        # print(f"gripper grad step {step}: {self.gripper_grad_step.to_numpy()}")

    @ti.kernel
    def check_nan(self, step: int):
        ret = False
        for i in range(self.tot_NV):
            if ti.math.isnan(self.pos_grad[step, i, 0]):
                ret = True
            if ti.math.isnan(self.pos_grad[step, i, 1]):
                ret = True
            if ti.math.isnan(self.pos_grad[step, i, 2]):
                ret = True

        for i in range(self.NF):
            for j in range(self.cloth_cnt):
                for l in range(3):
                    if ti.math.isnan(self.angleref_grad[step, j, i][l]):
                        ret = True
        if ret:
            print("has nan", step)

    @ti.kernel
    def max_abs(self, step: int, sys: ti.template()):
        ret = 0.0
        for i in range(sys.elastics[1].offset, self.tot_NV):
            if not sys.frozen[i * 3 + 0]:
                ti.atomic_max(ret, ti.abs(self.x_hat_grad[i * 3 + 0]))
            if not sys.frozen[i * 3 + 1]:
                ti.atomic_max(ret, ti.abs(self.x_hat_grad[i * 3 + 1]))
            if not sys.frozen[i * 3 + 2]:
                ti.atomic_max(ret, ti.abs(self.x_hat_grad[i * 3 + 2]))

        print("max_abs:", step, ret)

    @ti.kernel
    def clamp_grad(self, step: int):
        for i in range(self.tot_NV):
            self.pos_grad[step, i, 0] = ti.math.clamp(self.pos_grad[step, i, 0], -1000, 1000)
            self.pos_grad[step, i, 1] = ti.math.clamp(self.pos_grad[step, i, 1], -1000, 1000)
            self.pos_grad[step, i, 2] = ti.math.clamp(self.pos_grad[step, i, 2], -1000, 1000)
        for i in range(self.NF):
            for j in range(self.cloth_cnt):
                for l in range(3):
                    self.angleref_grad[step, j, i][l] = ti.math.clamp(self.angleref_grad[step, j, i][l], -1000, 1000)

    # @ti.kernel
    def print_grad(self, step: int, sys: ti.template()):
        ret = 0.0
        print("contact points grad on x, step", step)
        idx = sys.const_idx[0]
        print("elastic grad pos:", self.pos_grad[step, idx[3], 0])
        print("elastic grad x_hat:", self.x_hat_grad[idx[3] * 3 + 0])
        print("cloth grad x_hat:")
        print(self.x_hat_grad[idx[0] * 3 + 0], self.x_hat_grad[idx[1] * 3 + 0], self.x_hat_grad[idx[2] * 3 + 0])
        # for i in range(sys.nc1[None]):
        #     idx = sys.const_idx[i]
        print("cloth grad pos:")
        print(self.pos_grad[step, idx[0], 0], self.pos_grad[step, idx[1], 0], self.pos_grad[step, idx[2], 0])


    @ti.kernel
    def calc_norm(self, p: ti.template()) -> ti.f64:
        F_norm = 0.0
        for i in range(self.tot_NV):
            ti.atomic_max(F_norm, ti.abs(p[i * 3 + 0]))
            ti.atomic_max(F_norm, ti.abs(p[i * 3 + 1]))
            ti.atomic_max(F_norm, ti.abs(p[i * 3 + 2]))
        return F_norm

    @ti.kernel
    def copy_z(self, sys: ti.template(), p: ti.types.ndarray()):
        for i in range(self.tot_NV*3):
            sys.tmp_z_not_frozen[i] = p[i]


    def transfer_grad(self, step, sys, f_contact):

        self.clamp_grad(step)
        if_print = False
        sys.copy_pos_only(self.pos_buffer, step - 1)
        sys.calc_vn()
        f_contact(sys)
        sys.contact_analysis()

        sys.copy_pos_and_refangle(self, step)
        sys.gripper.set(self.gripper_pos_buffer, self.gripper_rot_buffer, step)
        sys.init_folding()
        sys.ref_angle_backprop_a2ax(self, step)

        # sys.static_friction_loss(self, step)
        # sys.copy_refangle(self, step - 1)
        # sys.init_folding()
        sys.H.clear_all()
        sys.compute_Hessian(False)
        self.get_F(step, sys)
        F_array = self.F.to_torch(device='cuda:0')
        p = sys.H.solve(F_array)
        self.copy_z(sys, p)
        sys.tmp_z_frozen.fill(0)
        sys.counting_z_frozen[None] = True
        sys.compute_Hessian(False)
        sys.counting_z_frozen[None] = False
        self.get_grad(p)
        if if_print:
            self.max_abs(step, sys)
        sys.contact_energy_backprop(True, self, step - 1, p)
        sys.ref_angle_backprop_x2a(self, step, p)
        if if_print:
            self.print_grad(step, sys)
        if step > 0:
            self.get_prev_grad(sys, step)
            self.get_gripper_grad(step, sys)
        if step > 1:
            self.get_prev_prev_grad(sys, step)

        # self.check_nan(step)

    @ti.kernel
    def get_loss(self, sys: ti.template()):
        for i, j in ti.ndrange(sys.cloths[0].NV, self.tot_timestep):
            # self.pos_grad[j, i, 2] = -0.5
            self.pos_grad[j, i, 0] = -1

    @ti.kernel
    def get_loss_sheet(self, sys: ti.template()):
        for i, j in ti.ndrange(sys.cloths[0].NV, self.tot_timestep - 1):
            # self.pos_grad[j * 10 + 9, i, 2] = -0.05 * j / 8.0
            self.pos_grad[j + 1, i, 0] = 1
        # for i in ti.ndrange(sys.cloths[0].NV):
        #     # self.pos_grad[j * 10 + 9, i, 2] = -0.05 * j / 8.0
        #     self.pos_grad[29, i, 0] = 1

    @ti.kernel
    def get_loss_book(self, sys: ti.template()):
        for i, j in ti.ndrange(sys.cloths[0].NV, self.tot_timestep - 1):
            # self.pos_grad[j, i, 2] = -0.5
            self.pos_grad[j + 1, i, 0] = -1

    @ti.kernel
    def get_loss_fold(self, sys: ti.template(), curve7: ti.f64, curve8: ti.f64):
        # for i, j in ti.ndrange(sys.cloths[0].NV, self.tot_timestep):
            # self.pos_grad[j * 10 + 9, i, 2] = fr-0.05 * j / 8.0
        for i in range(sys.cloths[0].NF):
            for l in range(3):
                if sys.cloths[0].counter_face[i][l] > i:
                    p1 = sys.cloths[0].f2v[i][l]
                    p2 = sys.cloths[0].f2v[sys.cloths[0].counter_face[i][l]][sys.cloths[0].counter_point[i][l]]
                    if ti.cast(p1 / (sys.cloths[0].M + 1), ti.i32) == 6 and ti.cast(p2 / (sys.cloths[0].M + 1), ti.i32) == 8:
                        self.angleref_grad[self.tot_timestep - 1, 0, i][l] = curve7
                    if ti.cast(p1 / (sys.cloths[0].M + 1), ti.i32) == 7 and ti.cast(p2 / (sys.cloths[0].M + 1), ti.i32) == 9:
                        self.angleref_grad[self.tot_timestep - 1, 0, i][l] = curve8
                    # else:
                    #     self.angleref_grad[self.tot_timestep - 1, 0, i][l] = sys.cloths[0].ref_angle[i][l]

    @ti.kernel
    def get_loss_push(self, sys: ti.template(), target_pos: ti.types.ndarray()):
        for j, k in ti.ndrange(sys.cloths[0].NV, 3):
            self.pos_grad[self.tot_timestep - 1, sys.cloths[0].offset + j, k] = 2 * (
                        self.pos_buffer[self.tot_timestep - 1, sys.cloths[0].offset + j, k] - target_pos[j, k])

    @ti.kernel
    def get_loss_lift(self, sys: ti.template()):
        j = self.tot_timestep - 1
        for i in range(sys.elastics[0].n_verts):
            self.pos_grad[j, sys.elastics[0].offset + i, 0] = (self.pos_buffer[j, sys.elastics[0].offset + i, 0] - self.pos_buffer[0, sys.elastics[0].offset + i, 0] + 0.012)
            self.pos_grad[j, sys.elastics[0].offset + i, 1] = (
                        self.pos_buffer[j, sys.elastics[0].offset + i, 1] - self.pos_buffer[
                    0, sys.elastics[0].offset + i, 1] + 0.012)
            self.pos_grad[j, sys.elastics[0].offset + i, 2] = (
                        self.pos_buffer[j, sys.elastics[0].offset + i, 2] - self.pos_buffer[
                    0, sys.elastics[0].offset + i, 2])

    @ti.kernel
    def get_loss_sep(self, sys: ti.template()):
        for i, j in ti.ndrange(sys.cloths[0].NV, self.tot_timestep):
            self.pos_grad[j, sys.cloths[0].offset + i, 0] = 1
            # self.pos_grad[j, sys.cloths[0].offset + i, 2] = self.pos_buffer[j, sys.cloths[0].offset + i, 2] * 10
        for i, j in ti.ndrange(sys.cloths[1].NV, self.tot_timestep):
            self.pos_grad[j, sys.cloths[1].offset + i, 0] = -1
            # self.pos_grad[j, sys.cloths[1].offset + i, 2] = (self.pos_buffer[j, sys.cloths[1].offset + i, 2] + 0.0004) * 10

    @ti.kernel
    def get_loss_pick(self, sys: ti.template()):
        for i, j in ti.ndrange(sys.cloths[0].NV, self.tot_timestep):
            if ti.cast(i / (sys.cloths[0].M + 1), ti.i32) == 8:
                self.pos_grad[j, sys.cloths[0].offset + i, 2] = -1

    def get_loss_bounce(self, sys: ti.template()):
        # for i in range(sys.cloths[0].M + 1):
        #     self.pos_grad[self.tot_timestep - 1, sys.cloths[0].offset + i, 2] = 2 * (self.pos_buffer[self.tot_timestep - 1, sys.cloths[0].offset + i, 2] - sys.target)

        tt = self.tot_timestep - 1
        max_z = -1.0
        for j in range(40, self.tot_timestep):
            now_z = 0
            for i in range(sys.cloths[0].M + 1):
                now_z += self.pos_buffer[j, i + sys.cloths[0].offset, 2]
            if now_z > max_z:
                max_z = now_z
                tt = j

        if (tt < self.tot_timestep - 1):

            z_prev = 0.0
            z_next = 0.0
            for i in range(sys.cloths[0].M + 1):
                z_prev += self.pos_buffer[tt - 1, i + sys.cloths[0].offset, 2]
                z_next += self.pos_buffer[tt + 1, i + sys.cloths[0].offset, 2]

            if z_prev > z_next:
                for i in range(sys.cloths[0].M + 1):
                    self.pos_grad[tt - 1, sys.cloths[0].offset + i, 2] = 2 * (
                            self.pos_buffer[tt - 1, sys.cloths[0].offset + i, 2] - sys.target)
            else:
                for i in range(sys.cloths[0].M + 1):
                    self.pos_grad[tt + 1, sys.cloths[0].offset + i, 2] = 2 * (
                            self.pos_buffer[tt + 1, sys.cloths[0].offset + i, 2] - sys.target)

        # for i in range(sys.cloths[0].M + 1):
        #     self.pos_grad[tt - 1, sys.cloths[0].offset + i, 2] = 2 * (
        #             self.pos_buffer[tt - 1, sys.cloths[0].offset + i, 2] - sys.target)
        #
        # if (tt < self.tot_timestep - 1):
        #     for i in range(sys.cloths[0].M + 1):
        #         self.pos_grad[tt, sys.cloths[0].offset + i, 2] = 2 * (
        #                     self.pos_buffer[tt, sys.cloths[0].offset + i, 2] - sys.target)
        #     tt = tt + 1
        for i in range(sys.cloths[0].M + 1):
            self.pos_grad[tt, sys.cloths[0].offset + i, 2] = 2 * (self.pos_buffer[tt, sys.cloths[0].offset + i, 2] - sys.target)
        return tt

    @ti.kernel
    def get_loss_pick_fold(self, sys: ti.template()):
        for i, j in ti.ndrange(sys.cloths[0].NF, self.tot_timestep):
            for l in range(3):
                if sys.cloths[0].counter_face[i][l] > i:
                    p1 = sys.cloths[0].f2v[i][l]
                    p2 = sys.cloths[0].f2v[sys.cloths[0].counter_face[i][l]][sys.cloths[0].counter_point[i][l]]
                    if ti.cast(p1 / (sys.cloths[0].M + 1), ti.i32) == 7 and ti.cast(p2 / (sys.cloths[0].M + 1),
                                                                                    ti.i32) == 9:
                        self.angleref_grad[j, 0, i][l] = -1

    @ti.kernel
    def get_loss_card(self, sys: ti.template()):
        for i, j in ti.ndrange(sys.cloths[0].NV, self.tot_timestep):
            if ti.cast(i / (sys.cloths[0].M + 1), ti.i32) == 8:
                self.pos_grad[j, sys.cloths[0].offset + i, 2] = -1

    @ti.kernel
    def get_loss_slide_simple(self, sys: ti.template()):
        for i in range(sys.cloths[0].NV):
            self.pos_grad[self.tot_timestep - 1, sys.cloths[0].offset + i, 0] = 1

    @ti.kernel
    def get_loss_deliver(self, sys: ti.template()):
        for i in range(sys.cloths[0].NV):
            self.pos_grad[self.tot_timestep - 1, sys.cloths[0].offset + i, 0] = 2 * (
                        self.pos_buffer[self.tot_timestep - 1, sys.cloths[0].offset + i, 0] - self.pos_buffer[
                    69, sys.cloths[0].offset + i, 0] - 0.01)
            self.pos_grad[self.tot_timestep - 1, sys.cloths[0].offset + i, 1] = 2 * (
                        self.pos_buffer[self.tot_timestep - 1, sys.cloths[0].offset + i, 1] - self.pos_buffer[
                    69, sys.cloths[0].offset + i, 1] - 0.01)
            self.pos_grad[self.tot_timestep - 1, sys.cloths[0].offset + i, 2] = 2 * (
                        self.pos_buffer[self.tot_timestep - 1, sys.cloths[0].offset + i, 2] - self.pos_buffer[
                    69, sys.cloths[0].offset + i, 2] - 0.01)

    @ti.kernel
    def get_loss_interact(self, sys: ti.template()):
        # for i, j in ti.ndrange(sys.cloths[0].NV, self.tot_timestep):
        #     self.pos_grad[j, sys.cloths[0].offset + i, 0] = 1
        #
        # for i, j in ti.ndrange(sys.elastics[3].n_verts, self.tot_timestep):
        #     self.pos_grad[j, sys.elastics[3].offset + i, 0] = -1 * 256.0 / 144.0

        for i in range(sys.cloths[0].NV):
            self.pos_grad[self.tot_timestep - 1, sys.cloths[0].offset + i, 0] = 1

        for i in range(sys.elastics[3].n_verts):
            self.pos_grad[self.tot_timestep - 1, sys.elastics[3].offset + i, 0] = -1 * 256.0 / 144.0

    @ti.kernel
    def get_loss_interact_1(self, sys: ti.template()):

        for i in range(sys.elastics[3].n_verts):
            self.pos_grad[self.tot_timestep - 1, sys.elastics[3].offset + i, 0] = 1

    @ti.kernel
    def get_loss_balance(self, sys: ti.template()):
        tt = (sys.cloth_N + 1) // 2 * (sys.cloth_M + 1) + (sys.cloth_M + 1) // 2
        for i, j in ti.ndrange(sys.elastics[0].n_verts, self.tot_timestep - 1):
            self.pos_grad[j + 1, sys.elastics[0].offset + i, 0] = 2 * (
                        self.pos_buffer[j + 1, sys.elastics[0].offset + i, 0] - self.pos_buffer[
                    j + 1, sys.cloths[0].offset + tt, 0])
            self.pos_grad[j + 1, sys.elastics[0].offset + i, 1] = 2 * (
                        self.pos_buffer[j + 1, sys.elastics[0].offset + i, 1] - self.pos_buffer[
                    j + 1, sys.cloths[0].offset + tt, 1])
            self.pos_grad[j + 1, sys.cloths[0].offset + tt, 0] = -2 * (
                        self.pos_buffer[j + 1, sys.elastics[0].offset + i, 0] - self.pos_buffer[
                    j + 1, sys.cloths[0].offset + tt, 0])
            self.pos_grad[j + 1, sys.cloths[0].offset + tt, 1] = -2 * (
                    self.pos_buffer[j + 1, sys.elastics[0].offset + i, 1] - self.pos_buffer[
                j + 1, sys.cloths[0].offset + tt, 1])

    @ti.kernel
    def get_loss_side(self, sys: ti.template()):
        tt = (sys.cloth_N + 1) // 4 * (sys.cloth_M + 1) + (sys.cloth_M + 1) // 2
        for i, j in ti.ndrange(sys.elastics[0].n_verts, self.tot_timestep - 1):
            self.pos_grad[j + 1, sys.elastics[0].offset + i, 0] = 2 * (
                    self.pos_buffer[j + 1, sys.elastics[0].offset + i, 0] - self.pos_buffer[
                j + 1, sys.cloths[0].offset + tt, 0])
            self.pos_grad[j + 1, sys.elastics[0].offset + i, 1] = 2 * (
                    self.pos_buffer[j + 1, sys.elastics[0].offset + i, 1] - self.pos_buffer[
                j + 1, sys.cloths[0].offset + tt, 1])
            self.pos_grad[j + 1, sys.cloths[0].offset + tt, 0] = -2 * (
                    self.pos_buffer[j + 1, sys.elastics[0].offset + i, 0] - self.pos_buffer[
                j + 1, sys.cloths[0].offset + tt, 0])
            self.pos_grad[j + 1, sys.cloths[0].offset + tt, 1] = -2 * (
                    self.pos_buffer[j + 1, sys.elastics[0].offset + i, 1] - self.pos_buffer[
                j + 1, sys.cloths[0].offset + tt, 1])

    @ti.kernel
    def get_loss_throwing(self, sys: ti.template()):

        for i, j in ti.ndrange(sys.elastics[0].n_verts, self.tot_timestep - 1):
            self.pos_grad[j + 1, sys.elastics[0].offset + i, 2] = -1

        for i, j in ti.ndrange(sys.cloth_M, self.tot_timestep - 1):
            self.pos_grad[j + 1, sys.cloths[0].offset + i, 2] = 20 * self.pos_buffer[j + 1, sys.cloths[0].offset + i, 2]
            self.pos_grad[j + 1, sys.cloths[0].offset + i + sys.cloth_N * (sys.cloth_M + 1), 2] = 20 * self.pos_buffer[
                j + 1, sys.cloths[0].offset + i + sys.cloth_N * (sys.cloth_M + 1), 2]


    @ti.kernel
    def print_out(self, sys: ti.template()):
        cnt = 0
        for i in range(sys.cloths[0].NV):
            if self.pos_grad[self.tot_timestep - 2, i, 0] < 0:
               cnt += 1
        print("inverse cnt", cnt)

    @ti.kernel
    def smooth_grad_kernel(self):
        for i in range(1, 46):
            for j in range(7):
                ret = 0.0
                for k in range(4):
                    ret += self.gripper_grad_old[i + k, j]
                ret /= 4.0
                self.gripper_grad[i, j] = ret

    def accumulate_gripper_grad(self, traj, max_dist):
        for step in range(self.tot_timestep - 2, 1, -1):
            for j in range(self.n_part):
                # print(j, end=" ")
                if traj.calculate_dist(step + 1, max_dist, j) > traj.max_moving_dist - 0.00005:
                    self.gripper_grad[step, j, 0] += self.gripper_grad[step + 1, j, 0]
                    self.gripper_grad[step, j, 1] += self.gripper_grad[step + 1, j, 1]
                    self.gripper_grad[step, j, 2] += self.gripper_grad[step + 1, j, 2]
                    self.gripper_grad[step, j, 3] += self.gripper_grad[step + 1, j, 3]
                    self.gripper_grad[step, j, 4] += self.gripper_grad[step + 1, j, 4]
                    self.gripper_grad[step, j, 5] += self.gripper_grad[step + 1, j, 5]

    def apply_action_limit_grad(self, traj, max_dist):
        for step in range(1, self.tot_timestep):
            for j in range(self.n_part):
                # print(j, end=" ")
                dist = traj.calculate_dist(step, max_dist, j)
                if dist > traj.max_moving_dist:
                    self.gripper_grad[step, j, 0] += (traj.traj[step, j, 0] - traj.traj[step - 1, j, 0]) * (dist - traj.max_moving_dist) * 10000000
                    # print(f"step : {step}, part {j}, grad: {self.gripper_grad[step, j, 0]}")
                    self.gripper_grad[step, j, 1] += (traj.traj[step, j, 1] - traj.traj[step - 1, j, 1]) * (dist - traj.max_moving_dist) * 10000000
                    self.gripper_grad[step, j, 2] += (traj.traj[step, j, 2] - traj.traj[step - 1, j, 2]) * (dist - traj.max_moving_dist) * 10000000
                    self.gripper_grad[step, j, 3] += (traj.traj[step, j, 3] - traj.traj[step - 1, j, 3]) * (dist - traj.max_moving_dist) * 100000
                    self.gripper_grad[step, j, 4] += (traj.traj[step, j, 4] - traj.traj[step - 1, j, 4]) * (dist - traj.max_moving_dist) * 100000
                    self.gripper_grad[step, j, 5] += (traj.traj[step, j, 5] - traj.traj[step - 1, j, 5]) * (dist - traj.max_moving_dist) * 100000




