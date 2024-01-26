import taichi as ti

@ti.data_oriented
class Grad:
    def __init__(self, sys, tot_timestep, n_parts):
        self.tot_NV = sys.tot_NV
        self.n_part = n_parts
        self.pos_buffer = ti.field(ti.f64, (tot_timestep, sys.tot_NV, 3))
        if n_parts > 0:
            self.gripper_pos_buffer = ti.Vector.field(3, dtype=ti.f64, shape=(tot_timestep, n_parts))
            self.gripper_rot_buffer = ti.Vector.field(4, dtype=ti.f64, shape=(tot_timestep, n_parts))
        self.ref_angle_buffer = ti.Vector.field(3, ti.f64, shape=(tot_timestep, sys.cloth_cnt, sys.cloths[0].NF))
        self.dt = sys.dt
        self.pos_grad = ti.field(ti.f64, (tot_timestep, sys.tot_NV, 3))
        self.x_hat_grad = ti.field(ti.f64, sys.tot_NV * 3)
        self.gripper_grad = ti.field(dtype=ti.f64, shape=(tot_timestep, 7))
        self.mass = ti.field(ti.f64, sys.tot_NV)
        self.F = ti.field(ti.f64, shape=(self.tot_NV * 3,))
        self.tot_timestep = tot_timestep
        self.grad_lam = ti.field(ti.f64, ())
        self.grad_mu = ti.field(ti.f64, ())
        self.grad_friction_coef = ti.field(ti.f64, ())
        self.grad_kb = ti.field(ti.f64, ())
        self.angleref_grad = ti.Vector.field(3, ti.f64, shape=(tot_timestep, sys.cloth_cnt, sys.cloths[0].NF))
        self.cloth_cnt = sys.cloth_cnt
        self.damping = 1.0
        self.count_friction_grad = False
        self.count_mu_lam_grad = False
        self.count_kb_grad = True

    def reset(self):
        self.pos_buffer.fill(0)
        self.pos_grad.fill(0)
        self.grad_mu[None] = 0
        self.grad_lam[None] = 0
        self.grad_friction_coef[None] = 0
        self.grad_kb[None] = 0

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

    @ti.kernel
    def get_parameters_grad(self, p: ti.types.ndarray(), sys: ti.template()):
        for i in range(self.tot_NV):
            for j in ti.static(range(3)):
                if self.count_mu_lam_grad:
                    if not sys.frozen[i * 3 + j]:
                        self.grad_mu[None] += p[i * 3 + j] * sys.d_mu[i][j]

                        self.grad_lam[None] += p[i * 3 + j] * sys.d_lam[i][j]
                if self.count_kb_grad:
                    if not sys.frozen[i * 3 + j]:
                        self.grad_kb[None] += p[i * 3 + j] * sys.d_kb[i][j]


    @ti.kernel
    def get_grad(self, p: ti.types.ndarray()):
        for i in range(self.tot_NV):
            self.x_hat_grad[i * 3 + 0] = p[i * 3 + 0] * self.mass[i] / (self.dt ** 2)
            self.x_hat_grad[i * 3 + 1] = p[i * 3 + 1] * self.mass[i] / (self.dt ** 2)
            self.x_hat_grad[i * 3 + 2] = p[i * 3 + 2] * self.mass[i] / (self.dt ** 2)

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
    def clamp_grad(self, step: int):
        for i in range(self.tot_NV):
            self.pos_grad[step, i, 0] = ti.math.clamp(self.pos_grad[step, i, 0], -1, 1)
            self.pos_grad[step, i, 1] = ti.math.clamp(self.pos_grad[step, i, 1], -1, 1)
            self.pos_grad[step, i, 2] = ti.math.clamp(self.pos_grad[step, i, 2], -1, 1)

    @ti.kernel
    def copy_z(self, sys: ti.template(), p: ti.types.ndarray()):
        for i in range(self.tot_NV * 3):
            sys.tmp_z_not_frozen[i] = p[i]

    def transfer_grad(self, step, sys, f_contact):

        self.clamp_grad(step)
        # self.print_grad(step, sys)
        sys.copy_pos_only(self.pos_buffer, step - 1)
        sys.calc_vn()
        f_contact(sys)
        sys.contact_analysis()

        sys.copy_pos_and_refangle(self, step)

        sys.init_folding()
        sys.ref_angle_backprop_a2ax(self, step)

        # sys.copy_refangle(self, step - 1)
        # sys.init_folding()
        sys.get_paramters_grad()
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
        # self.max_abs(step, sys)
        sys.contact_energy_backprop(True, self, step - 1, p)
        sys.ref_angle_backprop_x2a(self, step, p)
        if self.count_friction_grad:
            sys.contact_energy_backprop_friction(True, self, step - 1, p)
        else:
            self.get_parameters_grad(p, sys)

        # print(self.grad_friction_coef[None], end=" ")

        if step > 0:
            self.get_prev_grad(sys, step)
        if step > 1:
            self.get_prev_prev_grad(sys, step)

    def get_loss(self, sys, pos_grad=False):
        for step in range(1, self.tot_timestep):
            # print(step)
            sys.copy_pos_and_refangle(self, step)
            if pos_grad:
                sys.compute_pos_grad(self, step)
            else:
                sys.compute_force_grad(self, step)

    def get_loss_slide(self, sys, pos_grad=False):
        for i, j in ti.ndrange(sys.cloths[0].NV, self.tot_timestep - 1):
            self.pos_grad[j + 1, sys.cloths[0].offset + i, 0] = 1

    def get_loss_card(self, sys):
        for i in range(sys.cloths[0].NV):
            self.pos_grad[self.tot_timestep - 1, sys.cloths[0].offset + i, 0] = 1

    @ti.kernel
    def get_loss_table(self, sys: ti.template()):
        for i, j in ti.ndrange(sys.cloths[0].NV, self.tot_timestep - 1):
            if (ti.cast(i / (sys.cloths[0].N + 1), ti.i32) == 5) or (ti.cast(i / (sys.cloths[0].N + 1), ti.i32) == 10):
                self.pos_grad[j + 1, sys.cloths[0].offset + i, 2] = -1


