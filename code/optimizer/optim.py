import taichi as ti

@ti.data_oriented
class Adam:
    def __init__(self, parameters_shape, lr, beta_1, beta_2, eps):
        self.tot_timestep = parameters_shape[0]
        self.action_dim = parameters_shape[1]
        self.lr = lr
        self.beta_1 = float(beta_1)
        self.beta_2 = beta_2
        self.eps = eps
        self.momentum_buffer = ti.field(ti.f64, shape=(self.tot_timestep, self.action_dim))
        self.v_buffer = ti.field(ti.f64, shape=(self.tot_timestep, self.action_dim))
        self.iter = ti.field(ti.f64, shape=())

    @ti.kernel
    def step(self, parameters: ti.template(), grads: ti.template()):
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        epsilon = self.eps
        for i, j in ti.ndrange(self.tot_timestep, self.action_dim):
            mij = self.momentum_buffer[i, j]
            grad = grads[i, j]
            self.momentum_buffer[i, j] = beta_1 * mij + (1 - beta_1) * grad
            self.v_buffer[i, j] = beta_2 * self.v_buffer[i, j] + (1 - beta_2) * (grad * grad)
            m_cap = self.momentum_buffer[i, j] / (1 - (beta_1 ** (self.iter[None] + 1)))  # calculates the bias-corrected estimates
            v_cap = self.v_buffer[i, j] / (1 - (beta_2 ** (self.iter[None] + 1)))  # calculates the bias-corrected estimates
            parameters[i, j] -= (self.lr * m_cap) / (ti.math.sqrt(v_cap + epsilon))
        self.iter[None] += 1.0

    def reset(self):
        self.iter[None] = 0.0
        self.momentum_buffer.fill(0)
        self.v_buffer.fill(0)

@ti.data_oriented
class Adam_single:
    def __init__(self, parameters_shape, lr, beta_1, beta_2, eps, discount=0.9):
        self.tot_timestep = parameters_shape[0]
        self.action_dim1 = parameters_shape[1]
        self.action_dim2 = parameters_shape[2]
        self.beta_1 = float(beta_1)
        self.beta_2 = beta_2
        self.eps = eps
        self.momentum_buffer = ti.field(ti.f64, shape=(self.tot_timestep, self.action_dim1, self.action_dim2))
        self.v_buffer = ti.field(ti.f64, shape=(self.tot_timestep, self.action_dim1, self.action_dim2))
        self.iter = ti.field(ti.f64, shape=())
        self.lr = ti.field(ti.f64, shape=())
        self.lr[None] = lr
        self.ori_lr = lr
        self.discount = discount

    @ti.kernel
    def step(self, parameters: ti.template(), grads: ti.template()):
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        epsilon = self.eps
        has_nan = False
        for i, j, k in ti.ndrange(self.tot_timestep, self.action_dim1, self.action_dim2):
            mij = self.momentum_buffer[i, j, k]
            grad = grads[i, j, k]
            self.momentum_buffer[i, j, k] = beta_1 * mij + (1 - beta_1) * grad
            self.v_buffer[i, j, k] = beta_2 * self.v_buffer[i, j, k] + (1 - beta_2) * (grad * grad)
            m_cap = self.momentum_buffer[i, j, k] / (1 - (beta_1 ** (self.iter[None] + 1)))  # calculates the bias-corrected estimates
            v_cap = self.v_buffer[i, j, k] / (1 - (beta_2 ** (self.iter[None] + 1)))  # calculates the bias-corrected estimates
            parameters[i, j, k] -= (self.lr[None] * m_cap) / (ti.math.sqrt(v_cap + epsilon))
            if ti.math.isnan(grads[i, j, k]):
                has_nan = True

        if has_nan:
            print("nan in gripper grid!!")
        self.iter[None] += 1.0

        if int(self.iter[None]) % 10 == 0:
            self.lr[None] *= self.discount

    def reset(self):
        self.iter[None] = 0.0
        self.lr[None] = self.ori_lr
        self.momentum_buffer.fill(0)
        self.v_buffer.fill(0)


@ti.data_oriented
class SGD_single:
    def __init__(self, parameters_shape, lr, beta_1, beta_2, eps):
        self.tot_timestep = parameters_shape[0]
        self.action_dim1 = parameters_shape[1]
        self.action_dim2 = parameters_shape[2]
        self.lr = lr
        self.beta_1 = float(beta_1)
        self.beta_2 = beta_2
        self.eps = eps
        # self.momentum_buffer = ti.field(ti.f64, shape=(self.tot_timestep, self.action_dim1, self.action_dim2))
        # self.v_buffer = ti.field(ti.f64, shape=(self.tot_timestep, self.action_dim1, self.action_dim2))
        # self.iter = ti.field(ti.f64, shape=())

    @ti.kernel
    def step(self, parameters: ti.template(), grads: ti.template()):

        for i, j, k in ti.ndrange(self.tot_timestep, self.action_dim1, self.action_dim2):
            parameters[i, j, k] -= self.lr * grads[i, j, k]

    def reset(self):
        pass