import taichi as ti

@ti.data_oriented
class agent_trajopt:
    def __init__(self, tot_timestep, max_moving_dist=0.0005):
        self.traj = ti.field(ti.f64, (tot_timestep, 7))
        self.tmp_action = ti.field(ti.f64, (7, ))
        self.action_dim = 7
        self.tot_timestep = tot_timestep
        self.max_moving_dist = max_moving_dist
    def fix_action(self, max_dist):
        for i in range(1, self.tot_timestep):
            delta_pos = ti.Vector([self.traj[i, 0] - self.traj[i - 1, 0], self.traj[i, 1] - self.traj[i - 1, 1],
                                        self.traj[i, 2] - self.traj[i - 1, 2]])
            delta_rot = ti.Vector([self.traj[i, 3] - self.traj[i - 1, 3], self.traj[i, 4] - self.traj[i - 1, 4],
                                        self.traj[i, 5] - self.traj[i - 1, 5]])
            delta_dist = self.traj[i, 6] - self.traj[i - 1, 6]
            moving_dist = ti.math.sqrt(delta_pos.dot(delta_pos)) + ti.math.sqrt(delta_rot.dot(delta_rot)) * max_dist + ti.abs(delta_dist)
            weight = self.max_moving_dist / (moving_dist + 1e-8)
            if weight < 1.0:
                for j in range(self.action_dim):
                    self.traj[i, j] = self.traj[i - 1, j] + (self.traj[i, j] - self.traj[i - 1, j]) * weight

    def get_action(self, step):
        i = step
        delta_pos = ti.Vector([self.traj[i, 0] - self.traj[i - 1, 0], self.traj[i, 1] - self.traj[i - 1, 1],
                               self.traj[i, 2] - self.traj[i - 1, 2]])
        delta_rot = ti.Vector([self.traj[i, 3] - self.traj[i - 1, 3], self.traj[i, 4] - self.traj[i - 1, 4],
                               self.traj[i, 5] - self.traj[i - 1, 5]])
        delta_dist = self.traj[i, 6] - self.traj[i - 1, 6]
        return delta_pos, delta_rot, delta_dist

    def get_action_field(self, step):
        for i in range(7):
            self.tmp_action[i] = self.traj[step, i] - self.traj[step - 1, i]

    def init_traj(self):
        for i in range(5):
            self.traj[i, 6] = -0.001 * i
        self.traj[5, 6] = -0.005
        for i in range(6, 20):
            self.traj[i, 6] = -0.005
            self.traj[i, 2] = self.traj[i - 1, 2] + 0.0003

    def init_traj_1(self):
        for i in range(5):
            self.traj[i, 2] = -0.001 * i
        self.traj[5, 2] = -0.001 * 5 + 0.0007
        for i in range(6, 20):
            self.traj[i, 0] = self.traj[i - 1, 0] - 0.001
            self.traj[i, 2] = self.traj[i - 1, 2]
        for i in range(20, 35):
            self.traj[i, 0] = self.traj[i - 1, 0]
            self.traj[i, 2] = self.traj[i - 1, 2] + 0.00029
            self.traj[i, 6] = self.traj[i - 1, 6] - 0.00029
        for i in range(35, 50):
            self.traj[i, 0] = self.traj[i - 1, 0]
            self.traj[i, 2] = self.traj[i - 1, 2] + 0.0005
            self.traj[i, 6] = self.traj[i - 1, 6]
        # for i in range(15, 20):
        #     self.traj[i, 0] = self.traj[i - 1, 0] + 0.0003
        #     self.traj[i, 2] = self.traj[i - 1, 2] + 0.0001
        #     self.traj[i, 6] = self.traj[i - 1, 6] - 0.0003

    def init_traj_2(self):
        for i in range(2):
            self.traj[i, 2] = -0.0006 * i
        for i in range(6, 20):
            self.traj[i, 0] = self.traj[i - 1, 0] - 0.001
            self.traj[i, 2] = self.traj[i - 1, 2]
        for i in range(20, 35):
            self.traj[i, 0] = self.traj[i - 1, 0]
            self.traj[i, 2] = self.traj[i - 1, 2] + 0.00029
            self.traj[i, 6] = self.traj[i - 1, 6] - 0.00029
        for i in range(35, 55):
            self.traj[i, 0] = self.traj[i - 1, 0] - 0.0003
            self.traj[i, 2] = self.traj[i - 1, 2] + 0.0005
            self.traj[i, 6] = self.traj[i - 1, 6]
            self.traj[i, 4] = self.traj[i - 1, 4] + 0.005
        for i in range(55, 70):
            self.traj[i, 0] = self.traj[i - 1, 0] - 0.0003
            self.traj[i, 2] = self.traj[i - 1, 2] + 0.0005
            self.traj[i, 6] = self.traj[i - 1, 6]
            self.traj[i, 4] = self.traj[i - 1, 4]

    def init_traj_tactile_2(self):
        for i in range(10):
            self.traj[i, 2] = -0.0003 * i
        for i in range(10, 50):
            self.traj[i, 0] = self.traj[i - 1, 0] - 0.0005
            self.traj[i, 2] = self.traj[i - 1, 2]
        for i in range(50, 60):
            self.traj[i, 0] = self.traj[i - 1, 0]
            self.traj[i, 2] = self.traj[i - 1, 2] + 0.0003
            self.traj[i, 6] = self.traj[i - 1, 6] - 0.0003
        for i in range(60, 90):
            self.traj[i, 0] = self.traj[i - 1, 0] - 0.0005
            self.traj[i, 2] = self.traj[i - 1, 2]
            self.traj[i, 6] = self.traj[i - 1, 6]
            self.traj[i, 4] = self.traj[i - 1, 4]

    def init_traj_3(self):
        for i in range(15):
            self.traj[i, 2] = -0.00045 * i
        for i in range(15, 30):
            self.traj[i, 0] = self.traj[i - 1, 0] + 0.0005
            self.traj[i, 2] = self.traj[i - 1, 2]
            self.traj[i, 4] = self.traj[i - 1, 4] - 0.005
        for i in range(30, 90):
            self.traj[i, 0] = self.traj[i - 1, 0] + 0.0004
            self.traj[i, 2] = self.traj[i - 1, 2] + 0.0001
            self.traj[i, 4] = self.traj[i - 1, 4]


    def print_traj(self):
        for i in range(self.tot_timestep):
            print("step:", i, "pos", f"[{self.traj[i, 0]}, {self.traj[i, 1]}, {self.traj[i, 2]}]", "rot", f"[{self.traj[i, 3]}, {self.traj[i, 4]}, {self.traj[i, 5]}]", "dist", self.traj[i, 6])


