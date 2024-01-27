import taichi as ti

@ti.data_oriented
class agent_trajopt:
    def __init__(self, tot_timestep, cnt, max_moving_dist=0.0005):
        self.traj = ti.field(ti.f64, (tot_timestep, cnt, 6))
        self.tmp_action = ti.field(ti.f64, (cnt, 6))
        self.delta_pos = ti.Vector.field(3, ti.f64, shape=cnt)
        self.delta_rot = ti.Vector.field(3, ti.f64, shape=cnt)
        self.action_dim = 6 * cnt
        self.tot_timestep = tot_timestep
        self.max_moving_dist = max_moving_dist
        self.n_part = cnt

    def fix_action(self, max_dist):
        for i in range(1, self.tot_timestep):
            for j in ti.static(range(self.n_part)):
                delta_pos = ti.Vector([self.traj[i, j, 0] - self.traj[i - 1, j, 0], self.traj[i, j, 1] - self.traj[i - 1, j, 1],
                                            self.traj[i, j, 2] - self.traj[i - 1, j, 2]])
                delta_rot = ti.Vector([self.traj[i, j, 3] - self.traj[i - 1, j, 3], self.traj[i, j, 4] - self.traj[i - 1, j, 4],
                                            self.traj[i, j, 5] - self.traj[i - 1, j, 5]])

                moving_dist = ti.math.sqrt(delta_pos.dot(delta_pos)) + ti.math.sqrt(delta_rot.dot(delta_rot)) * max_dist
                weight = self.max_moving_dist / (moving_dist + 1e-8)
                if weight < 1.0:
                    for k in range(self.action_dim):
                        self.traj[i, j, k] = self.traj[i - 1, j, k] + (self.traj[i, j, k] - self.traj[i - 1, j, k]) * weight

    def calculate_dist(self, frame, max_dist, j):
        i = frame

        delta_pos = ti.Vector(
            [self.traj[i, j, 0] - self.traj[i - 1, j, 0], self.traj[i, j, 1] - self.traj[i - 1, j, 1],
             self.traj[i, j, 2] - self.traj[i - 1, j, 2]])
        delta_rot = ti.Vector(
            [self.traj[i, j, 3] - self.traj[i - 1, j, 3], self.traj[i, j, 4] - self.traj[i - 1, j, 4],
             self.traj[i, j, 5] - self.traj[i - 1, j, 5]])

        moving_dist = ti.math.sqrt(delta_pos.dot(delta_pos)) + ti.math.sqrt(delta_rot.dot(delta_rot)) * max_dist
        return moving_dist

    def get_action(self, step):
        i = step
        for j in ti.static(range(self.n_part)):
            self.delta_pos[j] = ti.Vector([self.traj[i, j, 0] - self.traj[i - 1, j, 0], self.traj[i, j, 1] - self.traj[i - 1, j, 1],
                                            self.traj[i, j, 2] - self.traj[i - 1, j, 2]])
            self.delta_rot[j] = ti.Vector([self.traj[i, j, 3] - self.traj[i - 1, j, 3], self.traj[i, j, 4] - self.traj[i - 1, j, 4],
                                            self.traj[i, j, 5] - self.traj[i - 1, j, 5]])

    def init_traj_forming(self):

        for i in range(1, 20):
            self.traj[i, 0, 2] = -0.00011 * i
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.00023
        for i in range(20, 35):
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] - 0.0002
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.00027
        for i in range(35, 50):
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0002

    def init_traj_pick_fold(self):

        for i in range(8):
            self.traj[i, 0, 2] = -0.0006 * i
            self.traj[i, 1, 2] = -0.0006 * i
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0]
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]
        for i in range(8, 50):
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2]
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0]
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]

    def init_traj_card(self):
        for i in range(5):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0003
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.0003

        for i in range(5, 20):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0001
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] + 0.0003
            self.traj[i, 0, 4] = self.traj[i - 1, 0, 4]
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]

        for i in range(20, 35):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0001
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] + 0.0002
            self.traj[i, 0, 4] = self.traj[i - 1, 0, 4]
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]

        for i in range(35, 50):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0002
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] + 0.0005
            self.traj[i, 0, 4] = self.traj[i - 1, 0, 4] + 0.02
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]

        for i in range(50, 150):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0]
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            self.traj[i, 0, 4] = self.traj[i - 1, 0, 4]
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]

    def init_traj_slide(self):
        for i in range(10):
            self.traj[i, 0, 2] = -0.00035 * i
        for i in range(10, 50):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] - 0.0005
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
