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

    # def get_action_field(self, step):
    #     for i in range(7):
    #         self.tmp_action[i] = self.traj[step, i] - self.traj[step - 1, i]

    def init_traj_lift(self):
        for i in range(5):
            self.traj[i, 0, 2] = -0.0004 * i
            # self.traj[i, 1, 2] = 0.0004 * i
            # self.traj[i, 2, 2] = 0.0004 * i
            self.traj[i, 1, 4] = 0.001 * i
            self.traj[i, 2, 4] = -0.001 * i
        # for i in range(5, 60):
        #     self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
        #     self.traj[i, 1, 4] = self.traj[i - 1, 1, 4]
        #     self.traj[i, 2, 4] = self.traj[i - 1, 2, 4]
        for i in range(5, 10):
            # self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] - 0.0004
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2]
            self.traj[i, 2, 2] = self.traj[i - 1, 2, 2]
            self.traj[i, 1, 4] = self.traj[i - 1, 1, 4]
            self.traj[i, 2, 4] = self.traj[i - 1, 2, 4]

        for i in range(10, 50):
            # self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] - 0.0004
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2]
            self.traj[i, 2, 2] = self.traj[i - 1, 2, 2]
            self.traj[i, 1, 4] = self.traj[i - 1, 1, 4]
            self.traj[i, 2, 4] = self.traj[i - 1, 2, 4]
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] - 0.0003
            self.traj[i, 0, 1] = self.traj[i - 1, 0, 1] - 0.0003
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.0003
            self.traj[i, 1, 1] = self.traj[i - 1, 1, 1] - 0.0003
            self.traj[i, 2, 0] = self.traj[i - 1, 2, 0] - 0.0003
            self.traj[i, 2, 1] = self.traj[i - 1, 2, 1] - 0.0003

    def init_traj_7(self):
        for i in range(6):
            self.traj[i, 0, 2] = -0.0003 * i
            self.traj[i, 1, 2] = 0.0003 * i
        for i in range(6, 40):
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2]
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] - 0.00002
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] + 0.00002
        # for i in range(20, 40):
        #     self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] + 0.0001
        #     self.traj[i, 1, 2] = self.traj[i - 1, 1, 2] - 0.0001
        #     self.traj[i, 0, 0] = self.traj[i - 1, 0, 0]
        #     self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]

    def init_traj_fold(self):
        for i in range(5):
            self.traj[i, 0, 2] = -0.0003 * i
        for i in range(5, 60):
            # self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] - 0.0004
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] + 0.00044
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2] - 0.00029
            self.traj[i, 1, 4] = self.traj[i - 1, 1, 4] - 0.0175

    def init_traj_bounce(self):
        for i in range(5):
            self.traj[i, 0, 2] = -0.0005 * i
        for i in range(5, 90):
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            # self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] - 0.0001
        # for i in range(40, 60):
        #     self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
        #     self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] - 0.0008
        # for i in range(60, 90):
        #     self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
        #     self.traj[i, 0, 0] = self.traj[i - 1, 0, 0]

    def regen_traj_bounce(self):
        for i in range(40, 60):
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] - 0.0008
        for i in range(60, 90):
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0]


    def init_traj_fold_2(self, a=0.00006, b=0.00007, c=0.0003, d=0.0001):

        for i in range(3):
            self.traj[i, 0, 2] = -a * i
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + b
        # for i in range(3, 50):
        #     self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
        #     self.traj[i, 0, 0] = self.traj[i - 1, 0, 0]
        for i in range(3, 20):
            self.traj[i, 1, 2] = -a * i
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.00003
        for i in range(20, 35):
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2] - 0.00003
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - d
        for i in range(35, 50):
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2] - 0.0001
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.0003

    def init_traj_fold_3(self, a=0.00027, b=0.00024, c=0.00025, d=0.0001):

        for i in range(3):
            self.traj[i, 0, 2] = -a * i
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0]
        # for i in range(3, 50):
        #     self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
        #     self.traj[i, 0, 0] = self.traj[i - 1, 0, 0]
        for i in range(3, 20):
            self.traj[i, 1, 2] = -a * i
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]
        for i in range(20, 35):
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2] + 0.0003
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - d
        for i in range(35, 50):
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2]
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - d

    def init_traj_push(self):

        for i in range(1, 20):
            self.traj[i, 0, 2] = -0.00011 * i
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.00023
        for i in range(20, 35):
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] - 0.0002
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.00027
        for i in range(35, 50):
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0002

    def init_traj_pick(self):

        for i in range(8):
            self.traj[i, 0, 2] = -0.0006 * i
            self.traj[i, 1, 2] = -0.0006 * i
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0008
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.0008
        for i in range(8, 50):
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2]
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0]
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]
        # for i in range(10, 21):
        #     self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0003
        #     self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.0003
        #     self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] - 0.00005
        #     self.traj[i, 1, 2] = self.traj[i - 1, 1, 2] - 0.00005
        # for i in range(21, 40):
        #     self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0003
        #     self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.0003
        #     self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] + 0.0001
        #     self.traj[i, 1, 2] = self.traj[i - 1, 1, 2] + 0.0001
        #     self.traj[i, 0, 4] = self.traj[i - 1, 0, 4] - 0.025
        #     self.traj[i, 1, 4] = self.traj[i - 1, 1, 4] + 0.025
        # for i in range(40, 80):
        #     self.traj[i, 0, 0] = self.traj[i - 1, 0, 0]
        #     self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]
        #     self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] + 0.0003
        #     self.traj[i, 1, 2] = self.traj[i - 1, 1, 2] + 0.0003
        #     self.traj[i, 0, 4] = self.traj[i - 1, 0, 4]
        #     self.traj[i, 1, 4] = self.traj[i - 1, 1, 4]

    def init_traj_pick_fold(self):

        for i in range(8):
            self.traj[i, 0, 2] = -0.0006 * i
            self.traj[i, 1, 2] = -0.0006 * i
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0001
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.0001
        # for i in range(8, 50):
        #     self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
        #     self.traj[i, 1, 2] = self.traj[i - 1, 1, 2]
        #     self.traj[i, 0, 0] = self.traj[i - 1, 0, 0]
        #     self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]
        for i in range(8, 21):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.00015
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.00015
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] - 0.0001
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2] - 0.0001
        for i in range(21, 40):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0002
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.0002
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2]
            self.traj[i, 0, 4] = self.traj[i - 1, 0, 4] - 0.003
            self.traj[i, 1, 4] = self.traj[i - 1, 1, 4] + 0.003
        for i in range(40, 50):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0003
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.0003
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2]
            self.traj[i, 0, 4] = self.traj[i - 1, 0, 4] - 0.01
            self.traj[i, 1, 4] = self.traj[i - 1, 1, 4] + 0.01
        # for i in range(50, 80):
        #     self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.00015
        #     self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.00015
        #     self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] + 0.00003
        #     self.traj[i, 1, 2] = self.traj[i - 1, 1, 2] + 0.00003
        #     self.traj[i, 0, 4] = self.traj[i - 1, 0, 4]
        #     self.traj[i, 1, 4] = self.traj[i - 1, 1, 4]

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

    def init_traj_shuffle(self):
        for i in range(5):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0003
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.0003
            self.traj[i, 3, 0] = self.traj[i - 1, 3, 0] - 0.0003
            self.traj[i, 4, 0] = self.traj[i - 1, 4, 0] + 0.0003

        for i in range(5, 20):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0001
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] + 0.0003
            self.traj[i, 0, 4] = self.traj[i - 1, 0, 4]
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]
            self.traj[i, 3, 0] = self.traj[i - 1, 3, 0] - 0.0001
            self.traj[i, 3, 2] = self.traj[i - 1, 3, 2] + 0.0003
            self.traj[i, 3, 4] = self.traj[i - 1, 3, 4]
            self.traj[i, 4, 0] = self.traj[i - 1, 4, 0]

        for i in range(20, 35):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0001
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] + 0.0002
            self.traj[i, 0, 4] = self.traj[i - 1, 0, 4]
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]
            self.traj[i, 3, 0] = self.traj[i - 1, 3, 0] - 0.0001
            self.traj[i, 3, 2] = self.traj[i - 1, 3, 2] + 0.0002
            self.traj[i, 3, 4] = self.traj[i - 1, 3, 4]
            self.traj[i, 4, 0] = self.traj[i - 1, 4, 0]

        for i in range(35, 90):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0002
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] + 0.0005
            self.traj[i, 0, 4] = self.traj[i - 1, 0, 4] + 0.02
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]
            self.traj[i, 3, 0] = self.traj[i - 1, 3, 0] - 0.0002
            self.traj[i, 3, 2] = self.traj[i - 1, 3, 2] + 0.0005
            self.traj[i, 3, 4] = self.traj[i - 1, 3, 4] - 0.02
            self.traj[i, 4, 0] = self.traj[i - 1, 4, 0]

    def init_traj_book(self):
        for i in range(15):
            self.traj[i, 0, 2] = -0.0004 * i

        for i in range(15, 50):
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]

        # for i in range(15, 30):
        #     self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0003
        #     self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
        #     self.traj[i, 0, 4] = self.traj[i - 1, 0, 4] - 0.01
        # for i in range(30, 50):
        #     self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0003
        #     self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] + 0.0002
        #     self.traj[i, 0, 4] = self.traj[i - 1, 0, 4] - 0.01
        # for i in range(50, 90):
        #     self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0003
        #     self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
        #     self.traj[i, 0, 4] = self.traj[i - 1, 0, 4]
        #
        # for i in range(90, 150):
        #     self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0003
        #     self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] + 0.0003
        #     self.traj[i, 0, 4] = self.traj[i - 1, 0, 4]

    def init_traj_slide(self):
        for i in range(10):
            self.traj[i, 0, 2] = -0.00035 * i
        for i in range(10, 50):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] - 0.0005
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]

    def init_traj_slide_simple(self):
        for i in range(10):
            self.traj[i, 0, 2] = -0.00029 * i
        for i in range(10, 30):
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]

    def init_traj_deliver(self):

        for i in range(8):
            self.traj[i, 0, 2] = -0.0003 * i
            self.traj[i, 1, 2] = -0.0003 * i
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0003
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.0003
        for i in range(8, 21):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0003
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.0003
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] - 0.0001
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2] - 0.0001
        for i in range(21, 40):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0001
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.0001
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] + 0.00008
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2] + 0.00008
            self.traj[i, 0, 4] = self.traj[i - 1, 0, 4] - 0.01
            self.traj[i, 1, 4] = self.traj[i - 1, 1, 4] + 0.01
        for i in range(40, 50):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0001
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.0001
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] + 0.0001
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2] + 0.0001
            self.traj[i, 0, 4] = self.traj[i - 1, 0, 4] - 0.01
            self.traj[i, 1, 4] = self.traj[i - 1, 1, 4] + 0.01
        for i in range(50, 70):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.00015
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.00015
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] + 0.0001
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2] + 0.0001
            self.traj[i, 0, 4] = self.traj[i - 1, 0, 4]
            self.traj[i, 1, 4] = self.traj[i - 1, 1, 4]
        for i in range(70, 120):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0]
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]
            self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2]
            self.traj[i, 0, 4] = self.traj[i - 1, 0, 4]
            self.traj[i, 1, 4] = self.traj[i - 1, 1, 4]

    def init_traj_interact(self):
        self.traj.fill(0)
        for i in range(5, 30):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] - 0.002

    def init_traj_flatlift(self):
        # self.traj.fill(0)
        for i in range(1, 5):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.0003
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.0003
        for i in range(5, 50):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0]
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]

    def init_traj_real(self):
        # self.traj.fill(0)
        for i in range(10, 25):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.001
        for i in range(25, 40):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] + 0.002
        for i in range(40, 70):
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] - 0.002
        # for i in range(30, 60):
        #     self.traj[i, 0, 0] = self.traj[i - 1, 0, 0]
        #     self.traj[i, 1, 0] = self.traj[i - 1, 1, 0]

    def smooth(self):
        for i in range(10, 50):
            if i < 20:
                self.traj[i, 0, 2] = self.traj[i - 1, 0, 2] - 0.00013
            else:
                self.traj[i, 0, 2] = self.traj[i - 1, 0, 2]
            self.traj[i, 1, 2] = self.traj[i - 1, 1, 2]
            self.traj[i, 2, 2] = self.traj[i - 1, 2, 2]
            self.traj[i, 0, 4] = self.traj[i - 1, 0, 4]
            self.traj[i, 1, 4] = self.traj[i - 1, 1, 4]
            self.traj[i, 2, 4] = self.traj[i - 1, 2, 4]
            self.traj[i, 0, 3] = self.traj[i - 1, 0, 3]
            self.traj[i, 1, 3] = self.traj[i - 1, 1, 3]
            self.traj[i, 2, 3] = self.traj[i - 1, 2, 3]
            self.traj[i, 0, 5] = self.traj[i - 1, 0, 5]
            self.traj[i, 1, 5] = self.traj[i - 1, 1, 5]
            self.traj[i, 2, 5] = self.traj[i - 1, 2, 5]
            self.traj[i, 0, 0] = self.traj[i - 1, 0, 0] - 0.00032
            self.traj[i, 0, 1] = self.traj[i - 1, 0, 1] - 0.00032
            self.traj[i, 1, 0] = self.traj[i - 1, 1, 0] - 0.00032
            self.traj[i, 1, 1] = self.traj[i - 1, 1, 1] - 0.00032
            self.traj[i, 2, 0] = self.traj[i - 1, 2, 0] - 0.00032
            self.traj[i, 2, 1] = self.traj[i - 1, 2, 1] - 0.00032
