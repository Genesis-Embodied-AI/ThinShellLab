import taichi as ti
# import torch
import time
# from PIL import Image
import numpy as np
import torch
import imageio
import os
from thinshelllab.agent.traj_opt_single import agent_trajopt
from thinshelllab.optimizer.optim import Adam_single
import random
from argparse import ArgumentParser
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--l', type=int, default=0)
parser.add_argument('--r', type=int, default=5)
parser.add_argument('--iter', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--tot_step', type=int, default=5)
parser.add_argument('--Kb', type=float, default=120.0)
parser.add_argument('--render_option', type=str, default="Taichi")
args = parser.parse_args()

ti.init(ti.cpu, device_memory_fraction=0.5, default_fp=ti.f64, default_ip=ti.i32, fast_math=False,
        offline_cache=True, offline_cache_max_size_of_files=1024 ** 3,
        offline_cache_cleaning_policy='version')

from thinshelllab.task_scene.Scene_bouncing import Scene, Body
from thinshelllab.engine.geometry import projection_query
from thinshelllab.engine.render_engine import Renderer
from thinshelllab.engine import linalg
from thinshelllab.engine.analytic_grad_system import Grad

M = 2
N = 16
k_spring = 1000
nv = M * (N ** 2)
nf = M * 2 * (N - 1) ** 2
# nv = (N**2)
# nf = 2*(N-1)**2
total_m = 0.005 * M
h = 0.005

tot_timestep = args.tot_step
sys = Scene(cloth_size=0.06)
sys.cloths[0].Kb[None] = args.Kb
analy_grad = Grad(sys, tot_timestep, sys.elastic_cnt - 1)
sys.init_all()
analy_grad.init_mass(sys)
renderer = Renderer(sys, "bouncing", option=args.render_option)

now_reward = 0
for ww in range(args.l, args.r):
    save_path = f"../imgs/traj_opt_table_{ww}"
    # sys.init_pos = [(random.random() - 0.5) * 0.002, (random.random() - 0.5) * 0.002, (random.random() - 0.5) * 0.0006]
    renderer.set_save_dir(save_path)
    print(f"Saving Path: {save_path}")

    sys.reset()
    sys.mu_cloth_elastic[None] = 0.5
    plot_x = []
    plot_y = []
    kb_list = []

    for i in range(args.iter):
        print("iter: ", i)
        analy_grad.copy_pos(sys, 0)
        obs_list = []
        action_list = []
        start_time = time.time()
        renderer.render("0")
        for frame in range(1, tot_timestep):
            print("frame:", frame)
            # agent.get_action_field(frame)
            # obs_list.append(sys.observation.to_torch('cpu'))
            # action_list.append(agent.tmp_action.to_torch('cpu'))
            sys.time_step(projection_query, frame)
            analy_grad.copy_pos(sys, frame)
            renderer.render(str(frame))

        end_time = time.time()
        print("tot_time:", end_time - start_time)
        tot_reward = sys.compute_reward()
        # if tot_reward > now_reward:
        #     now_reward = tot_reward
        #     data_path = f"../data/data_traj_opt_fold_{ww}"
        #     if not os.path.exists(data_path):
        #         os.mkdir(data_path)
        #     for frame in range(tot_timestep - 1):
        #         torch.save({
        #             'obs': obs_list[frame],
        #             'action': action_list[frame]
        #         }, os.path.join(data_path, f"data_{frame + 1}"))

        plot_x.append(i)
        plot_y.append(tot_reward)
        np.save(os.path.join(save_path, "plot_data.npy"), np.array(plot_y))
        print("total_reward:", plot_y)
        renderer.end_rendering(i)

        analy_grad.get_loss_table(sys)

        for j in range(tot_timestep - 1, 0, -1):
            # print("back prop step:", i)
            analy_grad.transfer_grad(j, sys, projection_query)

        loss_grad = analy_grad.grad_kb[None] * args.lr
        if loss_grad < -100:
            loss_grad = -100
        if loss_grad > 100:
            loss_grad = 100
        sys.cloths[0].Kb[None] -= loss_grad
        kb_list.append(sys.cloths[0].Kb[None])
        print("done grad")
        sys.reset()

        print("prev kbs", kb_list)
        print("now kb", sys.cloths[0].Kb[None], "now grad", analy_grad.grad_kb[None] * args.lr)
        analy_grad.reset()
        args.lr *= 0.95
        plt.plot(plot_x, plot_y)
        plt.savefig(os.path.join(save_path, f"plot.png"))