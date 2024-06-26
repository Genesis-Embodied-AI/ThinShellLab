import taichi as ti
import time
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
parser.add_argument('--sep', action="store_true", default=False)
parser.add_argument('--Kb', type=float, default=100)
parser.add_argument('--mu', type=float, default=5.0)
parser.add_argument('--discount', type=float, default=0.9)
parser.add_argument('--load_traj', type=str, default=None)
parser.add_argument('--soft', action="store_true", default=False)
parser.add_argument('--dense', type=float, default=10000.0)
parser.add_argument('--render', type=int, default=10)
parser.add_argument('--render_option', type=str, default="Taichi")
args = parser.parse_args()

ti.init(ti.cpu, device_memory_fraction=0.5, default_fp=ti.f64, default_ip=ti.i32, fast_math=False,
        offline_cache=True, offline_cache_max_size_of_files=1024 ** 3,
        offline_cache_cleaning_policy='version')

from thinshelllab.task_scene.Scene_interact import Scene, Body
from thinshelllab.engine.geometry import projection_query
from thinshelllab.engine.render_engine import Renderer
from thinshelllab.engine import linalg
from thinshelllab.engine.analytic_grad_single import Grad

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
sys = Scene(cloth_size=0.06, soft=args.soft, dense=args.dense)
sys.cloths[0].Kb[None] = args.Kb
if args.soft:
    sys.cloths[0].Kb[None] = 0.1
gripper_cnt = sys.elastic_cnt - 1
if sys.enable_gripper:
    gripper_cnt = int((sys.effector_cnt - 1) // 2)

analy_grad = Grad(sys, tot_timestep, gripper_cnt)
adam = Adam_single((tot_timestep, gripper_cnt, 6), args.lr, 0.9, 0.9999, 1e-8, discount=args.discount)
agent = agent_trajopt(tot_timestep, gripper_cnt, max_moving_dist=0.002)
sys.init_all()
analy_grad.init_mass(sys)
renderer = Renderer(sys, "interact", option=args.render_option)

now_reward = -100000
for ww in range(args.l, args.r):
    save_path = f"../imgs/traj_opt_interact_{ww}"
    # sys.init_pos = [(random.random() - 0.5) * 0.002, (random.random() - 0.5) * 0.002, (random.random() - 0.5) * 0.0006]
    renderer.set_save_dir(save_path)
    print(f"Saving Path: {save_path}")

    sys.reset()
    sys.mu_cloth_elastic[None] = args.mu
    plot_x = []
    plot_y = []
    if args.load_traj is not None:
        traj_np = np.load(args.load_traj)
        agent.fix_action(0.015)
    adam.reset()

    if not args.sep:
        tot_reward = sys.compute_reward_1()
    else:
        tot_reward = sys.compute_reward()
    print("init reward:", tot_reward)

    for i in range(args.iter):
        render = False
        if i % args.render == 0:
            render = True
        print("iter: ", i)
        analy_grad.copy_pos(sys, 0)
        obs_list = []
        action_list = []
        start_time = time.time()
        if render:
            renderer.render("0")
        for frame in range(1, tot_timestep):
            # print("frame:", frame)
            agent.get_action(frame)
            # sys.get_observation()
            sys.action(frame, agent.delta_pos, agent.delta_rot)
            # agent.get_action_field(frame)
            # obs_list.append(sys.observation.to_torch('cpu'))
            # action_list.append(agent.tmp_action.to_torch('cpu'))
            sys.time_step(projection_query, frame)
            analy_grad.copy_pos(sys, frame)
            if render:
                renderer.render(str(frame))

        end_time = time.time()
        print("tot_time:", end_time - start_time)

        if not args.sep:
            tot_reward = sys.compute_reward_1()
        else:
            tot_reward = sys.compute_reward()


        plot_x.append(i)
        plot_y.append(tot_reward)
        print("total_reward:", plot_y)
        if tot_reward > now_reward:
            now_reward = tot_reward
            np.save(os.path.join(save_path, "best_traj.npy"), agent.traj.to_numpy())
        np.save(os.path.join(save_path, "plot_data.npy"), np.array(plot_y))
        if args.sep:
            xx = []
            yy = []
            minx = 0.0
            for a in range(tot_timestep):
                minx = ti.min(agent.traj[a, 0, 0], minx)
                xx.append(a)
                yy.append(agent.traj[a, 0, 0])
            print("min traj x:", minx)
            plt.clf()
            plt.plot(xx, yy)
            plt.savefig(os.path.join(save_path, f"move_{i}.png"))
            plt.clf()

        if render:
            renderer.end_rendering(i)

        if not args.sep:
            analy_grad.get_loss_interact_1(sys)
        else:
            analy_grad.get_loss_interact(sys)

        for i in range(tot_timestep - 1, 5, -1):
            # print("back prop step:", i)
            analy_grad.transfer_grad(i, sys, projection_query)
        print("done grad")
        sys.reset()
        analy_grad.apply_action_limit_grad(agent, 0.015)
        # analy_grad.apply_gripper_pos_grad_interact(agent)
        adam.step(agent.traj, analy_grad.gripper_grad)
        # agent.fix_action(0.015)
        analy_grad.reset()
        # agent.print_traj()
        plt.plot(plot_x, plot_y)
        plt.savefig(os.path.join(save_path, f"plot.png"))