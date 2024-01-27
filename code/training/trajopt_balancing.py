import taichi as ti
# import torch
import time
# from PIL import Image
import numpy as np
import torch
import imageio
import os
from agent.traj_opt_single import agent_trajopt
from optimizer.optim import Adam_single
import random
from argparse import ArgumentParser
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--l', type=int, default=0)
parser.add_argument('--r', type=int, default=5)
parser.add_argument('--iter', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--tot_step', type=int, default=5)
parser.add_argument('--throwing', action="store_true", default=False)
parser.add_argument('--save', action="store_true", default=False)
parser.add_argument('--side', action="store_true", default=False)
parser.add_argument('--load_traj', type=str, default=None)
parser.add_argument('--render', type=int, default=50)
parser.add_argument('--Kb', type=float, default=100)
parser.add_argument('--load_state', type=str, default="../data/balance_state")
args = parser.parse_args()

ti.init(ti.cpu, device_memory_fraction=0.5, default_fp=ti.f64, default_ip=ti.i32, fast_math=False,
        offline_cache=True, offline_cache_max_size_of_files=1024 ** 3,
        offline_cache_cleaning_policy='version')

from Scene_balancing import Scene, Body
from geometry import projection_query
import linalg
from analytic_grad_single import Grad

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
gripper_cnt = sys.elastic_cnt - 1
if sys.enable_gripper:
    gripper_cnt = int((sys.effector_cnt - 1) // 2)

analy_grad = Grad(sys, tot_timestep, gripper_cnt)
adam = Adam_single((tot_timestep, gripper_cnt, 6), args.lr, 0.9, 0.9999, 1e-8)
agent = agent_trajopt(tot_timestep, gripper_cnt, max_moving_dist=0.001)

colors = ti.Vector.field(3, dtype=float, shape=sys.tot_NV)

sys.init_all()
sys.get_colors(colors)
analy_grad.init_mass(sys)

window = ti.ui.Window('surface test', res=(800, 800), vsync=True, show_window=False)
canvas = window.get_canvas()
canvas.set_background_color((0.5, 0.5, 0.5))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(-0.2, 0.2, 0.05)
camera.lookat(0, 0, 0)
camera.up(0, 0, 1)

now_reward = -100000

print("tot_NV:", sys.tot_NV)

for ww in range(args.l, args.r):
    save_path = f"../imgs/traj_opt_balance_{ww}"
    if args.throwing:
        save_path = f"../imgs/traj_opt_balance_throwing_{ww}"
    # sys.init_pos = [(random.random() - 0.5) * 0.002, (random.random() - 0.5) * 0.002, (random.random() - 0.5) * 0.0006]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print(f"Saving Path: {save_path}")

    sys.reset()
    sys.mu_cloth_elastic[None] = 5.0
    scene.set_camera(camera)
    scene.ambient_light([0.8, 0.8, 0.8])
    scene.point_light((2, 2, 2), (1, 1, 1))
    scene.mesh(sys.x32, indices=sys.f_vis, per_vertex_color=colors)  # , index_offset=(nf//2)*3, index_count=(nf//2)*3)
    canvas.scene(scene)
    window.save_image(os.path.join(save_path, f"0.png"))
    plot_x = []
    plot_y = []

    if args.load_traj is not None:
        agent.traj.from_numpy(np.load(args.load_traj))
        agent.fix_action(0.015)

    adam.reset()
    for i in range(args.iter):
        render = False
        if i % args.render == 0:
            render = True
        print("iter: ", i)
        analy_grad.copy_pos(sys, 0)
        obs_list = []
        action_list = []
        start_time = time.time()
        state_path = args.load_state
        if args.throwing:
            state_path = "../data/throwing_state"
        # if not os.path.exists(state_path):
        #     os.mkdir(state_path)
        if not args.save:
            sys.load_all(state_path)

        print("pos:", sys.gripper.pos.to_numpy())

        if not os.path.exists(state_path):
            os.mkdir(state_path)

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
                scene.set_camera(camera)
                scene.ambient_light([0.8, 0.8, 0.8])
                scene.point_light((2, 2, 2), (1, 1, 1))
                # sys.cloths[0].update_visual()
                scene.mesh(sys.x32, indices=sys.f_vis,
                           per_vertex_color=colors)  # , index_offset=(nf//2)*3, index_count=(nf//2)*3)
                canvas.scene(scene)
                window.save_image(os.path.join(save_path, f"{frame}.png"))

        if args.save:
            sys.save_all(state_path)

        end_time = time.time()
        print("tot_time:", end_time - start_time)
        if args.throwing:
            tot_reward = sys.compute_reward_throwing(analy_grad)
        else:
            tot_reward = sys.compute_reward_all(analy_grad)
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
        print("total_reward:", plot_y)
        if tot_reward > now_reward:
            now_reward = tot_reward
            np.save(os.path.join(save_path, "best_traj.npy"), agent.traj.to_numpy())
        np.save(os.path.join(save_path, "plot_data.npy"), np.array(plot_y))

        if render:
            frames = []
            for j in range(1, tot_timestep):
                filename = os.path.join(save_path, f"{j}.png")
                frames.append(imageio.imread(filename))

            gif_name = filename = os.path.join(save_path, f"GIF{i}.gif")
            imageio.mimsave(gif_name, frames, 'GIF', duration=0.02)

        if args.throwing:
            analy_grad.get_loss_throwing(sys)
        else:
            analy_grad.get_loss_balance(sys)

        for i in range(tot_timestep - 1, 0, -1):
            # print("back prop step:", i)
            analy_grad.transfer_grad(i, sys, projection_query)

        analy_grad.apply_action_limit_grad(agent, 0.015)
        print("done grad")
        sys.reset()

        adam.step(agent.traj, analy_grad.gripper_grad)
        # agent.fix_action(0.015)
        analy_grad.reset()
        # agent.print_traj()
        plt.plot(plot_x, plot_y)
        plt.savefig(os.path.join(save_path, f"plot.png"))