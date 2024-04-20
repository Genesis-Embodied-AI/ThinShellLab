import taichi as ti
# import torch
import time
# from PIL import Image
import numpy as np
import torch
import imageio
import os

import random
from argparse import ArgumentParser
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--pop_size', type=int, default=8)
parser.add_argument('--iter', type=int, default=10)
parser.add_argument('--tot_step', type=int, default=60)
parser.add_argument('--abs_step', type=int, default=60)
parser.add_argument('--sigma', type=float, default=1.0)
parser.add_argument('--trial', type=str, default="0")
parser.add_argument('--env', type=str, default="")
parser.add_argument('--Kb', type=float, default=100.0)
parser.add_argument('--mu', type=float, default=1.0)
parser.add_argument('--reward_name', type=str, default=None)
parser.add_argument('--load_dir', type=str, default=None)
parser.add_argument('--max_dist', type=float, default=0.002)
parser.add_argument('--curve7', type=float, default=1.0)
parser.add_argument('--curve8', type=float, default=1.0)
parser.add_argument('--dense', type=float, default=10000.0)
parser.add_argument('--target_dir', type=str, default=None)
args = parser.parse_args()

ti.init(ti.cpu, device_memory_fraction=0.5, default_fp=ti.f64, default_ip=ti.i32, fast_math=False,
        offline_cache=True, offline_cache_max_size_of_files=1024 ** 3,
        offline_cache_cleaning_policy='version')

from thinshelllab.engine.geometry import projection_query
from thinshelllab.engine import linalg
from thinshelllab.agent.traj_opt_single import agent_trajopt
import cma
import importlib

from thinshelllab.engine.analytic_grad_single import Grad

importlib.invalidate_caches()
Scene = importlib.import_module(f'thinshelllab.task_scene.Scene_{args.env}')

tot_timestep = args.tot_step
cloth_size=0.06
if args.env == "folding_2" or args.env == "forming":
    cloth_size = 0.1

if args.Kb < 2:
    sys = Scene.Scene(cloth_size=cloth_size, soft=True, dense=args.dense)
else:
    if args.env == "interact":
        sys = Scene.Scene(cloth_size=cloth_size, dense=args.dense)
    else:
        sys = Scene.Scene(cloth_size=cloth_size)
sys.cloths[0].Kb[None] = args.Kb

colors = ti.Vector.field(3, dtype=float, shape=sys.tot_NV)

sys.init_all()
sys.get_colors(colors)
sys.mu_cloth_elastic[None] = args.mu


window = ti.ui.Window('surface test', res=(800, 800), vsync=True, show_window=False)
canvas = window.get_canvas()
canvas.set_background_color((0.5, 0.5, 0.5))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(-0.2, 0.2, 0.05)
camera.lookat(0, 0, 0)
camera.up(0, 0, 1)

save_path = f"../data/cmaes_traj_{args.env}_{args.trial}"
if not os.path.exists(save_path):
    os.mkdir(save_path)

gripper_cnt = sys.elastic_cnt - 1
if sys.enable_gripper:
    gripper_cnt = int((sys.effector_cnt - 1) // 2)

analy_grad = Grad(sys, tot_timestep, gripper_cnt)

x0 = (args.abs_step * 6 * gripper_cnt) * [5]
sigma0 = args.sigma

es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize': args.pop_size})
agent = agent_trajopt(tot_timestep, gripper_cnt, max_moving_dist=args.max_dist)

sub_steps = int(tot_timestep / args.abs_step)
scaling = 5.0 / (sub_steps * 0.0003)
scaling_angle = 5.0 / (sub_steps * 0.01)

def evaluate(x):
    sys.reset()
    agent.traj.fill(0)
    if not (args.load_dir is None):
        sys.load_all(args.load_dir)
    for ii in range(args.abs_step):
        for jj in range(sub_steps):
            if ii == 0 and jj == 0:
                continue
            i = ii * sub_steps + jj
            if not (i < 5 and args.env == "interact"):
                for j in range(gripper_cnt):
                    for k in range(3):
                        agent.traj[i, j, k] = agent.traj[i - 1, j, k] + (x[ii * 6 * gripper_cnt + j * 6 + k] - 5) / sub_steps / scaling
                        agent.traj[i, j, k + 3] = agent.traj[i - 1, j, k + 3] + (x[ii * 6 * gripper_cnt + j * 6 + k + 3] - 5) / sub_steps / scaling_angle

    agent.fix_action(0.015)

    early_stop = False
    stop_step = 0
    tot_reward = 0
    target = None
    if args.env == "forming":
        target = np.load(args.target_dir)
    if args.env == "balancing" or args.env == "bounce":
        analy_grad.copy_pos(sys, 0)
    for frame in range(1, tot_timestep):
        agent.get_action(frame)
        sys.action(frame, agent.delta_pos, agent.delta_rot)
        sys.time_step(projection_query, frame)
        early_stop = sys.check_early_stop(frame)
        if early_stop:
            break

        stop_step = frame + 1
        if args.env == "balancing" or args.env == "bounce":
            analy_grad.copy_pos(sys, frame)

    reward = stop_step / tot_timestep * 0.1
    if not early_stop:
        if (args.reward_name is None) or (args.env == "balancing"):
            if (args.env == "balancing"):
                if args.reward_name == "compute_reward_throwing":
                    reward += sys.compute_reward_throwing(analy_grad) + 10
                else:
                    func = getattr(sys, args.reward_name)
                    if callable(func):
                        reward += func(analy_grad) + 5
                    else:
                        print(f"{args.reward_name}, not a callable function!!")
                        exit(0)
            elif (args.env == "forming"):
                reward += sys.compute_reward(target) + 5
            elif (args.env == "bounce"):
                reward += sys.compute_reward(analy_grad) + 5
            else:
                reward += sys.compute_reward() + 5
        else:
            func = getattr(sys, args.reward_name)
            if callable(func):
                reward += func() + 5
            else:
                print(f"{args.reward_name}, not a callable function!!")
                exit(0)

    return -reward

tot_count = 0
plot_x = []
plot_y = []

for ww in range(args.iter):
    X = es.ask()  # sample len(X) candidate solutions
    # print(len(X))
    # print(X[:5])
    tell_list = []
    for x in X:
        eva = evaluate(x)
        tell_list.append(eva)
        tot_count += 1
        plot_x.append(tot_count)
        plot_y.append(eva)
    # print("value:", tell_list)
    es.tell(X, tell_list)
    es.disp()
    plt.plot(plot_x, plot_y)
    plt.savefig(os.path.join(save_path, f"plot.png"))
    np.save(os.path.join(save_path, "plot_Data.npy"), np.array(plot_y))
    # print("fbest:", es.result.fbest)

    result = es.result
    x = result.xbest

    agent.traj.fill(0)
    for ii in range(args.abs_step):
        for jj in range(sub_steps):
            if ii == 0 and jj == 0:
                continue
            i = ii * sub_steps + jj
            if not (i < 5 and args.env == "interact"):
                for j in range(gripper_cnt):
                    for k in range(3):
                        agent.traj[i, j, k] = agent.traj[i - 1, j, k] + (
                                    x[ii * 6 * gripper_cnt + j * 6 + k] - 5) / sub_steps / scaling
                        agent.traj[i, j, k + 3] = agent.traj[i - 1, j, k + 3] + (
                                    x[ii * 6 * gripper_cnt + j * 6 + k + 3] - 5) / sub_steps / scaling_angle
    agent.fix_action(0.015)
    np_traj = agent.traj.to_numpy()
    np.save(os.path.join(save_path, f"traj_{ww}.npy"), np_traj)

    sys.reset()
    if not (args.load_dir is None):
        sys.load_all(args.load_dir)

    scene.set_camera(camera)
    scene.ambient_light([0.8, 0.8, 0.8])
    scene.point_light((2, 2, 2), (1, 1, 1))
    # sys.cloths[0].update_visual()
    scene.mesh(sys.x32, indices=sys.f_vis,
               per_vertex_color=colors)  # , index_offset=(nf//2)*3, index_count=(nf//2)*3)
    canvas.scene(scene)
    window.save_image(os.path.join(save_path, f"{0}.png"))

    for frame in range(1, tot_timestep):
        agent.get_action(frame)
        sys.action(frame, agent.delta_pos, agent.delta_rot)
        sys.time_step(projection_query, frame)
        if sys.check_early_stop(frame, True):
            print("should stop in", frame)
        scene.set_camera(camera)
        scene.ambient_light([0.8, 0.8, 0.8])
        scene.point_light((2, 2, 2), (1, 1, 1))
        # sys.cloths[0].update_visual()
        scene.mesh(sys.x32, indices=sys.f_vis,
                   per_vertex_color=colors)  # , index_offset=(nf//2)*3, index_count=(nf//2)*3)
        canvas.scene(scene)
        window.save_image(os.path.join(save_path, f"{frame}.png"))

    frames = []
    for j in range(tot_timestep):
        filename = os.path.join(save_path, f"{j}.png")
        frames.append(imageio.imread(filename))

    gif_name = os.path.join(save_path, f"GIF_{ww}.gif")
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.02)