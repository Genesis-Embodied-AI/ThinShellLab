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
parser.add_argument('--pop_size', type=int, default=5)
parser.add_argument('--iter', type=int, default=10)
parser.add_argument('--tot_step', type=int, default=60)
parser.add_argument('--sigma', type=float, default=1.0)
parser.add_argument('--trial', type=str, default="0")
parser.add_argument('--env', type=str, default="")
parser.add_argument('--traj', type=str, default="")
parser.add_argument('--Kb', type=float, default=100.0)
parser.add_argument('--max_dist', type=float, default=0.002)
parser.add_argument('--mu', type=float, default=1.0)
parser.add_argument('--mu_cloth', type=float, default=1.0) \
args = parser.parse_args()

ti.init(ti.cpu, device_memory_fraction=0.5, default_fp=ti.f64, default_ip=ti.i32, fast_math=False,
        offline_cache=True, offline_cache_max_size_of_files=1024 ** 3,
        offline_cache_cleaning_policy='version')

from geometry import projection_query
import linalg
from analytic_grad_single import Grad
from agent.traj_opt_single import agent_trajopt
import cma
import importlib

from analytic_grad_single import Grad

importlib.invalidate_caches()
Scene = importlib.import_module(f'Scene_{args.env}')

tot_timestep = args.tot_step
cloth_size=0.06

sys = Scene.Scene(cloth_size=cloth_size)
sys.cloths[0].Kb[None] = args.Kb

colors = ti.Vector.field(3, dtype=float, shape=sys.tot_NV)

sys.init_all()
sys.get_colors(colors)
sys.mu_cloth_elastic[None] = args.mu




# window = ti.ui.Window('surface test', res=(800, 800), vsync=True, show_window=False)
# canvas = window.get_canvas()
# canvas.set_background_color((0.5, 0.5, 0.5))
# scene = ti.ui.Scene()
# camera = ti.ui.Camera()
# camera.position(-0.2, 0.2, 0.05)
# camera.lookat(0, 0, 0)
# camera.up(0, 0, 1)

save_path = f"../data/cmaes_traj_{args.env}_{args.trial}"
if not os.path.exists(save_path):
    os.mkdir(save_path)

gripper_cnt = sys.elastic_cnt - 1
if sys.enable_gripper:
    gripper_cnt = int((sys.effector_cnt - 1) // 2)

x0 = [0, 0]
sigma0 = args.sigma

es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize': args.pop_size})
if gripper_cnt > 0:
    agent = agent_trajopt(tot_timestep, gripper_cnt, max_moving_dist=args.max_dist)
    func = getattr(agent, args.traj)
    if callable(func):
        func()
else:
    agent = agent_trajopt(tot_timestep, 1, max_moving_dist=args.max_dist)



def evaluate(x):
    sys.reset()
    if args.env == "slide":
        sys.mu_cloth_cloth[None] = args.mu_cloth + x[0]
        sys.mu_cloth_cloth[None] = max(0.0001, sys.mu_cloth_cloth[None])
    else:
        sys.cloths[0].Kb[None] = args.Kb + x[0] * 200
        sys.cloths[0].Kb[None] = max(0.0001, sys.cloths[0].Kb[None])

    for frame in range(1, tot_timestep):
        if gripper_cnt > 0:
            agent.get_action(frame)
            sys.action(frame, agent.delta_pos, agent.delta_rot)
        sys.time_step(projection_query, frame)

    reward = sys.compute_reward()
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

    print(x)