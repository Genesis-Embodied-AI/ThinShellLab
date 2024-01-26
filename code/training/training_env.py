""" 
    multi_env.py
    Implemented by Elgce August, 2023
    Use implemented class TactileEnv to train ppo policy
    For use of stable-baselines3
"""

import numpy as np
import taichi as ti
import os

off_screen = True
if off_screen:
    os.environ["PYTHON_PLATFORM"] = 'egl'
ti.init(ti.cpu, device_memory_fraction=0.5, default_fp=ti.f64, default_ip=ti.i32, fast_math=False,
        offline_cache=True, offline_cache_max_size_of_files=1024 ** 3,
        offline_cache_cleaning_policy='version')

from RL_env import Env
from RL_eval_env import EvalEnv
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import torch
device_count = torch.cuda.device_count()
print(f"Number of available devices: {device_count}")

for i in range(device_count):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

##################################################################
# init taichi env & use implemented Contact class
##################################################################
# if trained on servers, off_screen = True, otherwise False


##################################################################


##################################################################
# A class for train taichi_tasks with ppo algorithm
# num_envs: number of envs for multi-processing training
# task_type: name of task to do
# num_sub_steps: num of sub_steps for update iteration
# num_total_step: num of total_step, maximum step length
# dt: dt of the simulated env as in taichi envs
# n_sensors: related to how many sensors you have (noted as n), 2*n
##################################################################



class Trainer():
    def __init__(self, num_envs, num_eval_envs, env_name, time_step, training_iterations, resume, off_screen, algo, reward_name, load_dir, task_name, Kb, mu, model):
        self.num_envs = num_envs
        self.num_eval_envs = num_eval_envs
        self.training_iterations = training_iterations
        self.tot_step = time_step
        # self.env = TactileEvalEnv(env_name, time_step, reward_name, load_dir, task_name, Kb, mu, model=None)
        self.env_kwargs = {"sys_name": env_name, \
                           "time_step": time_step, "reward_name": reward_name,
                           "load_dir": load_dir, "task_name": task_name, "Kb": Kb, "mu": mu, "model": model}
        self.vec_env = make_vec_env(Env, n_envs=self.num_envs, env_kwargs=self.env_kwargs)
        self.checkpoint_path = "../data/checkpoints/" + task_name + "/"
        self.eval_env_kwargs = {"sys_name": env_name, \
                           "time_step": time_step, "reward_name": reward_name,
                           "load_dir": load_dir, "task_name": task_name, "Kb": Kb, "mu": mu, "model": None}
        self.eval_env = make_vec_env(EvalEnv, n_envs=self.num_eval_envs, env_kwargs=self.eval_env_kwargs)
        self.resume = resume
        self.off_screen = off_screen
        self.algo = algo
        # render guis as taichi envs do
        self.eval_rwd = 0
        self.model_name = model

    # use multiprocessing ppo env to train for self.training_iterations
    def train(self):
        if not resume:
            if self.model_name == "RecurrentPPO":
                self.model = self.algo("MlpLstmPolicy", self.vec_env, verbose=1, tensorboard_log=self.checkpoint_path)
            else:
                self.model = self.algo("MlpPolicy", self.vec_env, verbose=1, tensorboard_log=self.checkpoint_path)
        else:
            self.load_model()
        self.eval_callback = EvalCallback(eval_env=self.eval_env, best_model_save_path=self.checkpoint_path,
                                          log_path=self.checkpoint_path, eval_freq=self.tot_step * 50)
        self.model.learn(total_timesteps=self.training_iterations * self.tot_step, \
                         callback=self.eval_callback)
        print("================= Training process has finished! =================")

    def load_model(self):
        try:
            self.model = self.algo.load(self.checkpoint_path + "best_model")
        except FileNotFoundError as e:
            print(f"model file does not exist!: {e}")

    # if there is no previous saved ckp, raise error, else load it and evaluate it with single env
    # def evaluate(self):
    #     # resume and evaluate
    #     self.load_model()
    #     obs, _ = self.env.reset()
    #     for item in range(self.task.total_steps):
    #         action, _states = self.model.predict(obs)
    #         obs, rewards, dones, truncated, infos = self.env.step(action)
    #         print("action:", action)
    #         print("rewards:", rewards)
    #         print("obs:", obs)
    #         self.eval_rwd = rewards
    #     print("total reward:", self.eval_rwd)

    # call tactile env's render function to render forward process
    def render(self):
        pass


##################################################################
from argparse import ArgumentParser
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--num_env', type=int, default=4)
parser.add_argument('--num_eval_envs', type=int, default=4)
parser.add_argument('--tot_step', type=int, default=60)
parser.add_argument('--env', type=str, default="")
parser.add_argument('--Kb', type=float, default=100.0)
parser.add_argument('--mu', type=float, default=1.0)
parser.add_argument('--reward_name', type=str, default=None)
parser.add_argument('--load_dir', type=str, default=None)
parser.add_argument('--task_name', type=str, default=None)
parser.add_argument('--model', type=str, default="RecurrentPPO")
args = parser.parse_args()

if __name__ == "__main__":
    # choose tasks from registered tasks
    task_type = "surface_following"
    # set number of envs to train
    num_env = 2
    num_eval_envs = 2
    training_iterations = 100000

    resume = False  # this parameter determines whether or not to resume previous trained ckp, True for use
    if args.model == "RecurrentPPO":
        algo = RecurrentPPO  # SAC
    elif args.model == "PPO":
        algo = PPO
    else:
        algo = SAC
    trainer = Trainer(num_envs=args.num_env, num_eval_envs=args.num_eval_envs, env_name=args.env, time_step=args.tot_step, training_iterations=training_iterations, resume=False,
                      off_screen=True, algo=algo, reward_name=args.reward_name, load_dir=args.load_dir, task_name=args.task_name, Kb=args.Kb, mu=args.mu, model=args.model)
    trainer.train()
##################################################################