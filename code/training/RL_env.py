""" 
    tactile_env.py
    Implemented by Elgce August, 2023
    TactileEnv class inherited from gym.Env
    Wrap implemented Contact_models like Surface_follow
    For use of stable-baselines3
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

import os
import taichi as ti

from matplotlib import pyplot as plt
from geometry import projection_query
import importlib

##################################################################
# init taichi env & use implemented Contact class
##################################################################
# if trained on servers, off_screen = True, otherwise False
off_screen = True
if off_screen:
    os.environ["PYTHON_PLATFORM"] = 'egl'
##################################################################



##################################################################
# implement TactileEnv, 
# a wrapper of implemented taichi contact classes
# for use of stable_baselines based DRL training
##################################################################

class Env(gym.Env):
    """_TactileEnv_
    A class implemented for RL training of current tasks
    init with implemented contact_model
        required ti.kernel of contact_model:
        (since it's all Python scope here, all these should not be ti.func)
            a. (Python) prepare_env: including any code between __init__ and ts loop
            b. (Python) apply_action: input action, apply such action in taichi env, end with self.memory_to_cache
            c. (Python) calculate_force: do whatever needed to calculate force of mpm and fem objects
            d. (Python) compute_loss: run all funcs to get rwd in taichi env
        required fields and vectors of contact_model:
            a. ()
            b. ()
    PPO related parameters:
        a. obs: use the image of tactile info currently
        b. rewards:
    NOTE: this version, we use CNN (obs: images of tactile) as input
    """

    count = 0
    def __init__(self, sys_name, time_step, reward_name=None, load_dir=None, task_name=None, Kb=100.0, mu=5.0, model="PPO"):
        super().__init__()
        importlib.invalidate_caches()
        Scene = importlib.import_module(f'Scene_{sys_name}')

        cloth_size = 0.06
        self.model = model
        if sys_name == "folding_2" or  sys_name == "push":
            cloth_size = 0.1
        self.sys_name = sys_name

        now_cnt = Env.count + 1
        Env.count += 1
        if sys_name == "interact":
            sys = Scene.Scene(cloth_size=cloth_size, dense=20000)
        else:
            sys = Scene.Scene(cloth_size=cloth_size)

        self.target_pos = None
        if sys_name == "push":
            self.target_pos = np.load("../data/push_pos_save/cloth_pos.npy")
        self.sys = sys
        self.sys.init_all()
        self.sys.cloths[0].Kb[None] = Kb
        self.sys.mu_cloth_elastic[None] = mu
        self.n_actions = self.sys.action_dim
        self.n_observations = self.sys.obs_dim
        self.action_space = spaces.Box(low=-0.001, high=0.001, shape=(self.n_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(self.n_observations,),
                                            dtype=np.float32)
        self.time_step = 0 # record the time of this env, add 1 each step
        self.time_limit = time_step
        self.total_rewards = 0
        self.task_name = task_name
        self.reward_name = reward_name
        self.load_dir = load_dir
        # arrays for matplotlib
        self.iters = []
        self.rewards = []
        self.reset()
        self.model = model # for model.save
        self.max_rewards = 0 # reward the maximum rewards
        self.last_reward = 0
        if model is None:
            self.save_dir = None
        else:
            self.save_dir = os.path.join("../data/checkpoints", f"{self.task_name}_plot")
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            for ii in range(10):
                save_dir = os.path.join(self.save_dir, f"{self.model}_{ii}")
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                    self.save_dir = save_dir
                    break
    
    def step(self, action):
        """_summary_
        Args:
            action (tuple with 2 elements): actions got by ppo network, (d_pos, d_ori)
        Returns:
            obs: observation got in taichi env after apply action
            rewards: rewards of current observation
            dones: if the env if done, now only check the time_steps of env, may check more later
            infos: []
        """
        real_rewards = self.compute_real_rewards()
        self.time_step += 1
        if self.time_step <= self.time_limit - 1 and self.task_name == "balance_RL":
            real_rewards -= 0.5
        for i in range(self.sys.gripper.n_part):
            for j in range(3):
                xx = i * 6 + j
                self.sys.delta_pos[i][j] = action[xx]
                self.sys.delta_rot[i][j] = action[xx + 3]
        self.sys.action(self.time_step, self.sys.delta_pos, self.sys.delta_rot)

        self.sys.time_step(projection_query, self.time_step)
        obs = self.get_observations()
        rewards = self.compute_rewards()
        dones = self.check_termination()
        infos = {} # TODO: if need more infos, add here
        truncated = dones
        if truncated:
            obs = np.zeros_like(obs) # otherwise, will get error for nan X float
            rewards = 0
            # NOTE: Below part used for single env and draw training curve
            # NOTE: May not use any more
            # self.rewards.append(self.total_rewards)
            # self.iters.append(len(self.rewards))
            # self.render_rewards()
            # if self.total_rewards >= self.max_rewards:
            #     self.max_rewards = self.total_rewards
            #     self.model.save("checkpoints/" + self.task_name + "/best")
            self.rewards.append(real_rewards)
            leng = len(self.rewards)
            if leng % 10 == 0:
                print(f"task: {self.task_name}", "episode: ", leng, "reward: ", real_rewards, "history max: ", np.max(np.array(self.rewards)))
                if not (self.save_dir is None):
                    np.save(os.path.join(self.save_dir, "plot_data.npy"), np.array(self.rewards))
        else:
            self.total_rewards += rewards
        return obs, rewards, dones, truncated, infos
    
    def reset(self, seed=None, options=None):
        """_summary_
        Args:
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.
        Returns:
            obs: observation of initial state given by self.get_observations
            infos: []
        """
        self.sys.reset()
        if self.load_dir is not None:
            self.sys.load_all(self.load_dir)
        infos = {}
        obs = self.get_observations()
        self.time_step = 0
        self.total_rewards = 0
        self.last_reward = 0
        return obs, infos
    
    def get_observations(self):
        """_summary_
        Returns:
            obs : we only use the marker points' position of fem_sensor1 here
        """
        self.sys.get_observation_kernel()
        obs = self.sys.observation.to_numpy().reshape(-1) # put all inputs into 1-D
        
        return obs
    
    def compute_rewards(self):
        """_summary_
        Args:
            obs (array from self.get_observations): tactile image
        Returns:
            rewards : rewards of present obs, same as taichi env
        """
        # calculate detailed rewards --> the same as taichi env is OK
        if self.reward_name is None:
            if self.sys_name == "push":
                rewards = self.sys.compute_reward(self.target_pos)
            else:
                rewards = self.sys.compute_reward()
        else:
            func = getattr(self.sys, self.reward_name)
            if callable(func):
                rewards = func()
            else:
                print(f"{self.reward_name}, not a callable function!!")
                exit(0)
        # print(rewards)
        rewards = np.exp(rewards)
        return rewards

    def compute_real_rewards(self):
        """_summary_
        Args:
            obs (array from self.get_observations): tactile image
        Returns:
            rewards : rewards of present obs, same as taichi env
        """
        # calculate detailed rewards --> the same as taichi env is OK
        if self.reward_name is None:
            if self.sys_name == "push":
                rewards = self.sys.compute_reward(self.target_pos)
            else:
                rewards = self.sys.compute_reward()
        else:
            func = getattr(self.sys, self.reward_name)
            if callable(func):
                rewards = func()
            else:
                print(f"{self.reward_name}, not a callable function!!")
                exit(0)

        return rewards
    
    def check_termination(self):
        if self.time_step >= self.time_limit:
            return True
        if self.sys.check_early_stop(self.time_step):
            return True
        return False
    
    def close(self):
        pass
    
    def render_rewards(self):
        """_summary_
            according to self.rewards & self.iters, render rewards curve
        """
        plt.figure(figsize=(20, 12))
        plt.plot(self.iters, self.rewards, marker='o')
        plt.title("mean rewards for " + self.task_name) # NOTE: change titles to what you need
        plt.xlabel("training iter")
        plt.ylabel("total reward")
        plt.savefig("../data/checkpoints/" + self.task_name + "/" +"iter_reward.png")
        
##################################################################
