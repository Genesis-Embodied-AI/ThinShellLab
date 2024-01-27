""" 
    tactile_env.py
    Implemented by Elgce August, 2023
    TactileEnv class inherited from gym.Env
    Wrap implemented Contact_models like Surface_follow
    For use of stable-baselines3
"""
from RL_env import Env

##################################################################
# implement TactileEvalEnv, 
# inherit from TactileEnv 
# for eval usage in rl algos
##################################################################
class EvalEnv(Env):

    def __init__(self, sys_name, time_step, reward_name=None, load_dir=None, task_name=None, Kb=100, mu=5.0, model=None):
        super().__init__(sys_name, time_step, reward_name, load_dir, task_name, Kb, mu, model)
    
    def compute_rewards(self):
        """_summary_
        Args:
            obs (array from self.get_observations): tactile image
        Returns:
            rewards : taichi env's -loss
        """
        if self.reward_name is None:
            if self.sys_name == "forming":
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
        ret_rewards = rewards - self.last_reward
        self.last_reward = rewards
        if self.sys.check_early_stop(self.time_step, RL=True):
            ret_rewards = 0.0
        return ret_rewards
    