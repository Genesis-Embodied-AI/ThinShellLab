import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import taichi as ti


class Policy(nn.Module):
    def __init__(self, feat_dim, obs_dim, action_dim, tot_timestep, num_layers=2):
        super(Policy, self).__init__()
        self.tot_timestep = tot_timestep
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.feat_dim = feat_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.obs_dim, self.feat_dim, num_layers, batch_first=True, dtype=torch.double)
        self.mlp2 = nn.Linear(feat_dim, feat_dim, dtype=torch.double)
        self.mlp3 = nn.Linear(feat_dim, action_dim, dtype=torch.double)
        self.max_dist = 0.12
        self.max_move = 0.0005

    def forward(self, observations):
        net, _ = self.lstm(observations)
        net = F.leaky_relu(self.mlp2(net))
        net = self.mlp3(net)
        return net

    def forward_action(self, observations):
        net = self.forward(observations)
        batch_size = net.shape[0]
        sequence_length = net.shape[1]
        net = net.reshape(-1, self.action_dim)
        # pos, rot, dist = net[:, :3], net[:, 3:6], net[:, 6]
        # norm = torch.norm(pos, dim=1) + torch.norm(rot, dim=1) * self.max_dist + torch.abs(dist)
        # norm, _ = torch.max(
        #     torch.cat([norm.reshape(-1, 1) / self.max_move, torch.ones((net.shape[0], 1), dtype=torch.double, device='cuda:0')],
        #               dim=1), dim=1)
        # norm = norm.reshape(-1, 1).repeat(1, self.action_dim)
        # net = torch.div(net, norm)
        return net.reshape(batch_size, sequence_length, self.action_dim)

    def get_action(self, observations):
        observations = observations.reshape(1, -1, self.action_dim)
        actions = self.forward_action(observations)
        return actions[0, -1]

    def get_action_taichi(self, observations):

        with torch.no_grad():
            action = self.forward_action(observations).detach().cpu().numpy()[0, -1]
        # print(time_step, action)
        delta_pos = ti.Vector([action[0], action[1], action[2]])
        delta_rot = ti.Vector([action[3], action[4], action[5]])
        delta_dist = action[6]
        return delta_pos, delta_rot, delta_dist


