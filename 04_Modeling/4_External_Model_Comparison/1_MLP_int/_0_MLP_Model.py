import torch
import torch.nn as nn
import torch.nn.functional as F


class Transient_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_dim = 12
        self.flux_dim = 12
        self.cooling_dim = 2
        self.net = nn.Sequential(
            nn.Linear(self.flux_dim + self.cooling_dim + self.theta_dim + 1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.theta_dim),
        )

    def forward(self, time_D_v, flux_v, cooling_v, envtemp_v, v_curr):
        theta = v_curr - envtemp_v
        int = self.net(torch.cat([flux_v, cooling_v, theta, time_D_v], dim=1))
        theta = theta + int
        temp_v_next = theta + envtemp_v
        return temp_v_next
