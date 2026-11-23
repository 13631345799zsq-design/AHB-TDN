import torch
import torch.nn as nn
import torch.nn.functional as F


class ODEFunc(nn.Module):
    def __init__(self, flux_dim, cooling_dim, theta_dim):
        super(ODEFunc, self).__init__()
        self.flux_dim = flux_dim
        self.cooling_dim = cooling_dim
        self.theta_dim = theta_dim

        self.net = nn.Sequential(
            nn.Linear(self.flux_dim + self.cooling_dim + self.theta_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.theta_dim),
        )

    def forward(self, theta, flux_v, cooling_v):
        dtheta_dt = self.net(torch.cat([flux_v, cooling_v, theta], dim=1))
        return dtheta_dt


class Transient_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_dim = 12
        self.flux_dim = 12
        self.cooling_dim = 2
        self.N = 1
        self.ode_func = ODEFunc(self.flux_dim, self.cooling_dim, self.theta_dim)

    def forward(self, time_D_v, flux_v, cooling_v, envtemp_v, v_curr):
        theta = v_curr - envtemp_v
        h = time_D_v / float(self.N)
        for _ in range(self.N):
            dT_dt = self.ode_func(theta, flux_v, cooling_v)
            theta = theta + h * dT_dt
        temp_v_next = theta + envtemp_v
        return temp_v_next
