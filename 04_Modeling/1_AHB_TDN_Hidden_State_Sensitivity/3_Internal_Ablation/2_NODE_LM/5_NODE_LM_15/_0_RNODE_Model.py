import torch
import torch.nn as nn
import torch.nn.functional as F


class ODEFunc(nn.Module):
    def __init__(self, flux_dim, cooling_dim, theta_dim, augment_dim):
        super(ODEFunc, self).__init__()
        self.beta = 50.0
        self.flux_dim = flux_dim
        self.cooling_dim = cooling_dim
        self.theta_dim = theta_dim
        self.augment_dim = augment_dim

        self.net = nn.Sequential(
            nn.Linear(
                self.flux_dim + self.cooling_dim + self.theta_dim + self.augment_dim, 64
            ),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.theta_dim + self.augment_dim),
        )

    def forward(self, theta, flux_v, cooling_v, h_old):
        dt_dt = self.net(torch.cat([flux_v, cooling_v, theta, h_old], dim=1))
        dtheta_dt = dt_dt[:, : self.theta_dim]
        a = dt_dt[:, self.theta_dim :]
        a_curr = (
            a + F.softplus(-a, beta=self.beta) - F.softplus(a - 1.0, beta=self.beta)
        )
        return dtheta_dt, a_curr


class Transient_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_dim = 12
        self.flux_dim = 12
        self.cooling_dim = 2
        self.augment_dim = 16
        self.N = 1
        self.ode_func = ODEFunc(
            self.flux_dim, self.cooling_dim, self.theta_dim, self.augment_dim
        )

    def forward(self, time_D_v, flux_v, cooling_v, envtemp_v, v_curr, a_old=None):
        theta = v_curr - envtemp_v
        if a_old is None:
            a_old = theta.new_zeros(theta.shape[0], self.augment_dim)
        h = time_D_v / float(self.N)
        for _ in range(self.N):
            dtheta_dt, a_curr = self.ode_func(theta, flux_v, cooling_v, a_old)
            temp_v_next = theta + h * dtheta_dt + envtemp_v
        return temp_v_next, a_curr
