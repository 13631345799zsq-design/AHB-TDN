import torch
import torch.nn as nn


class GRU_Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net_z = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.net_r = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.net_h = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)

    def forward(self, x, h_old):
        xh = torch.cat([x, h_old], dim=1)
        z = torch.sigmoid(self.net_z(xh))
        r = torch.sigmoid(self.net_r(xh))
        xrh = torch.cat([x, r * h_old], dim=1)
        h_tilde = torch.tanh(self.net_h(xrh))
        h_curr = (1.0 - z) * h_old + z * h_tilde
        return h_curr


class Transient_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_dim = 12
        self.flux_dim = 12
        self.cooling_dim = 2
        self.hidden_dim = 16
        self.gru_cell = GRU_Cell(
            self.flux_dim + self.cooling_dim + self.theta_dim + 1, self.hidden_dim
        )
        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.theta_dim),
        )

    def forward(self, time_D_v, flux_v, cooling_v, envtemp_v, v_curr, h_old=None):
        theta = v_curr - envtemp_v
        if h_old is None:
            h_old = theta.new_zeros(theta.shape[0], self.hidden_dim)
        x = torch.cat([flux_v, cooling_v, theta, time_D_v], dim=1)
        h_curr = self.gru_cell(x, h_old)
        int = self.net(h_curr)
        temp_v_next = v_curr + int
        return temp_v_next, h_curr
