import torch
import torch.nn as nn
import torch.nn.functional as F


class ThermalDiffusionLayer(nn.Module):
    def __init__(self, theta_dim, augment_dim):
        super().__init__()
        self.theta_dim = theta_dim
        edge_index = torch.tensor(
            [
                (target, source)
                for target in range(theta_dim)
                for source in range(theta_dim)
                if target != source
            ],
            dtype=torch.long,
        ).t()
        self.register_buffer("edge_index", edge_index)
        edge_num = theta_dim * (theta_dim - 1)
        self.theta_weight = nn.Parameter(torch.full((edge_num,), 0.01))
        self.hidden_weight = nn.Parameter(torch.full((theta_dim, augment_dim), 0.001))

    def project_positive_weights(self):
        with torch.no_grad():
            self.theta_weight.clamp_(min=0.0)
            self.hidden_weight.clamp_(min=0.0)

    def forward_components(self, theta, h_old):
        diffusion_matrix = theta.new_zeros(self.theta_dim, self.theta_dim)
        target, source = self.edge_index
        diffusion_matrix[target, source] = self.theta_weight
        theta_diffusion = theta @ diffusion_matrix.t() - theta * diffusion_matrix.sum(
            dim=1
        )
        hidden_diffusion = (
            h_old @ self.hidden_weight.t() - theta * self.hidden_weight.sum(dim=1)
        )
        return theta_diffusion, hidden_diffusion

    def forward(self, theta, h_old):
        theta_diffusion, hidden_diffusion = self.forward_components(theta, h_old)
        return theta_diffusion + hidden_diffusion


class ODEFunc(nn.Module):
    def __init__(self, flux_dim, cooling_dim, theta_dim, augment_dim):
        super(ODEFunc, self).__init__()
        self.flux_dim = flux_dim
        self.cooling_dim = cooling_dim
        self.theta_dim = theta_dim
        self.augment_dim = augment_dim
        # ODE network for the augmented state
        alpha_init = 0.01
        self.flux_alpha = nn.Parameter(alpha_init * torch.ones(self.theta_dim))

        self.net_cooling = nn.Sequential(
            nn.Linear(self.cooling_dim, self.theta_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.theta_dim, self.theta_dim, bias=False),
        )
        self.net_theta_1 = nn.Linear(
            self.theta_dim + self.augment_dim, self.theta_dim, bias=False
        )
        self.net_theta_2 = ThermalDiffusionLayer(self.theta_dim, self.augment_dim)

    def project_positive_weights(self):
        with torch.no_grad():
            for layer in self.net_cooling:
                if isinstance(layer, nn.Linear):
                    layer.weight.clamp_(min=0.0)
            self.net_theta_1.weight.clamp_(min=0.0)
        self.net_theta_2.project_positive_weights()

    def forward(self, theta, flux_v, cooling_v, h_old):
        dtheta_dt = (
            self.flux_alpha * flux_v
            - self.net_cooling(cooling_v)
            * self.net_theta_1(torch.cat([theta, h_old], dim=1))
            + self.net_theta_2(theta, h_old)
        )
        return dtheta_dt


class MLP_Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.beta = 50.0
        self.mlp_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, h_old):
        h = self.mlp_net(torch.cat([x, h_old], dim=1))
        h_curr = (
            h + F.softplus(-h, beta=self.beta) - F.softplus(h - 1.0, beta=self.beta)
        )
        return h_curr


class Transient_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_dim = 12
        self.flux_dim = 12
        self.cooling_dim = 2
        self.augment_dim = 16
        self.N = 1
        self.ode_func_core = ODEFunc(
            self.flux_dim, self.cooling_dim, self.theta_dim, self.augment_dim
        )
        self.mlp_cell = MLP_Cell(
            (self.theta_dim + self.flux_dim + self.cooling_dim), self.augment_dim
        )

    def forward(self, time_D_v, flux_v, cooling_v, envtemp_v, v_curr, h_old=None):
        theta = v_curr - envtemp_v
        if h_old is None:
            h_old = theta.new_zeros(theta.shape[0], self.augment_dim)
        h = time_D_v / float(self.N)
        for _ in range(self.N):
            dT_dt = self.ode_func_core(theta, flux_v, cooling_v, h_old)
            temp_v_next = v_curr + h * dT_dt
        h_curr = self.mlp_cell(torch.cat([theta, flux_v, cooling_v], dim=1), h_old)
        return temp_v_next, h_curr
