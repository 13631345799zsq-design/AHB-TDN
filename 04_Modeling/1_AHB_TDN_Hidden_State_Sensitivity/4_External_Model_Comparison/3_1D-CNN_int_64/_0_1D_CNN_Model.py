import torch
import torch.nn as nn

HISTORY_LENGTH = 10


class _1D_CNN_Model(nn.Module):
    def __init__(self, input_dim, temp_dim=12, kernel_size=4):
        super(_1D_CNN_Model, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size),
            nn.GroupNorm(1, 64),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size),
            nn.GroupNorm(1, 64),
            nn.SiLU(),
            nn.Conv1d(64, temp_dim, kernel_size),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        correction = self.feature_extractor(x).transpose(1, 2)
        return correction


class Transient_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flux_dim = 12
        self.cooling_dim = 2
        self.envtemp_dim = 1
        self.temp_dim = 12
        self.time_dim = 1
        self.feature_dim = (
            self.flux_dim
            + self.cooling_dim
            + self.envtemp_dim
            + self.temp_dim
            + self.time_dim
        )
        self.kernel_size = 4

        # 1D-CNN backbone.
        self.cnn = _1D_CNN_Model(
            input_dim=self.feature_dim,
            temp_dim=self.temp_dim,
            kernel_size=self.kernel_size,
        )

    def forward(self, time_v, flux_v, cooling_v, envtemp_v, temp_v_curr):
        x = torch.cat([flux_v, cooling_v, envtemp_v, temp_v_curr, time_v], dim=-1)
        int = self.cnn(x)
        temp_next = temp_v_curr[:, -1:, :] + int
        return temp_next
