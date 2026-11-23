import torch
import torch.nn as nn


class Transient_Model(nn.Module):
    def __init__(self, init_seed=42, init_log_bound=1.0, power_reference=30.0):
        super().__init__()
        self.theta_dim = 12
        self.flux_dim = 12
        self.cooling_dim = 2
        self.sink_dim = 12
        self.temp_scale = 80.0
        self.time_scale = 5.0
        self.power_reference = power_reference
        self.init_seed = init_seed
        self.init_log_bound = init_log_bound
        generator = torch.Generator().manual_seed(init_seed)

        # Areas only decode the existing dataset inputs; they do not initialize RC parameters.
        area = torch.tensor([
            50 * 50, 30 * 30, 20 * 20, 20 * 20, 20 * 20, 20 * 20,
            10 * 10, 10 * 10, 10 * 10, 10 * 10, 10 * 10, 10 * 10,
        ], dtype=torch.float32)

        # Dataset builder: flux_v = nominal_power * P_W / (100 * area_mm2 * 0.03).
        nominal_power = torch.tensor([30, 10, 6, 6, 6, 6, 3, 3, 3, 3, 3, 3])
        self.power_scale = 100.0 * area * 0.03 / nominal_power

        # Dimensionless C_bar = C * temp_scale / (time_scale * power_reference).
        self.log_C_heater = self._log_parameter((self.theta_dim,), generator, init_log_bound)
        self.log_C_sink = self._log_parameter((self.sink_dim,), generator, init_log_bound)
        
        # Dimensionless R_bar = R * power_reference / temp_scale.
        self.log_R_tim = self._log_parameter((self.theta_dim,), generator, init_log_bound)
        self.log_R_fan = self._log_parameter((self.sink_dim, self.cooling_dim), generator, init_log_bound)
        self.log_fan_exponent = self._log_parameter((self.cooling_dim,), generator, init_log_bound)
        self.log_R_base = self._log_parameter((self.sink_dim,), generator, init_log_bound)

        # Complete undirected graph: one fitted resistance for each sink-node pair.
        source, target = torch.triu_indices(self.sink_dim, self.sink_dim, offset=1)
        edge_count = source.numel()
        self.sink_incidence = torch.zeros(edge_count, self.sink_dim)
        self.sink_incidence[torch.arange(edge_count), source] = 1.0
        self.sink_incidence[torch.arange(edge_count), target] = -1.0
        # Internal resistances use the same dimensionless random initialization.
        self.log_R_sink = self._log_parameter((edge_count,), generator, init_log_bound)
        
    @staticmethod
    def _log_parameter(shape, generator, init_log_bound):
        return nn.Parameter(torch.empty(shape).uniform_(
            -init_log_bound, init_log_bound, generator=generator
        ))

    def get_extra_state(self):
        # Save units and scales; old physical-unit checkpoints must not load silently.
        return {
            'parameter_units': 'dimensionless_RC_v1',
            'temp_scale': self.temp_scale,
            'time_scale': self.time_scale,
            'power_reference': self.power_reference,
            'init_seed': self.init_seed,
            'init_log_bound': self.init_log_bound,
        }

    def set_extra_state(self, state):
        if state['parameter_units'] != 'dimensionless_RC_v1':
            raise ValueError('Checkpoint does not contain dimensionless RC parameters')
        for name in ('temp_scale', 'time_scale', 'power_reference', 'init_seed', 'init_log_bound'):
            setattr(self, name, state[name])

    def init_sink_state(self, v_curr):
        # Each local sink starts from its paired heater temperature as an unmeasured-state estimate.
        return v_curr.clone()

    def forward(self, time_D_v, flux_v, cooling_v, envtemp_v, v_curr, sink_curr=None):
        # Treat sink-node temperatures as hidden latent states.
        if sink_curr is None:
            sink_curr = self.init_sink_state(v_curr)
        # Dataset dt is already scaled by time_scale; theta and ambient share (T - 20) / 80.
        dt = time_D_v
        # Convert flux_v to actual power.
        power = flux_v * self.power_scale.to(flux_v)
        power = power / self.power_reference
        # Exponentiate parameters to enforce positivity in the forward pass and provide a scaling effect during backpropagation.
        # exp enforces positivity; it does not itself add a regularization penalty.
        C_heater, C_sink = self.log_C_heater.exp(), self.log_C_sink.exp()
        R_tim = self.log_R_tim.exp()
        R_fan, R_base = self.log_R_fan.exp(), self.log_R_base.exp()
        R_sink = self.log_R_sink.exp()
        # Fan-dependent thermal resistance response.
        airflow_ratio = cooling_v.clamp_min(0)
        exponent = self.log_fan_exponent.exp()
        offset = 0.01
        fan_response = ((airflow_ratio + offset).pow(exponent) - offset ** exponent
        ) / ((1 + offset) ** exponent - offset ** exponent)
        R_air = 1.0 / (1.0 / R_base + (fan_response[:, None, :] / R_fan).sum(dim=2))

        # L = B.T @ diag(1/R_sink) @ B: symmetric internal conduction with zero net heat generation.
        incidence = self.sink_incidence.to(R_sink)
        sink_laplacian = incidence.T @ (incidence / R_sink[:, None])

        # RC energy-balance equations.
        # Heat transferred per unit temperature difference.
        dt_over_R_tim = dt / R_tim
        dt_over_R_air = dt / R_air

        # Rearrange the heater equation: heater_coeff * temp_v_next - dt_over_R_tim * sink_next = heater_rhs.
        heater_coeff = C_heater + dt_over_R_tim
        heater_rhs = C_heater * v_curr + dt * power

        # Substitute the heater expression into the sink equation and eliminate variables to obtain sink_coeff * sink_next = sink_rhs.
        sink_coeff = (
            C_sink + dt_over_R_air
            + C_heater * dt_over_R_tim / heater_coeff
        )
        sink_rhs = (
            C_sink * sink_curr + dt_over_R_air * envtemp_v
            + dt_over_R_tim * heater_rhs / heater_coeff
        )
        sink_coeff = torch.diag_embed(sink_coeff) + dt.unsqueeze(-1) * sink_laplacian

        # Solve the sink temperatures first, then substitute into the heater balance; both belong to the coupled solution at the same instant.
        sink_next = torch.linalg.solve(sink_coeff, sink_rhs.unsqueeze(-1)).squeeze(-1)
        temp_v_next = (heater_rhs + dt_over_R_tim * sink_next) / heater_coeff
        return temp_v_next, sink_next
