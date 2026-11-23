import torch
import torch.nn as nn


class DBC_Branch(nn.Module):
    TREND_WINDOW_SIZE = 10

    def __init__(self, alpha_int=0.001):
        super().__init__()
        self.alpha_int = float(alpha_int)

    def forward(self, v_measurement, time_step, dT_dt_core, state_old=None):
        batch_size = v_measurement.shape[0]
        if state_old is None:
            state_old = {
                "samples_since_update": torch.zeros(
                    batch_size, 1, dtype=torch.long, device=v_measurement.device
                ),
                "global_time": v_measurement.new_zeros(batch_size, 1),
                "prev_v": v_measurement.new_zeros(batch_size, v_measurement.shape[1]),
                "prev_time": v_measurement.new_zeros(batch_size, 1),
                "has_prev_point": torch.zeros(
                    batch_size, 1, dtype=torch.bool, device=v_measurement.device
                ),
                "dbc": torch.zeros_like(v_measurement),
                "trend": torch.zeros_like(v_measurement),
            }

        core = dT_dt_core.detach()
        dt = time_step.clamp_min(1e-6).view(batch_size, 1)
        global_time = state_old["global_time"] + dt
        samples_since_update = state_old["samples_since_update"] + 1
        update_ready = samples_since_update >= self.TREND_WINDOW_SIZE

        # Backward difference over TREND_WINDOW_SIZE points: (v_curr - v_prev) / (t_curr - t_prev).
        diff_time = (global_time - state_old["prev_time"]).clamp_min(1e-12)
        point_trend = (v_measurement - state_old["prev_v"]) / diff_time
        # Do not apply another EMA to point_trend; retain the original value when the first reference has no valid trend.
        trend_new = torch.where(
            state_old["has_prev_point"], point_trend, state_old["trend"]
        )
        trend = torch.where(update_ready, trend_new, state_old["trend"])
        dbc_new = state_old["dbc"] + self.alpha_int * (trend - core - state_old["dbc"])
        # Do not update dbc when no valid trend is available.
        dbc_new = torch.where(state_old["has_prev_point"], dbc_new, state_old["dbc"])
        dbc_curr = torch.where(update_ready, dbc_new, state_old["dbc"])
        samples_since_update = torch.where(
            update_ready, torch.zeros_like(samples_since_update), samples_since_update
        )
        prev_v = torch.where(update_ready, v_measurement, state_old["prev_v"])
        prev_time = torch.where(update_ready, global_time, state_old["prev_time"])
        has_prev_point = torch.where(
            update_ready,
            torch.ones_like(state_old["has_prev_point"]),
            state_old["has_prev_point"],
        )

        state_curr = {
            "samples_since_update": samples_since_update,
            "global_time": global_time,
            "prev_v": prev_v,
            "prev_time": prev_time,
            "has_prev_point": has_prev_point,
            "dbc": dbc_curr,
            "trend": trend,
        }
        return dbc_curr, state_curr, update_ready
