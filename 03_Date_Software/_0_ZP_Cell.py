import torch
import torch.nn as nn

class ZP_Cell(nn.Module):
    def __init__(self, trend_alpha=0.05, zp_alpha=0.03):
        super().__init__()
        if not 0.0 < trend_alpha <= 1.0:
            raise ValueError("trend_alpha must be in (0, 1]")
        if not 0.0 < zp_alpha <= 1.0:
            raise ValueError("zp_alpha must be in (0, 1]")
        self.trend_alpha = float(trend_alpha)
        self.zp_alpha = float(zp_alpha)

    def forward(self, v_measurement, time_step, dT_dt_core, zp_update_available, state_old=None):
        batch_size = v_measurement.shape[0]
        if state_old is None:
            state_old = {
                "temp": v_measurement,   # 温度状态为测量值
                "measurement_prev": v_measurement,   # 上一时刻测量值(用于计算测量温升率)
                "zp": torch.zeros_like(v_measurement),   # 零点纠偏
                "previous_available": torch.zeros(batch_size, 1, dtype=torch.bool, device=v_measurement.device),
                "trend": torch.zeros_like(v_measurement),   # EMA 温升趋势估计
                "update_active": torch.zeros(batch_size, 1, dtype=torch.bool, device=v_measurement.device,),
            }

        core = dT_dt_core.detach()
        dt = time_step.clamp_min(1e-6).expand(-1, v_measurement.shape[1])

        # ---- 测量温升率(差分) ----
        measured_rate = (v_measurement - state_old["measurement_prev"]) / dt
        measured_rate = torch.where(
            state_old["previous_available"],
            measured_rate,
            core,                       # 首帧无差分，以物理模型温升率为初始估计
        )

        # ---- EMA 更新(仅在校正窗口内, 窗口外保持) ----
        trend_alpha = self.trend_alpha
        trend_new = (1.0 - trend_alpha) * state_old["trend"] + trend_alpha * measured_rate
        trend = torch.where(zp_update_available, trend_new, state_old["trend"])
        # Keep ZP as a slowly varying zero-point state instead of replacing it
        # with the full correction whenever the core derivative changes.
        zp_target = trend - core
        zp_new = state_old["zp"] + self.zp_alpha * (zp_target - state_old["zp"])
        zp_curr = torch.where(zp_update_available, zp_new, state_old["zp"])

        state_curr = {
            "temp": v_measurement,
            "measurement_prev": v_measurement,
            "zp": zp_curr,
            "previous_available": torch.ones(batch_size, 1, dtype=torch.bool, device=v_measurement.device),
            "trend": trend,
            "update_active": zp_update_available,
        }
        return zp_curr, state_curr
