import math
import os
import time
from pathlib import Path
import numpy as np
import torch
import scipy.optimize as opt
from _0_RNODE_Model_Euler import Transient_Model
from _0_ZP_Cell import ZP_Cell


TEMP_OFFSET_C = 20.0
TEMP_SCALE_C = 80.0
FAN_SCALE_RPM = 6300.0
NODE_TIME_SCALE_S = 5.0
FLUX_MIN = 0.0
FLUX_MAX = 0.03
SERIES_LIMITS = {
    1: 30, 2: 10, 3: 6, 4: 6, 5: 6, 6: 6,
    7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3
}
FLUX_POWER_COEFF = np.array([
    30.0 / (100.0 * 50.0 * 50.0),
    10.0 / (100.0 * 30.0 * 30.0),
    6.0 / (100.0 * 20.0 * 20.0),
    6.0 / (100.0 * 20.0 * 20.0),
    6.0 / (100.0 * 20.0 * 20.0),
    6.0 / (100.0 * 20.0 * 20.0),
    3.0 / (100.0 * 10.0 * 10.0),
    3.0 / (100.0 * 10.0 * 10.0),
    3.0 / (100.0 * 10.0 * 10.0),
    3.0 / (100.0 * 10.0 * 10.0),
    3.0 / (100.0 * 10.0 * 10.0),
    3.0 / (100.0 * 10.0 * 10.0),
], dtype=np.float32)

# MPC配置参数
MPC_TUNED_CONFIG = {
    "model_path": str(Path(__file__).resolve().parent / "models_1" / "model_epoch_186.pth"),
    "device": "cpu",
    # 优化器参数
    "horizon": 20,
    "optimizer_iter": 10,
    "ftol": 1.0e-6,
    "default_dt_s": 1.0,
    "enable_zp_correction": True,
    # "enable_zp_correction": False,
    # 风扇参数边界条件
    "fan_min_rpm": 1000.0,
    "fan_max_rpm": 6300.0,
    "initial_fan_rpm": 1000.0,
    # MPC权重参数
    "weight_heat": 5.0e-2,
    "weight_grad": 0.0,
    # "weight_grad": 1.0e-4,
    "weight_cooling": 2.0e-3,  # 基准
    # "weight_slew": 0.0e0,
    "weight_slew": 5.0e-4,
    "slew_delta_rpm": 1000.0,
    "grad_target_c_per_s": 0.1,
}

# 转换代码
def temp_c_to_norm(temp_c):
    return (np.asarray(temp_c, dtype=np.float32) - TEMP_OFFSET_C) / TEMP_SCALE_C
def temp_norm_to_c(temp_norm):
    return TEMP_OFFSET_C + TEMP_SCALE_C * np.asarray(temp_norm, dtype=np.float32)
def fan_rpm_to_norm(fan_rpm):
    return np.asarray(fan_rpm, dtype=np.float32) / FAN_SCALE_RPM
def fan_norm_to_rpm(fan_norm):
    return FAN_SCALE_RPM * np.asarray(fan_norm, dtype=np.float32)
def power_w_to_flux_norm(power_w):
    flux = FLUX_POWER_COEFF * np.asarray(power_w, dtype=np.float32)
    return (flux - FLUX_MIN) / (FLUX_MAX - FLUX_MIN)
def flux_norm_to_power_w(flux_norm):
    flux = FLUX_MIN + (FLUX_MAX - FLUX_MIN) * np.asarray(flux_norm, dtype=np.float32)
    return flux / FLUX_POWER_COEFF


class NODEMPCFanController:
    RPM_OFF = 0
    def __init__(self, config=None):
        # MPC配置参数读取
        cfg = dict(MPC_TUNED_CONFIG)
        if config:
            cfg.update(config)
        self.horizon = int(cfg["horizon"])
        self.optimizer_iter = int(cfg["optimizer_iter"])
        self.ftol = float(cfg["ftol"])
        self.default_dt_s = float(cfg["default_dt_s"])
        self.enable_zp_correction = bool(cfg["enable_zp_correction"])
        self.fan_min_rpm = float(cfg["fan_min_rpm"])
        self.fan_max_rpm = float(cfg["fan_max_rpm"])
        self.fan_min_norm = float(fan_rpm_to_norm(self.fan_min_rpm))
        self.fan_max_norm = float(fan_rpm_to_norm(self.fan_max_rpm))
        self.bounds = [(self.fan_min_norm, self.fan_max_norm)] * 2
        self.initial_fan_norm = float(np.clip(
            fan_rpm_to_norm(float(cfg["initial_fan_rpm"])),
            self.fan_min_norm,
            self.fan_max_norm,
        ))
        self.grad_target_norm = float(cfg["grad_target_c_per_s"]) / TEMP_SCALE_C
        self.slew_delta_norm = float(cfg["slew_delta_rpm"]) / FAN_SCALE_RPM
        self.weights = {
            "heat": float(cfg["weight_heat"]),
            "grad": float(cfg["weight_grad"]),
            "cooling": float(cfg["weight_cooling"]),
            "slew": float(cfg["weight_slew"]),
        }
        # 控制标志位
        self._enabled = False
        # 运行设备配置
        device_cfg = str(cfg["device"]).lower()
        if device_cfg == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = device_cfg
        self._device = torch.device(device_name)
        # 模型导入
        model_path = Path(cfg["model_path"])
        self._model = Transient_Model().to(self._device)
        self._model.load_state_dict(torch.load(model_path, map_location=self._device, weights_only=True), strict=True)
        self._model.ode_func_core.project_positive_weights()
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad_(False)
        self._zp_cell = ZP_Cell().to(self._device)
        self._zp_cell.eval()
        # 风扇转速初始化
        self._last_output_rpm = np.array([self.fan_min_rpm, self.fan_min_rpm], dtype=np.float32)
        self._previous_u = fan_rpm_to_norm(self._last_output_rpm).reshape(1, 2)
        self._u = self._previous_u.reshape(2).astype(np.float32)
        # 目标温度初始化
        self._t_aim = 60.0
        self._channel_t_aim = None
        # 运行状态
        self._last_parts = {}
        self._last_loss_drpm = {}
        self._last_total_drpm = (0.0, 0.0)
        self._last_success = None
        self._last_message = ""
        self._last_fan_plan_rpm = np.repeat(self._last_output_rpm.reshape(1, 2), self.horizon, axis=0).astype(np.float32)
        self._actual_v_curr = None
        self._actual_h = None
        self._actual_zp_state = None
        self._actual_step_index = 0
        self._actual_elapsed_s = 0.0
        self._last_zp_status = {}
        self._last_augment_status = {}

    # MPC优化器启动停止函数蔟 ====================
    def enable(self):
        self._enabled = True
        self.reset()
    def disable(self):
        self._enabled = False
        self._last_output_rpm[:] = self.RPM_OFF
    def is_enabled(self):
        return self._enabled
    def reset(self):
        self._previous_u = np.array([[self.initial_fan_norm, self.initial_fan_norm]], dtype=np.float32)
        self._u = self._previous_u.reshape(2).astype(np.float32)
        self._last_output_rpm = fan_norm_to_rpm(self._previous_u.reshape(2)).astype(np.float32)
        self._last_parts = {}
        self._last_loss_drpm = {}
        self._last_total_drpm = (0.0, 0.0)
        self._last_success = None
        self._last_message = ""
        self._last_fan_plan_rpm = np.repeat(self._last_output_rpm.reshape(1, 2), self.horizon, axis=0).astype(np.float32)
        self._actual_v_curr = None
        self._actual_h = None
        self._actual_zp_state = None
        self._actual_step_index = 0
        self._actual_elapsed_s = 0.0
        self._last_zp_status = {}
        self._last_augment_status = {}
    # ==========================================

    # 目标温度设置函数簇 =========================
    def set_t_aim(self, t_aim):
        self._t_aim = float(t_aim)
        self._channel_t_aim = None
    def set_channel_t_aim(self, t_aim_by_channel):
        targets = self._coerce_vector(t_aim_by_channel, fill_value=self._t_aim)
        self._channel_t_aim = targets.astype(np.float32)
    # ==========================================

    @property
    def last_output(self):
        return (int(round(self._last_output_rpm[0])), int(round(self._last_output_rpm[1])))

    @property
    def last_status(self):
        return {
            "success": self._last_success,
            "message": self._last_message,
            "optimize_elapsed_s": float(self._last_parts.get("optimize_elapsed_s", math.nan)),
            "loss_drpm": {name: tuple(values) for name, values in self._last_loss_drpm.items()},
            "total_drpm": tuple(self._last_total_drpm),
            "fan_plan_rpm": self._last_fan_plan_rpm.copy(),
            "zero_point_correction": dict(self._last_zp_status),
            "augment_state": dict(self._last_augment_status),
        }

    def _prepare_actual_state(self, measured_v):
        if self._actual_v_curr is None:
            self._actual_v_curr = measured_v.detach().clone()

        actual_v = self._actual_v_curr.detach().clone()
        actual_h = None if self._actual_h is None else self._actual_h.detach().clone()
        actual_zp_state = self._clone_zp_state(self._actual_zp_state)
        correction = measured_v - actual_v
        actual_v = measured_v.detach().clone()
        if not self.enable_zp_correction:
            actual_zp_state = None

        return actual_v, actual_h, actual_zp_state, correction.detach()

    def _advance_actual_state(self, actual_v, actual_h, actual_zp_state, flux, cooling, env, measured_v, dt):
        time_d = self._to_tensor(np.array([[dt / NODE_TIME_SCALE_S]], dtype=np.float32))
        zp_update_available = torch.full(
            (actual_v.shape[0], 1),
            self.enable_zp_correction,
            dtype=torch.bool,
            device=self._device,
        )
        with torch.no_grad():
            h_for_core = actual_h
            if h_for_core is None:
                h_for_core = actual_v.mean(dim=1, keepdim=True).repeat(
                    1, self._model.augment_dim
                )
            core_dtheta_dt = self._model.ode_func_core(
                actual_v, flux, cooling, env, h_for_core
            )
            v_next, h_next = self._model(
                time_d,
                flux,
                cooling,
                env,
                actual_v,
                actual_h,
            )
            if self.enable_zp_correction:
                zp_curr, zp_state_next = self._zp_cell(
                    measured_v,
                    time_d,
                    core_dtheta_dt,
                    zp_update_available,
                    actual_zp_state,
                )
                v_next = v_next + time_d * zp_curr
            else:
                zp_state_next = None
        self._actual_v_curr = v_next.detach()
        self._actual_h = h_next.detach()
        self._actual_zp_state = (
            self._clone_zp_state(zp_state_next)
            if self.enable_zp_correction
            else None
        )
        self._actual_step_index += 1
        self._actual_elapsed_s += dt
        return core_dtheta_dt.detach()

    def compute(self, temperatures, heater_power=None, actual_fan_rpm=None, ambient_temp_c=25.0, t_aim=None, dt_s=None):
        # 启动检查
        if not self._enabled:
            return (self.RPM_OFF, self.RPM_OFF)
        # 采样数据的处理
        temps_c = self._coerce_vector(temperatures, fill_value=math.nan)
        valid_mask = np.isfinite(temps_c)
        if not np.any(valid_mask):
            return self.last_output
        if not np.all(valid_mask):
            fill = float(np.nanmax(temps_c))
            temps_c = np.where(valid_mask, temps_c, fill)
        power = self._coerce_vector(heater_power, fill_value=0.0)
        power = power_w_to_flux_norm(power)

        sampled_fan_rpm = self._coerce_fan_rpm(actual_fan_rpm)
        self._previous_u = fan_rpm_to_norm(sampled_fan_rpm).reshape(1, 2).astype(np.float32)
        u_plan_start = np.clip(self._previous_u, self.fan_min_norm, self.fan_max_norm)
        self._u = u_plan_start.reshape(2).astype(np.float32)


        target_c = self._resolve_target(t_aim)
        dt = self.default_dt_s if dt_s is None else max(float(dt_s), 1.0e-6)
        measured_v = self._to_tensor(temp_c_to_norm(temps_c).reshape(1, 12))
        v0, h0, zp0, correction = self._prepare_actual_state(measured_v)
        flux_h = self._to_tensor(np.repeat(power.reshape(1, 12), self.horizon, axis=0))
        env_norm = temp_c_to_norm(np.array([[float(ambient_temp_c)]], dtype=np.float32))
        env_h = self._to_tensor(np.repeat(env_norm, self.horizon, axis=0))
        dt_h = self._to_tensor(np.repeat(np.array([dt], dtype=np.float32), self.horizon, axis=0))
        target_h = self._to_tensor(np.repeat(temp_c_to_norm(target_c).reshape(1, 12), self.horizon, axis=0))
        sampled_u = self._to_tensor(self._previous_u.reshape(1, 2))

        # 优化启动
        previous_rpm = fan_norm_to_rpm(self._previous_u.reshape(2)).astype(np.float32)
        prediction_counter_before = self._actual_step_index
        actual_v_before_prediction = self._actual_v_curr.detach().clone()
        actual_h_before_prediction = None if self._actual_h is None else self._actual_h.detach().clone()
        actual_zp_before_prediction = self._clone_zp_state(self._actual_zp_state)
        optimize_start_s = time.perf_counter()
        best_plan, result, parts = self._optimize(v0, h0, zp0, flux_h, env_h, dt_h, target_h)
        optimize_elapsed_s = time.perf_counter() - optimize_start_s
        loss_effects = self._marginal_loss_rpm_effects(
            v0,
            h0,
            zp0,
            flux_h,
            env_h,
            dt_h,
            target_h,
            previous_rpm,
            fan_norm_to_rpm(best_plan[0]).astype(np.float32),
        )
        future_tmax_c = self._rollout_future_tmax_c(v0, h0, zp0, flux_h, env_h, dt_h, best_plan)
        future_fan_rpm = np.clip(fan_norm_to_rpm(best_plan), self.fan_min_rpm, self.fan_max_rpm)
        prediction_counter_after = self._actual_step_index
        actual_state_preserved = (
            torch.equal(self._actual_v_curr, actual_v_before_prediction)
            and self._optional_tensor_equal(self._actual_h, actual_h_before_prediction)
            and self._zp_state_equal(self._actual_zp_state, actual_zp_before_prediction)
        )

        core_dtheta_dt = self._advance_actual_state(
            v0, h0, zp0, flux_h[0:1], sampled_u, env_h[0:1], measured_v, dt
        )
        u_act = best_plan[0].copy()
        rpm_cmd = np.clip(fan_norm_to_rpm(u_act), self.fan_min_rpm, self.fan_max_rpm).astype(np.float32)

        self._last_output_rpm = rpm_cmd
        self._previous_u = fan_rpm_to_norm(rpm_cmd).reshape(1, 2).astype(np.float32)
        self._u = best_plan[0].astype(np.float32)
        self._last_parts = parts
        self._last_parts["optimize_elapsed_s"] = float(optimize_elapsed_s)
        self._last_success = bool(result.success)
        self._last_message = str(result.message)
        self._last_fan_plan_rpm = future_fan_rpm.astype(np.float32)
        if self._actual_zp_state is None:
            zp_output_norm_rate = [math.nan] * 12
            zp_trend_norm_rate = [math.nan] * 12
        else:
            zp_output_norm_rate = (
                self._actual_zp_state["zp"]
                .detach()
                .cpu()
                .numpy()
                .reshape(12)
                .astype(float)
                .tolist()
            )
            zp_trend_norm_rate = (
                self._actual_zp_state["trend"]
                .detach()
                .cpu()
                .numpy()
                .reshape(12)
                .astype(float)
                .tolist()
            )
        core_dtheta_dt_norm_rate = (
            core_dtheta_dt
            .cpu()
            .numpy()
            .reshape(12)
            .astype(float)
            .tolist()
        )
        augment_h_values = (
            self._actual_h.detach().cpu().numpy().reshape(self._model.augment_dim).astype(float).tolist()
            if self._actual_h is not None
            else [math.nan] * self._model.augment_dim
        )
        self._last_augment_status = {
            "h": augment_h_values,
        }
        if zp0 is None:
            prediction_zp_norm_rate = [math.nan] * 12
        else:
            prediction_zp_norm_rate = (
                zp0["zp"]
                .detach()
                .cpu()
                .numpy()
                .reshape(12)
                .astype(float)
                .tolist()
            )
        self._last_zp_status = {
            "actual_correction_enabled": bool(self.enable_zp_correction),
            "actual_step_before": int(prediction_counter_before),
            "actual_step_after_prediction": int(prediction_counter_after),
            "actual_step_after_update": int(self._actual_step_index),
            "actual_elapsed_s": float(self._actual_elapsed_s),
            "correction_max_abs": float(torch.max(torch.abs(correction)).detach().cpu().item()),
            "prediction_correction_enabled": bool(self.enable_zp_correction and zp0 is not None),
            "actual_state_preserved_during_prediction": bool(actual_state_preserved),
            "zp_output_norm_rate": zp_output_norm_rate,
            "zp_trend_norm_rate": zp_trend_norm_rate,
            "core_dtheta_dt_norm_rate": core_dtheta_dt_norm_rate,
        }
        future_fan1_rpm = int(round(future_fan_rpm[0, 0]))
        future_fan2_rpm = int(round(future_fan_rpm[0, 1]))
        correction_norm_values = correction.detach().cpu().numpy().reshape(12)
        correction_c_values = TEMP_SCALE_C * correction_norm_values
        correction_norm_text = "[" + ", ".join(f"{v:+.6f}" for v in correction_norm_values) + "]"
        correction_c_text = "[" + ", ".join(f"{v:+.4f}" for v in correction_c_values) + "]"
        h_text = "[" + ", ".join(f"{v:+.6f}" for v in augment_h_values) + "]"
        prediction_zp_text = "[" + ", ".join(f"{v:+.6f}" for v in prediction_zp_norm_rate) + "]"
        updated_zp_text = "[" + ", ".join(f"{v:+.6f}" for v in zp_output_norm_rate) + "]"
        zp_trend_text = "[" + ", ".join(f"{v:+.6f}" for v in zp_trend_norm_rate) + "]"
        core_dtheta_dt_text = "[" + ", ".join(f"{v:+.6f}" for v in core_dtheta_dt_norm_rate) + "]"
        corrected_dtheta_dt_text = "[" + ", ".join(
            f"{core + zp:+.6f}"
            for core, zp in zip(core_dtheta_dt_norm_rate, zp_output_norm_rate)
        ) + "]"
        effect_text = ", ".join(
            f"{name}=({effect['delta_rpm'][0]:+.0f}, {effect['delta_rpm'][1]:+.0f})"
            for name, effect in loss_effects.items()
        )
        total_delta_rpm = np.rint(rpm_cmd).astype(np.int32) - np.rint(previous_rpm).astype(np.int32)
        self._last_loss_drpm = {
            name: tuple(float(value) for value in effect["delta_rpm"])
            for name, effect in loss_effects.items()
        }
        self._last_total_drpm = tuple(float(value) for value in total_delta_rpm)

        print(
            "[MPC] =============================================\n"
            f"time={optimize_elapsed_s:.4f}s, sampled_rpm=({int(round(sampled_fan_rpm[0]))}, {int(round(sampled_fan_rpm[1]))})\n"
            f"loss_dRPM(order=heat->grad->cooling->slew): {effect_text}\n"
            f"total_dRPM=({total_delta_rpm[0]:+.0f}, {total_delta_rpm[1]:+.0f})\n"
            f"augment_state={h_text}\n"
            f"corrected_dTheta_dt_norm_rate={corrected_dtheta_dt_text}\n"
            f"future_Tmax_C={[round(v, 2) for v in future_tmax_c]}\n"
            f"future_Fan1_RPM={future_fan1_rpm}, future_Fan2_RPM={future_fan2_rpm}"
        )
        return (int(round(rpm_cmd[0])), int(round(rpm_cmd[1])))

    # ------------------------------------------------------------------
    # MPC internals
    # ------------------------------------------------------------------
    def _rollout_loss(self, v0, h0, zp0, flux_h, env_h, dt_h, target_h, u_plan, previous_u):
        v_curr = v0.clone()
        h_curr = None if h0 is None else h0.clone()
        zp_state_curr = self._clone_zp_state(zp0)
        heat_cost = v0.new_tensor(0.0)
        grad_cost = v0.new_tensor(0.0)
        cooling_cost = v0.new_tensor(0.0)
        slew_cost = v0.new_tensor(0.0)
        total_dt = torch.clamp(dt_h.sum(), min=1.0e-6)
        last_u = previous_u

        for i in range(self.horizon):
            u_i = u_plan[i:i + 1]
            dt_i = torch.clamp(dt_h[i:i + 1], min=1.0e-6)
            time_d = (dt_i / NODE_TIME_SCALE_S).view(1, 1)
            v_next, h_next = self._model(
                time_d,
                flux_h[i:i + 1],
                u_i,
                env_h[i:i + 1],
                v_curr,
                h_curr,
            )
            if zp_state_curr is not None:
                v_next = v_next + time_d * zp_state_curr["zp"]

            time_weight = torch.sqrt(torch.tensor((i + 1) / self.horizon))
            heat_error = torch.max(torch.clamp(v_next - target_h[i:i + 1], min=0.0))
            epsilon = 0.05 / 80.0
            heat_cost = heat_cost + time_weight * epsilon * (torch.sqrt(1 + (heat_error / epsilon)**2) - 1)

            heating_rate = (v_next - v_curr) / dt_i.view(1, 1)
            grad_excess = torch.clamp(heating_rate - self.grad_target_norm, min=0.0)
            grad_cost = grad_cost + torch.mean(grad_excess.pow(2))

            cooling_cost = cooling_cost + torch.mean(u_i.pow(3)).pow(1.0/3.0) * dt_i.squeeze() / total_dt

            if last_u is not None:
                delta_u = u_i - last_u
                delta = self.slew_delta_norm
                smooth_epsilon = 1.0 / 6300.0

                slew_cost = slew_cost + torch.mean(
                    ((delta_u / delta).pow(2) + smooth_epsilon**2).pow(0.25)
                )

            v_curr = v_next
            h_curr = h_next
            last_u = u_i

        slew_cost = slew_cost / max(self.horizon, 1)


        total_cost = (
            self.weights["heat"] * heat_cost
            + self.weights["cooling"] * cooling_cost
            + self.weights["grad"] * grad_cost
            + self.weights["slew"] * slew_cost
        )
        return total_cost, {
            "heat": heat_cost,
            "grad": grad_cost,
            "cooling": cooling_cost,
            "slew": slew_cost,
        }

    def _optimize(self, v0, h0, zp0, flux_h, env_h, dt_h, target_h):
        u0 = self._u.astype(np.float64).reshape(2)
        previous_u = self._to_tensor(fan_rpm_to_norm(self._last_output_rpm).reshape(1, 2))

        def scipy_objective(u_flat):
            u_plan = torch.as_tensor(
                np.repeat(u_flat.reshape(1, 2), self.horizon, axis=0),
                dtype=torch.float32,
                device=self._device,
            ).clone().detach().requires_grad_(True)
            loss, _ = self._rollout_loss(v0, h0, zp0, flux_h, env_h, dt_h, target_h, u_plan, previous_u)
            loss.backward()
            grad = u_plan.grad.detach().cpu().numpy().astype(np.float64).sum(axis=0)
            return float(loss.detach().cpu().item()), grad

        result = opt.minimize(
            scipy_objective,
            x0=u0,
            method="SLSQP",
            jac=True,
            bounds=self.bounds,
            options={"maxiter": self.optimizer_iter, "ftol": self.ftol, "disp": False},
        )
        best_u = np.asarray(result.x, dtype=np.float32).reshape(1, 2)
        best_u = np.clip(best_u, self.fan_min_norm, self.fan_max_norm)
        best_plan = np.repeat(best_u, self.horizon, axis=0).astype(np.float32)
        with torch.no_grad():
            _, parts = self._rollout_loss(v0, h0, zp0, flux_h, env_h, dt_h, target_h, self._to_tensor(best_plan), previous_u)
        parts_np = {key: float(value.detach().cpu().item()) for key, value in parts.items()}
        return best_plan, result, parts_np

    def _marginal_loss_rpm_effects(self, v0, h0, zp0, flux_h, env_h, dt_h, target_h, previous_rpm, total_rpm):
        u0 = self._u.astype(np.float64).reshape(2)
        previous_u = self._to_tensor(fan_rpm_to_norm(self._last_output_rpm).reshape(1, 2))
        effects = {}

        loss_order = ("heat", "grad", "cooling", "slew")
        active_indices = [index for index, name in enumerate(loss_order) if self.weights[name] != 0.0]
        last_active_index = active_indices[-1] if active_indices else None
        previous_endpoint_rpm = np.rint(previous_rpm).astype(np.int32)
        cumulative_names = []

        for index, loss_name in enumerate(loss_order):
            if self.weights[loss_name] == 0.0:
                endpoint_rpm = previous_endpoint_rpm.copy()
            else:
                cumulative_names.append(loss_name)
                if index == last_active_index:
                    endpoint_rpm = np.rint(total_rpm).astype(np.int32)
                else:

                    def scipy_objective(u_flat):
                        u_plan = torch.as_tensor(
                            np.repeat(u_flat.reshape(1, 2), self.horizon, axis=0),
                            dtype=torch.float32,
                            device=self._device,
                        ).clone().detach().requires_grad_(True)
                        _, parts = self._rollout_loss(
                            v0, h0, zp0, flux_h, env_h, dt_h, target_h, u_plan, previous_u
                        )
                        loss = sum(self.weights[name] * parts[name] for name in cumulative_names)
                        loss.backward()
                        grad = u_plan.grad.detach().cpu().numpy().astype(np.float64).sum(axis=0)
                        return float(loss.detach().cpu().item()), grad

                    result = opt.minimize(
                        scipy_objective,
                        x0=u0,
                        method="SLSQP",
                        jac=True,
                        bounds=self.bounds,
                        options={"maxiter": self.optimizer_iter, "ftol": self.ftol, "disp": False},
                    )
                    best_u = np.asarray(result.x, dtype=np.float32).reshape(2)
                    best_u = np.clip(best_u, self.fan_min_norm, self.fan_max_norm)
                    endpoint_rpm = np.rint(fan_norm_to_rpm(best_u)).astype(np.int32)

            delta_rpm = endpoint_rpm - previous_endpoint_rpm
            effects[loss_name] = {
                "rpm": endpoint_rpm.astype(np.float32),
                "delta_rpm": delta_rpm.astype(np.float32),
            }
            previous_endpoint_rpm = endpoint_rpm

        return effects

    def _rollout_future_tmax_c(self, v0, h0, zp0, flux_h, env_h, dt_h, u_plan_np):
        future_tmax_c = []
        v_curr = v0.clone()
        h_curr = None if h0 is None else h0.clone()
        zp_state_curr = self._clone_zp_state(zp0)
        u_plan = self._to_tensor(u_plan_np)
        with torch.no_grad():
            for i in range(self.horizon):
                dt_i = torch.clamp(dt_h[i:i + 1], min=1.0e-6)
                time_d = (dt_i / NODE_TIME_SCALE_S).view(1, 1)
                v_curr, h_curr = self._model(
                    time_d,
                    flux_h[i:i + 1],
                    u_plan[i:i + 1],
                    env_h[i:i + 1],
                    v_curr,
                    h_curr,
                )
                if zp_state_curr is not None:
                    v_curr = v_curr + time_d * zp_state_curr["zp"]
                future_tmax_c.append(float(np.max(temp_norm_to_c(v_curr.detach().cpu().numpy().reshape(12)))))
        return future_tmax_c

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clone_zp_state(state):
        if state is None:
            return None
        return {name: value.detach().clone() for name, value in state.items()}

    @staticmethod
    def _optional_tensor_equal(left, right):
        if left is None or right is None:
            return left is right
        return torch.equal(left, right)

    @classmethod
    def _zp_state_equal(cls, left, right):
        if left is None or right is None:
            return left is right
        if left.keys() != right.keys():
            return False
        return all(cls._optional_tensor_equal(left[key], right[key]) for key in left)

    def _to_tensor(self, array):
        return torch.as_tensor(array, dtype=torch.float32, device=self._device)

    def _resolve_target(self, t_aim):
        if t_aim is None:
            if self._channel_t_aim is not None:
                return self._channel_t_aim.copy()
            return np.full(12, self._t_aim, dtype=np.float32)
        if np.isscalar(t_aim):
            self.set_t_aim(float(t_aim))
            return np.full(12, float(t_aim), dtype=np.float32)
        self.set_channel_t_aim(t_aim)
        return self._channel_t_aim.copy()

    @staticmethod
    def _coerce_vector(values, fill_value=0.0):
        if values is None:
            return np.full(12, fill_value, dtype=np.float32)
        if isinstance(values, dict):
            return np.array([values.get(ch, fill_value) for ch in range(1, 13)], dtype=np.float32)
        arr = np.asarray(list(values), dtype=np.float32).reshape(-1)
        if len(arr) < 12:
            arr = np.concatenate([arr, np.full(12 - len(arr), fill_value, dtype=np.float32)])
        return arr[:12].astype(np.float32)

    def _coerce_fan_rpm(self, values):
        if values is None:
            return self._last_output_rpm.astype(np.float32)
        if isinstance(values, dict):
            arr = np.array([values.get(fan, self._last_output_rpm[fan - 1]) for fan in [1, 2]], dtype=np.float32)
        else:
            arr = np.asarray(list(values), dtype=np.float32).reshape(-1)
            if len(arr) < 2:
                arr = np.concatenate([arr, self._last_output_rpm[len(arr):]])
            arr = arr[:2]
        return np.clip(arr, 0.0, self.fan_max_rpm).astype(np.float32)

    @staticmethod
    def _shift_control_plan(best_plan):
        return np.vstack([best_plan[1:], best_plan[-1:]]).astype(np.float32)

# Alias with a short name for app-side imports.
MPCFanController = NODEMPCFanController
