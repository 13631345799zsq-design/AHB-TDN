import math
import time
import threading
from pathlib import Path
import numpy as np
import torch
import scipy.optimize as opt
from _0_RNODE_Model import Transient_Model
from _0_DBC_branch import DBC_Branch


TEMP_OFFSET_C = 20.0
TEMP_SCALE_C = 80.0
FAN_SCALE_RPM = 6300.0
NODE_TIME_SCALE_S = 5.0
FLUX_MIN = 0.0
FLUX_MAX = 0.03
FLUX_POWER_COEFF = np.array(
    [
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
    ],
    dtype=np.float32,
)

# MPC configuration parameters.
MPC_TUNED_CONFIG = {
    "model_path": str(
        Path(__file__).resolve().parent / "models_1" / "model_epoch_218.pth"
    ),
    "device": "cpu",
    # Optimizer parameters.
    "horizon": 20,
    "optimizer_iter": 10,
    "ftol": 1.0e-6,
    "prediction_dt_s": 1.0,
    "h_advance_step_s": 1.0,
    "enable_dbc_correction": True,
    # Fan-parameter boundaries.
    "fan_min_rpm": 1000.0,
    "fan_max_rpm": 6300.0,
    "initial_fan_rpm": 1000.0,
    "command_slew_attenuation": 0.6,
    # MPC weight parameters.
    "weight_heat": 1.0e0,
    "weight_grad": 0.0,
    "weight_cooling": 2.0e-3,  # Baseline.
    "grad_target_c_per_s": 0.1,
}

# Conversion code.
def temp_c_to_norm(temp_c):
    return (np.asarray(temp_c, dtype=np.float32) - TEMP_OFFSET_C) / TEMP_SCALE_C


def fan_rpm_to_norm(fan_rpm):
    return np.asarray(fan_rpm, dtype=np.float32) / FAN_SCALE_RPM


def fan_norm_to_rpm(fan_norm):
    return FAN_SCALE_RPM * np.asarray(fan_norm, dtype=np.float32)


def power_w_to_flux_norm(power_w):
    flux = FLUX_POWER_COEFF * np.asarray(power_w, dtype=np.float32)
    return (flux - FLUX_MIN) / (FLUX_MAX - FLUX_MIN)


class NODEMPCFanController:
    RPM_OFF = 0

    def __init__(self, config=None):
        # Read the MPC configuration.
        cfg = dict(MPC_TUNED_CONFIG)
        if config:
            cfg.update(config)
        self.horizon = int(cfg["horizon"])
        self.optimizer_iter = int(cfg["optimizer_iter"])
        self.ftol = float(cfg["ftol"])
        self.prediction_dt_s = float(cfg["prediction_dt_s"])
        self.h_advance_step_s = float(cfg["h_advance_step_s"])
        self.enable_dbc_correction = bool(cfg["enable_dbc_correction"])
        self.fan_min_rpm = float(cfg["fan_min_rpm"])
        self.fan_max_rpm = float(cfg["fan_max_rpm"])
        self.fan_min_norm = float(fan_rpm_to_norm(self.fan_min_rpm))
        self.fan_max_norm = float(fan_rpm_to_norm(self.fan_max_rpm))
        self.bounds = [(self.fan_min_norm, self.fan_max_norm)] * 2
        self.initial_fan_norm = float(
            np.clip(
                fan_rpm_to_norm(float(cfg["initial_fan_rpm"])),
                self.fan_min_norm,
                self.fan_max_norm,
            )
        )
        self.command_slew_attenuation = float(cfg["command_slew_attenuation"])
        if not 0.0 <= self.command_slew_attenuation <= 1.0:
            raise ValueError("command_slew_attenuation must be in [0, 1]")
        self.grad_target_norm = float(cfg["grad_target_c_per_s"]) / TEMP_SCALE_C
        self.weights = {
            "heat": float(cfg["weight_heat"]),
            "grad": float(cfg["weight_grad"]),
            "cooling": float(cfg["weight_cooling"]),
        }
        # Control flags.
        self._enabled = False
        # Runtime device configuration.
        device_cfg = str(cfg["device"]).lower()
        if device_cfg == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = device_cfg
        self._device = torch.device(device_name)
        # Model imports.
        model_path = Path(cfg["model_path"])
        self._model = Transient_Model().to(self._device)
        self._model.load_state_dict(
            torch.load(model_path, map_location=self._device, weights_only=True),
            strict=True,
        )
        self._model.ode_func_core.project_positive_weights()
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad_(False)
        self._dbc_branch = DBC_Branch().to(self._device)
        self._dbc_branch.eval()
        self._state_lock = threading.RLock()
        # Initialize fan speeds.
        self._last_output_rpm = np.array(
            [self.fan_min_rpm, self.fan_min_rpm], dtype=np.float32
        )
        self._u = fan_rpm_to_norm(self._last_output_rpm).astype(np.float32)
        # Initialize the target temperature.
        self._t_aim = 60.0
        self._channel_t_aim = None
        # Runtime state.
        self._last_loss_drpm = {}
        self._last_total_drpm = (0.0, 0.0)
        self._last_prediction = {}
        self._last_loss_values = {}
        self._last_success = None
        self._last_message = ""
        self._actual_v_curr = None
        self._actual_h = None
        self._actual_dbc_state = None
        self._actual_step_index = 0
        self._last_dbc_measurement_stamp = None
        self._last_core_dtheta_dt = None
        self._last_dbc_status = {}
        self._last_augment_status = {}

    # MPC optimizer start/stop functions ====================
    def enable(self):
        self._enabled = True
        self.reset()

    def disable(self):
        self._enabled = False
        self._last_output_rpm[:] = self.RPM_OFF

    def reset(self):
        self._u = np.full(2, self.initial_fan_norm, dtype=np.float32)
        self._last_output_rpm = fan_norm_to_rpm(self._u).astype(np.float32)
        self._last_loss_drpm = {}
        self._last_total_drpm = (0.0, 0.0)
        self._last_prediction = {}
        self._last_loss_values = {}
        self._last_success = None
        self._last_message = ""
        self._actual_v_curr = None
        self._actual_h = None
        self._actual_dbc_state = None
        self._actual_step_index = 0
        self._last_dbc_measurement_stamp = None
        self._last_core_dtheta_dt = None
        self._last_dbc_status = {}
        self._last_augment_status = {}

    # ==========================================

    # Target-temperature setter functions =========================
    def set_t_aim(self, t_aim):
        self._t_aim = float(t_aim)
        self._channel_t_aim = None

    def set_channel_t_aim(self, t_aim_by_channel):
        targets = self._coerce_vector(t_aim_by_channel, fill_value=self._t_aim)
        self._channel_t_aim = targets.astype(np.float32)

    # ==========================================

    @property
    def last_output(self):
        return (
            int(round(self._last_output_rpm[0])),
            int(round(self._last_output_rpm[1])),
        )

    @property
    def last_status(self):
        return {
            "success": self._last_success,
            "message": self._last_message,
            "loss_drpm": {
                name: tuple(values) for name, values in self._last_loss_drpm.items()
            },
            "total_drpm": tuple(self._last_total_drpm),
            "prediction": dict(self._last_prediction),
            "loss_values": dict(self._last_loss_values),
            "dbc_branch": dict(self._last_dbc_status),
            "augment_state": dict(self._last_augment_status),
        }

    def _prepare_actual_state(self, measured_v):
        if self._actual_v_curr is None:
            self._actual_v_curr = measured_v.detach().clone()

        actual_v = self._actual_v_curr.detach().clone()
        actual_h = None if self._actual_h is None else self._actual_h.detach().clone()
        actual_dbc_state = self._clone_dbc_state(self._actual_dbc_state)
        correction = measured_v - actual_v
        actual_v = measured_v.detach().clone()
        if not self.enable_dbc_correction:
            actual_dbc_state = None

        return actual_v, actual_h, actual_dbc_state, correction.detach()

    def update_dbc_measurement(
        self,
        temperatures,
        heater_power=None,
        actual_fan_rpm=None,
        ambient_temp_c=25.0,
        timestamp=None,
    ):
        if not self._enabled or not self.enable_dbc_correction:
            return False
        temps_c = self._coerce_temperatures(temperatures)
        if temps_c is None:
            return False
        measured_v = self._to_tensor(temp_c_to_norm(temps_c).reshape(1, 12))
        flux = self._to_tensor(
            power_w_to_flux_norm(
                self._coerce_vector(heater_power, fill_value=0.0)
            ).reshape(1, 12)
        )
        cooling = self._to_tensor(
            fan_rpm_to_norm(self._coerce_fan_rpm(actual_fan_rpm)).reshape(1, 2)
        )
        env = self._to_tensor(
            temp_c_to_norm(np.array([[float(ambient_temp_c)]], dtype=np.float32))
        )
        sample_stamp = time.perf_counter() if timestamp is None else float(timestamp)
        with self._state_lock:
            if not self._enabled:
                return False
            dt = (
                self.prediction_dt_s
                if self._last_dbc_measurement_stamp is None
                else max(sample_stamp - self._last_dbc_measurement_stamp, 1.0e-6)
            )
            time_d = self._to_tensor(
                np.array([[dt / NODE_TIME_SCALE_S]], dtype=np.float32)
            )
            h_for_core = self._actual_h
            if h_for_core is None:
                h_for_core = measured_v.new_zeros(
                    measured_v.shape[0], self._model.augment_dim
                )
            # MLP_Cell.forward() smoothly limits h with Softplus; use it directly to match model rollout.
            with torch.no_grad():
                core_dtheta_dt = self._model.ode_func_core(
                    measured_v - env, flux, cooling, h_for_core
                )
                _, dbc_state_next, dbc_updated = self._dbc_branch(
                    measured_v, time_d, core_dtheta_dt, self._actual_dbc_state
                )
            self._actual_dbc_state = dbc_state_next
            self._last_dbc_measurement_stamp = sample_stamp
            self._last_core_dtheta_dt = core_dtheta_dt
            self._actual_step_index += int(torch.any(dbc_updated).item())
        return True

    def _advance_state(
        self, v_curr, h_curr, flux, cooling, env, dbc_state, step_s, step_count=1
    ):
        time_d = self._to_tensor(
            np.array([[step_s / NODE_TIME_SCALE_S]], dtype=np.float32)
        )
        with torch.no_grad():
            for _ in range(step_count):
                v_next, h_next = self._model(time_d, flux, cooling, env, v_curr, h_curr)
                if dbc_state is not None:
                    v_next = v_next + time_d * dbc_state["dbc"]
                v_curr, h_curr = v_next, h_next
        return v_curr.detach(), h_curr.detach()

    def compute(
        self,
        temperatures,
        heater_power=None,
        actual_fan_rpm=None,
        ambient_temp_c=25.0,
        t_aim=None,
    ):
        # Startup checks.
        if not self._enabled:
            return (self.RPM_OFF, self.RPM_OFF)
        # Process sampled data.
        temps_c = self._coerce_temperatures(temperatures)
        if temps_c is None:
            return self.last_output
        power = power_w_to_flux_norm(self._coerce_vector(heater_power, fill_value=0.0))

        sampled_fan_rpm = self._coerce_fan_rpm(actual_fan_rpm)
        sampled_u_norm = (
            fan_rpm_to_norm(sampled_fan_rpm).reshape(1, 2).astype(np.float32)
        )
        self._u = np.clip(sampled_u_norm, self.fan_min_norm, self.fan_max_norm).reshape(
            2
        )
        target_c = self._resolve_target(t_aim)
        dt = self.prediction_dt_s
        measured_v = self._to_tensor(temp_c_to_norm(temps_c).reshape(1, 12))
        with self._state_lock:
            v0, h0, dbc0, correction = self._prepare_actual_state(measured_v)
            actual_v_before_prediction = self._actual_v_curr.detach().clone()
            actual_h_before_prediction = (
                None if self._actual_h is None else self._actual_h.detach().clone()
            )
        flux_h = self._to_tensor(np.repeat(power.reshape(1, 12), self.horizon, axis=0))
        env_norm = temp_c_to_norm(np.array([[float(ambient_temp_c)]], dtype=np.float32))
        env_h = self._to_tensor(np.repeat(env_norm, self.horizon, axis=0))
        dt_h = self._to_tensor(
            np.repeat(np.array([dt], dtype=np.float32), self.horizon, axis=0)
        )
        target_h = self._to_tensor(
            np.repeat(temp_c_to_norm(target_c).reshape(1, 12), self.horizon, axis=0)
        )
        sampled_u = self._to_tensor(sampled_u_norm)

        # Start optimization.
        previous_rpm = sampled_fan_rpm.astype(np.float32)
        optimize_started = time.perf_counter()
        best_u, result = self._optimize(v0, h0, dbc0, flux_h, env_h, dt_h, target_h)
        optimized_rpm = np.clip(
            fan_norm_to_rpm(best_u), self.fan_min_rpm, self.fan_max_rpm
        ).astype(np.float32)
        loss_effects = self._marginal_loss_rpm_effects(
            v0, h0, dbc0, flux_h, env_h, dt_h, target_h, previous_rpm, optimized_rpm
        )
        optimize_elapsed_s = time.perf_counter() - optimize_started
        h_advance_steps = max(0, int(round(optimize_elapsed_s / self.h_advance_step_s)))
        with self._state_lock:
            actual_state_preserved = torch.equal(
                self._actual_v_curr, actual_v_before_prediction
            ) and self._optional_tensor_equal(
                self._actual_h, actual_h_before_prediction
            )
            dbc_for_state = self._clone_dbc_state(self._actual_dbc_state)
            self._actual_v_curr, self._actual_h = self._advance_state(
                v0, h0, flux_h[0:1], sampled_u, env_h[0:1], dbc_for_state, dt
            )
            if h_advance_steps > 0:
                self._actual_v_curr, self._actual_h = self._advance_state(
                    self._actual_v_curr,
                    self._actual_h,
                    flux_h[0:1],
                    sampled_u,
                    env_h[0:1],
                    dbc_for_state,
                    self.h_advance_step_s,
                    h_advance_steps,
                )
            dbc_status_state = dbc_for_state
            core_dtheta_dt = self._last_core_dtheta_dt
            dbc_update_step = int(self._actual_step_index)
            augment_h_values = (
                self._actual_h.detach()
                .cpu()
                .numpy()
                .reshape(self._model.augment_dim)
                .astype(float)
                .tolist()
                if self._actual_h is not None
                else [math.nan] * self._model.augment_dim
            )
        previous_command_rpm = self._last_output_rpm.astype(np.float32)
        retained_change = 1.0 - self.command_slew_attenuation
        rpm_cmd = previous_command_rpm + retained_change * (
            optimized_rpm - previous_command_rpm
        )
        rpm_cmd = np.clip(rpm_cmd, self.fan_min_rpm, self.fan_max_rpm).astype(
            np.float32
        )
        loss_effects["slew"] = (
            np.rint(rpm_cmd).astype(np.int32) - np.rint(optimized_rpm).astype(np.int32)
        ).astype(np.float32)
        self._last_output_rpm = rpm_cmd
        self._u = fan_rpm_to_norm(rpm_cmd).astype(np.float32)

        # Roll out once more using the actual constant fan command for terminal-dashboard display.
        # This rollout only reads the initial prediction state; it neither participates in optimization nor modifies the actual state.
        display_u = self._to_tensor(
            np.repeat(self._u.reshape(1, 2), self.horizon, axis=0)
        )
        with torch.no_grad():
            display_total, display_parts, prediction_max_c = self._rollout_loss(
                v0,
                h0,
                dbc0,
                flux_h,
                env_h,
                dt_h,
                target_h,
                display_u,
                return_max_temperature=True,
            )
        loss_raw = {
            name: float(value.detach().cpu().item())
            for name, value in display_parts.items()
        }
        loss_weighted = {
            name: self.weights[name] * value for name, value in loss_raw.items()
        }
        peak_index = int(np.argmax(prediction_max_c)) if prediction_max_c else 0
        self._last_prediction = {
            "dt_s": float(dt),
            "max_temp_c": tuple(prediction_max_c),
            "peak_c": float(prediction_max_c[peak_index])
            if prediction_max_c
            else math.nan,
            "peak_step": peak_index + 1 if prediction_max_c else 0,
            "final_c": float(prediction_max_c[-1]) if prediction_max_c else math.nan,
        }
        self._last_loss_values = {
            "raw": loss_raw,
            "weighted": loss_weighted,
            "total": float(display_total.detach().cpu().item()),
        }
        self._last_success = bool(result.success)
        self._last_message = str(result.message)
        if dbc_status_state is None:
            dbc_output_norm_rate = [math.nan] * 12
            dbc_trend_norm_rate = [math.nan] * 12
        else:
            dbc_output_norm_rate = (
                dbc_status_state["dbc"]
                .detach()
                .cpu()
                .numpy()
                .reshape(12)
                .astype(float)
                .tolist()
            )
            dbc_trend_norm_rate = (
                dbc_status_state["trend"]
                .detach()
                .cpu()
                .numpy()
                .reshape(12)
                .astype(float)
                .tolist()
            )
        core_dtheta_dt_norm_rate = (
            core_dtheta_dt.cpu().numpy().reshape(12).astype(float).tolist()
            if core_dtheta_dt is not None
            else [math.nan] * 12
        )
        self._last_augment_status = {
            "h": augment_h_values,
            "compute_elapsed_s": float(optimize_elapsed_s),
            "advance_step_s": float(self.h_advance_step_s),
            "advance_steps": int(h_advance_steps),
        }
        self._last_dbc_status = {
            "actual_correction_enabled": bool(self.enable_dbc_correction),
            "actual_step_after_update": dbc_update_step,
            "correction_max_abs": float(
                torch.max(torch.abs(correction)).detach().cpu().item()
            ),
            "prediction_correction_enabled": bool(
                self.enable_dbc_correction and dbc0 is not None
            ),
            "actual_state_preserved_during_prediction": bool(actual_state_preserved),
            "dbc_output_norm_rate": dbc_output_norm_rate,
            "dbc_trend_norm_rate": dbc_trend_norm_rate,
            "core_dtheta_dt_norm_rate": core_dtheta_dt_norm_rate,
        }
        total_delta_rpm = np.rint(rpm_cmd).astype(np.int32) - np.rint(
            previous_rpm
        ).astype(np.int32)
        self._last_loss_drpm = {
            name: tuple(float(value) for value in values)
            for name, values in loss_effects.items()
        }
        self._last_total_drpm = tuple(float(value) for value in total_delta_rpm)
        return (int(round(rpm_cmd[0])), int(round(rpm_cmd[1])))

    # ------------------------------------------------------------------
    # MPC internals
    # ------------------------------------------------------------------
    def _rollout_loss(
        self,
        v0,
        h0,
        dbc0,
        flux_h,
        env_h,
        dt_h,
        target_h,
        u_plan,
        return_max_temperature=False,
    ):
        v_curr = v0.clone()
        h_curr = None if h0 is None else h0.clone()
        dbc_curr = None if dbc0 is None else dbc0["dbc"]
        heat_cost = v0.new_tensor(0.0)
        grad_cost = v0.new_tensor(0.0)
        cooling_cost = v0.new_tensor(0.0)
        max_temperature_c = []

        for i in range(self.horizon):
            u_i = u_plan[i : i + 1]
            dt_i = torch.clamp(dt_h[i : i + 1], min=1.0e-6)
            time_d = (dt_i / NODE_TIME_SCALE_S).view(1, 1)
            v_next, h_next = self._model(
                time_d, flux_h[i : i + 1], u_i, env_h[i : i + 1], v_curr, h_curr
            )
            if dbc_curr is not None:
                v_next = v_next + time_d * dbc_curr

            # heat_loss
            time_weight = torch.sqrt(torch.tensor((i + 1) / self.horizon))
            heat_error = torch.max(torch.clamp(v_next - target_h[i : i + 1], min=0.0))
            epsilon = 0.05 / 80.0
            heat_cost = heat_cost + time_weight * epsilon * (
                torch.sqrt(1 + (heat_error / epsilon) ** 2) - 1
            ) / max(self.horizon, 1)

            # gradient_loss
            heating_rate = (v_next - v_curr) / dt_i.view(1, 1)
            grad_excess = torch.clamp(heating_rate - self.grad_target_norm, min=0.0)
            grad_cost = grad_cost + torch.mean(grad_excess.pow(2)) / max(
                self.horizon, 1
            )

            # cooling_loss
            cooling_cost = cooling_cost + torch.mean(u_i.pow(3)).pow(1.0 / 3.0) / max(
                self.horizon, 1
            )

            if return_max_temperature:
                max_norm = float(torch.max(v_next).detach().cpu().item())
                max_temperature_c.append(TEMP_OFFSET_C + TEMP_SCALE_C * max_norm)

            v_curr = v_next
            h_curr = h_next

        total_cost = (
            self.weights["heat"] * heat_cost
            + self.weights["cooling"] * cooling_cost
            + self.weights["grad"] * grad_cost
        )
        parts = {"heat": heat_cost, "grad": grad_cost, "cooling": cooling_cost}
        if return_max_temperature:
            return total_cost, parts, max_temperature_c
        return total_cost, parts

    def _optimize(
        self, v0, h0, dbc0, flux_h, env_h, dt_h, target_h, active_losses=None
    ):
        u0 = self._u.astype(np.float64).reshape(2)

        def scipy_objective(u_flat):
            u_plan = (
                torch.as_tensor(
                    np.repeat(u_flat.reshape(1, 2), self.horizon, axis=0),
                    dtype=torch.float32,
                    device=self._device,
                )
                .clone()
                .detach()
                .requires_grad_(True)
            )
            loss, parts = self._rollout_loss(
                v0, h0, dbc0, flux_h, env_h, dt_h, target_h, u_plan
            )
            if active_losses is not None:
                loss = sum(self.weights[name] * parts[name] for name in active_losses)
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
        return best_u, result

    def _marginal_loss_rpm_effects(
        self, v0, h0, dbc0, flux_h, env_h, dt_h, target_h, previous_rpm, total_rpm
    ):
        effects = {}

        loss_order = ("heat", "grad", "cooling")
        active_names = [name for name in loss_order if self.weights[name] != 0.0]
        last_active_name = active_names[-1] if active_names else None
        previous_endpoint_rpm = np.rint(previous_rpm).astype(np.int32)
        cumulative_names = []

        for loss_name in loss_order:
            if self.weights[loss_name] == 0.0:
                endpoint_rpm = previous_endpoint_rpm.copy()
            else:
                cumulative_names.append(loss_name)
                if loss_name == last_active_name:
                    endpoint_rpm = np.rint(total_rpm).astype(np.int32)
                else:
                    best_u, _ = self._optimize(
                        v0, h0, dbc0, flux_h, env_h, dt_h, target_h, cumulative_names
                    )
                    endpoint_rpm = np.rint(fan_norm_to_rpm(best_u)).astype(np.int32)

            delta_rpm = endpoint_rpm - previous_endpoint_rpm
            effects[loss_name] = delta_rpm.astype(np.float32)
            previous_endpoint_rpm = endpoint_rpm

        return effects

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clone_dbc_state(state):
        if state is None:
            return None
        return {name: value.detach().clone() for name, value in state.items()}

    @staticmethod
    def _optional_tensor_equal(left, right):
        if left is None or right is None:
            return left is right
        return torch.equal(left, right)

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
            return np.array(
                [values.get(ch, fill_value) for ch in range(1, 13)], dtype=np.float32
            )
        arr = np.asarray(list(values), dtype=np.float32).reshape(-1)
        if len(arr) < 12:
            arr = np.concatenate(
                [arr, np.full(12 - len(arr), fill_value, dtype=np.float32)]
            )
        return arr[:12].astype(np.float32)

    def _coerce_temperatures(self, values):
        temperatures = self._coerce_vector(values, fill_value=math.nan)
        valid = np.isfinite(temperatures)
        if not np.any(valid):
            return None
        return np.where(valid, temperatures, float(np.nanmax(temperatures))).astype(
            np.float32
        )

    def _coerce_fan_rpm(self, values):
        if values is None:
            return self._last_output_rpm.astype(np.float32)
        if isinstance(values, dict):
            arr = np.array(
                [values.get(fan, self._last_output_rpm[fan - 1]) for fan in [1, 2]],
                dtype=np.float32,
            )
        else:
            arr = np.asarray(list(values), dtype=np.float32).reshape(-1)
            if len(arr) < 2:
                arr = np.concatenate([arr, self._last_output_rpm[len(arr) :]])
            arr = arr[:2]
        return np.clip(arr, 0.0, self.fan_max_rpm).astype(np.float32)


# Alias with a short name for app-side imports.
MPCFanController = NODEMPCFanController
