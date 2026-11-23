import math

PID_TUNED_CONFIG = {
    "Kp": 0.75,  # The ratio has a strong prior basis.
    "Ki": 0.005,
    "I_limit": 1.0,
    "rpm_step_limit": None,
    "control_margin_c": 0.0,  # Safety margin.
    "fan_weight_matrix": [
        [
            0.6031,
            0.5141,
            0.1043,
            0.1230,
            0.1187,
            0.0943,
            0.2533,
            0.4130,
            0.6070,
            0.6759,
            0.8556,
            1.0000,
        ],
        [
            0.6031,
            0.0956,
            0.4867,
            0.9572,
            0.8192,
            0.4634,
            0.6385,
            0.5132,
            0.2972,
            0.1616,
            0.1181,
            0.1210,
        ],
    ],
}


class PIDFanController:
    """
    PID controller that automatically adjusts fan speed from heat-source temperatures and target temperature T_aim.

    Control logic:
    - temperature <= T_aim - control_margin_c: retain the PI integral output; when control is enabled, keep the fans at least at minimum speed
    - temperature > T_aim - control_margin_c: compute fan speed from the weighted normalized temperature errors of all heat-source channels
    - compute(temperatures) accepts 12 heat-source temperatures and calculates Fan1/Fan2 separately using a 2 x 12 fan-weight matrix
    - valid output range is {0} union [RPM_MIN, RPM_MAX]
    """

    # Fan-speed constants.
    RPM_MIN = 1000
    RPM_MAX = 6300
    RPM_OFF = 0

    def __init__(
        self,
        t_aim: float = 40.0,
        kp: float = None,
        ki: float = None,
        i_limit: float = None,
    ):
        """
        Parameters
        ----------
        t_aim : float
            Target temperature [°C]. The fans remain off while the maximum heat-source temperature does not exceed this value.
        kp : float
            Proportional gain.
        ki : float
            Integral gain.
        i_limit : float
            Upper limit of the integral term. The control output is normalized to [0, 1], so the default is 1.
        """
        cfg = PID_TUNED_CONFIG
        self.t_aim = t_aim
        self.kp = cfg["Kp"] if kp is None else kp
        self.ki = cfg["Ki"] if ki is None else ki
        self.i_limit = cfg["I_limit"] if i_limit is None else i_limit
        self.rpm_step_limit = cfg["rpm_step_limit"]
        self.control_margin_c = cfg["control_margin_c"]
        self.fan_weight_matrix = self._validate_fan_weight_matrix(
            cfg["fan_weight_matrix"]
        )

        # Internal PI state.
        self._integral = [0.0, 0.0]

        # Control state.
        self._enabled = False
        self._last_output_rpm1 = self.RPM_OFF
        self._last_output_rpm2 = self.RPM_OFF

    # ------------------------------------------------------------------
    # Public interface.
    # ------------------------------------------------------------------
    def enable(self):
        """Enable PID control and reset its internal state."""
        self._enabled = True
        self.reset()

    def disable(self):
        """Disable PID control."""
        self._enabled = False
        self._last_output_rpm1 = self.RPM_OFF
        self._last_output_rpm2 = self.RPM_OFF

    def is_enabled(self) -> bool:
        return self._enabled

    def reset(self):
        """Reset the PI integral state."""
        self._integral = [0.0, 0.0]

    def set_t_aim(self, t_aim: float):
        """Update the target temperature."""
        self.t_aim = t_aim

    def set_kp(self, kp: float):
        """Update the proportional gain."""
        self.kp = kp

    def set_pid_gains(self, kp: float, ki: float = None, kd: float = None):
        """Maintain compatibility with the legacy interface by updating PI kp/ki and ignoring kd."""
        self.set_kp(kp)
        if ki is not None:
            self.ki = ki

    @property
    def last_output(self):
        """Return the most recently output (rpm1, rpm2) tuple."""
        return (self._last_output_rpm1, self._last_output_rpm2)

    # ------------------------------------------------------------------
    # Core computation.
    # ------------------------------------------------------------------
    def compute(self, temperatures) -> tuple:
        # PID enable guard.
        if not self._enabled:
            return (self.RPM_OFF, self.RPM_OFF)
        # Read temperature data and validate it.
        temps = self._coerce_temperatures(temperatures)
        valid_temps = [t for t in temps if self._is_valid_temp(t)]
        if not valid_temps:
            return (self._last_output_rpm1, self._last_output_rpm2)
        # ---- PI control: use T_max - T_aim as the sole temperature-error reference. ----
        control_start_c = self.t_aim - self.control_margin_c
        max_temp_idx, max_temp = max(
            (
                (idx, temp)
                for idx, temp in enumerate(temps)
                if self._is_valid_temp(temp)
            ),
            key=lambda item: item[1],
        )
        p_errors = []
        i_errors = []
        for weights in self.fan_weight_matrix:
            weight = weights[max_temp_idx] if max_temp_idx < len(weights) else 0.0
            p_errors.append(weight * max(max_temp - control_start_c, 0.0))
            i_errors.append(weight * (max_temp - control_start_c))

        # PI computation loop that converts error to fan speed.
        fan_norms = []
        for fan_idx, error in enumerate(p_errors):
            self._integral[fan_idx] += self.ki * i_errors[fan_idx]
            self._integral[fan_idx] = max(
                0.0, min(self.i_limit, self._integral[fan_idx])
            )
            pid_output = self.kp * error + self._integral[fan_idx]
            fan_norms.append(max(0.0, min(1.0, pid_output)))

        rpm1_raw = self._clamp_rpm(
            fan_norms[0] * (self.RPM_MAX - self.RPM_MIN) + self.RPM_MIN
        )
        rpm2_raw = self._clamp_rpm(
            fan_norms[1] * (self.RPM_MAX - self.RPM_MIN) + self.RPM_MIN
        )
        self._last_output_rpm1 = self._limit_rpm_step(self._last_output_rpm1, rpm1_raw)
        self._last_output_rpm2 = self._limit_rpm_step(self._last_output_rpm2, rpm2_raw)

        return (self._last_output_rpm1, self._last_output_rpm2)

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_rpm(rpm: float) -> int:
        """Clamp an RPM value to the valid range {0} union [1000, 6300]."""
        rpm = round(rpm)
        if rpm < 0:
            return 0
        elif rpm < 1000:
            # For 0 < RPM < 1000, raise values near 1000 to 1000; otherwise set them to zero.
            return 1000 if rpm >= 500 else 0
        elif rpm > 6300:
            return 6300
        return rpm

    def _limit_rpm_step(self, last_rpm: int, target_rpm: int) -> int:
        """Limit the RPM change per sample to reduce abrupt fan-command transitions."""
        if self.rpm_step_limit is None:
            return target_rpm
        delta = target_rpm - last_rpm
        if delta > self.rpm_step_limit:
            return self._clamp_rpm(last_rpm + self.rpm_step_limit)
        if delta < -self.rpm_step_limit:
            return self._clamp_rpm(last_rpm - self.rpm_step_limit)
        return target_rpm

    @staticmethod
    def _is_valid_temp(temp) -> bool:
        return temp is not None and not (isinstance(temp, float) and math.isnan(temp))

    @staticmethod
    def _coerce_temperatures(temperatures):
        """Convert temperatures to a list ordered from CH1 through CH12."""
        if isinstance(temperatures, dict):
            return [temperatures.get(ch, math.nan) for ch in range(1, 13)]
        values = list(temperatures or [])
        if len(values) < 12:
            values += [math.nan] * (12 - len(values))
        return values[:12]

    @staticmethod
    def _validate_fan_weight_matrix(fan_weight_matrix):
        """Read the 2 x 12 fan-to-heat-source influence-weight matrix."""
        matrix = []
        for row in fan_weight_matrix[:2]:
            values = [max(0.0, float(v)) for v in row[:12]]
            if len(values) < 12:
                values += [0.0] * (12 - len(values))
            matrix.append(values)
        while len(matrix) < 2:
            matrix.append([0.0] * 12)
        return matrix
