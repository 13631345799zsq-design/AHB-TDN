import math

PID_TUNED_CONFIG = {
    "Kp": 0.75, # 比例是先验性质极强的
    "Ki": 0.005,
    "I_limit": 1.0,
    "rpm_step_limit": None,
    "control_margin_c": 0.0,  # 安全裕度

    "fan_weight_matrix": [
        [0.6031, 0.5141, 0.1043, 0.1230, 0.1187, 0.0943, 0.2533, 0.4130, 0.6070, 0.6759, 0.8556, 1.0000],
        [0.6031, 0.0956, 0.4867, 0.9572, 0.8192, 0.4634, 0.6385, 0.5132, 0.2972, 0.1616, 0.1181, 0.1210]
    ],
}


class PIDFanController:
    """
    PID 控制器：根据热源温度和目标温度 T_aim 自动调节风扇转速。

    控制逻辑:
    - 温度 <= T_aim - control_margin_c : PI 输出为积分保持量；启用控制时风扇至少保持最低转速
    - 温度 > T_aim - control_margin_c  : 根据各热源通道的加权归一化温度误差输出风扇转速
    - compute(temperatures) 输入 12 个热源温度，并用 2x12 风扇权重矩阵分别计算 Fan1/Fan2
    - 有效输出范围限制为 {0} ∪ [RPM_MIN, RPM_MAX]
    """

    # 风扇转速常量
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
            目标温度 (°C)。热源最高温度不超过此值时风扇关闭。
        kp : float
            比例增益参数。
        ki : float
            积分增益参数。
        i_limit : float
            积分项上限，当前控制输出归一化到 [0, 1]，默认取 1。
        """
        cfg = PID_TUNED_CONFIG
        self.t_aim = t_aim
        self.kp = cfg["Kp"] if kp is None else kp
        self.ki = cfg["Ki"] if ki is None else ki
        self.i_limit = cfg["I_limit"] if i_limit is None else i_limit
        self.rpm_step_limit = cfg["rpm_step_limit"]
        self.control_margin_c = cfg["control_margin_c"]
        self.fan_weight_matrix = self._validate_fan_weight_matrix(cfg["fan_weight_matrix"])

        # PI 内部状态
        self._integral = [0.0, 0.0]

        # 控制状态
        self._enabled = False
        self._last_output_rpm1 = self.RPM_OFF
        self._last_output_rpm2 = self.RPM_OFF

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def enable(self):
        """启用 PID 控制，重置内部状态。"""
        self._enabled = True
        self.reset()

    def disable(self):
        """禁用 PID 控制。"""
        self._enabled = False
        self._last_output_rpm1 = self.RPM_OFF
        self._last_output_rpm2 = self.RPM_OFF

    def is_enabled(self) -> bool:
        return self._enabled

    def reset(self):
        """重置 PI 积分状态。"""
        self._integral = [0.0, 0.0]

    def set_t_aim(self, t_aim: float):
        """更新目标温度。"""
        self.t_aim = t_aim

    def set_kp(self, kp: float):
        """更新比例增益。"""
        self.kp = kp

    def set_pid_gains(self, kp: float, ki: float = None, kd: float = None):
        """兼容旧接口：PI 控制器更新 kp/ki，忽略 kd。"""
        self.set_kp(kp)
        if ki is not None:
            self.ki = ki

    @property
    def last_output(self):
        """返回上一次输出的 (rpm1, rpm2) 元组。"""
        return (self._last_output_rpm1, self._last_output_rpm2)

    # ------------------------------------------------------------------
    # 核心计算
    # ------------------------------------------------------------------
    def compute(self, temperatures) -> tuple:
        # PID 启用保护代码
        if not self._enabled:
            return (self.RPM_OFF, self.RPM_OFF)
        # 读取温度数据并进行有效性检查
        temps = self._coerce_temperatures(temperatures)
        valid_temps = [t for t in temps if self._is_valid_temp(t)]
        if not valid_temps:
            return (self._last_output_rpm1, self._last_output_rpm2)
        # ---- PI 控制：以 T_max 与 T_aim 的差值作为唯一温度误差基准 ----
        control_start_c = self.t_aim - self.control_margin_c  
        max_temp_idx, max_temp = max(
            ((idx, temp) for idx, temp in enumerate(temps) if self._is_valid_temp(temp)),
            key=lambda item: item[1]
        )
        p_errors = []
        i_errors = []
        for weights in self.fan_weight_matrix:
            weight = weights[max_temp_idx] if max_temp_idx < len(weights) else 0.0
            p_errors.append(weight * max(max_temp - control_start_c, 0.0))
            i_errors.append(weight * (max_temp - control_start_c))

        # PI 计算循环（将误差转换为风扇转速）
        fan_norms = []
        for fan_idx, error in enumerate(p_errors):
            self._integral[fan_idx] += self.ki * i_errors[fan_idx]
            self._integral[fan_idx] = max(0.0, min(self.i_limit, self._integral[fan_idx]))
            pid_output = self.kp * error + self._integral[fan_idx]
            fan_norms.append(max(0.0, min(1.0, pid_output)))

        rpm1_raw = self._clamp_rpm(fan_norms[0] * (self.RPM_MAX - self.RPM_MIN) + self.RPM_MIN)
        rpm2_raw = self._clamp_rpm(fan_norms[1] * (self.RPM_MAX - self.RPM_MIN) + self.RPM_MIN)
        self._last_output_rpm1 = self._limit_rpm_step(self._last_output_rpm1, rpm1_raw)
        self._last_output_rpm2 = self._limit_rpm_step(self._last_output_rpm2, rpm2_raw)

        print(f"T_max: CH{max_temp_idx + 1}={max_temp} || P error: {p_errors[0]}, {p_errors[1]} || I error: {i_errors[0]}, {i_errors[1]} || I: {self._integral[0]}, {self._integral[1]}")
        print(f"Fan norm: {fan_norms[0]}, {fan_norms[1]} || Fan RPM: {self._last_output_rpm1}, {self._last_output_rpm2}")
        
        return (self._last_output_rpm1, self._last_output_rpm2)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_rpm(rpm: float) -> int:
        """将 RPM 值钳位到有效范围 {0} ∪ [1000, 6300]。"""
        rpm = round(rpm)
        if rpm < 0:
            return 0
        elif rpm < 1000:
            # 低于 1000 但大于 0：如果很接近 1000 则拉到 1000，否则归零
            return 1000 if rpm >= 500 else 0
        elif rpm > 6300:
            return 6300
        return rpm

    def _limit_rpm_step(self, last_rpm: int, target_rpm: int) -> int:
        """限制单次采样的 RPM 变化量，降低风扇命令跳变。"""
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
        """转换为 CH1-CH12 顺序的温度列表。"""
        if isinstance(temperatures, dict):
            return [temperatures.get(ch, math.nan) for ch in range(1, 13)]
        values = list(temperatures or [])
        if len(values) < 12:
            values += [math.nan] * (12 - len(values))
        return values[:12]

    @staticmethod
    def _validate_fan_weight_matrix(fan_weight_matrix):
        """读取 2x12 风扇-热源影响权重矩阵。"""
        matrix = []
        for row in fan_weight_matrix[:2]:
            values = [max(0.0, float(v)) for v in row[:12]]
            if len(values) < 12:
                values += [0.0] * (12 - len(values))
            matrix.append(values)
        while len(matrix) < 2:
            matrix.append([0.0] * 12)
        return matrix
