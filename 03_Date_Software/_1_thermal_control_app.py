"""
Thermal Control System v2.2
Features: Heater power control + Fan speed control + GL860A temperature logging
Dependencies: pip install PyQt5 pyqtgraph pyserial numpy pymodbus
"""

import sys, os, re, time, math, csv, traceback, serial, serial.tools.list_ports
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QPushButton, QComboBox, QLineEdit,
    QSpinBox, QDoubleSpinBox, QTextEdit, QCheckBox, QSplitter,
    QFileDialog, QButtonGroup, QScrollArea, QFrame, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QTimer
from PyQt5.QtGui import QFont, QColor

import pyqtgraph as pg
from pymodbus.client import ModbusTcpClient
from PID_Control import PIDFanController
from MPC_Control import MPCFanController

# ============================================================
# Constants
# ============================================================
SERIES_LIMITS = {
    1: 30, 2: 10, 3: 6, 4: 6, 5: 6, 6: 6,
    7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3
}
MAX_PLOT_POINTS = 1200
PLOT_REFRESH_INTERVAL_MS = 500
FONT_1 = '14pt'
FONT_2 = '12pt'
FONT_3 = '12pt'
FONT_4 = '12pt'
FULL_DATA_FLUSH_THRESHOLD = MAX_PLOT_POINTS * 2
MPC_LOSS_NAMES = ("heat", "grad", "cooling", "slew")
PLOT_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
    '#fabed4', '#469990', '#dcbeff', '#ffe119'
]
TEMP_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
    '#fabed4', '#469990', '#dcbeff', '#ffe119', '#000000'
]

GL860_DEFAULT_IP = '192.168.0.2'
GL860_DEFAULT_PORT = 502
GL860_START_ADDRESS = 100
GL860_NUM_CHANNELS = 13
TEMP_EMA_ALPHA = 1.0

# Style for read-only data fields
SECTION_TITLE_STYLE = (
    "font-size:20px;font-weight:bold;color:#222;padding:2px 4px;"
    "border-bottom:2px solid #555;"
)
GROUPBOX_STYLE = (
    "QGroupBox{font-size:18px;font-weight:bold;border:1px solid #bbb;border-radius:4px;"
    "margin-top:6px;padding-top:10px;}"
    "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 3px;font-size:14px;}"
)

FIELD_STYLE = (
    "QLineEdit{background:#f5f5f5;border:1px solid #bbb;border-radius:2px;"
    "padding:1px 3px;font-family:Consolas,'Courier New',monospace;font-size:16px;color:#333;}"
    "QLineEdit:read-only{color:#333;}"
)
FIELD_FIXED_W = 64


class PlotHistorySeries:
    """Bounded peak-preserving series covering the complete recording time range."""
    def __init__(self, max_points=MAX_PLOT_POINTS):
        self.max_points = int(max_points)
        self.x = []
        self.y = []

    def clear(self):
        self.x.clear()
        self.y.clear()

    def append(self, x, y, compress=True):
        self.x.append(float(x))
        self.y.append(float(y))
        if compress and len(self.x) > self.max_points:
            self.compress()

    def compress(self):
        while len(self.x) > self.max_points:
            n = len(self.x)
            keep = {0, n - 1}
            for start in range(0, n, 4):
                end = min(start + 4, n)
                values = np.asarray(self.y[start:end], dtype=float)
                finite = np.flatnonzero(np.isfinite(values))
                if finite.size == 0:
                    keep.update((start, end - 1))
                    continue
                local = values[finite]
                min_idx = start + int(finite[int(np.argmin(local))])
                max_idx = start + int(finite[int(np.argmax(local))])
                if min_idx == max_idx:
                    keep.update((start, end - 1))
                else:
                    keep.update((min_idx, max_idx))
            indices = sorted(keep)
            self.x = [self.x[i] for i in indices]
            self.y = [self.y[i] for i in indices]

    def arrays(self):
        return np.asarray(self.x, dtype=float), np.asarray(self.y, dtype=float)


# ============================================================
# STM32 Serial Worker (with disconnect detection)
# ============================================================
class SerialWorker(QThread):
    data_received = pyqtSignal(dict)
    connection_status = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)

    def __init__(self, port, baudrate=115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = True
        self._lock = QMutex()

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.connection_status.emit(True, f"STM32 connected {self.port} @ {self.baudrate}")
            time.sleep(2)
            while self.running:
                try:
                    if self.ser and self.ser.is_open:
                        if self.ser.in_waiting > 0:
                            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                            if line and "V_Sys:" in line:
                                data = self._parse_line(line)
                                if data: self.data_received.emit(data)
                        else:
                            self.msleep(10)
                    else:
                        self.connection_status.emit(False, "STM32 disconnected")
                        break
                except (serial.SerialException, OSError):
                    self.connection_status.emit(False, "STM32 disconnected")
                    break
                except Exception as e:
                    self.log_message.emit(f"Parse error: {e}")
        except serial.SerialException as e:
            self.connection_status.emit(False, f"STM32 connect failed: {e}")

    def _parse_line(self, line):
        v_sys_match = re.search(r"V_Sys:([-+]?\d*\.\d+|\d+)", line)
        if not v_sys_match: return None
        v_sys = float(v_sys_match.group(1))
        channels = re.findall(r"Ch(\d+)_VD:([-+]?\d*\.\d+|\d+),IS:([-+]?\d*\.\d+|\d+)", line)
        if not channels: return None
        result = {'v_sys': v_sys, 'channels': {}, 'fans': {}}
        for ch in channels:
            ch_idx, v_d, i_s = int(ch[0]), float(ch[1]), float(ch[2])
            v_drop = v_sys - v_d
            power = v_drop * i_s
            result['channels'][ch_idx] = {'v_drop': v_drop, 'current': i_s, 'power': power}
        fan_matches = re.findall(r"F(\d+)_RPM:([\d.]+)", line)
        for fm in fan_matches:
            result['fans'][int(fm[0])] = float(fm[1])
        return result

    def send_command(self, command):
        self._lock.lock()
        try:
            if self.ser and self.ser.is_open:
                self.ser.write(command.encode('utf-8'))
                self.log_message.emit(f"TX: {command.strip()}")
                return True
        except Exception as e:
            self.log_message.emit(f"Send failed: {e}")
        finally:
            self._lock.unlock()
        return False

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            try: self.ser.close()
            except: pass


# ============================================================
# GL860A Modbus/TCP Worker (with disconnect detection)
# ============================================================
class GL860Worker(QThread):
    temp_received = pyqtSignal(dict)
    connection_status = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)

    def __init__(self, ip, port=502, poll_interval=0.1):
        super().__init__()
        self.ip = ip
        self.port = port
        self.poll_interval = poll_interval
        self.client = None
        self.running = True

    def run(self):
        try:
            self.client = ModbusTcpClient(self.ip, port=self.port)
            if self.client.connect():
                self.connection_status.emit(True, f"GL860A connected {self.ip}:{self.port}")
                fail_count = 0
                while self.running:
                    start_t = time.time()
                    if not self.client.connected:
                        self.connection_status.emit(False, "GL860A disconnected")
                        break
                    temps = self._read_temperatures()
                    if temps is not None:
                        fail_count = 0
                        self.temp_received.emit(temps)
                    else:
                        fail_count += 1
                        if fail_count >= 10:
                            self.connection_status.emit(False, "GL860A read failures, disconnected")
                            break
                    elapsed = time.time() - start_t
                    wait = max(0, self.poll_interval - elapsed)
                    self.msleep(int(wait * 1000))
            else:
                self.connection_status.emit(False, f"GL860A connect failed {self.ip}:{self.port}")
        except Exception as e:
            self.connection_status.emit(False, f"GL860A error: {e}")

    def _read_temperatures(self):
        try:
            response = self.client.read_input_registers(
                address=GL860_START_ADDRESS, count=GL860_NUM_CHANNELS * 2, slave=1)
            if not response.isError():
                raw = response.registers
                temps = {}
                for i in range(GL860_NUM_CHANNELS):
                    val = raw[i * 2]
                    if val == 16382 or val >= 16000:
                        temps[i + 1] = math.nan
                    else:
                        if val > 32767: val -= 65536
                        temps[i + 1] = val / 10.0
                return temps
            return None
        except Exception:
            return None

    def stop(self):
        self.running = False
        if self.client:
            try: self.client.close()
            except: pass


# ============================================================
# Test Sequence Worker
# ============================================================
class TestWorker(QThread):
    progress = pyqtSignal(str)
    command_sent = pyqtSignal(dict)
    test_finished = pyqtSignal()

    def __init__(self, test_type, serial_worker, coeff=1.0):
        super().__init__()
        self.test_type = test_type
        self.serial_worker = serial_worker
        self.coeff = coeff
        self.running = True

    def run(self):
        if self.test_type == 'step': self._step_test()
        elif self.test_type == 'cosine': self._cosine_test()
        self.test_finished.emit()

    def _step_test(self):
        self.progress.emit("Step test started")
        power_mag_list = [0.1, 0.3, 0.5, 0.8, 0.5, 0.3, 0, 0.8, 0, 0.5, 0]
        hold_time = 10.0
        for i, mag in enumerate(power_mag_list):
            if not self.running: break
            self.progress.emit(f"Phase {i+1}/{len(power_mag_list)}: {mag*self.coeff*100:.1f}%")
            targets = self._send_multi_power(mag)
            self.command_sent.emit(targets)
            self._wait(hold_time)
        self.progress.emit("Step test finished")

    def _cosine_test(self):
        self.progress.emit("Cosine test started")
        min_mag, max_mag = 0.2, 1.0
        period, total_duration = 30.0, 60.0
        offset = (max_mag + min_mag) / 2.0
        amp = (max_mag - min_mag) / 2.0
        start_time = time.time()
        while self.running:
            t = time.time() - start_time
            if t > total_duration: break
            mag = offset + amp * np.cos(2 * np.pi * t / period)
            if mag < 0: mag = 0.0
            self.progress.emit(f"Cosine: {mag*self.coeff*100:.1f}% ({t:.1f}s/{total_duration:.0f}s)")
            targets = self._send_multi_power(mag)
            self.command_sent.emit(targets)
            self._wait(0.5)
        self.progress.emit("Cosine test finished")

    def _send_multi_power(self, power_mag):
        p = {}
        for ch in range(1, 13):
            p[ch] = SERIES_LIMITS[ch] * power_mag * self.coeff
        cmd = (f"MH:{p[1]:.3f},{p[2]:.3f},{p[3]:.3f},{p[4]:.3f},"
               f"{p[5]:.3f},{p[6]:.3f},{p[7]:.3f},{p[8]:.3f},"
               f"{p[9]:.3f},{p[10]:.3f},{p[11]:.3f},{p[12]:.3f}\n")
        self.serial_worker.send_command(cmd)
        return p

    def _wait(self, seconds):
        elapsed = 0
        while elapsed < seconds and self.running:
            time.sleep(0.1); elapsed += 0.1

    def stop(self): self.running = False


# ============================================================
# Command Import Worker
# ============================================================
class CommandWorker(QThread):
    progress = pyqtSignal(str)
    heater_command = pyqtSignal(dict)
    fan_command = pyqtSignal(dict)
    command_finished = pyqtSignal()

    def __init__(self, cmd_data, serial_worker):
        super().__init__()
        self.cmd_data = cmd_data  # [(time, {ch: W}, {fan: RPM}), ...]
        self.serial_worker = serial_worker
        self.running = True

    def run(self):
        self.progress.emit("Command execution started")
        start_t = time.time()
        total = len(self.cmd_data)
        for i, (cmd_t, heater, fans) in enumerate(self.cmd_data):
            if not self.running: break
            # Wait until the scheduled time
            while self.running:
                elapsed = time.time() - start_t
                if elapsed >= cmd_t: break
                wait = min(0.02, cmd_t - elapsed)
                time.sleep(wait)
            if not self.running: break
            # Send combined MHF command (12 heaters + 2 fans in one packet)
            p = [heater.get(ch, 0.0) for ch in range(1, 13)]
            f1 = fans.get(1, 0); f2 = fans.get(2, 0)
            cmd = (f"MHF:{p[0]:.3f},{p[1]:.3f},{p[2]:.3f},{p[3]:.3f},"
                   f"{p[4]:.3f},{p[5]:.3f},{p[6]:.3f},{p[7]:.3f},"
                   f"{p[8]:.3f},{p[9]:.3f},{p[10]:.3f},{p[11]:.3f},"
                   f"{f1:.3f},{f2:.3f}\n")
            self.serial_worker.send_command(cmd)
            self.heater_command.emit(heater)
            self.fan_command.emit(fans)
            # Progress (every 50 steps or at key points)
            if i % 50 == 0 or i == total - 1:
                self.progress.emit(f"Command: {cmd_t:.1f}s ({i+1}/{total})")
        self.command_finished.emit()
        self.progress.emit("Command execution finished")

    def stop(self): self.running = False


class ControlProfileWorker(QThread):
    progress = pyqtSignal(str)
    control_command = pyqtSignal(dict, float)
    profile_finished = pyqtSignal()

    def __init__(self, profile_data):
        super().__init__()
        self.profile_data = profile_data  # [(time, {ch: W}, temp_aim), ...]
        self.running = True

    def run(self):
        self.progress.emit("Control profile execution started")
        start_t = time.time()
        total = len(self.profile_data)
        for i, (cmd_t, heater, temp_aim) in enumerate(self.profile_data):
            if not self.running: break
            while self.running:
                elapsed = time.time() - start_t
                if elapsed >= cmd_t: break
                time.sleep(min(0.02, cmd_t - elapsed))
            if not self.running: break

            self.control_command.emit(heater, float(temp_aim))
            if i % 50 == 0 or i == total - 1:
                self.progress.emit(f"Control profile: {cmd_t:.1f}s ({i+1}/{total}), T_aim={temp_aim:.1f}°C")
        self.profile_finished.emit()
        self.progress.emit("Control profile execution finished")

    def stop(self): self.running = False


class MPCComputeWorker(QThread):
    result_ready = pyqtSignal(int, int, dict, float, float, float)
    error = pyqtSignal(str)

    def __init__(self, controller, temperatures, heater_power, actual_fan_rpm, ambient_temp, t_aim, dt_s, max_temp):
        super().__init__()
        self.controller = controller
        self.temperatures = dict(temperatures)
        self.heater_power = dict(heater_power)
        self.actual_fan_rpm = dict(actual_fan_rpm)
        self.ambient_temp = float(ambient_temp)
        self.t_aim = float(t_aim)
        self.dt_s = float(dt_s)
        self.max_temp = float(max_temp)

    def run(self):
        try:
            rpm1, rpm2 = self.controller.compute(
                self.temperatures,
                heater_power=self.heater_power,
                actual_fan_rpm=self.actual_fan_rpm,
                ambient_temp_c=self.ambient_temp,
                t_aim=self.t_aim,
                dt_s=self.dt_s,
            )
            status = dict(self.controller.last_status)
            status['actual_fan_rpm'] = dict(self.actual_fan_rpm)
            self.result_ready.emit(
                int(rpm1),
                int(rpm2),
                status,
                self.t_aim,
                self.max_temp,
                self.ambient_temp,
            )
        except Exception as exc:
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


# ============================================================
# Main Window
# ============================================================
class ThermalControlApp(QMainWindow):
    VIEW_ALL = 'all'

    def __init__(self):
        super().__init__()
        self.serial_worker = None
        self.gl860_worker = None
        self.test_worker = None
        self.cmd_worker = None
        self.ctrl_profile_worker = None
        self.mpc_worker = None
        self._plot_refresh_suspended = False
        self._cmd_data = None
        self._cmd_save_dir = ""  # Auto-save directory for command results (empty = prompt)
        self.start_time = time.time()
        self._recording = True  # Controls whether incoming data is written to buffers

        self.time_data = []
        self.channel_power = {i: [] for i in range(1, 13)}
        self.channel_voltage = {i: [] for i in range(1, 13)}
        self.channel_current = {i: [] for i in range(1, 13)}
        self.channel_target = {i: [] for i in range(1, 13)}
        self.current_targets = {i: 0.0 for i in range(1, 13)}
        self.channel_fan_actual = {1: [], 2: []}
        self.channel_fan_target = {1: [], 2: []}
        self.fan_rpm_values = {1: 0, 2: 0}
        self.current_fan_targets = {1: 0.0, 2: 0.0}
        self.mpc_time = []
        self.mpc_optimize_elapsed = []
        self.mpc_loss_drpm = {name: {1: [], 2: []} for name in MPC_LOSS_NAMES}
        self.mpc_total_drpm = {1: [], 2: []}
        self.mpc_t_aim = []
        self.mpc_t_max = []
        self.mpc_ambient_temp = []
        self.mpc_fan1_actual = []
        self.mpc_fan2_actual = []
        self.mpc_fan1_target = []
        self.mpc_fan2_target = []
        self.mpc_zp_actual_enabled = []
        self.mpc_zp_actual_step = []
        self.mpc_zp_prediction_enabled = []
        self.mpc_zp_state_isolated = []
        self.mpc_zp_correction_max_abs = []
        self.mpc_zp_output_norm_rate = {i: [] for i in range(1, 13)}
        self.mpc_zp_trend_norm_rate = {i: [] for i in range(1, 13)}
        self.mpc_core_dtheta_dt_norm_rate = {i: [] for i in range(1, 13)}
        self.mpc_augment_h = {i: [] for i in range(1, 17)}

        self.temp_time = []
        self.channel_temp = {i: [] for i in range(1, GL860_NUM_CHANNELS + 1)}
        self.latest_temps = {}
        self._temp_ema_values = {}

        self._plot_power = {i: PlotHistorySeries() for i in range(1, 13)}
        self._plot_voltage = {i: PlotHistorySeries() for i in range(1, 13)}
        self._plot_current = {i: PlotHistorySeries() for i in range(1, 13)}
        self._plot_target = {i: PlotHistorySeries() for i in range(1, 13)}
        self._plot_fan_actual = {i: PlotHistorySeries() for i in [1, 2]}
        self._plot_fan_target = {i: PlotHistorySeries() for i in [1, 2]}
        self._plot_temp = {i: PlotHistorySeries() for i in range(1, GL860_NUM_CHANNELS + 1)}
        self._plot_heater_time_range = [None, None]
        self._plot_temp_time_range = [None, None]

        self._flush_files = []

        self.current_view = self.VIEW_ALL

        # Automatic fan controllers
        self._pid_controller = PIDFanController()
        self._mpc_controller = None
        self._pid_active_t_aim = 40.0
        self._mpc_min_interval_s = 1.0
        self._mpc_last_compute_wall = None
        self._mpc_cached_plan_rpm = np.empty((0, 2), dtype=float)
        self._mpc_cached_plan_index = 0
        self._mpc_cached_plan_last_wall = None
        self._control_mode = "none"  # "none" / "pid" / "mpc"
        self._ctrl_profile_time = np.array([])
        self._ctrl_profile_power = {ch: np.array([]) for ch in range(1, 13)}
        self._ctrl_profile_temp_aim = np.array([])
        self._ctrl_profile_data = None
        self._ctrl_profile_running = False

        self._setup_ui()
        self._setup_timer()
        self._rebuild_plot_layout()

    def _setup_ui(self):
        self.setWindowTitle("Thermal Control System v2.5")
        self.setMinimumSize(1500, 950)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.main_splitter)

        # ========== Left Panel ==========
        left_panel = QWidget()
        left_panel.setFixedWidth(520)
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area for vertical scrolling (horizontal compression preserved)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        scroll_content = QWidget()
        scroll_content.setStyleSheet(
            "QLabel{font-size:13px;color:#333;}"
            "QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox{min-height:24px;font-size:13px;}"
            "QPushButton{min-height:26px;font-size:13px;padding:3px 8px;}"
        )
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(8, 6, 8, 8)
        scroll_layout.setSpacing(6)

        # Section title
        left_title = QLabel("Control Panel")
        left_title.setStyleSheet(SECTION_TITLE_STYLE)
        scroll_layout.addWidget(left_title)

        # STM32 Serial
        conn_group = QGroupBox("STM32 Serial")
        conn_group.setStyleSheet(GROUPBOX_STYLE)
        conn_layout = QGridLayout()
        self._style_control_grid(conn_layout)
        self.port_combo = QComboBox(); self._refresh_ports()
        conn_layout.addWidget(QLabel("Port:"), 0, 0); conn_layout.addWidget(self.port_combo, 0, 1, 1, 3)
        self.baudrate_edit = QLineEdit("115200")
        conn_layout.addWidget(QLabel("Baud:"), 1, 0); conn_layout.addWidget(self.baudrate_edit, 1, 1, 1, 3)
        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Rescan available COM ports")
        self.refresh_btn.clicked.connect(self._refresh_ports)
        self.connect_btn = QPushButton("Connect"); self.connect_btn.clicked.connect(self._connect_serial)
        self.disconnect_btn = QPushButton("Disconnect"); self.disconnect_btn.clicked.connect(self._disconnect_serial)
        self.disconnect_btn.setEnabled(False)
        btn_row.addWidget(self.connect_btn); btn_row.addWidget(self.disconnect_btn); btn_row.addWidget(self.refresh_btn)
        conn_layout.addLayout(btn_row, 2, 0, 1, 4)
        self.status_label = QLabel("Status: Disconnected"); self.status_label.setStyleSheet("color:red;font-weight:bold;font-size:14px;")
        conn_layout.addWidget(self.status_label, 3, 0, 1, 4)
        conn_group.setLayout(conn_layout)
        scroll_layout.addWidget(conn_group)

        # GL860A
        gl_group = QGroupBox("GL860A Temp Logger (Modbus/TCP)")
        gl_group.setStyleSheet(GROUPBOX_STYLE)
        gl_layout = QGridLayout()
        self._style_control_grid(gl_layout)
        self.gl_ip_edit = QLineEdit(GL860_DEFAULT_IP)
        gl_layout.addWidget(QLabel("IP:"), 0, 0); gl_layout.addWidget(self.gl_ip_edit, 0, 1, 1, 3)
        self.gl_port_spin = QSpinBox(); self.gl_port_spin.setRange(1, 65535); self.gl_port_spin.setValue(GL860_DEFAULT_PORT)
        gl_layout.addWidget(QLabel("Port:"), 1, 0); gl_layout.addWidget(self.gl_port_spin, 1, 1, 1, 3)
        self.gl_poll_spin = QDoubleSpinBox(); self.gl_poll_spin.setRange(0.05, 10.0)
        self.gl_poll_spin.setValue(0.1); self.gl_poll_spin.setSingleStep(0.05); self.gl_poll_spin.setDecimals(2)
        gl_layout.addWidget(QLabel("Poll(s):"), 2, 0); gl_layout.addWidget(self.gl_poll_spin, 2, 1, 1, 3)
        gl_btn_row = QHBoxLayout()
        self.gl_connect_btn = QPushButton("Connect"); self.gl_connect_btn.clicked.connect(self._connect_gl860)
        self.gl_disconnect_btn = QPushButton("Disconnect"); self.gl_disconnect_btn.clicked.connect(self._disconnect_gl860)
        self.gl_disconnect_btn.setEnabled(False)
        gl_btn_row.addWidget(self.gl_connect_btn); gl_btn_row.addWidget(self.gl_disconnect_btn)
        gl_layout.addLayout(gl_btn_row, 3, 0, 1, 4)
        self.gl_status_label = QLabel("Status: Disconnected"); self.gl_status_label.setStyleSheet("color:red;font-weight:bold;font-size:14px;")
        gl_layout.addWidget(self.gl_status_label, 4, 0, 1, 4)
        gl_group.setLayout(gl_layout)
        scroll_layout.addWidget(gl_group)

        # Single Heater + Multi Heater (same row)
        heater_row = QHBoxLayout()
        heater_row.setSpacing(6)

        # Single Heater
        single_group = QGroupBox("Single Heater (H)")
        single_group.setStyleSheet(GROUPBOX_STYLE)
        single_layout = QGridLayout()
        self._style_control_grid(single_layout)
        self.ch_spin = QSpinBox(); self.ch_spin.setRange(1, 12)
        single_layout.addWidget(QLabel("Channel:"), 0, 0); single_layout.addWidget(self.ch_spin, 0, 1)
        self.single_power_spin = QDoubleSpinBox()
        self.single_power_spin.setRange(0.0, 1.0); self.single_power_spin.setSingleStep(0.05); self.single_power_spin.setDecimals(2)
        single_layout.addWidget(QLabel("Power Coeff:"), 1, 0); single_layout.addWidget(self.single_power_spin, 1, 1)
        self.single_send_btn = QPushButton("Send"); self.single_send_btn.clicked.connect(self._send_single)
        single_layout.addWidget(self.single_send_btn, 2, 0, 1, 2)
        single_group.setLayout(single_layout)
        heater_row.addWidget(single_group)

        # Multi Heater
        multi_group = QGroupBox("Multi Heater (MH)")
        multi_group.setStyleSheet(GROUPBOX_STYLE)
        multi_layout = QGridLayout()
        self._style_control_grid(multi_layout)
        self.multi_power_spin = QDoubleSpinBox()
        self.multi_power_spin.setRange(0.0, 1.0); self.multi_power_spin.setSingleStep(0.05); self.multi_power_spin.setDecimals(2)
        multi_layout.addWidget(QLabel("Power Coeff:"), 0, 0); multi_layout.addWidget(self.multi_power_spin, 0, 1)
        self.multi_send_btn = QPushButton("Send"); self.multi_send_btn.clicked.connect(self._send_multi)
        multi_layout.addWidget(self.multi_send_btn, 1, 0, 1, 2)
        multi_group.setLayout(multi_layout)
        heater_row.addWidget(multi_group)

        heater_row.setStretch(0, 1)
        heater_row.setStretch(1, 1)
        scroll_layout.addLayout(heater_row)

        # Fan Control
        fan_group = QGroupBox("Fan Control (MF, RPM)")
        fan_group.setStyleSheet(GROUPBOX_STYLE)
        fan_layout = QGridLayout()
        self._style_control_grid(fan_layout)
        fan_layout.addWidget(QLabel("Fan1 RPM:"), 0, 0)
        self.fan1_input = QLineEdit("0")
        fan_layout.addWidget(self.fan1_input, 0, 1)
        fan_layout.addWidget(QLabel("Fan2 RPM:"), 0, 2)
        self.fan2_input = QLineEdit("0")
        fan_layout.addWidget(self.fan2_input, 0, 3)
        self.fan_send_btn = QPushButton("Send"); self.fan_send_btn.clicked.connect(self._send_fan_multi)
        fan_layout.addWidget(self.fan_send_btn, 1, 0, 1, 4)
        fan_group.setLayout(fan_layout)
        scroll_layout.addWidget(fan_group)

        # Test Sequence
        test_group = QGroupBox("Test Sequence")
        test_group.setStyleSheet(GROUPBOX_STYLE)
        test_layout = QGridLayout()
        self._style_control_grid(test_layout)
        self.step_coeff_spin = QDoubleSpinBox()
        self.step_coeff_spin.setRange(0.01, 5.0); self.step_coeff_spin.setValue(1.0); self.step_coeff_spin.setSingleStep(0.1)
        test_layout.addWidget(QLabel("Step Coeff:"), 0, 0); test_layout.addWidget(self.step_coeff_spin, 0, 1, 1, 2)
        self.step_btn = QPushButton("Step Test"); self.step_btn.clicked.connect(self._run_step_test)
        test_layout.addWidget(self.step_btn, 0, 3)
        self.cosine_coeff_spin = QDoubleSpinBox()
        self.cosine_coeff_spin.setRange(0.01, 5.0); self.cosine_coeff_spin.setValue(1.0); self.cosine_coeff_spin.setSingleStep(0.1)
        test_layout.addWidget(QLabel("Cosine Coeff:"), 1, 0); test_layout.addWidget(self.cosine_coeff_spin, 1, 1, 1, 2)
        self.cosine_btn = QPushButton("Cosine Test"); self.cosine_btn.clicked.connect(self._run_cosine_test)
        test_layout.addWidget(self.cosine_btn, 1, 3)
        test_group.setLayout(test_layout)
        scroll_layout.addWidget(test_group)

        # Command Import
        cmd_group = QGroupBox("Command Import")
        cmd_group.setStyleSheet(GROUPBOX_STYLE)
        cmd_layout = QGridLayout()
        self._style_control_grid(cmd_layout)
        self.cmd_import_btn = QPushButton("Import CSV"); self.cmd_import_btn.clicked.connect(self._import_command)
        cmd_layout.addWidget(self.cmd_import_btn, 0, 0, 1, 4)
        self.cmd_file_label = QLabel("No file loaded")
        self.cmd_file_label.setStyleSheet("font-size:14px;color:#999;")
        cmd_layout.addWidget(self.cmd_file_label, 1, 0, 1, 4)
        # Save directory — row1: label + buttons, row2: path text box
        save_dir_row1 = QHBoxLayout(); save_dir_row1.setSpacing(4)
        save_dir_row1.addWidget(QLabel("Save Dir:"))
        save_dir_row1.addStretch(1)
        self.cmd_browse_btn = QPushButton("Browse")
        self.cmd_browse_btn.setFixedWidth(72)
        self.cmd_browse_btn.clicked.connect(self._browse_cmd_save_dir)
        save_dir_row1.addWidget(self.cmd_browse_btn)
        self.cmd_clear_dir_btn = QPushButton("✕")
        self.cmd_clear_dir_btn.setFixedWidth(24)
        self.cmd_clear_dir_btn.setToolTip("Clear save directory")
        self.cmd_clear_dir_btn.clicked.connect(self._clear_cmd_save_dir)
        save_dir_row1.addWidget(self.cmd_clear_dir_btn)
        cmd_layout.addLayout(save_dir_row1, 2, 0, 1, 4)
        self.cmd_save_dir_edit = QLineEdit()
        self.cmd_save_dir_edit.setPlaceholderText("(optional) auto-save folder...")
        self.cmd_save_dir_edit.setReadOnly(True)
        self.cmd_save_dir_edit.setStyleSheet("font-size:13px;color:#555;background:#f5f5f5;")
        cmd_layout.addWidget(self.cmd_save_dir_edit, 3, 0, 1, 4)
        cmd_btn_row2 = QHBoxLayout()
        self.cmd_confirm_btn = QPushButton("Confirm & Start"); self.cmd_confirm_btn.clicked.connect(self._start_command)
        self.cmd_confirm_btn.setEnabled(False)
        cmd_btn_row2.addWidget(self.cmd_confirm_btn)
        self.cmd_stop_btn = QPushButton("Stop"); self.cmd_stop_btn.clicked.connect(self._stop_command)
        self.cmd_stop_btn.setEnabled(False)
        cmd_btn_row2.addWidget(self.cmd_stop_btn)
        cmd_layout.addLayout(cmd_btn_row2, 4, 0, 1, 4)
        cmd_group.setLayout(cmd_layout)
        scroll_layout.addWidget(cmd_group)

        # Control Mode
        ctrl_mode_group = QGroupBox("Control Mode")
        ctrl_mode_group.setStyleSheet(GROUPBOX_STYLE)
        ctrl_mode_layout = QGridLayout()
        self._style_control_grid(ctrl_mode_layout)

        self.ctrl_mode_btn_group = QButtonGroup()
        self.ctrl_mode_btn_group.setExclusive(True)

        _mode_btn_style = (
            "QPushButton{font-size:14px;font-weight:bold;border-radius:4px;border:1px solid #999;"
            "background:#e8e8e8;color:#333;padding:6px 10px;}"
            "QPushButton:checked{background:#2a7fff;color:white;border:2px solid #1a5fcc;}"
            "QPushButton:hover{background:#d0d0d0;}"
            "QPushButton:checked:hover{background:#2570e6;}"
        )

        mode_btn_row = QHBoxLayout()
        self.mode_none_btn = QPushButton("不控制")
        self.mode_none_btn.setCheckable(True)
        self.mode_none_btn.setChecked(True)
        self.mode_none_btn.setStyleSheet(_mode_btn_style)
        self.ctrl_mode_btn_group.addButton(self.mode_none_btn, 0)
        mode_btn_row.addWidget(self.mode_none_btn)

        self.mode_pid_btn = QPushButton("PID 控制")
        self.mode_pid_btn.setCheckable(True)
        self.mode_pid_btn.setStyleSheet(_mode_btn_style)
        self.ctrl_mode_btn_group.addButton(self.mode_pid_btn, 1)
        mode_btn_row.addWidget(self.mode_pid_btn)

        self.mode_mpc_btn = QPushButton("MPC 控制")
        self.mode_mpc_btn.setCheckable(True)
        self.mode_mpc_btn.setStyleSheet(_mode_btn_style)
        self.ctrl_mode_btn_group.addButton(self.mode_mpc_btn, 2)
        mode_btn_row.addWidget(self.mode_mpc_btn)

        ctrl_mode_layout.addLayout(mode_btn_row, 0, 0, 1, 4)

        # PID target temperature
        ctrl_mode_layout.addWidget(QLabel("T_aim:"), 1, 0)
        self.pid_t_aim_spin = QDoubleSpinBox()
        self.pid_t_aim_spin.setRange(0.0, 300.0)
        self.pid_t_aim_spin.setValue(40.0)
        self.pid_t_aim_spin.setDecimals(1)
        self.pid_t_aim_spin.setSuffix(" °C")
        ctrl_mode_layout.addWidget(self.pid_t_aim_spin, 1, 1, 1, 3)

        self.ctrl_import_btn = QPushButton("Import Config")
        self.ctrl_import_btn.clicked.connect(self._import_control_config)
        ctrl_mode_layout.addWidget(self.ctrl_import_btn, 2, 0, 1, 2)
        self.ctrl_config_label = QLabel("No control config")
        self.ctrl_config_label.setStyleSheet("font-size:14px;color:#999;")
        ctrl_mode_layout.addWidget(self.ctrl_config_label, 2, 2, 1, 2)

        ctrl_profile_btn_row = QHBoxLayout()
        self.ctrl_start_btn = QPushButton("Start")
        self.ctrl_start_btn.clicked.connect(self._start_control_profile)
        self.ctrl_start_btn.setEnabled(False)
        ctrl_profile_btn_row.addWidget(self.ctrl_start_btn)
        self.ctrl_stop_btn = QPushButton("Stop")
        self.ctrl_stop_btn.clicked.connect(self._stop_control_profile)
        self.ctrl_stop_btn.setEnabled(False)
        ctrl_profile_btn_row.addWidget(self.ctrl_stop_btn)
        ctrl_mode_layout.addLayout(ctrl_profile_btn_row, 3, 0, 1, 4)

        self.pid_active_t_aim_label = QLabel(f"Current T_aim: {self._pid_active_t_aim:.1f} °C")
        self.pid_active_t_aim_label.setStyleSheet("font-size:14px;font-weight:bold;color:#444;")
        ctrl_mode_layout.addWidget(self.pid_active_t_aim_label, 4, 0, 1, 4)

        # Status label
        self.ctrl_status_label = QLabel("Mode: 不控制")
        self.ctrl_status_label.setStyleSheet("font-size:14px;font-weight:bold;color:#666;")
        ctrl_mode_layout.addWidget(self.ctrl_status_label, 5, 0, 1, 4)

        ctrl_mode_group.setLayout(ctrl_mode_layout)
        scroll_layout.addWidget(ctrl_mode_group)
        self.ctrl_mode_btn_group.buttonClicked[int].connect(self._on_control_mode_changed)

        # Action Buttons
        op1 = QHBoxLayout()
        self.zero_heat_btn = QPushButton("Heater Zero")
        self.zero_heat_btn.setStyleSheet("QPushButton{background:#e6a800;color:white;font-weight:bold;font-size:14px;padding:6px;border-radius:4px;}QPushButton:hover{background:#cc9400;}")
        self.zero_heat_btn.clicked.connect(self._zero_heat); op1.addWidget(self.zero_heat_btn)
        self.zero_fan_btn = QPushButton("Fan Zero")
        self.zero_fan_btn.setStyleSheet("QPushButton{background:#4a90d9;color:white;font-weight:bold;font-size:14px;padding:6px;border-radius:4px;}QPushButton:hover{background:#3a7cc4;}")
        self.zero_fan_btn.clicked.connect(self._zero_fan); op1.addWidget(self.zero_fan_btn)
        scroll_layout.addLayout(op1)

        self.emergency_btn = QPushButton("EMERGENCY STOP")
        self.emergency_btn.setStyleSheet("QPushButton{background:#dd2222;color:white;font-weight:bold;font-size:14px;padding:8px;border-radius:4px;}QPushButton:hover{background:#ff0000;}")
        self.emergency_btn.clicked.connect(self._emergency_stop)
        scroll_layout.addWidget(self.emergency_btn)

        op2 = QHBoxLayout()
        self.clear_btn = QPushButton("Clear Curves"); self.clear_btn.clicked.connect(self._clear_plot_data); op2.addWidget(self.clear_btn)
        self.export_btn = QPushButton("Export Data"); self.export_btn.clicked.connect(self._export_data); op2.addWidget(self.export_btn)
        scroll_layout.addLayout(op2)
        scroll_layout.addStretch()

        # Set scroll content and finalize scroll area
        scroll_area.setWidget(scroll_content)
        left_panel_layout.addWidget(scroll_area)

        # ========== Right Side Sections ==========

        # ---- Sampling Section ----
        sampling_section = QWidget()
        sampling_section.setFixedWidth(430)
        sampling_section_vlayout = QVBoxLayout(sampling_section)
        sampling_section_vlayout.setContentsMargins(4,4,4,4)
        sampling_section_vlayout.setSpacing(4)
        sample_title = QLabel("Real-time Sampling")
        sample_title.setStyleSheet(SECTION_TITLE_STYLE)
        sampling_section_vlayout.addWidget(sample_title)

        # Sampling data (inside scroll)
        sample_container = QWidget()
        sample_vlayout = QVBoxLayout(sample_container)
        sample_vlayout.setContentsMargins(4,4,4,4)
        sample_vlayout.setSpacing(20)

        # Ambient Temperature row
        amb_row = QHBoxLayout(); amb_row.setSpacing(8)
        amb_row.addWidget(self._field_label("Ambient Temp:"))
        self.ambient_field = QLineEdit("--"); self.ambient_field.setReadOnly(True)
        self.ambient_field.setFixedWidth(90); self.ambient_field.setStyleSheet(FIELD_STYLE)
        amb_row.addWidget(self.ambient_field)
        amb_row.addStretch()
        sample_vlayout.addLayout(amb_row)

        # Fan row
        fan_row = QHBoxLayout(); fan_row.setSpacing(8)
        fan_row.addWidget(self._field_label("Fan1:"))
        self.fan1_display = QLineEdit("--"); self.fan1_display.setReadOnly(True)
        self.fan1_display.setFixedWidth(90); self.fan1_display.setStyleSheet(FIELD_STYLE)
        fan_row.addWidget(self.fan1_display)
        fan_row.addWidget(self._field_label("Fan2:"))
        self.fan2_display = QLineEdit("--"); self.fan2_display.setReadOnly(True)
        self.fan2_display.setFixedWidth(90); self.fan2_display.setStyleSheet(FIELD_STYLE)
        fan_row.addWidget(self.fan2_display)
        fan_row.addStretch()
        sample_vlayout.addLayout(fan_row)

        # Separator
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color:#ccc;")
        sample_vlayout.addWidget(sep)

        # 12 Channel rows
        self.ch_p_fields = {}; self.ch_t_fields = {}; self.ch_v_fields = {}; self.ch_i_fields = {}
        for ch in range(1, 13):
            row = QHBoxLayout(); row.setSpacing(8)
            # CH label
            ch_lbl = QLabel(f"CH{ch:02d}")
            ch_lbl.setFixedWidth(44)
            ch_lbl.setStyleSheet("font-size:14px;font-weight:bold;color:#333;")
            row.addWidget(ch_lbl)
            # P(W)
            row.addWidget(self._field_label("P:"))
            p_f = QLineEdit("--"); p_f.setReadOnly(True); p_f.setFixedWidth(FIELD_FIXED_W); p_f.setStyleSheet(FIELD_STYLE)
            self.ch_p_fields[ch] = p_f; row.addWidget(p_f)
            # T(°C)
            row.addWidget(self._field_label("T:"))
            t_f = QLineEdit("--"); t_f.setReadOnly(True); t_f.setFixedWidth(FIELD_FIXED_W); t_f.setStyleSheet(FIELD_STYLE)
            self.ch_t_fields[ch] = t_f; row.addWidget(t_f)
            # V(V)
            row.addWidget(self._field_label("V:"))
            v_f = QLineEdit("--"); v_f.setReadOnly(True); v_f.setFixedWidth(FIELD_FIXED_W); v_f.setStyleSheet(FIELD_STYLE)
            self.ch_v_fields[ch] = v_f; row.addWidget(v_f)
            # I(A)
            row.addWidget(self._field_label("I:"))
            i_f = QLineEdit("--"); i_f.setReadOnly(True); i_f.setFixedWidth(FIELD_FIXED_W); i_f.setStyleSheet(FIELD_STYLE)
            self.ch_i_fields[ch] = i_f; row.addWidget(i_f)
            row.addStretch()
            sample_vlayout.addLayout(row)

        sample_vlayout.addStretch()

        # Wrap in scroll area
        scroll = QScrollArea()
        scroll.setWidget(sample_container)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:1px solid #ccc;border-radius:4px;background:#fff;}")
        sampling_section_vlayout.addWidget(scroll, stretch=1)

        # Top splitter: sampling + curves
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.setChildrenCollapsible(False)
        self.top_splitter.addWidget(sampling_section)

        # ---- Curve Visualization ----
        plot_widget = QWidget()
        plot_vlayout = QVBoxLayout(plot_widget)
        plot_vlayout.setContentsMargins(4,4,8,8)
        plot_title = QLabel("Curve Visualization")
        plot_title.setStyleSheet(SECTION_TITLE_STYLE)
        plot_vlayout.addWidget(plot_title)

        # ---- Row 1: View mode buttons ----
        mode_row = QHBoxLayout(); mode_row.setSpacing(3)
        self.view_btn_group = QButtonGroup()
        self.view_btn_group.setExclusive(True)

        self.all_btn = QPushButton("All"); self.all_btn.setCheckable(True); self.all_btn.setChecked(True); self.all_btn.setFixedHeight(26)
        self.all_btn.setStyleSheet(self._view_btn_style())
        self.view_btn_group.addButton(self.all_btn, 0); mode_row.addWidget(self.all_btn)

        self.ch_view_btns = {}
        for i in range(1, 13):
            btn = QPushButton(f"{i}"); btn.setCheckable(True); btn.setFixedWidth(34); btn.setFixedHeight(26)
            btn.setStyleSheet(self._view_btn_style())
            self.view_btn_group.addButton(btn, i); self.ch_view_btns[i] = btn; mode_row.addWidget(btn)

        mode_row.addStretch(1)
        plot_vlayout.addLayout(mode_row)

        # ---- Row 2: Subplots / Lock / X Range ----
        ctrl_row = QHBoxLayout(); ctrl_row.setSpacing(6)

        self.subplot_label = QLabel("Subplots:"); self.subplot_label.setStyleSheet("font-weight:bold;font-size:14px;")
        ctrl_row.addWidget(self.subplot_label)
        self.p_cb = QCheckBox("P"); self.p_cb.setChecked(True); self.p_cb.setStyleSheet("font-weight:bold;font-size:14px;color:#cc0000;")
        ctrl_row.addWidget(self.p_cb)
        self.v_cb = QCheckBox("V"); self.v_cb.setChecked(False); self.v_cb.setStyleSheet("font-weight:bold;font-size:14px;color:#2ca02c;")
        ctrl_row.addWidget(self.v_cb)
        self.i_cb = QCheckBox("I"); self.i_cb.setChecked(False); self.i_cb.setStyleSheet("font-weight:bold;font-size:14px;color:#1f77b4;")
        ctrl_row.addWidget(self.i_cb)
        self.t_cb = QCheckBox("Temp"); self.t_cb.setChecked(True); self.t_cb.setStyleSheet("font-weight:bold;font-size:14px;color:#ff7f0e;")
        ctrl_row.addWidget(self.t_cb)
        self.fan_cb = QCheckBox("Fan"); self.fan_cb.setChecked(True); self.fan_cb.setStyleSheet("font-weight:bold;font-size:14px;color:#4363d8;")
        ctrl_row.addWidget(self.fan_cb)

        ctrl_row.addSpacing(16)
        self.lock_cb = QCheckBox("Lock View"); self.lock_cb.setStyleSheet("font-weight:bold;font-size:14px;")
        ctrl_row.addWidget(self.lock_cb)

        ctrl_row.addSpacing(16)
        x_range_lbl = QLabel("X Range:"); x_range_lbl.setStyleSheet("font-weight:bold;font-size:14px;")
        ctrl_row.addWidget(x_range_lbl)
        self.x_min_spin = QDoubleSpinBox(); self.x_min_spin.setRange(0, 999999); self.x_min_spin.setDecimals(1)
        self.x_min_spin.setSuffix(" s"); self.x_min_spin.setValue(0); self.x_min_spin.setFixedWidth(80)
        ctrl_row.addWidget(self.x_min_spin)
        ctrl_row.addWidget(QLabel("~"))
        self.x_max_spin = QDoubleSpinBox(); self.x_max_spin.setRange(0, 999999); self.x_max_spin.setDecimals(1)
        self.x_max_spin.setSuffix(" s"); self.x_max_spin.setValue(0); self.x_max_spin.setFixedWidth(80)
        self.x_max_spin.setSpecialValueText("Auto")
        ctrl_row.addWidget(self.x_max_spin)
        self.x_range_btn = QPushButton("Set"); self.x_range_btn.setFixedWidth(44); self.x_range_btn.setFixedHeight(26)
        self.x_range_btn.clicked.connect(self._apply_x_range)
        ctrl_row.addWidget(self.x_range_btn)
        self.x_auto_btn = QPushButton("Auto"); self.x_auto_btn.setFixedWidth(52); self.x_auto_btn.setFixedHeight(26)
        self.x_auto_btn.clicked.connect(self._set_x_auto)
        ctrl_row.addWidget(self.x_auto_btn)
        ctrl_row.addStretch(1)

        plot_vlayout.addLayout(ctrl_row)

        # All mode channel select
        self.multi_sel_row = QHBoxLayout(); self.multi_sel_row.setSpacing(3)
        self.multi_sel_label = QLabel("Select:"); self.multi_sel_label.setStyleSheet("font-weight:bold;font-size:14px;")
        self.multi_sel_row.addWidget(self.multi_sel_label)
        self.ch_cbs = {}
        for i in range(1, 13):
            cb = QPushButton(f"{i}"); cb.setCheckable(True); cb.setFixedWidth(34); cb.setFixedHeight(26)
            cb.setStyleSheet(
                "QPushButton{font-size:14px;font-weight:bold;border-radius:3px;border:1px solid #999;background:white;color:#333;padding:2px 4px;}"
                "QPushButton:checked{background:#555;color:white;border:2px solid #555;}"
                "QPushButton:hover{background:#e0e0e0;}")
            cb.setChecked(True)
            self.ch_cbs[i] = cb; self.multi_sel_row.addWidget(cb)
        self.multi_sel_row.addStretch(1)
        plot_vlayout.addLayout(self.multi_sel_row)

        # Temp mode channel select
        self.temp_sel_row = QHBoxLayout(); self.temp_sel_row.setSpacing(3)
        self.temp_sel_label = QLabel("Temp CH:"); self.temp_sel_label.setStyleSheet("font-weight:bold;font-size:14px;")
        self.temp_sel_row.addWidget(self.temp_sel_label)
        self.temp_cbs = {}
        for i in range(1, GL860_NUM_CHANNELS + 1):
            cb = QPushButton(f"{i}"); cb.setCheckable(True); cb.setFixedWidth(34); cb.setFixedHeight(26)
            cb.setStyleSheet(
                "QPushButton{font-size:14px;font-weight:bold;border-radius:3px;border:1px solid #999;background:white;color:#333;padding:2px 4px;}"
                "QPushButton:checked{background:#555;color:white;border:2px solid #555;}"
                "QPushButton:hover{background:#e0e0e0;}")
            cb.setChecked(i <= 5)
            self.temp_cbs[i] = cb; self.temp_sel_row.addWidget(cb)
        self.temp_show_all_btn = QPushButton("Show ALL")
        self.temp_show_all_btn.setFixedHeight(26)
        self.temp_show_all_btn.setStyleSheet(
            "QPushButton{font-size:13px;font-weight:bold;border-radius:3px;border:1px solid #777;background:#f0f0f0;color:#333;padding:2px 8px;}"
            "QPushButton:hover{background:#e0e0e0;}")
        self.temp_show_all_btn.clicked.connect(self._show_all_temp_channels)
        self.temp_sel_row.addWidget(self.temp_show_all_btn)
        self.temp_sel_row.addStretch(1)
        plot_vlayout.addLayout(self.temp_sel_row)

        self.plot_container = QVBoxLayout()
        plot_vlayout.addLayout(self.plot_container, stretch=1)

        # PlotWidgets
        self.p_plot = pg.PlotWidget(); self.p_plot.setBackground('w')
        self.p_plot.setLabel('left', 'Power (W)'); self.p_plot.showGrid(x=True, y=True, alpha=0.3)
        self.p_legend = pg.LegendItem(offset=(60, 10), labelTextSize='8pt', colCount=4)
        self.p_legend.setParentItem(self.p_plot.plotItem)

        self.v_plot = pg.PlotWidget(); self.v_plot.setBackground('w')
        self.v_plot.setLabel('left', 'Voltage (V)'); self.v_plot.showGrid(x=True, y=True, alpha=0.3)
        self.v_plot.addLegend(offset=(60, 10))  # x-axis controlled manually, no setXLink

        self.i_plot = pg.PlotWidget(); self.i_plot.setBackground('w')
        self.i_plot.setLabel('left', 'Current (A)'); self.i_plot.showGrid(x=True, y=True, alpha=0.3)
        self.i_plot.addLegend(offset=(60, 10))  # x-axis controlled manually, no setXLink

        self.fan_plot = pg.PlotWidget(); self.fan_plot.setBackground('w')
        self.fan_plot.setLabel('left', 'RPM'); self.fan_plot.setLabel('bottom', 'Time (s)')
        self.fan_plot.showGrid(x=True, y=True, alpha=0.3); self.fan_plot.addLegend(offset=(60, 10))

        self.temp_plot = pg.PlotWidget(); self.temp_plot.setBackground('w')
        self.temp_plot.setLabel('left', 'Temp (°C)'); self.temp_plot.setLabel('bottom', 'Time (s)')
        self.temp_plot.showGrid(x=True, y=True, alpha=0.3)
        self.temp_legend = pg.LegendItem(offset=(60, 10), labelTextSize='8pt', colCount=2)
        self.temp_legend.setParentItem(self.temp_plot.plotItem)
        # NOTE: temp_plot x-axis is controlled manually to match heater time range

        # Curves
        self.p_curves = {}; self.p_tgt_curves = {}; self.v_curves = {}; self.i_curves = {}
        for i in range(1, 13):
            c = PLOT_COLORS[i-1]
            self.p_curves[i] = self.p_plot.plot(pen=pg.mkPen(c, width=2))
            self.p_tgt_curves[i] = self.p_plot.plot(pen=pg.mkPen(c, width=1.5, style=Qt.DashLine))
            self.p_legend.addItem(self.p_curves[i], f'CH{i}')
            self.p_legend.addItem(self.p_tgt_curves[i], f'CH{i} Cmd')
            self.v_curves[i] = self.v_plot.plot(pen=pg.mkPen(c, width=2), name=f'CH{i}')
            self.i_curves[i] = self.i_plot.plot(pen=pg.mkPen(c, width=2), name=f'CH{i}')

        self.fan1_actual_curve = self.fan_plot.plot(pen=pg.mkPen('#4363d8', width=2), name='Fan1')
        self.fan1_cmd_curve = self.fan_plot.plot(pen=pg.mkPen('#4363d8', width=1.5, style=Qt.DashLine), name='Fan1 Cmd')
        self.fan2_actual_curve = self.fan_plot.plot(pen=pg.mkPen('#e6194b', width=2), name='Fan2')
        self.fan2_cmd_curve = self.fan_plot.plot(pen=pg.mkPen('#e6194b', width=1.5, style=Qt.DashLine), name='Fan2 Cmd')

        # Command preview curves (full preset profile, shown during command execution)
        self.cmd_preview_p = {}
        for i in range(1, 13):
            c = PLOT_COLORS[i-1]
            curve = self.p_plot.plot(pen=pg.mkPen(c, width=1, style=Qt.DashDotLine))
            curve.hide()
            self.cmd_preview_p[i] = curve
        self.cmd_preview_fan = {}
        for fan_idx, fc in [(1, '#4363d8'), (2, '#e6194b')]:
            curve = self.fan_plot.plot(pen=pg.mkPen(fc, width=1, style=Qt.DashDotLine))
            curve.hide()
            self.cmd_preview_fan[fan_idx] = curve

        # Control-mode profile preview curves (imported control targets)
        self.ctrl_preview_p = {}
        for i in range(1, 13):
            c = PLOT_COLORS[i-1]
            curve = self.p_plot.plot(pen=pg.mkPen(c, width=1.2, style=Qt.DotLine))
            curve.hide()
            self.ctrl_preview_p[i] = curve

        self.temp_curves = {}
        for i in range(1, GL860_NUM_CHANNELS + 1):
            c = TEMP_COLORS[i-1]
            name = 'Ambient Temperature' if i == 13 else f'CH{i}'
            self.temp_curves[i] = self.temp_plot.plot(pen=pg.mkPen(c, width=2))
            self.temp_legend.addItem(self.temp_curves[i], name)
        self.ctrl_preview_temp_aim = self.temp_plot.plot(
            pen=pg.mkPen('#222222', width=2, style=Qt.DashDotLine)
        )
        self.ctrl_preview_temp_aim.hide()
        self.temp_legend.addItem(self.ctrl_preview_temp_aim, 'T_aim Cmd')

        self.plot_container.addWidget(self.p_plot, stretch=4)
        self.v_plot.hide(); self.plot_container.addWidget(self.v_plot)
        self.i_plot.hide(); self.plot_container.addWidget(self.i_plot)
        self.plot_container.addWidget(self.temp_plot, stretch=3)
        self.fan_plot.hide(); self.plot_container.addWidget(self.fan_plot, stretch=3)
        # V and I hidden by default in All mode
        self.v_cb.hide(); self.i_cb.hide()

        self.view_btn_group.buttonClicked.connect(self._on_view_changed)
        self.p_cb.toggled.connect(self._rebuild_plot_layout)
        self.v_cb.toggled.connect(self._rebuild_plot_layout)
        self.i_cb.toggled.connect(self._rebuild_plot_layout)
        self.t_cb.toggled.connect(self._on_temp_cb_toggled)
        self.fan_cb.toggled.connect(self._rebuild_plot_layout)

        self.top_splitter.addWidget(plot_widget)
        self.top_splitter.setSizes([300, 900])
        self.top_splitter.setStretchFactor(0, 3)
        self.top_splitter.setStretchFactor(1, 7)

        # ---- Command Log Section ----
        log_section = QWidget()
        log_section_vlayout = QVBoxLayout(log_section)
        log_section_vlayout.setContentsMargins(4,4,4,4)
        log_title = QLabel("Command Log")
        log_title.setStyleSheet(SECTION_TITLE_STYLE)
        log_section_vlayout.addWidget(log_title)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "QTextEdit{font-family:Consolas,'Courier New',monospace;font-size:20px;background:#f8f8f8;border:1px solid #ddd;border-radius:4px;padding:4px;}")
        log_section_vlayout.addWidget(self.log_text)

        # ---- Assemble all sections with resizable splitters ----
        self.right_splitter = QSplitter(Qt.Vertical)
        self.right_splitter.setChildrenCollapsible(False)
        self.right_splitter.addWidget(self.top_splitter)
        self.right_splitter.addWidget(log_section)
        self.right_splitter.setSizes([750, 200])
        self.right_splitter.setStretchFactor(0, 4)
        self.right_splitter.setStretchFactor(1, 1)

        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(self.right_splitter)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setSizes([100, 1400])
        self.main_splitter.setStretchFactor(0, 0) 
        self.main_splitter.setStretchFactor(1, 1)

    @staticmethod
    def _style_control_grid(layout):
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(7)
        for col in range(4):
            layout.setColumnStretch(col, 1 if col % 2 else 0)

    @staticmethod
    def _field_label(text):
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size:14px;color:#555;")
        return lbl

    @staticmethod
    def _view_btn_style():
        return (
            "QPushButton{font-size:14px;font-weight:bold;border-radius:3px;border:1px solid #999;background:#e8e8e8;color:#333;padding:2px 4px;}"
            "QPushButton:checked{background:#555;color:white;border:2px solid #333;}"
            "QPushButton:hover{background:#d0d0d0;}")

    def _setup_timer(self):
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._update_plots)
        self.plot_timer.start(PLOT_REFRESH_INTERVAL_MS)

        self.output_command_timer = QTimer()
        self.output_command_timer.setSingleShot(True)
        self.output_command_timer.setInterval(30)
        self.output_command_timer.timeout.connect(self._flush_output_command)

    # ----------------------------------------------------------
    # Control config import
    # ----------------------------------------------------------
    def _import_control_config(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Control Config CSV", "", "CSV Files (*.csv)")
        if not filepath:
            return
        try:
            profile = self._parse_control_config_csv(filepath)
            self._ctrl_profile_time = profile["time"]
            self._ctrl_profile_power = profile["power"]
            self._ctrl_profile_temp_aim = profile["temp_aim"]
            self._ctrl_profile_data = [
                (
                    float(self._ctrl_profile_time[i]),
                    {ch: float(self._ctrl_profile_power[ch][i]) for ch in range(1, 13)},
                    float(self._ctrl_profile_temp_aim[i]),
                )
                for i in range(len(self._ctrl_profile_time))
            ]
            self._ctrl_profile_data.sort(key=lambda row: row[0])

            valid_t_aim = self._ctrl_profile_temp_aim[np.isfinite(self._ctrl_profile_temp_aim)]
            if len(valid_t_aim) == 0:
                raise ValueError("missing valid Temp_Aim values")
            t_aim = float(valid_t_aim[0])
            if not (self.pid_t_aim_spin.minimum() <= t_aim <= self.pid_t_aim_spin.maximum()):
                raise ValueError(f"T_aim out of range: {t_aim}")
            self.pid_t_aim_spin.setValue(t_aim)

            self.ctrl_config_label.setText(os.path.basename(filepath))
            self.ctrl_config_label.setStyleSheet("font-size:14px;color:#333;")
            self.ctrl_start_btn.setEnabled(True)
            self._show_control_profile_preview()

            duration = self._ctrl_profile_time[-1] if len(self._ctrl_profile_time) > 0 else 0.0
            t_min = float(np.nanmin(valid_t_aim))
            t_max = float(np.nanmax(valid_t_aim))
            self._on_log(
                f"Control profile imported: {len(self._ctrl_profile_time)} steps, "
                f"{duration:.1f}s, Temp_Aim={t_min:.1f}-{t_max:.1f}°C, pending T_aim={t_aim:.1f}°C")
        except Exception as e:
            self.ctrl_config_label.setText("Import failed")
            self.ctrl_config_label.setStyleSheet("font-size:14px;color:red;")
            self._on_log(f"Control config import error: {e}")

    def _parse_control_config_csv(self, filepath):
        """Parse control profile CSV by header.

        Expected columns: Time, CH1-CH12, Temp_Aim.
        """
        temp_aliases = {"T_AIM", "TEMP_AIM", "TARGET_TEMP", "TARGET_T", "T_TARGET"}
        with open(filepath, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            header = [h.strip().upper() for h in next(reader)]

            col_map = {}
            for i, col in enumerate(header):
                if col == "TIME":
                    col_map["time"] = i
                elif col in temp_aliases:
                    col_map["temp_aim"] = i
                elif col.startswith("CH"):
                    try:
                        ch = int(col[2:])
                        if 1 <= ch <= 12:
                            col_map[f"ch{ch}"] = i
                    except ValueError:
                        pass

            if "time" not in col_map:
                raise ValueError("missing Time column")
            if "temp_aim" not in col_map:
                raise ValueError("missing Temp_Aim column")

            times = []
            powers = {ch: [] for ch in range(1, 13)}
            temp_aim = []
            for row in reader:
                if not row or not any(cell.strip() for cell in row):
                    continue
                times.append(float(row[col_map["time"]]))
                for ch in range(1, 13):
                    key = f"ch{ch}"
                    if key in col_map and col_map[key] < len(row) and row[col_map[key]].strip():
                        powers[ch].append(float(row[col_map[key]]))
                    else:
                        powers[ch].append(0.0)
                temp_aim.append(float(row[col_map["temp_aim"]]))

        if not times:
            raise ValueError("no command rows found")
        return {
            "time": np.array(times, dtype=float),
            "power": {ch: np.array(powers[ch], dtype=float) for ch in range(1, 13)},
            "temp_aim": np.array(temp_aim, dtype=float),
        }

    def _show_control_profile_preview(self):
        """Show imported control profile targets on the plot area."""
        if len(self._ctrl_profile_time) == 0:
            return
        for ch in range(1, 13):
            self.ctrl_preview_p[ch].setData(self._ctrl_profile_time, self._ctrl_profile_power[ch])
            self.ctrl_preview_p[ch].show()
        self.ctrl_preview_temp_aim.setData(self._ctrl_profile_time, self._ctrl_profile_temp_aim)
        self.ctrl_preview_temp_aim.show()

        if not self.lock_cb.isChecked():
            duration = self._ctrl_profile_time[-1] if len(self._ctrl_profile_time) > 0 else 1.0
            for pw in [self.p_plot, self.v_plot, self.i_plot, self.fan_plot, self.temp_plot]:
                pw.disableAutoRange(axis='x')
                pw.setXRange(0, duration, padding=0.02)
        self._rebuild_plot_layout()

    def _start_control_profile(self):
        if not self.serial_worker:
            self._on_log("Error: STM32 not connected"); return
        if not self._ctrl_profile_data:
            self._on_log("Error: No control profile loaded"); return
        if self.ctrl_profile_worker and self.ctrl_profile_worker.isRunning():
            self._on_log("Control profile is already running"); return

        self._recording = False
        self._clear_plot_data()
        self._unlock_view()
        self.start_time = time.time()
        self._recording = True
        self._on_log("Control profile start — timestamp zero point set (t=0), recording started")

        self.x_min_spin.setValue(0)
        self.x_max_spin.setValue(0)
        self._show_control_profile_preview()

        first_t_aim = float(self._ctrl_profile_data[0][2])
        self.pid_t_aim_spin.setValue(first_t_aim)
        if self._control_mode == "pid":
            self._pid_active_t_aim = first_t_aim
            self._pid_controller.set_t_aim(first_t_aim)
            self.pid_active_t_aim_label.setText(f"Current T_aim: {self._pid_active_t_aim:.1f} °C")
            self.ctrl_status_label.setText(f"Mode: PID (T_aim={first_t_aim}°C)")
        elif self._control_mode == "mpc" and self._mpc_controller:
            self._pid_active_t_aim = first_t_aim
            self._mpc_controller.set_t_aim(first_t_aim)
            self.pid_active_t_aim_label.setText(f"Current T_aim: {self._pid_active_t_aim:.1f} °C")
            self.ctrl_status_label.setText(f"Mode: MPC (T_aim={first_t_aim}°C)")

        self._ctrl_profile_running = True
        self.ctrl_profile_worker = ControlProfileWorker(self._ctrl_profile_data)
        self.ctrl_profile_worker.progress.connect(self._on_log)
        self.ctrl_profile_worker.control_command.connect(self._on_ctrl_profile_command)
        self.ctrl_profile_worker.profile_finished.connect(self._on_ctrl_profile_finished)
        self.ctrl_profile_worker.start()
        self.ctrl_start_btn.setEnabled(False)
        self.ctrl_stop_btn.setEnabled(True)
        self.ctrl_import_btn.setEnabled(False)

    def _stop_control_profile(self):
        if self.ctrl_profile_worker and self.ctrl_profile_worker.isRunning():
            self.ctrl_profile_worker.stop()
            self.ctrl_profile_worker.wait(3000)
        self._ctrl_profile_running = False
        self.ctrl_start_btn.setEnabled(bool(self._ctrl_profile_data))
        self.ctrl_stop_btn.setEnabled(False)
        self.ctrl_import_btn.setEnabled(True)
        self._zero_heat()
        self._on_log("Control profile stopped")

    def _on_ctrl_profile_command(self, targets, temp_aim):
        if not self._ctrl_profile_running or not self.serial_worker:
            return
        for ch, tgt in targets.items():
            self.current_targets[ch] = float(tgt)
        self.pid_t_aim_spin.setValue(float(temp_aim))
        if self._control_mode == "pid":
            self._pid_active_t_aim = float(temp_aim)
            self._pid_controller.set_t_aim(float(temp_aim))
            self.pid_active_t_aim_label.setText(f"Current T_aim: {self._pid_active_t_aim:.1f} °C")
            self.ctrl_status_label.setText(f"Mode: PID (T_aim={self._pid_active_t_aim:.1f}°C)")
        elif self._control_mode == "mpc" and self._mpc_controller:
            self._pid_active_t_aim = float(temp_aim)
            self._mpc_controller.set_t_aim(float(temp_aim))
            self.pid_active_t_aim_label.setText(f"Current T_aim: {self._pid_active_t_aim:.1f} °C")
            self.ctrl_status_label.setText(f"Mode: MPC (T_aim={self._pid_active_t_aim:.1f}°C)")

        self._queue_output_command("control profile")

    def _on_ctrl_profile_finished(self):
        was_running = self._ctrl_profile_running
        self._ctrl_profile_running = False
        self.ctrl_start_btn.setEnabled(bool(self._ctrl_profile_data))
        self.ctrl_stop_btn.setEnabled(False)
        self.ctrl_import_btn.setEnabled(True)
        if was_running:
            self._zero_heat()
            self._recording = False
            self._on_log("Control profile complete; heater zeroed and recording stopped")
            QApplication.beep()
            QMessageBox.information(
                self,
                "Control Profile Complete",
                "Control profile execution has finished.\nAll heaters have been zeroed."
            )

    # ----------------------------------------------------------
    # Control mode switching
    # ----------------------------------------------------------
    def _on_control_mode_changed(self, btn_id):
        """切换控制模式: 0=不控制, 1=PID, 2=MPC"""
        # 先停止当前控制
        self._pid_controller.disable()
        if self._mpc_controller:
            self._mpc_controller.disable()
        self._mpc_last_compute_wall = None
        self._reset_mpc_cached_plan()
        self._control_mode = "none"

        if btn_id == 0:
            self._control_mode = "none"
            self.ctrl_status_label.setText("Mode: 不控制")
            self.ctrl_status_label.setStyleSheet("font-size:14px;font-weight:bold;color:#666;")
            self._on_log("Control mode: OFF")

        elif btn_id == 1:
            # PID 控制
            t_aim = self.pid_t_aim_spin.value()
            self._pid_active_t_aim = t_aim
            self._pid_controller.set_t_aim(t_aim)
            self._pid_controller.enable()
            self._control_mode = "pid"
            self.pid_active_t_aim_label.setText(f"Current T_aim: {self._pid_active_t_aim:.1f} °C")
            self.ctrl_status_label.setText(f"Mode: PID (T_aim={t_aim}°C)")
            self.ctrl_status_label.setStyleSheet("font-size:14px;font-weight:bold;color:#2a7fff;")
            self._on_log(f"Control mode: PID enabled (T_aim={t_aim}°C, triggered by temperature samples)")

        elif btn_id == 2:
            # MPC 控制
            try:
                mpc = self._get_mpc_controller()
                t_aim = self.pid_t_aim_spin.value()
                self._pid_active_t_aim = t_aim
                mpc.set_t_aim(t_aim)
                mpc.enable()
                self._control_mode = "mpc"
                self.pid_active_t_aim_label.setText(f"Current T_aim: {self._pid_active_t_aim:.1f} °C")
                self.ctrl_status_label.setText(f"Mode: MPC (T_aim={t_aim}°C)")
                self.ctrl_status_label.setStyleSheet("font-size:14px;font-weight:bold;color:#00a06a;")
                self._on_log(
                    f"Control mode: MPC enabled (T_aim={t_aim}°C, "
                    f"interval={self._mpc_min_interval_s:.1f}s, triggered by temperature samples)")
            except Exception as e:
                self._control_mode = "none"
                self.mode_none_btn.setChecked(True)
                self.ctrl_status_label.setText("Mode: MPC unavailable")
                self.ctrl_status_label.setStyleSheet("font-size:14px;font-weight:bold;color:#dd2222;")
                self._on_log(f"Control mode: MPC unavailable: {e}")

    def _get_mpc_controller(self):
        """Create/load the NODE-MPC controller on first use."""
        if self._mpc_controller:
            return self._mpc_controller
        self._mpc_controller = MPCFanController()
        return self._mpc_controller

    def _reset_mpc_cached_plan(self):
        self._mpc_cached_plan_rpm = np.empty((0, 2), dtype=float)
        self._mpc_cached_plan_index = 0
        self._mpc_cached_plan_last_wall = None

    def _cache_mpc_plan(self, status, now):
        plan = np.asarray(status.get("fan_plan_rpm", []), dtype=float)
        self._mpc_cached_plan_rpm = plan.reshape(-1, 2)
        self._mpc_cached_plan_index = 1
        self._mpc_cached_plan_last_wall = now

    def _apply_cached_mpc_plan_step(self, now):
        if self._mpc_cached_plan_index >= len(self._mpc_cached_plan_rpm):
            return
        if self._mpc_cached_plan_last_wall is not None:
            if now - self._mpc_cached_plan_last_wall < self._mpc_min_interval_s:
                return
        rpm = self._mpc_cached_plan_rpm[self._mpc_cached_plan_index]
        rpm1 = int(round(rpm[0]))
        rpm2 = int(round(rpm[1]))
        self.fan1_input.setText(str(rpm1))
        self.fan2_input.setText(str(rpm2))
        self.current_fan_targets[1] = float(rpm1)
        self.current_fan_targets[2] = float(rpm2)
        self._mpc_cached_plan_index += 1
        self._mpc_cached_plan_last_wall = now
        self._queue_output_command("mpc cached fan")

    def _run_mpc_control(self):
        """启动一次后台 NODE-MPC 控制计算。"""
        now = time.time()
        if self.mpc_worker and self.mpc_worker.isRunning():
            self._apply_cached_mpc_plan_step(now)
            return

        if self._mpc_last_compute_wall is not None:
            elapsed = now - self._mpc_last_compute_wall
            if elapsed < self._mpc_min_interval_s:
                return
            dt_s = self._mpc_min_interval_s
        else:
            dt_s = self._mpc_min_interval_s
        self._mpc_last_compute_wall = now

        try:
            mpc = self._get_mpc_controller()
            source_temps = {}
            max_temp = -math.inf
            for ch in range(1, 13):
                t = self.latest_temps.get(ch, math.nan)
                source_temps[ch] = t
                if not (isinstance(t, float) and math.isnan(t)):
                    max_temp = max(max_temp, t)

            if max_temp == -math.inf:
                return

            ambient_temp = self.latest_temps.get(13, math.nan)
            if isinstance(ambient_temp, float) and math.isnan(ambient_temp):
                ambient_temp = 25.0

            heater_power = {}
            for ch in range(1, 13):
                if self.channel_power[ch]:
                    heater_power[ch] = self.channel_power[ch][-1]
                else:
                    heater_power[ch] = self.current_targets[ch]
            actual_fan_rpm = {fan: self.fan_rpm_values[fan] for fan in [1, 2]}
            t_aim = self._pid_active_t_aim
            self.mpc_worker = MPCComputeWorker(
                mpc,
                source_temps,
                heater_power,
                actual_fan_rpm,
                ambient_temp,
                t_aim,
                dt_s,
                max_temp,
            )
            self.mpc_worker.result_ready.connect(self._on_mpc_result)
            self.mpc_worker.error.connect(self._on_mpc_error)
            self.mpc_worker.finished.connect(self._on_mpc_worker_finished)
            self._plot_refresh_suspended = True
            self.plot_timer.stop()
            self.mpc_worker.start()
        except Exception as e:
            self._plot_refresh_suspended = False
            self.plot_timer.start(PLOT_REFRESH_INTERVAL_MS)
            self._on_mpc_error(str(e))

    def _on_mpc_result(self, rpm1, rpm2, status, t_aim, max_temp, ambient_temp):
        if self._control_mode != "mpc":
            return

        now = time.time()
        self.fan1_input.setText(str(rpm1))
        self.fan2_input.setText(str(rpm2))
        self.current_fan_targets[1] = float(rpm1)
        self.current_fan_targets[2] = float(rpm2)
        self._queue_output_command("mpc fan")
        self._cache_mpc_plan(status, now)

        loss_drpm = status.get("loss_drpm", {})
        total_drpm = status.get("total_drpm", (math.nan, math.nan))
        self.mpc_time.append(now - self.start_time)
        self.mpc_optimize_elapsed.append(float(status.get("optimize_elapsed_s", math.nan)))
        for name in MPC_LOSS_NAMES:
            values = loss_drpm.get(name, (math.nan, math.nan))
            self.mpc_loss_drpm[name][1].append(float(values[0]))
            self.mpc_loss_drpm[name][2].append(float(values[1]))
        self.mpc_total_drpm[1].append(float(total_drpm[0]))
        self.mpc_total_drpm[2].append(float(total_drpm[1]))
        self.mpc_t_aim.append(float(t_aim))
        self.mpc_t_max.append(float(max_temp))
        self.mpc_ambient_temp.append(float(ambient_temp))
        actual_fan_rpm = status.get('actual_fan_rpm', {})
        self.mpc_fan1_actual.append(float(actual_fan_rpm.get(1, math.nan)))
        self.mpc_fan2_actual.append(float(actual_fan_rpm.get(2, math.nan)))
        self.mpc_fan1_target.append(int(rpm1))
        self.mpc_fan2_target.append(int(rpm2))
        zp_status = status.get('zero_point_correction', {})
        self.mpc_zp_actual_enabled.append(int(bool(zp_status.get('actual_correction_enabled', False))))
        self.mpc_zp_actual_step.append(int(zp_status.get('actual_step_before', -1)))
        self.mpc_zp_prediction_enabled.append(int(bool(zp_status.get('prediction_correction_enabled', False))))
        self.mpc_zp_state_isolated.append(int(bool(zp_status.get('actual_state_preserved_during_prediction', False))))
        self.mpc_zp_correction_max_abs.append(float(zp_status.get('correction_max_abs', math.nan)))
        zp_output = zp_status.get('zp_output_norm_rate', [])
        zp_trend = zp_status.get('zp_trend_norm_rate', [])
        core_dtheta_dt = zp_status.get('core_dtheta_dt_norm_rate', [])
        for ch in range(1, 13):
            value = zp_output[ch - 1] if ch <= len(zp_output) else math.nan
            self.mpc_zp_output_norm_rate[ch].append(float(value))
            value = zp_trend[ch - 1] if ch <= len(zp_trend) else math.nan
            self.mpc_zp_trend_norm_rate[ch].append(float(value))
            value = core_dtheta_dt[ch - 1] if ch <= len(core_dtheta_dt) else math.nan
            self.mpc_core_dtheta_dt_norm_rate[ch].append(float(value))
        augment_status = status.get('augment_state', {})
        augment_h = augment_status.get('h', [])
        for index in range(1, 17):
            value = augment_h[index - 1] if index <= len(augment_h) else math.nan
            self.mpc_augment_h[index].append(float(value))

        if not hasattr(self, '_mpc_log_time'):
            self._mpc_log_time = 0
        now = time.time()
        if now - self._mpc_log_time >= 5.0:
            self._mpc_log_time = now
            temp_str = f"{max_temp:.1f}" if not math.isnan(max_temp) else "N/A"
            opt_state = "ok" if status.get("success") else "warn"
            loss_drpm_text = ", ".join(
                f"{name}=({loss_drpm.get(name, (math.nan, math.nan))[0]:+.0f}, "
                f"{loss_drpm.get(name, (math.nan, math.nan))[1]:+.0f})"
                for name in MPC_LOSS_NAMES
            )
            self._on_log(
                f"[MPC] T_aim={t_aim:.1f}°C, T_max={temp_str}°C, "
                f"Amb={ambient_temp:.1f}°C → Fan1={rpm1}, Fan2={rpm2} RPM "
                f"({opt_state}, loss_dRPM: {loss_drpm_text})")

    def _on_mpc_error(self, message):
        if self._control_mode == "mpc":
            self._control_mode = "none"
            if self._mpc_controller:
                self._mpc_controller.disable()
            self._reset_mpc_cached_plan()
            self.mode_none_btn.setChecked(True)
            self.ctrl_status_label.setText("Mode: MPC error")
            self.ctrl_status_label.setStyleSheet("font-size:14px;font-weight:bold;color:#dd2222;")
            self._on_log(f"[MPC] error, control disabled: {message}")

    def _on_mpc_worker_finished(self):
        if self.sender() is self.mpc_worker:
            self.mpc_worker = None
            self._plot_refresh_suspended = False
            self._check_auto_flush()
            self._update_plots()
            self.plot_timer.start(PLOT_REFRESH_INTERVAL_MS)

    def _run_pid_control(self):
        """执行一次 PID 控制计算并发送风扇命令。"""
        # 获取当前 12 个热源温度，并保留最高温用于日志显示
        source_temps = {}
        max_temp = -math.inf
        for ch in range(1, 13):
            t = self.latest_temps.get(ch, math.nan)
            source_temps[ch] = t
            if not (isinstance(t, float) and math.isnan(t)):
                if t > max_temp:
                    max_temp = t

        if max_temp == -math.inf:
            max_temp = math.nan  # 无有效温度数据

        t_aim = self._pid_active_t_aim

        # PID 计算
        rpm1, rpm2 = self._pid_controller.compute(source_temps)

        # 更新 UI 输入框（仅显示，不影响手动操作）
        self.fan1_input.setText(str(rpm1))
        self.fan2_input.setText(str(rpm2))

        # 更新风扇目标；实际串口输出由统一 MHF 队列合并发送。
        self.fan_rpm_values[1] = rpm1
        self.fan_rpm_values[2] = rpm2
        self.current_fan_targets[1] = float(rpm1)
        self.current_fan_targets[2] = float(rpm2)
        self._queue_output_command("pid fan")

        # 日志（每 5 秒打印一次，避免刷屏）
        if not hasattr(self, '_pid_log_time'):
            self._pid_log_time = 0
        now = time.time()
        if now - self._pid_log_time >= 5.0:
            self._pid_log_time = now
            temp_str = f"{max_temp:.1f}" if not math.isnan(max_temp) else "N/A"
            self._on_log(f"[PID] T_aim={t_aim:.1f}°C, T_max={temp_str}°C → Fan1={rpm1}, Fan2={rpm2} RPM")

    # ----------------------------------------------------------
    # View switching
    # ----------------------------------------------------------
    def _on_view_changed(self, btn):
        btn_id = self.view_btn_group.id(btn)
        if btn_id == 0: self.current_view = self.VIEW_ALL
        elif 1 <= btn_id <= 12: self.current_view = f'ch{btn_id}'

        show_multi = (self.current_view == self.VIEW_ALL)
        self.multi_sel_label.setVisible(show_multi)
        for i in range(1, 13): self.ch_cbs[i].setVisible(show_multi)

        show_temp_sel = (self.current_view == self.VIEW_ALL and self.t_cb.isChecked())
        self.temp_sel_label.setVisible(show_temp_sel)
        for i in range(1, GL860_NUM_CHANNELS + 1): self.temp_cbs[i].setVisible(show_temp_sel)
        self.temp_show_all_btn.setVisible(show_temp_sel)

        is_all = (self.current_view == self.VIEW_ALL)
        self.subplot_label.setVisible(True)
        self.p_cb.setVisible(True); self.t_cb.setVisible(True)
        self.v_cb.setVisible(not is_all); self.i_cb.setVisible(not is_all)
        self.fan_cb.setVisible(is_all)

        self._rebuild_plot_layout()

    def _rebuild_plot_layout(self):
        view = self.current_view
        show_p = self.p_cb.isChecked()
        is_all = (view == self.VIEW_ALL)
        show_v = self.v_cb.isChecked() if not is_all else False
        show_i = self.i_cb.isChecked() if not is_all else False
        show_t = self.t_cb.isChecked()
        show_fan = self.fan_cb.isChecked() if is_all else False
        self.fan_plot.setVisible(show_fan)
        self.p_plot.setVisible(show_p); self.v_plot.setVisible(show_v)
        self.i_plot.setVisible(show_i); self.temp_plot.setVisible(show_t)
        if not (show_p or show_v or show_i or show_t):
            self.p_plot.setVisible(True); self.p_cb.setChecked(True)

    def _on_temp_cb_toggled(self, checked):
        if self.current_view == self.VIEW_ALL:
            self.temp_sel_label.setVisible(checked)
            for i in range(1, GL860_NUM_CHANNELS + 1): self.temp_cbs[i].setVisible(checked)
            self.temp_show_all_btn.setVisible(checked)
        self._rebuild_plot_layout()

    def _show_all_temp_channels(self):
        for i in range(1, GL860_NUM_CHANNELS + 1):
            self.temp_cbs[i].setChecked(True)
        self._rebuild_plot_layout()

    # ----------------------------------------------------------
    # Auto-flush to disk for memory optimization
    # ----------------------------------------------------------
    def _build_npz_from_memory(self):
        """Build npz data dict from current in-memory lists."""
        data = {'series_limits': np.array([SERIES_LIMITS[i] for i in range(1, 13)])}
        if len(self.time_data) > 0:
            data['time'] = np.array(self.time_data)
            for ch in range(1, 13):
                data[f'CH{ch}_power'] = np.array(self.channel_power[ch])
                data[f'CH{ch}_voltage'] = np.array(self.channel_voltage[ch])
                data[f'CH{ch}_current'] = np.array(self.channel_current[ch])
                data[f'CH{ch}_target'] = np.array(self.channel_target[ch])
            data['Fan1_rpm_actual'] = np.array(self.channel_fan_actual[1])
            data['Fan2_rpm_actual'] = np.array(self.channel_fan_actual[2])
            data['Fan1_rpm_target'] = np.array(self.channel_fan_target[1])
            data['Fan2_rpm_target'] = np.array(self.channel_fan_target[2])
        if len(self.temp_time) > 0:
            data['temp_time'] = np.array(self.temp_time)
            for i in range(1, GL860_NUM_CHANNELS + 1):
                data[f'T{i}_temp'] = np.array(self.channel_temp[i])
        return data

    def _write_mpc_csv(self, csv_base):
        mpc_csv = csv_base + '_mpc.csv'
        with open(mpc_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Time(s)', 'Optimize_Time(s)',
                *[f'{name.title()}_dRPM_Fan{fan}' for name in MPC_LOSS_NAMES for fan in (1, 2)],
                'Total_dRPM_Fan1', 'Total_dRPM_Fan2',
                'T_Aim(C)', 'T_Max(C)', 'Ambient_Temp(C)',
                'Fan1_Actual_RPM', 'Fan2_Actual_RPM',
                'Fan1_Command_RPM', 'Fan2_Command_RPM',
                'ZP_Actual_Enabled', 'ZP_Actual_Step', 'ZP_MPC_Enabled',
                'ZP_State_Isolated', 'ZP_Correction_MaxAbs_Norm',
                *[f'ZP_CH{ch}_NormRate' for ch in range(1, 13)],
                *[f'ZP_Trend_CH{ch}_NormRate' for ch in range(1, 13)],
                *[f'Core_dThetaDt_CH{ch}_NormRate' for ch in range(1, 13)],
                *[f'Augment_H{index}' for index in range(1, 17)]
            ])
            for i in range(len(self.mpc_time)):
                writer.writerow([
                    f'{self.mpc_time[i]:.4f}',
                    f'{self.mpc_optimize_elapsed[i]:.6f}',
                    *[f'{self.mpc_loss_drpm[name][fan][i]:.9g}' for name in MPC_LOSS_NAMES for fan in (1, 2)],
                    f'{self.mpc_total_drpm[1][i]:.9g}',
                    f'{self.mpc_total_drpm[2][i]:.9g}',
                    f'{self.mpc_t_aim[i]:.4f}',
                    f'{self.mpc_t_max[i]:.4f}',
                    f'{self.mpc_ambient_temp[i]:.4f}',
                    f'{self.mpc_fan1_actual[i]:.4f}',
                    f'{self.mpc_fan2_actual[i]:.4f}',
                    f'{self.mpc_fan1_target[i]}',
                    f'{self.mpc_fan2_target[i]}',
                    f'{self.mpc_zp_actual_enabled[i]}',
                    f'{self.mpc_zp_actual_step[i]}',
                    f'{self.mpc_zp_prediction_enabled[i]}',
                    f'{self.mpc_zp_state_isolated[i]}',
                    f'{self.mpc_zp_correction_max_abs[i]:.9g}',
                    *[f'{self.mpc_zp_output_norm_rate[ch][i]:.9g}' for ch in range(1, 13)],
                    *[f'{self.mpc_zp_trend_norm_rate[ch][i]:.9g}' for ch in range(1, 13)],
                    *[f'{self.mpc_core_dtheta_dt_norm_rate[ch][i]:.9g}' for ch in range(1, 13)],
                    *[f'{self.mpc_augment_h[index][i]:.9g}' for index in range(1, 17)],
                ])
        self._on_log(f"CSV MPC exported: {mpc_csv}")

    @staticmethod
    def _merge_npz(base, new_data):
        """Merge new_data arrays into base dict by concatenation."""
        for key, arr in new_data.items():
            if key == 'series_limits':
                if key not in base:
                    base[key] = arr
                continue
            if key in base:
                base[key] = np.concatenate([base[key], arr])
            else:
                base[key] = arr

    def _check_auto_flush(self):
        """Check if data exceeds threshold and flush older data to disk."""
        if self._plot_refresh_suspended:
            return
        if len(self.time_data) >= FULL_DATA_FLUSH_THRESHOLD:
            self._auto_flush_to_disk('heater')
        if len(self.temp_time) >= FULL_DATA_FLUSH_THRESHOLD:
            self._auto_flush_to_disk('temp')

    def _auto_flush_to_disk(self, data_type):
        """Flush older data to a temp file, keeping last MAX_PLOT_POINTS in memory for display."""
        if data_type == 'heater':
            n = len(self.time_data)
            if n <= MAX_PLOT_POINTS * 2: return
            flush_n = n - MAX_PLOT_POINTS
            data = {'series_limits': np.array([SERIES_LIMITS[i] for i in range(1, 13)])}
            data['time'] = np.array(self.time_data[:flush_n])
            for ch in range(1, 13):
                data[f'CH{ch}_power'] = np.array(self.channel_power[ch][:flush_n])
                data[f'CH{ch}_voltage'] = np.array(self.channel_voltage[ch][:flush_n])
                data[f'CH{ch}_current'] = np.array(self.channel_current[ch][:flush_n])
                data[f'CH{ch}_target'] = np.array(self.channel_target[ch][:flush_n])
            data['Fan1_rpm_actual'] = np.array(self.channel_fan_actual[1][:flush_n])
            data['Fan2_rpm_actual'] = np.array(self.channel_fan_actual[2][:flush_n])
            data['Fan1_rpm_target'] = np.array(self.channel_fan_target[1][:flush_n])
            data['Fan2_rpm_target'] = np.array(self.channel_fan_target[2][:flush_n])
            # Trim lists: keep last MAX_PLOT_POINTS
            self.time_data = self.time_data[flush_n:]
            for ch in range(1, 13):
                self.channel_power[ch] = self.channel_power[ch][flush_n:]
                self.channel_voltage[ch] = self.channel_voltage[ch][flush_n:]
                self.channel_current[ch] = self.channel_current[ch][flush_n:]
                self.channel_target[ch] = self.channel_target[ch][flush_n:]
            self.channel_fan_actual[1] = self.channel_fan_actual[1][flush_n:]
            self.channel_fan_actual[2] = self.channel_fan_actual[2][flush_n:]
            self.channel_fan_target[1] = self.channel_fan_target[1][flush_n:]
            self.channel_fan_target[2] = self.channel_fan_target[2][flush_n:]
        else:  # temp
            n = len(self.temp_time)
            if n <= MAX_PLOT_POINTS * 2: return
            flush_n = n - MAX_PLOT_POINTS
            data = {}
            data['temp_time'] = np.array(self.temp_time[:flush_n])
            for i in range(1, GL860_NUM_CHANNELS + 1):
                data[f'T{i}_temp'] = np.array(self.channel_temp[i][:flush_n])
            # Trim lists
            self.temp_time = self.temp_time[flush_n:]
            for i in range(1, GL860_NUM_CHANNELS + 1):
                self.channel_temp[i] = self.channel_temp[i][flush_n:]

        # Save to temp file
        ts = time.strftime("%Y%m%d_%H%M%S")
        temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"_temp_flush_{data_type}_{ts}_{len(self._flush_files)}.npz")
        np.savez(temp_path, **data)
        self._flush_files.append(temp_path)
        self._on_log(f"Auto-flushed {flush_n} {data_type} samples to disk (memory optimized)")

    def _cleanup_flush_files(self):
        """Delete all temporary flush files."""
        for fp in self._flush_files:
            try: os.remove(fp)
            except: pass
        self._flush_files = []

    # ----------------------------------------------------------
    # Serial operations
    # ----------------------------------------------------------
    def _refresh_ports(self):
        self.port_combo.clear()
        for p in serial.tools.list_ports.comports():
            self.port_combo.addItem(f"{p.device} - {p.description}")
        if not self.port_combo.count(): self.port_combo.addItem("COM4")

    def _get_port_name(self):
        text = self.port_combo.currentText()
        return text.split(" - ")[0].strip() if " - " in text else text.strip()

    def _connect_serial(self):
        port = self._get_port_name()
        try: baudrate = int(self.baudrate_edit.text())
        except: baudrate = 115200
        self.serial_worker = SerialWorker(port, baudrate)
        self.serial_worker.data_received.connect(self._on_data_received)
        self.serial_worker.connection_status.connect(self._on_serial_status)
        self.serial_worker.log_message.connect(self._on_log)
        self.serial_worker.start()
        self.connect_btn.setEnabled(False); self.disconnect_btn.setEnabled(True)
        self.start_time = time.time()

    def _disconnect_serial(self):
        self._emergency_stop()
        if self.serial_worker:
            self.serial_worker.stop(); self.serial_worker.wait(3000); self.serial_worker = None
        self.connect_btn.setEnabled(True); self.disconnect_btn.setEnabled(False)
        self.status_label.setText("Status: Disconnected"); self.status_label.setStyleSheet("color:red;font-weight:bold;")

    def _on_serial_status(self, success, msg):
        if success:
            self.status_label.setText(f"Status: {msg}"); self.status_label.setStyleSheet("color:green;font-weight:bold;")
        else:
            self.status_label.setText(f"Status: {msg}"); self.status_label.setStyleSheet("color:red;font-weight:bold;")
            self.connect_btn.setEnabled(True); self.disconnect_btn.setEnabled(False)
            if self.serial_worker: self.serial_worker = None
        self._on_log(msg)

    # ----------------------------------------------------------
    # GL860A operations
    # ----------------------------------------------------------
    def _connect_gl860(self):
        ip = self.gl_ip_edit.text().strip()
        port = self.gl_port_spin.value(); poll = self.gl_poll_spin.value()
        self.gl860_worker = GL860Worker(ip, port, poll)
        self.gl860_worker.temp_received.connect(self._on_temp_received)
        self.gl860_worker.connection_status.connect(self._on_gl860_status)
        self.gl860_worker.log_message.connect(self._on_log)
        self.gl860_worker.start()
        self.gl_connect_btn.setEnabled(False); self.gl_disconnect_btn.setEnabled(True)

    def _disconnect_gl860(self):
        if self.gl860_worker:
            self.gl860_worker.stop(); self.gl860_worker.wait(3000); self.gl860_worker = None
        self.gl_connect_btn.setEnabled(True); self.gl_disconnect_btn.setEnabled(False)
        self.gl_status_label.setText("Status: Disconnected"); self.gl_status_label.setStyleSheet("color:red;font-weight:bold;")

    def _on_gl860_status(self, success, msg):
        if success:
            self.gl_status_label.setText(f"Status: {msg}"); self.gl_status_label.setStyleSheet("color:green;font-weight:bold;")
        else:
            self.gl_status_label.setText(f"Status: {msg}"); self.gl_status_label.setStyleSheet("color:red;font-weight:bold;")
            self.gl_connect_btn.setEnabled(True); self.gl_disconnect_btn.setEnabled(False)
            if self.gl860_worker: self.gl860_worker = None
        self._on_log(msg)

    # ----------------------------------------------------------
    # Data reception
    # ----------------------------------------------------------
    def _on_data_received(self, data):
        # Always update display; only write to recording buffers when recording
        for fan_idx in [1, 2]:
            if fan_idx in data['fans']:
                self.fan_rpm_values[fan_idx] = data['fans'][fan_idx]
        self._update_data_display()
        if not self._recording:
            return
        t = time.time() - self.start_time
        self.time_data.append(t)
        for ch_idx, ch_data in data['channels'].items():
            if 1 <= ch_idx <= 12:
                self.channel_power[ch_idx].append(ch_data['power'])
                self.channel_voltage[ch_idx].append(ch_data['v_drop'])
                self.channel_current[ch_idx].append(ch_data['current'])
        for ch in range(1, 13):
            if ch not in data['channels']:
                self.channel_power[ch].append(0); self.channel_voltage[ch].append(0); self.channel_current[ch].append(0)
            self.channel_target[ch].append(self.current_targets[ch])
        # Fan data at heater sampling rate
        for fan_idx in [1, 2]:
            actual_rpm = data['fans'].get(fan_idx, self.fan_rpm_values[fan_idx])
            self.channel_fan_actual[fan_idx].append(actual_rpm)
            self.channel_fan_target[fan_idx].append(self.current_fan_targets[fan_idx])
        compress_plot = not self._plot_refresh_suspended
        if self._plot_heater_time_range[0] is None:
            self._plot_heater_time_range[0] = t
        self._plot_heater_time_range[1] = t
        for ch in range(1, 13):
            self._plot_power[ch].append(t, self.channel_power[ch][-1], compress_plot)
            self._plot_voltage[ch].append(t, self.channel_voltage[ch][-1], compress_plot)
            self._plot_current[ch].append(t, self.channel_current[ch][-1], compress_plot)
            self._plot_target[ch].append(t, self.channel_target[ch][-1], compress_plot)
        for fan_idx in [1, 2]:
            self._plot_fan_actual[fan_idx].append(t, self.channel_fan_actual[fan_idx][-1], compress_plot)
            self._plot_fan_target[fan_idx].append(t, self.channel_fan_target[fan_idx][-1], compress_plot)
        self._check_auto_flush()

    def _on_temp_received(self, temps):
        filtered_temps = self._filter_temp_ema(temps)
        self.latest_temps = filtered_temps
        self._update_data_display()
        if self._control_mode == "pid":
            self._run_pid_control()
        elif self._control_mode == "mpc":
            self._run_mpc_control()
        if not self._recording:
            return
        t = time.time() - self.start_time
        self.temp_time.append(t)
        for i in range(1, GL860_NUM_CHANNELS + 1):
            self.channel_temp[i].append(filtered_temps.get(i, math.nan))
        compress_plot = not self._plot_refresh_suspended
        if self._plot_temp_time_range[0] is None:
            self._plot_temp_time_range[0] = t
        self._plot_temp_time_range[1] = t
        for i in range(1, GL860_NUM_CHANNELS + 1):
            self._plot_temp[i].append(t, self.channel_temp[i][-1], compress_plot)
        self._check_auto_flush()

    def _filter_temp_ema(self, temps):
        alpha = TEMP_EMA_ALPHA
        filtered = {}
        for ch in range(1, GL860_NUM_CHANNELS + 1):
            raw = temps.get(ch, math.nan)
            prev = self._temp_ema_values.get(ch, math.nan)
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                value = prev
            elif prev is None or (isinstance(prev, float) and math.isnan(prev)):
                value = float(raw)
            else:
                value = alpha * float(raw) + (1.0 - alpha) * float(prev)
            self._temp_ema_values[ch] = value
            filtered[ch] = value
        return filtered

    # ----------------------------------------------------------
    # Sampling panel update
    # ----------------------------------------------------------
    def _update_data_display(self):
        # Ambient
        amb = self.latest_temps.get(13, math.nan) if self.latest_temps else math.nan
        self.ambient_field.setText(f"{amb:.1f}" if not math.isnan(amb) else "--")

        # Fans
        self.fan1_display.setText(f"{self.fan_rpm_values[1]:.0f}")
        self.fan2_display.setText(f"{self.fan_rpm_values[2]:.0f}")

        # Channels
        for ch in range(1, 13):
            # P(W)
            if len(self.channel_power[ch]) > 0:
                p = self.channel_power[ch][-1]
                ratio = abs(p) / SERIES_LIMITS[ch] if SERIES_LIMITS[ch] > 0 else 0
                self.ch_p_fields[ch].setText(f"{p:.2f}")
                if ratio > 0.8: self.ch_p_fields[ch].setStyleSheet(FIELD_STYLE + "QLineEdit{color:#cc0000;font-weight:bold;}")
                elif ratio > 0.5: self.ch_p_fields[ch].setStyleSheet(FIELD_STYLE + "QLineEdit{color:#cc6600;font-weight:bold;}")
                elif ratio > 0.1: self.ch_p_fields[ch].setStyleSheet(FIELD_STYLE + "QLineEdit{color:#006600;}")
                else: self.ch_p_fields[ch].setStyleSheet(FIELD_STYLE + "QLineEdit{color:#888;}")
            # T(°C)
            t_val = self.latest_temps.get(ch, math.nan) if self.latest_temps else math.nan
            if math.isnan(t_val):
                self.ch_t_fields[ch].setText("--")
                self.ch_t_fields[ch].setStyleSheet(FIELD_STYLE + "QLineEdit{color:#999;}")
            else:
                self.ch_t_fields[ch].setText(f"{t_val:.2f}")
                if t_val > 80: self.ch_t_fields[ch].setStyleSheet(FIELD_STYLE + "QLineEdit{color:#cc0000;font-weight:bold;}")
                elif t_val > 50: self.ch_t_fields[ch].setStyleSheet(FIELD_STYLE + "QLineEdit{color:#cc6600;font-weight:bold;}")
                elif t_val > 30: self.ch_t_fields[ch].setStyleSheet(FIELD_STYLE + "QLineEdit{color:#006600;}")
                else: self.ch_t_fields[ch].setStyleSheet(FIELD_STYLE + "QLineEdit{color:#333;}")
            # V(V)
            if len(self.channel_voltage[ch]) > 0:
                self.ch_v_fields[ch].setText(f"{self.channel_voltage[ch][-1]:.2f}")
            # I(A)
            if len(self.channel_current[ch]) > 0:
                self.ch_i_fields[ch].setText(f"{self.channel_current[ch][-1]:.2f}")

    # ----------------------------------------------------------
    # Curve update
    # ----------------------------------------------------------
    def _iter_plot_history_series(self):
        groups = (
            self._plot_power, self._plot_voltage, self._plot_current, self._plot_target,
            self._plot_fan_actual, self._plot_fan_target, self._plot_temp,
        )
        for group in groups:
            yield from group.values()

    def _compress_plot_history(self):
        for series in self._iter_plot_history_series():
            series.compress()

    @staticmethod
    def _set_history_curve(curve, series):
        x, y = series.arrays()
        curve.setData(x, y)

    def _update_plots(self):
        if self._plot_refresh_suspended:
            return
        self._compress_plot_history()
        locked = self.lock_cb.isChecked()
        view = self.current_view
        cmd_preview_active = any(self.cmd_preview_p[ch].isVisible() for ch in range(1, 13))
        ctrl_preview_active = (
            any(self.ctrl_preview_p[ch].isVisible() for ch in range(1, 13))
            or self.ctrl_preview_temp_aim.isVisible()
        )
        x_auto = self.x_max_spin.value() == 0  # 0 means Auto
        all_plots = [self.p_plot, self.v_plot, self.i_plot, self.fan_plot, self.temp_plot]

        # ---- Compute target x-range from actual data (never from auto-range) ----
        if locked:
            target_x = None  # don't touch x-axis when locked
        elif cmd_preview_active or ctrl_preview_active:
            preview_durations = []
            if cmd_preview_active and len(self._cmd_profile_time) > 0:
                preview_durations.append(self._cmd_profile_time[-1])
            if ctrl_preview_active and len(self._ctrl_profile_time) > 0:
                preview_durations.append(self._ctrl_profile_time[-1])
            target_x = (0, max(preview_durations) if preview_durations else 1)
        elif not x_auto:
            target_x = (self.x_min_spin.value(), self.x_max_spin.value())
        elif self._plot_heater_time_range[1] is not None or self._plot_temp_time_range[1] is not None:
            starts = [v for v in [self._plot_heater_time_range[0], self._plot_temp_time_range[0]] if v is not None]
            ends = [v for v in [self._plot_heater_time_range[1], self._plot_temp_time_range[1]] if v is not None]
            x_lo = min(starts)
            x_hi = max(ends)
            target_x = (x_lo, x_hi)
        else:
            target_x = None

        # Apply x-range to all plots before any setData calls
        if target_x is not None:
            for pw in all_plots:
                pw.disableAutoRange(axis='x')
                pw.setXRange(target_x[0], target_x[1], padding=0.02)

        # Fan curves
        if view == self.VIEW_ALL:
            for fan_idx, actual_curve, cmd_curve in [
                (1, self.fan1_actual_curve, self.fan1_cmd_curve),
                (2, self.fan2_actual_curve, self.fan2_cmd_curve)
            ]:
                self._set_history_curve(actual_curve, self._plot_fan_actual[fan_idx])
                self._set_history_curve(cmd_curve, self._plot_fan_target[fan_idx])
            for fan in [1, 2]:
                if self.cmd_preview_fan[fan].isVisible():
                    self.cmd_preview_fan[fan].setData(self._cmd_profile_time, self._cmd_profile_fan[fan])
                else:
                    self.cmd_preview_fan[fan].setData([], [])

        # Heater / channel curves (VIEW_ALL and single-channel views)
        if view == self.VIEW_ALL:
            vis = [ch for ch in range(1, 13) if self.ch_cbs[ch].isChecked()]
        else:
            vis = [int(view[2:])]
        for ch in range(1, 13):
            if ch in vis:
                self._set_history_curve(self.p_curves[ch], self._plot_power[ch])
                self._set_history_curve(self.p_tgt_curves[ch], self._plot_target[ch])
                self._set_history_curve(self.v_curves[ch], self._plot_voltage[ch])
                self._set_history_curve(self.i_curves[ch], self._plot_current[ch])
                if self.cmd_preview_p[ch].isVisible():
                    self.cmd_preview_p[ch].setData(self._cmd_profile_time, self._cmd_profile_power[ch])
                else:
                    self.cmd_preview_p[ch].setData([], [])
                if self.ctrl_preview_p[ch].isVisible():
                    self.ctrl_preview_p[ch].setData(self._ctrl_profile_time, self._ctrl_profile_power[ch])
                else:
                    self.ctrl_preview_p[ch].setData([], [])
            else:
                self.p_curves[ch].setData([], []); self.p_tgt_curves[ch].setData([], [])
                self.v_curves[ch].setData([], []); self.i_curves[ch].setData([], [])
                self.cmd_preview_p[ch].setData([], [])
                self.ctrl_preview_p[ch].setData([], [])
        # Temperature curves
        if self.t_cb.isChecked() and self.temp_plot.isVisible():
            if view == self.VIEW_ALL:
                temp_vis = [i for i in range(1, GL860_NUM_CHANNELS + 1) if self.temp_cbs[i].isChecked()]
            else:
                temp_vis = [int(view[2:])]
            for i in range(1, GL860_NUM_CHANNELS + 1):
                if i in temp_vis:
                    self._set_history_curve(self.temp_curves[i], self._plot_temp[i])
                else:
                    self.temp_curves[i].setData([], [])
            if self.ctrl_preview_temp_aim.isVisible():
                self.ctrl_preview_temp_aim.setData(self._ctrl_profile_time, self._ctrl_profile_temp_aim)
        else:
            self.ctrl_preview_temp_aim.setData([], [])

    def _update_control_profile_preview_visibility(self):
        """Keep imported control profile visible even before live samples arrive."""
        if len(self._ctrl_profile_time) == 0:
            return
        if self.current_view == self.VIEW_ALL:
            vis = [ch for ch in range(1, 13) if self.ch_cbs[ch].isChecked()]
        else:
            vis = [int(self.current_view[2:])]
        for ch in range(1, 13):
            if ch in vis and self.ctrl_preview_p[ch].isVisible():
                self.ctrl_preview_p[ch].setData(self._ctrl_profile_time, self._ctrl_profile_power[ch])
            else:
                self.ctrl_preview_p[ch].setData([], [])
        if self.t_cb.isChecked() and self.temp_plot.isVisible() and self.ctrl_preview_temp_aim.isVisible():
            self.ctrl_preview_temp_aim.setData(self._ctrl_profile_time, self._ctrl_profile_temp_aim)
        else:
            self.ctrl_preview_temp_aim.setData([], [])


    # ----------------------------------------------------------
    # X-axis range control
    # ----------------------------------------------------------
    def _apply_x_range(self):
        x_min = self.x_min_spin.value()
        x_max = self.x_max_spin.value()
        if x_max <= x_min:
            self._on_log("X Range error: Max must be > Min"); return
        for pw in [self.p_plot, self.v_plot, self.i_plot, self.fan_plot, self.temp_plot]:
            pw.disableAutoRange(axis='x')
            pw.setXRange(x_min, x_max, padding=0)
        self._on_log(f"X range set: [{x_min:.1f}, {x_max:.1f}] s")

    def _set_x_auto(self):
        self.x_max_spin.setValue(0)  # Reset to Auto — _update_plots will compute range from data
        self._on_log("X range: Auto")

    # ----------------------------------------------------------
    # Command sending
    # ----------------------------------------------------------
    def _unlock_view(self):
        """Auto-unlock view when any command is sent."""
        if self.lock_cb.isChecked():
            self.lock_cb.setChecked(False)

    def _queue_output_command(self, reason=""):
        """Coalesce heater and fan target updates into one MHF command."""
        if not self.serial_worker:
            self._on_log("Error: STM32 not connected"); return False
        self.output_command_timer.start()
        return True

    def _flush_output_command(self):
        if not self.serial_worker:
            return
        p = [self.current_targets[ch] for ch in range(1, 13)]
        f1 = self.current_fan_targets.get(1, 0.0)
        f2 = self.current_fan_targets.get(2, 0.0)
        cmd = (f"MHF:{p[0]:.3f},{p[1]:.3f},{p[2]:.3f},{p[3]:.3f},"
               f"{p[4]:.3f},{p[5]:.3f},{p[6]:.3f},{p[7]:.3f},"
               f"{p[8]:.3f},{p[9]:.3f},{p[10]:.3f},{p[11]:.3f},"
               f"{f1:.3f},{f2:.3f}\n")
        if not self.serial_worker.send_command(cmd):
            self._on_log("Output command failed: STM32 serial unavailable")

    def _send_single(self):
        if not self.serial_worker: self._on_log("Error: STM32 not connected"); return
        self._unlock_view()
        ch = self.ch_spin.value(); mag = self.single_power_spin.value()
        v_target = SERIES_LIMITS[ch] * mag
        self.current_targets[ch] = v_target
        self._queue_output_command("single heater")
        self._on_log(f"Single: CH{ch} -> {v_target:.3f}W (coeff={mag})")

    def _send_multi(self):
        if not self.serial_worker: self._on_log("Error: STM32 not connected"); return
        self._unlock_view()
        mag = self.multi_power_spin.value()
        targets = {ch: SERIES_LIMITS[ch] * mag for ch in range(1, 13)}
        for ch in range(1, 13): self.current_targets[ch] = targets[ch]
        self._queue_output_command("multi heater")
        self._on_log(f"Multi heater: coeff={mag}")

    def _send_fan_multi(self):
        if not self.serial_worker: self._on_log("Error: STM32 not connected"); return
        self._unlock_view()
        try:
            rpm1 = int(self.fan1_input.text()); rpm2 = int(self.fan2_input.text())
        except ValueError:
            self._on_log("Error: Fan RPM must be integer"); return
        # Valid range: {0} ∪ [1000, 6300]
        for idx, v in [(1, rpm1), (2, rpm2)]:
            if v < 0: v = 0
            elif v < 1000: v = 0
            elif v > 6300: v = 6300
            if idx == 1: rpm1 = v
            else: rpm2 = v
        self.fan_rpm_values[1] = rpm1; self.fan_rpm_values[2] = rpm2
        self.current_fan_targets[1] = float(rpm1); self.current_fan_targets[2] = float(rpm2)
        self._queue_output_command("fan")
        self._on_log(f"Fan cmd: Fan1={rpm1} RPM, Fan2={rpm2} RPM")
        self._update_data_display()

    # ----------------------------------------------------------
    # Test sequence
    # ----------------------------------------------------------
    def _run_step_test(self):
        if not self.serial_worker: self._on_log("Error: STM32 not connected"); return
        self._unlock_view()
        coeff = self.step_coeff_spin.value()
        self.step_btn.setEnabled(False); self.cosine_btn.setEnabled(False)
        self.test_worker = TestWorker('step', self.serial_worker, coeff)
        self.test_worker.progress.connect(self._on_log)
        self.test_worker.command_sent.connect(self._on_test_command)
        self.test_worker.test_finished.connect(self._on_test_finished)
        self.test_worker.start()

    def _run_cosine_test(self):
        if not self.serial_worker: self._on_log("Error: STM32 not connected"); return
        self._unlock_view()
        coeff = self.cosine_coeff_spin.value()
        self.step_btn.setEnabled(False); self.cosine_btn.setEnabled(False)
        self.test_worker = TestWorker('cosine', self.serial_worker, coeff)
        self.test_worker.progress.connect(self._on_log)
        self.test_worker.command_sent.connect(self._on_test_command)
        self.test_worker.test_finished.connect(self._on_test_finished)
        self.test_worker.start()

    def _on_test_command(self, targets):
        for ch, tgt in targets.items(): self.current_targets[ch] = tgt

    def _on_test_finished(self):
        self.step_btn.setEnabled(True); self.cosine_btn.setEnabled(True)
        self._on_log("Test sequence completed")

    # ----------------------------------------------------------
    # Command import & execution
    # ----------------------------------------------------------
    def _browse_cmd_save_dir(self):
        """Browse for auto-save directory for command results."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Auto-Save Directory", self._cmd_save_dir or os.path.dirname(os.path.abspath(__file__)))
        if directory:
            self._cmd_save_dir = directory
            self.cmd_save_dir_edit.setText(directory)
            self._on_log(f"Command save directory set: {directory}")

    def _clear_cmd_save_dir(self):
        """Clear the auto-save directory (revert to prompt-on-finish behaviour)."""
        self._cmd_save_dir = ""
        self.cmd_save_dir_edit.clear()
        self._on_log("Command save directory cleared — will prompt on finish")

    def _import_command(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Command CSV", "", "CSV Files (*.csv)")
        if not filepath: return
        try:
            self._cmd_data = self._parse_command_csv(filepath)
            self.cmd_file_label.setText(os.path.basename(filepath))
            self.cmd_file_label.setStyleSheet("font-size:14px;color:#333;")
            self.cmd_confirm_btn.setEnabled(True)
            duration = self._cmd_data[-1][0] if self._cmd_data else 0
            self._on_log(f"Command imported: {len(self._cmd_data)} steps, {duration:.1f}s duration")
            # Pre-compute profile arrays for preview curves
            self._cmd_profile_time = np.array([row[0] for row in self._cmd_data])
            self._cmd_profile_power = {ch: np.array([row[1].get(ch, 0) for row in self._cmd_data]) for ch in range(1, 13)}
            self._cmd_profile_fan = {fan: np.array([row[2].get(fan, 0) for row in self._cmd_data]) for fan in [1, 2]}
        except Exception as e:
            self._on_log(f"Import error: {e}")
            self.cmd_file_label.setText("Import failed")
            self.cmd_file_label.setStyleSheet("font-size:14px;color:red;")

    def _parse_command_csv(self, filepath):
        """Parse command CSV → [(time, {ch: W}, {fan: RPM}), ...]."""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = [h.strip().upper() for h in next(reader)]
            col_map = {}
            for i, col in enumerate(header):
                if col == 'TIME': col_map['time'] = i
                elif col.startswith('CH'):
                    try: col_map[f'ch{int(col[2:])}'] = i
                    except ValueError: pass
                elif col.startswith('FAN'):
                    try: col_map[f'fan{int(col[3:])}'] = i
                    except ValueError: pass
            for row in reader:
                if not row or not row[0].strip(): continue
                t = float(row[col_map['time']])
                heater = {}
                for ch in range(1, 13):
                    key = f'ch{ch}'
                    heater[ch] = float(row[col_map[key]]) if key in col_map else 0.0
                fans = {}
                for fan in [1, 2]:
                    key = f'fan{fan}'
                    fans[fan] = float(row[col_map[key]]) if key in col_map else 0.0
                data.append((t, heater, fans))
        return data

    def _start_command(self):
        if not self.serial_worker:
            self._on_log("Error: STM32 not connected"); return
        if not self._cmd_data:
            self._on_log("Error: No command data loaded"); return
        # Set timestamp zero point & clear old data
        self._recording = False  # Pause during clear to avoid partial data
        self._clear_plot_data()
        self._unlock_view()
        self.start_time = time.time()
        self._recording = True  # Start recording from t=0
        self._on_log(f"Command start — timestamp zero point set (t=0), recording started")
        # Auto-set X range to command duration
        cmd_dur = self._cmd_profile_time[-1] if len(self._cmd_profile_time) > 0 else 1
        self.x_min_spin.setValue(0)
        self.x_max_spin.setValue(0)  # Auto mode – _update_plots will use cmd_dur
        # Show command preview curves on P plot and Fan plot
        for ch in range(1, 13):
            self.cmd_preview_p[ch].setData(self._cmd_profile_time, self._cmd_profile_power[ch])
            self.cmd_preview_p[ch].show()
        for fan in [1, 2]:
            self.cmd_preview_fan[fan].setData(self._cmd_profile_time, self._cmd_profile_fan[fan])
            self.cmd_preview_fan[fan].show()
        # Start command worker
        self.cmd_worker = CommandWorker(self._cmd_data, self.serial_worker)
        self.cmd_worker.progress.connect(self._on_log)
        self.cmd_worker.heater_command.connect(self._on_cmd_heater)
        self.cmd_worker.fan_command.connect(self._on_cmd_fan)
        self.cmd_worker.command_finished.connect(self._on_cmd_finished)
        self.cmd_worker.start()
        self.cmd_confirm_btn.setEnabled(False)
        self.cmd_stop_btn.setEnabled(True)
        self.cmd_import_btn.setEnabled(False)

    def _stop_command(self):
        if self.cmd_worker and self.cmd_worker.isRunning():
            self.cmd_worker.stop(); self.cmd_worker.wait(3000)
        # Hide and clear command preview curves (must clear data to remove from auto-range)
        for ch in range(1, 13): self.cmd_preview_p[ch].setData([], []); self.cmd_preview_p[ch].hide()
        for fan in [1, 2]: self.cmd_preview_fan[fan].setData([], []); self.cmd_preview_fan[fan].hide()
        self.cmd_confirm_btn.setEnabled(True)
        self.cmd_stop_btn.setEnabled(False)
        self.cmd_import_btn.setEnabled(True)
        # Auto-zero after manual stop
        self._zero_heat()
        self._zero_fan()

    def _on_cmd_heater(self, targets):
        # Only update if command worker is still running (prevent stale queued signals)
        if self.cmd_worker and self.cmd_worker.isRunning():
            for ch, tgt in targets.items():
                self.current_targets[ch] = tgt

    def _on_cmd_fan(self, targets):
        if self.cmd_worker and self.cmd_worker.isRunning():
            for fan, rpm in targets.items():
                self.current_fan_targets[fan] = float(rpm)
                self.fan_rpm_values[fan] = int(rpm)

    def _on_cmd_finished(self):
        # Hide and clear command preview curves (must clear data to remove from auto-range)
        for ch in range(1, 13): self.cmd_preview_p[ch].setData([], []); self.cmd_preview_p[ch].hide()
        for fan in [1, 2]: self.cmd_preview_fan[fan].setData([], []); self.cmd_preview_fan[fan].hide()
        self.cmd_confirm_btn.setEnabled(True)
        self.cmd_stop_btn.setEnabled(False)
        self.cmd_import_btn.setEnabled(True)
        # Auto-zero all heaters and fans after command execution
        self._zero_heat()
        self._zero_fan()
        # Stop recording immediately after zeroing
        self._recording = False
        self._on_log("Recording stopped")
        # Reset x-range to actual data extent before locking, so preview range doesn't persist
        if self._plot_heater_time_range[1] is not None or self._plot_temp_time_range[1] is not None:
            starts = [v for v in [self._plot_heater_time_range[0], self._plot_temp_time_range[0]] if v is not None]
            ends = [v for v in [self._plot_heater_time_range[1], self._plot_temp_time_range[1]] if v is not None]
            x_lo = min(starts)
            x_hi = max(ends)
            for pw in [self.p_plot, self.v_plot, self.i_plot, self.fan_plot, self.temp_plot]:
                pw.disableAutoRange(axis='x')
                pw.setXRange(x_lo, x_hi, padding=0.02)
        # Auto lock view to preserve final plot state
        self.lock_cb.setChecked(True)
        # Notification + beep
        QApplication.beep()
        QMessageBox.information(self, "Command Complete",
                                "Command execution has finished.\n" +
                                "All heaters and fans have been zeroed.")
        # Export: auto-save if directory is set, otherwise prompt
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if self._cmd_save_dir and os.path.isdir(self._cmd_save_dir):
            # Auto-save directly to the specified directory
            filepath = os.path.join(self._cmd_save_dir, f"cmd_result_{timestamp}")
            try:
                self._do_export(filepath)
                self._on_log(f"Auto-saved to: {filepath}")
                QMessageBox.information(self, "Data Exported", f"Data auto-saved to:\n{filepath}")
            except Exception as e:
                QMessageBox.warning(self, "Export Failed", f"Auto-save failed: {e}")
        else:
            # No directory set — show save dialog
            base_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(base_dir, f"cmd_result_{timestamp}")
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Command Result", default_path,
                "Data Files (*.npz *.csv);;NumPy Files (*.npz);;CSV Files (*.csv)")
            if filepath:
                try:
                    self._do_export(filepath)
                    QMessageBox.information(self, "Data Exported", f"Data exported to:\n{filepath}")
                except Exception as e:
                    QMessageBox.warning(self, "Export Failed", f"Export failed: {e}")

    # ----------------------------------------------------------
    # Safety operations
    # ----------------------------------------------------------
    def _zero_heat(self):
        if self.test_worker and self.test_worker.isRunning():
            self.test_worker.stop(); self.test_worker.wait(2000)
        if self.cmd_worker and self.cmd_worker.isRunning():
            self.cmd_worker.stop(); self.cmd_worker.wait(2000)
        if self.ctrl_profile_worker and self.ctrl_profile_worker.isRunning():
            self.ctrl_profile_worker.stop(); self.ctrl_profile_worker.wait(2000)
            self._ctrl_profile_running = False
            self.ctrl_start_btn.setEnabled(bool(self._ctrl_profile_data))
            self.ctrl_stop_btn.setEnabled(False)
            self.ctrl_import_btn.setEnabled(True)
        for ch in range(1, 13): self.current_targets[ch] = 0.0
        if self.serial_worker:
            self.output_command_timer.stop()
            self._flush_output_command()
        self._on_log("Heater zeroed")

    def _zero_fan(self):
        self.fan_rpm_values[1] = 0; self.fan_rpm_values[2] = 0
        self.current_fan_targets[1] = 0.0; self.current_fan_targets[2] = 0.0
        self._reset_mpc_cached_plan()
        self.fan1_input.setText("0"); self.fan2_input.setText("0")
        if self.serial_worker:
            self.output_command_timer.stop()
            self._flush_output_command()
        self._on_log("Fan zeroed")
        self._update_data_display()

    def _emergency_stop(self):
        if self.test_worker and self.test_worker.isRunning():
            self.test_worker.stop(); self.test_worker.wait(2000)
        if self.cmd_worker and self.cmd_worker.isRunning():
            self.cmd_worker.stop(); self.cmd_worker.wait(2000)
        if self.ctrl_profile_worker and self.ctrl_profile_worker.isRunning():
            self.ctrl_profile_worker.stop(); self.ctrl_profile_worker.wait(2000)
            self._ctrl_profile_running = False
            self.ctrl_start_btn.setEnabled(bool(self._ctrl_profile_data))
            self.ctrl_stop_btn.setEnabled(False)
            self.ctrl_import_btn.setEnabled(True)
        if self.serial_worker:
            self.output_command_timer.stop()
            # Send combined MHF zero command: 12 heaters + 2 fans in one packet
            self.serial_worker.send_command(
                "MHF:0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000\n")
        for ch in range(1, 13): self.current_targets[ch] = 0.0
        self.fan_rpm_values[1] = 0; self.fan_rpm_values[2] = 0
        self.current_fan_targets[1] = 0.0; self.current_fan_targets[2] = 0.0
        self.fan1_input.setText("0"); self.fan2_input.setText("0")
        self._on_log("EMERGENCY STOP - all zeroed")
        # 停止任何自动控制模式
        self._pid_controller.disable()
        if self._mpc_controller:
            self._mpc_controller.disable()
        self._mpc_last_compute_wall = None
        self._reset_mpc_cached_plan()
        self._control_mode = "none"
        self.mode_none_btn.setChecked(True)
        self.ctrl_status_label.setText("Mode: 不控制 (Emergency)")
        self.ctrl_status_label.setStyleSheet("font-size:14px;font-weight:bold;color:#dd2222;")
        self._update_data_display()

    def _clear_plot_data(self):
        self.time_data.clear(); self.temp_time.clear()
        for series in self._iter_plot_history_series():
            series.clear()
        self._plot_heater_time_range = [None, None]
        self._plot_temp_time_range = [None, None]
        for ch in range(1, 13):
            self.channel_power[ch].clear(); self.channel_voltage[ch].clear()
            self.channel_current[ch].clear(); self.channel_target[ch].clear()
        self.channel_fan_actual[1].clear(); self.channel_fan_actual[2].clear()
        self.channel_fan_target[1].clear(); self.channel_fan_target[2].clear()
        self._reset_mpc_cached_plan()
        self.mpc_time.clear()
        self.mpc_optimize_elapsed.clear()
        for name in MPC_LOSS_NAMES:
            self.mpc_loss_drpm[name][1].clear()
            self.mpc_loss_drpm[name][2].clear()
        self.mpc_total_drpm[1].clear()
        self.mpc_total_drpm[2].clear()
        self.mpc_t_aim.clear()
        self.mpc_t_max.clear()
        self.mpc_ambient_temp.clear()
        self.mpc_fan1_actual.clear()
        self.mpc_fan2_actual.clear()
        self.mpc_fan1_target.clear()
        self.mpc_fan2_target.clear()
        self.mpc_zp_actual_enabled.clear()
        self.mpc_zp_actual_step.clear()
        self.mpc_zp_prediction_enabled.clear()
        self.mpc_zp_state_isolated.clear()
        self.mpc_zp_correction_max_abs.clear()
        for ch in range(1, 13):
            self.mpc_zp_output_norm_rate[ch].clear()
            self.mpc_zp_trend_norm_rate[ch].clear()
            self.mpc_core_dtheta_dt_norm_rate[ch].clear()
        for index in range(1, 17):
            self.mpc_augment_h[index].clear()
        for i in range(1, GL860_NUM_CHANNELS + 1): self.channel_temp[i].clear()
        self.latest_temps = {}
        self._temp_ema_values = {}
        # Clear all curve visuals
        for ch in range(1, 13):
            self.p_curves[ch].setData([], []); self.p_tgt_curves[ch].setData([], [])
            self.v_curves[ch].setData([], []); self.i_curves[ch].setData([], [])
            self.cmd_preview_p[ch].setData([], [])
            self.ctrl_preview_p[ch].setData([], []); self.ctrl_preview_p[ch].hide()
        self.fan1_actual_curve.setData([], []); self.fan1_cmd_curve.setData([], [])
        self.fan2_actual_curve.setData([], []); self.fan2_cmd_curve.setData([], [])
        for fan in [1, 2]: self.cmd_preview_fan[fan].setData([], [])
        for i in range(1, GL860_NUM_CHANNELS + 1): self.temp_curves[i].setData([], [])
        self.ctrl_preview_temp_aim.setData([], []); self.ctrl_preview_temp_aim.hide()
        self._cleanup_flush_files()
        self._on_log("Curve data cleared")

    def _export_data(self):
        has_memory = len(self.time_data) > 0 or len(self.temp_time) > 0
        has_flushed = len(self._flush_files) > 0
        if not has_memory and not has_flushed:
            self._on_log("No data to export"); return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(base_dir, f"thermal_data_{timestamp}")
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Data", default_path,
            "Data Files (*.npz *.csv);;NumPy Files (*.npz);;CSV Files (*.csv)")
        if not filepath: return
        self._do_export(filepath)

    def _do_export(self, filepath):
        """Core export logic — filepath without extension, adds .npz/_heater.csv/_temp.csv."""
        # ---- Merge all data: flushed files + current memory ----
        all_data = {'series_limits': np.array([SERIES_LIMITS[i] for i in range(1, 13)])}
        for fp in self._flush_files:
            try:
                self._merge_npz(all_data, dict(np.load(fp)))
            except Exception as e:
                self._on_log(f"Warning: could not load flush file {fp}: {e}")
        self._merge_npz(all_data, self._build_npz_from_memory())

        # ---- Write NPZ ----
        npz_path = filepath if filepath.endswith('.npz') else filepath + '.npz'
        np.savez(npz_path, **all_data)
        self._on_log(f"NPZ exported: {npz_path}")

        # ---- Write CSV files (two separate files for different time steps) ----
        csv_base = filepath
        if csv_base.endswith('.npz') or csv_base.endswith('.csv'): csv_base = csv_base[:-4]
        has_heater = 'time' in all_data
        has_temp = 'temp_time' in all_data
        if has_heater:
            heater_csv = csv_base + '_heater.csv'
            t_arr = all_data['time']
            header1 = ['Time(s)']
            for ch in range(1, 13):
                header1 += [f'CH{ch}_P(W)', f'CH{ch}_V(V)', f'CH{ch}_I(A)']
            header1 += ['Fan1_RPM', 'Fan2_RPM']
            with open(heater_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header1)
                n1 = len(t_arr)
                for row_i in range(n1):
                    row = [f'{t_arr[row_i]:.4f}']
                    for ch in range(1, 13):
                        for key, fmt in [(f'CH{ch}_power', '.4f'), (f'CH{ch}_voltage', '.4f'), (f'CH{ch}_current', '.6f')]:
                            if key in all_data and row_i < len(all_data[key]):
                                row.append(f'{all_data[key][row_i]:{fmt}}')
                            else: row.append('')
                    for fan_key in ['Fan1_rpm_actual', 'Fan2_rpm_actual']:
                        if fan_key in all_data and row_i < len(all_data[fan_key]):
                            row.append(f'{all_data[fan_key][row_i]}')
                        else: row.append('')
                    writer.writerow(row)
            self._on_log(f"CSV heater exported: {heater_csv}")
        if has_temp:
            temp_csv = csv_base + '_temp.csv'
            tt_arr = all_data['temp_time']
            header2 = ['Time(s)']
            for i in range(1, 13): header2.append(f'CH{i}_Temp(C)')
            header2.append('Ambient_Temp(C)')
            with open(temp_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header2)
                n2 = len(tt_arr)
                for row_i in range(n2):
                    row = [f'{tt_arr[row_i]:.4f}']
                    for i in range(1, GL860_NUM_CHANNELS + 1):
                        k = f'T{i}_temp'
                        if k in all_data and row_i < len(all_data[k]):
                            v = all_data[k][row_i]
                            row.append(f'{v:.2f}' if not math.isnan(v) else '')
                        else: row.append('')
                    writer.writerow(row)
            self._on_log(f"CSV temp exported: {temp_csv}")
        if len(self.mpc_time) > 0:
            self._write_mpc_csv(csv_base)

    # ----------------------------------------------------------
    # Log
    # ----------------------------------------------------------
    def _on_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self._emergency_stop()
        if self.serial_worker: self.serial_worker.stop(); self.serial_worker.wait(3000)
        if self.gl860_worker: self.gl860_worker.stop(); self.gl860_worker.wait(3000)
        if self.cmd_worker: self.cmd_worker.stop(); self.cmd_worker.wait(3000)
        if self.mpc_worker and self.mpc_worker.isRunning(): self.mpc_worker.wait(3000)
        self._cleanup_flush_files()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont("Microsoft YaHei", 10))
    window = ThermalControlApp(); window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
