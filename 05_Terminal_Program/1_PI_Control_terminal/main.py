#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PID thermal-control terminal (streamlined version with PID control only)

Communication:
  1) STM32 serial: acquire power/voltage/current from 12 heater channels and actual speed from two fans; transmit MHF commands
  2) GL860A network: acquire CH1-CH13 temperatures over Modbus/TCP (CH13 is ambient temperature)
Control:
  - PID mode: calculate Fan1/Fan2 target speeds automatically from 12 heat-source temperatures
  - Command file: import CSV data (Time,CH1..CH12[,Fan1,Fan2|Temp_Aim]) and execute it according to its time schedule

Dependencies: pip install pyserial pymodbus
Run: python main.py --port COM5 --ip 192.168.0.2 [command.csv]
"""

import argparse
import csv
import logging
import math
import os
import re
import shutil
import shlex
import sys
import threading
import time
import unicodedata
from collections import deque

import serial
import serial.tools.list_ports
from pymodbus.client import ModbusTcpClient

from PID_Control import PIDFanController

# ------------------------------------------------------------------ Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GL860_START_ADDRESS = 100  # Starting address of the GL860A temperature registers.
GL860_NUM_CHANNELS = 13  # CH1-CH12 are heat sources; CH13 is ambient temperature.
TEMP_EMA_ALPHA = 0.5  # Temperature EMA coefficient.
PID_START_T_AIM = 60.0  # Initial target temperature when PID is enabled [°C].


def fmt(v, spec=".2f"):
    """Format numeric values, displaying NaN or missing values as --."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = math.nan
    return f"{f:{spec}}" if not math.isnan(f) else "--"


# ------------------------------------------------------------------ Data parsing
def parse_stm32(line):
    """Parse an STM32 report line: V_Sys:.. Ch1_VD:..,IS:.. F1_RPM:.."""
    v_m = re.search(r"V_Sys:([-+]?\d*\.\d+|\d+)", line)
    if not v_m:
        return None
    v_sys = float(v_m.group(1))
    channels = re.findall(
        r"Ch(\d+)_VD:([-+]?\d*\.\d+|\d+),IS:([-+]?\d*\.\d+|\d+)", line
    )
    if not channels:
        return None
    data = {"v_sys": v_sys, "channels": {}, "fans": {}}
    for ch, v_d, i_s in channels:
        ch_idx = int(ch)
        v_drop = v_sys - float(v_d)
        data["channels"][ch_idx] = {
            "v_drop": v_drop,
            "current": float(i_s),
            "power": v_drop * float(i_s),
        }
    for fan, rpm in re.findall(r"F(\d+)_RPM:([\d.]+)", line):
        data["fans"][int(fan)] = float(rpm)
    return data


def read_temps(client):
    """Read 13 GL860A temperatures at 0.1 °C resolution and set invalid channels to NaN."""
    try:
        # Support pymodbus 2.x (unit), 3.x (slave), and 3.7+ (device_id) argument names.
        resp = None
        kwargs = {"address": GL860_START_ADDRESS, "count": GL860_NUM_CHANNELS * 2}
        for kw in ("device_id", "slave", "unit"):
            try:
                resp = client.read_input_registers(**kwargs, **{kw: 1})
                break
            except TypeError:
                continue
        if resp is None or resp.isError():
            return None
        raw = resp.registers
    except Exception:
        return None
    temps = {}
    for i in range(GL860_NUM_CHANNELS):
        val = raw[i * 2]
        if val == 16382 or val >= 16000:
            temps[i + 1] = math.nan
        else:
            if val > 32767:
                val -= 65536
            temps[i + 1] = val / 10.0
    return temps


def find_ch340_port():
    """Find the serial port by CH340 characteristics: hardware ID containing 1A86 (VID:PID), or description/hardware ID containing CH340."""
    candidates = []
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").upper()
        hwid = (p.hwid or "").upper()
        if "1A86" in hwid or "CH340" in desc or "CH340" in hwid:
            candidates.append((0 if "1A86" in hwid else 1, p.device))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


# ------------------------------------------------------------------ Shared state
class SharedState:
    def __init__(self, port, baud, ip, tcp_port, t_aim):
        self.lock = threading.Lock()
        self.port = port
        self.baud = baud
        self.ip = ip
        self.tcp_port = tcp_port
        self.t_aim = t_aim
        self.data_dir = os.path.join(SCRIPT_DIR, "data")
        self.recording = False
        self.record_interval = 1.0
        self.rec_base = (
            ""  # Base path for three-file recording: thermal_data_YYYYMMDD_HHMMSS.
        )
        self.rec_rows = {"temp": 0, "heater": 0, "pid": 0}
        self.rec_t0 = (
            time.time()
        )  # Recording time origin: the instant start/record is enabled.
        self.rec_lock = (
            threading.Lock()
        )  # File-write lock to serialize writes from multiple command threads.
        self.last_temp_saved = (
            0.0  # last_temp_t at the most recently saved temperature row.
        )
        self.last_heater_saved = (
            0.0  # last_serial_t at the most recently saved controller row.
        )
        self.auto_record = (
            False  # Flag for automatic recording during command execution.
        )
        self.active_serial_port = (
            ""  # Currently connected serial port, displayed in AUTO mode.
        )
        self.serial_connected = False
        self.serial_status = "Not connected"
        self.gl_connected = False
        self.gl_status = "Not connected"
        self.pid = None  # Import the PID controller for Kp/Ki display.
        self.v_sys = math.nan
        self.channels = {}  # ch -> {v_drop, current, power}
        self.fans_actual = {1: math.nan, 2: math.nan}
        self.temps = {}  # 1..13
        self.temp_ema = {}
        self.last_serial_t = 0.0
        self.last_temp_t = 0.0
        self.heater_targets = {ch: 0.0 for ch in range(1, 13)}
        self.fan_targets = {1: 0.0, 2: 0.0}
        self.pid_enabled = False
        self.logs = deque(maxlen=8)
        self.cmd_rows = []
        self.cmd_name = ""
        self.cmd_path = ""  # Full path of the current command file, used to select the automatic save location.
        self.cmd_running = False
        self.cmd_t0 = 0.0
        self.cmd_duration = 0.0
        self.flash_active = False  # Background flashing in progress.
        self.flash_hex = ""  # Name of the HEX file currently being flashed.
        self.flash_thread = None  # Current FlashThread, used for abort operations.
        self.flash_new_serial = (
            None  # Serial thread returned to the main loop after flashing.
        )

    def log(self, msg):
        with self.lock:
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ------------------------------------------------------------------ Serial thread
class SerialLink(threading.Thread):
    """STM32 serial link: find the CH340 automatically, maintain reconnection, acquire power/current/fan speed, and transmit MHF commands."""

    RETRY_SEC = 3.0

    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state
        self.ser = None
        self._stop_evt = threading.Event()
        self._write_lock = threading.Lock()
        self._last_warn = 0.0

    def run(self):
        s = self.state
        while not self._stop_evt.is_set():
            self._connect_once()
            if not self._stop_evt.is_set():
                self._stop_evt.wait(self.RETRY_SEC)

    def _connect_once(self):
        s = self.state
        port = self._resolve_port()
        if port is None:
            now = time.time()
            with s.lock:
                s.serial_status = "CH340 not found, auto-retrying..."
            if now - self._last_warn >= 10.0:
                s.log(
                    f"CH340 serial not found, auto-retrying every {self.RETRY_SEC:.0f}s"
                )
                self._last_warn = now
            return
        try:
            # dtr/rts=False prevents the CH340 auto-download circuit from resetting the MCU or driving BOOT0 high on connection.
            # Note: pyserial 3.5 does not accept dtr/rts constructor arguments; set them after opening.
            self.ser = serial.Serial(port, s.baud, timeout=0.1)
            self.ser.dtr = False
            self.ser.rts = False
        except Exception as e:
            with s.lock:
                s.serial_status = f"Connect failed: {e}"
            s.log(f"Serial {port} connect failed: {e}, auto-retrying...")
            return
        with s.lock:
            s.serial_connected = True
            s.active_serial_port = port
            s.serial_status = f"OK {port} @ {s.baud}"
        s.log(f"STM32 connected {port} @ {s.baud}")
        self.send(
            build_mhf(s)
        )  # Zero all outputs immediately after connection to establish a safe initial state.
        try:
            while not self._stop_evt.is_set():
                try:
                    if self.ser.in_waiting > 0:
                        line = (
                            self.ser.readline().decode("utf-8", errors="ignore").strip()
                        )
                        if line and "V_Sys:" in line:
                            data = parse_stm32(line)
                            if data:
                                with s.lock:
                                    s.v_sys = data["v_sys"]
                                    s.channels = data["channels"]
                                    if data["fans"]:
                                        s.fans_actual = data["fans"]
                                    s.last_serial_t = time.time()
                    else:
                        time.sleep(0.01)
                except (serial.SerialException, OSError):
                    s.log("STM32 disconnected, auto-reconnecting...")
                    break
        finally:
            with s.lock:
                s.serial_connected = False
                s.active_serial_port = ""
                s.serial_status = "Disconnected, auto-reconnecting..."
            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                except Exception:
                    pass
            self.ser = None

    def _resolve_port(self):
        """Continuously scan for CH340 devices in AUTO mode; use an explicitly specified COM port directly."""
        with self.state.lock:
            port = self.state.port
        if port and port.upper() != "AUTO":
            return port
        return find_ch340_port()

    def send(self, cmd):
        ser = self.ser
        if not ser or not ser.is_open:
            return False
        with self._write_lock:
            try:
                ser.write(cmd.encode("utf-8"))
                return True
            except Exception as e:
                self.state.log(f"Send failed: {e}")
        return False

    def stop(self):
        self._stop_evt.set()
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass


# ------------------------------------------------------------------ Network thread
class ModbusLink(threading.Thread):
    """Poll temperatures from the GL860A over Modbus/TCP and apply EMA filtering."""

    RETRY_SEC = 3.0

    def __init__(self, state, poll=0.1):
        super().__init__(daemon=True)
        self.state = state
        self.poll = poll
        self.client = None
        self._stop_evt = threading.Event()

    def run(self):
        s = self.state
        last_warn = 0.0
        while not self._stop_evt.is_set():
            try:
                self.client = ModbusTcpClient(s.ip, port=s.tcp_port)
                if not self.client.connect():
                    with s.lock:
                        s.gl_connected = False
                        s.gl_status = f"Connect failed {s.ip}:{s.tcp_port}"
                    now = time.time()
                    if now - last_warn >= 10.0:
                        s.log(f"GL860A connect failed {s.ip}:{s.tcp_port}, retrying...")
                        last_warn = now
                    self._stop_evt.wait(self.RETRY_SEC)
                    continue
            except Exception as e:
                with s.lock:
                    s.gl_connected = False
                    s.gl_status = f"Connect failed {s.ip}:{s.tcp_port}"
                now = time.time()
                if now - last_warn >= 10.0:
                    s.log(f"GL860A error: {e}, retrying...")
                    last_warn = now
                self._stop_evt.wait(self.RETRY_SEC)
                continue
            with s.lock:
                s.gl_connected = True
                s.gl_status = f"OK {s.ip}:{s.tcp_port}"
            s.log(f"GL860A connected {s.ip}:{s.tcp_port}")
            fail = 0
            try:
                while not self._stop_evt.is_set():
                    if not self.client.connected:
                        s.log("GL860A disconnected, retrying...")
                        break
                    try:
                        temps = read_temps(self.client)
                    except Exception:
                        temps = None
                    if temps is not None:
                        fail = 0
                        with s.lock:
                            alpha = TEMP_EMA_ALPHA
                            for ch in range(1, GL860_NUM_CHANNELS + 1):
                                raw = temps.get(ch, math.nan)
                                last = s.temp_ema.get(ch, math.nan)
                                s.temp_ema[ch] = (
                                    last
                                    if math.isnan(raw)
                                    else raw
                                    if math.isnan(last)
                                    else alpha * raw + (1 - alpha) * last
                                )
                            s.temps = dict(s.temp_ema)
                            s.last_temp_t = time.time()
                    else:
                        fail += 1
                        if fail >= 10:
                            s.log("GL860A read failures, reconnecting...")
                            break
                    self._stop_evt.wait(self.poll)
            finally:
                with s.lock:
                    s.gl_connected = False
                    s.gl_status = "Disconnected, retrying..."
                if self.client:
                    try:
                        self.client.close()
                    except Exception:
                        pass
                self.client = None

    def stop(self):
        self._stop_evt.set()
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass


# ------------------------------------------------------------------ Command execution
def load_commands(path):
    """Parse a command CSV into [{t, heaters{ch:W}, fans{1,2:RPM}, t_aim}], supporting both header formats."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.strip().upper() for h in next(reader)]
        time_col, ch_cols, fan_cols, aim_col = None, {}, {}, None
        for i, h in enumerate(header):
            if h == "TIME":
                time_col = i
            elif h.startswith("CH") and h[2:].isdigit():
                ch_cols[int(h[2:])] = i
            elif h.startswith("FAN") and h[3:].isdigit():
                fan_cols[int(h[3:])] = i
            elif h == "TEMP_AIM":
                aim_col = i
        if time_col is None:
            raise ValueError("CSV missing TIME column")
        for r in reader:
            if not r or not r[0].strip():
                continue
            row = {"t": float(r[time_col]), "heaters": {}, "fans": {}, "t_aim": None}
            for ch, ci in ch_cols.items():
                if 1 <= ch <= 12 and ci < len(r) and r[ci].strip():
                    row["heaters"][ch] = float(r[ci])
            for fan, ci in fan_cols.items():
                if fan in (1, 2) and ci < len(r) and r[ci].strip():
                    row["fans"][fan] = float(r[ci])
            if aim_col is not None and aim_col < len(r) and r[aim_col].strip():
                row["t_aim"] = float(r[aim_col])
            rows.append(row)
    if not rows:
        raise ValueError("no command rows found")
    return rows


def clean_path(p):
    """Remove redundant backslashes and quotes introduced by pasting or escaping."""
    p = p.strip().strip('"').strip("'")
    prev = None
    while prev != p:  # Collapse consecutive backslashes to one (\\ -> \).
        prev = p
        p = p.replace("\\\\", "\\")
    return p


def path_candidates(p):
    """Generate plausible path representations to try in sequence."""
    p = clean_path(p)
    cands = [p]
    alt = re.sub(
        r"\\(?=_)", "", p
    )  # Remove redundant backslashes before underscores, e.g., 1\_transient -> 1_transient.
    if alt != p:
        cands.append(alt)
    fwd = p.replace("\\", "/")  # Windows also accepts forward slashes.
    if fwd != p:
        cands.append(fwd)
    # For relative paths, also try the script directory because the SSH login directory may differ.
    if not os.path.isabs(p) and not re.match(r"^[A-Za-z]:", p):
        cands.append(os.path.join(SCRIPT_DIR, p))
        if alt != p:
            cands.append(os.path.join(SCRIPT_DIR, alt))
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def load_command_file(arg):
    """Try multiple path representations when loading a CSV file and return (rows, resolved path)."""
    last_err = None
    for cand in path_candidates(arg):
        try:
            return load_commands(cand), cand
        except OSError as e:
            last_err = e
    raise last_err if last_err else OSError(f"file not found: {arg}")


def build_mhf(s):
    """Build an MHF command containing 12 heater powers and two fan speeds."""
    with s.lock:
        p = [s.heater_targets[ch] for ch in range(1, 13)]
        f1, f2 = s.fan_targets[1], s.fan_targets[2]
    body = ",".join(f"{v:.3f}" for v in p)
    return f"MHF:{body},{f1:.3f},{f2:.3f}\n"


def send_mhf(s, link):
    sent = bool(link) and link.send(build_mhf(s))
    if sent:
        log_pid(
            s
        )  # Append one row to the control-command timeline after each successful command.
    return sent


def zero_output(s, link):
    with s.lock:
        for ch in range(1, 13):
            s.heater_targets[ch] = 0.0
        s.fan_targets[1] = 0.0
        s.fan_targets[2] = 0.0
    send_mhf(s, link)


def csv_val(v, spec):
    """Format CSV values, writing missing or NaN values as empty fields."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return "" if math.isnan(f) else f"{f:{spec}}"


# ------------------------------------------------------------------ Three-file data export (using thermal_data_* naming)
# Temperature acquisition, controller acquisition, and command transmission use independent time axes.
TEMP_HEADER = (
    ["Time(s)"] + [f"CH{i}_Temp(C)" for i in range(1, 13)] + ["Ambient_Temp(C)"]
)
HEATER_HEADER = (
    ["Time(s)"]
    + [f"CH{i}_{k}" for i in range(1, 13) for k in ("P(W)", "V(V)", "I(A)")]
    + ["Fan1_RPM", "Fan2_RPM"]
)
PID_HEADER = [
    "Time(s)",
    "T_Aim(C)",
    "T_Max(C)",
    "Ambient_Temp(C)",
    "Fan1_Actual_RPM",
    "Fan2_Actual_RPM",
    "Fan1_Command_RPM",
    "Fan2_Command_RPM",
] + [f"CH{i}_P_Cmd(W)" for i in range(1, 13)]


def rec_start(s, folder=None):
    """Start three-file recording with base name thermal_data_YYYYMMDD_HHMMSS."""
    folder = folder or s.data_dir
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, f"thermal_data_{time.strftime('%Y%m%d_%H%M%S')}")
    with s.lock:
        s.rec_base = base
        s.rec_rows = {"temp": 0, "heater": 0, "pid": 0}
        s.rec_t0 = time.time()
        s.last_temp_saved = 0.0
        s.last_heater_saved = 0.0
    return base


def rec_summary(s):
    """Return a recording summary containing the base name and row counts of the three files."""
    with s.lock:
        base, rows = s.rec_base, dict(s.rec_rows)
    if not base:
        return "no files"
    return f"{os.path.basename(base)}_* " + ", ".join(
        f"{k}:{rows.get(k, 0)}" for k in ("temp", "heater", "pid")
    )


def _append_row(s, path, header, row):
    """Append one CSV row, writing the header for a new file; rec_lock serializes multithreaded writes."""
    with s.rec_lock:
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(header)
            writer.writerow(row)


def sample_temp(s):
    """Append one row to the temperature timeline whenever the GL860A provides new temperatures."""
    if not s.recording:
        return
    with s.lock:
        if s.last_temp_t <= s.last_temp_saved:
            return
        temps = dict(s.temps)
        now_t = time.time() - s.rec_t0
        s.last_temp_saved = s.last_temp_t
    row = [f"{now_t:.4f}"]
    for ch in range(1, 13):
        row.append(csv_val(temps.get(ch), ".2f"))
    row.append(csv_val(temps.get(13), ".2f"))
    try:
        _append_row(s, s.rec_base + "_temp.csv", TEMP_HEADER, row)
        s.rec_rows["temp"] += 1
    except Exception as e:
        s.recording = False
        s.log(f"Recording error (temp), stopped: {e}")


def sample_heater(s):
    """Append one row to the controller timeline whenever new STM32 serial data arrives."""
    if not s.recording:
        return
    with s.lock:
        if s.last_serial_t <= s.last_heater_saved:
            return
        channels = dict(s.channels)
        fans = dict(s.fans_actual)
        now_t = time.time() - s.rec_t0
        s.last_heater_saved = s.last_serial_t
    row = [f"{now_t:.4f}"]
    for ch in range(1, 13):
        d = channels.get(ch, {})
        row.append(csv_val(d.get("power"), ".4f"))
        row.append(csv_val(d.get("v_drop"), ".4f"))
        row.append(csv_val(d.get("current"), ".6f"))
    row.append(csv_val(fans.get(1), ".0f"))
    row.append(csv_val(fans.get(2), ".0f"))
    try:
        _append_row(s, s.rec_base + "_heater.csv", HEATER_HEADER, row)
        s.rec_rows["heater"] += 1
    except Exception as e:
        s.recording = False
        s.log(f"Recording error (heater), stopped: {e}")


def log_pid(s):
    """Append one row to the control-command timeline after each successful MHF transmission."""
    if not s.recording:
        return
    with s.lock:
        temps = dict(s.temps)
        fans_act = dict(s.fans_actual)
        fans_cmd = dict(s.fan_targets)
        heat_cmd = dict(s.heater_targets)
        t_aim = s.t_aim
        now_t = time.time() - s.rec_t0
    valid = {ch: t for ch, t in temps.items() if ch != 13 and not math.isnan(t)}
    t_max = max(valid.values()) if valid else math.nan
    row = [
        f"{now_t:.4f}",
        csv_val(t_aim, ".4f"),
        csv_val(t_max, ".4f"),
        csv_val(temps.get(13), ".4f"),
        csv_val(fans_act.get(1), ".0f"),
        csv_val(fans_act.get(2), ".0f"),
        csv_val(fans_cmd.get(1), ".0f"),
        csv_val(fans_cmd.get(2), ".0f"),
    ]
    for ch in range(1, 13):
        row.append(csv_val(heat_cmd.get(ch), ".4f"))
    try:
        _append_row(s, s.rec_base + "_pid.csv", PID_HEADER, row)
        s.rec_rows["pid"] += 1
    except Exception as e:
        s.recording = False
        s.log(f"Recording error (pid), stopped: {e}")


def clamp_fan(v):
    v = int(round(float(v)))
    return 0 if v < 1000 else min(v, 6300)


def run_pid(s, pid, link):
    """Perform one PID calculation and transmit the fan targets."""
    with s.lock:
        if not s.pid_enabled:
            return
        temps = {ch: s.temps.get(ch, math.nan) for ch in range(1, 13)}
    rpm1, rpm2 = pid.compute(temps)
    with s.lock:
        s.fan_targets[1], s.fan_targets[2] = rpm1, rpm2
    send_mhf(s, link)


def send_cooling(s, link):
    """Cooling command (one-time transmission): zero heater power and set the fans to full speed.
    If PID is enabled, it takes control of fan speed on the next cycle."""
    with s.lock:
        for ch in range(1, 13):
            s.heater_targets[ch] = 0.0
        s.fan_targets[1] = PIDFanController.RPM_MAX
        s.fan_targets[2] = PIDFanController.RPM_MAX
    send_mhf(s, link)
    s.log("Cooling sent: heaters 0, fans 100% (PID takes over next cycle)")


class CommandPlayer(threading.Thread):
    """Execute an imported command CSV according to its time schedule."""

    def __init__(self, state, link, pid, rows, name):
        super().__init__(daemon=True)
        self.state = state
        self.link = link
        self.pid = pid
        self.rows = rows
        self.name = name
        self._stop_evt = threading.Event()

    def run(self):
        s = self.state
        t0 = time.time()
        total = len(self.rows)
        with s.lock:
            s.cmd_name = self.name
            s.cmd_running = True
            s.cmd_t0 = t0
            s.cmd_duration = self.rows[-1]["t"]
        s.log(f"Command started: {self.name} ({total} steps)")
        for i, row in enumerate(self.rows):
            while not self._stop_evt.is_set():
                elapsed = time.time() - t0
                if elapsed >= row["t"]:
                    break
                self._stop_evt.wait(min(0.02, row["t"] - elapsed))
            if self._stop_evt.is_set():
                break
            with s.lock:
                if row["heaters"]:
                    s.heater_targets.update(row["heaters"])
                if row["fans"]:
                    s.fan_targets.update(row["fans"])
                if row["t_aim"] is not None:
                    s.t_aim = row["t_aim"]
                    if s.pid_enabled:
                        self.pid.set_t_aim(row["t_aim"])
            send_mhf(s, self.link)
            if i % 50 == 0 or i == total - 1:
                s.log(f"Step {i + 1}/{total} @ {row['t']:.1f}s")
        with s.lock:
            s.cmd_running = False
        s.log("Command finished/stopped, outputs zeroed")
        zero_output(s, self.link)
        if s.auto_record:
            s.auto_record = False
            if s.recording:
                s.recording = False
                s.log(f"Command done, data saved: {rec_summary(s)}")

    def stop(self):
        self._stop_evt.set()


# ------------------------------------------------------------------ Terminal display
def build_lines(state, now, log_limit=4, show_hints=True):
    s = state
    with s.lock:
        temps = dict(s.temps)
        channels = dict(s.channels)
        fans_actual = dict(s.fans_actual)
        fan_t = dict(s.fan_targets)
        pid_on = s.pid_enabled
        t_aim = s.t_aim
        kp = s.pid.kp if s.pid else math.nan
        ki = s.pid.ki if s.pid else math.nan
        serial_ok = s.serial_connected
        gl_ok = s.gl_connected
        ser_port = s.active_serial_port or s.port
        ser_age = now - s.last_serial_t if s.last_serial_t else math.nan
        gl_age = now - s.last_temp_t if s.last_temp_t else math.nan
        running = s.cmd_running
        t0, dur, name = s.cmd_t0, s.cmd_duration, s.cmd_name
        rec_on = s.recording
        logs = list(s.logs)
        flash_active = s.flash_active
        flash_hex = s.flash_hex

    valid = {ch: t for ch, t in temps.items() if ch != 13 and not math.isnan(t)}
    t_max = max(valid.values()) if valid else math.nan
    ch_max = max(valid, key=valid.get) if valid else None
    elapsed = time.time() - t0 if running else 0.0

    cmd_line = f" CMD    : {name or '-'}"
    if running:
        pct = min(100.0, elapsed / dur * 100.0) if dur > 0 else 0.0
        left = max(0.0, dur - elapsed)
        bar_w = 26
        filled = int(round(pct / 100.0 * bar_w))
        bar = "#" * filled + "-" * (bar_w - filled)
        cmd_line += (
            f"  RUNNING {elapsed:6.1f}/{dur:.1f}s  {pct:5.1f}%  left {left:5.1f}s"
        )
        prog_line = f" PROGRESS: [{bar}]  {pct:5.1f}%"
    else:
        cmd_line += "  STOPPED"
        prog_line = None
    if rec_on:
        cmd_line += "  REC ON"

    lines = [
        f" SERIAL : {ser_port} @ {s.baud:<7d}  {'OK ' if serial_ok else 'FAIL'}   last {fmt(ser_age, '.1f')} s",
        f" GL860A : {s.ip}:{s.tcp_port}  {'OK ' if gl_ok else 'FAIL'}   last {fmt(gl_age, '.1f')} s",
        f" PID    : {'ENABLED' if pid_on else 'OFF '}    T_aim={t_aim:5.1f} C    "
        f"T_max={fmt(t_max):>6} C" + (f" (CH{ch_max})" if ch_max else ""),
        f" PARAM  : Kp={fmt(kp, '.4f'):>7}    Ki={fmt(ki, '.6f')}",
        cmd_line,
    ]
    if prog_line:
        lines.append(prog_line)
    lines += [
        "-" * 66,
        f"{' CH':>3}  {'Temp(C)':>9}  {'P(W)':>9}  {'Vdrop(V)':>9}  {'I(A)':>8}",
    ]
    if flash_active:
        lines.insert(2, f" FLASH  : {flash_hex} - in progress (BOOT0=1 + reset)")
    for ch in range(1, 13):
        chd = channels.get(ch, {})
        star = "*" if ch == ch_max else " "
        lines.append(
            f" {star}{ch:>2}  {fmt(temps.get(ch)):>9}  {fmt(chd.get('power')):>9}  "
            f"{fmt(chd.get('v_drop')):>9}  {fmt(chd.get('current')):>8}"
        )
    lines.append(f" {'AMB':>3}  {fmt(temps.get(13)):>9}")
    lines.append("-" * 66)
    lines.append(" FAN   ACTUAL   TARGET")
    lines.append(
        f"  1    {fmt(fans_actual.get(1), '.0f'):>6}   {fmt(fan_t.get(1), '.0f'):>6}"
    )
    lines.append(
        f"  2    {fmt(fans_actual.get(2), '.0f'):>6}   {fmt(fan_t.get(2), '.0f'):>6}"
    )
    if show_hints:
        lines.append("-" * 66)
        lines.append(
            " q | pid on/off | t_aim 60 | gains Kp Ki | heater CH W | fan R1 R2"
        )
        lines.append(
            " import FILE | start cooling | stop | zero | flash | ports | help"
        )
    for log in logs[-max(0, log_limit) :]:
        lines.append(f" {log}")
    return lines


def _truncate_disp(line, width):
    """Truncate to terminal width, counting CJK/full-width characters as two columns to prevent wrapping misalignment."""
    out, cols = [], 0
    for ch in line:
        w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if cols + w > width:
            break
        out.append(ch)
        cols += w
    return "".join(out)


def draw(state, input_text=""):
    """Refresh in place by returning the cursor to the upper-left and redrawing the screen without clearing or scrolling the buffer."""
    size = shutil.get_terminal_size((80, 24))
    width = max(20, size.columns - 1)
    height = max(10, size.lines)
    show_hints = height >= 27
    with state.lock:
        extra = (1 if state.flash_active else 0) + (1 if state.cmd_running else 0)
    fixed = (27 if show_hints else 24) + extra
    log_limit = max(0, height - fixed - 1)  # Reserve one line for the input prompt.
    body = "\n".join(
        _truncate_disp(l, width)
        for l in build_lines(state, time.time(), log_limit, show_hints)
    )
    prompt = _truncate_disp(f"> {input_text}", width)
    if not sys.stdout.isatty():
        # In non-interactive mode (ssh host "cmd" or redirection), print changed plain-text snapshots without ANSI redraw codes.
        snap = body + "\n" + prompt
        if getattr(draw, "_last_plain", None) != snap:
            draw._last_plain = snap
            sys.stdout.write(snap + "\n")
            sys.stdout.flush()
        return
    out = "\x1b[H\x1b[2J\x1b[?25l" + body + "\x1b[?25h" + f"\n{prompt}"
    sys.stdout.write(out)
    sys.stdout.flush()


# ------------------------------------------------------------------ Keyboard input
try:
    import msvcrt
except ImportError:
    msvcrt = None


def poll_input(buf):
    """Read keys non-blockingly from the local console using msvcrt."""
    lines = []
    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch == "\r":
            lines.append("".join(buf))
            buf.clear()
        elif ch == "\b":
            if buf:
                buf.pop()
        elif ch == "\x03":
            lines.append("__QUIT__")
        elif ch in ("\x00", "\xe0"):
            msvcrt.getwch()  # Ignore function and arrow keys.
        elif ch == "\t":
            buf.append(
                " "
            )  # Convert tabs to spaces because tabs disrupt line layout in raw mode.
        else:
            buf.append(ch)
    return lines


def _pipe_avail():
    """Return the number of readable standard-input bytes using PeekNamedPipe on Windows and select on Linux."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.windll.kernel32
            h = k32.GetStdHandle(-10)  # STD_INPUT_HANDLE
            avail = wintypes.DWORD(0)
            if k32.PeekNamedPipe(h, None, 0, None, ctypes.byref(avail), None):
                return int(avail.value)
            return 0
        except Exception:
            return 0
    import select

    r, _, _ = select.select([sys.stdin], [], [], 0)
    return 1 if r else 0


def _pipe_read_one():
    """Read one byte non-blockingly to skip arrow-key escape sequences."""
    try:
        if _pipe_avail() > 0:
            return os.read(sys.stdin.fileno(), 1)
    except (OSError, ValueError):
        pass
    return None


def _handle_bytes(data, buf, lines):
    """Process a block of input bytes, update the display buffer, and emit complete command lines."""
    for ch in data.decode("utf-8", errors="ignore"):
        if ch in "\r\n":
            lines.append("".join(buf))
            buf.clear()
        elif ch in "\x7f\x08":
            if buf:
                buf.pop()
        elif ch in "\x03\x04":  # Exit with Ctrl+C or Ctrl+D.
            lines.append("__QUIT__")
        elif ch == "\x1b":  # Skip escape sequences such as arrow-key input.
            for _ in range(8):
                nxt = _pipe_read_one()
                if nxt is None:
                    break
                if 0x40 <= nxt[0] <= 0x7E:
                    break
        elif ch == "\t":
            buf.append(
                " "
            )  # Convert tabs to spaces because they expand lines and cause wrapping under SSH/raw mode.
        else:
            buf.append(ch)


def poll_pipe(buf):
    """Poll SSH/pipe input in the main loop without a separate thread to avoid interpreter crashes during shutdown."""
    lines = []
    try:
        avail = _pipe_avail()
        if avail > 0:
            data = os.read(sys.stdin.fileno(), avail if os.name == "nt" else 1024)
            if data:
                _handle_bytes(data, buf, lines)
    except (OSError, ValueError):
        pass
    return lines


# ------------------------------------------------------------------ STM32F1 serial flashing (system bootloader, ST AN3155)
FLASH_BAUD = 115200  # The bootloader uses fixed 8E1 framing at 115200 baud.
FLASH_APP_START = 0x08000000  # STM32F1 application start address.
FLASH_SYNC_WAIT = 60.0  # Maximum time to wait for BOOT0 selection and reset [s].
FLASH_ERASE_TIMEOUT = 30.0  # Timeout while waiting for the mass-erase ACK [s].


def default_hex_path():
    """Default firmware: prefer project.hex in the script directory for cross-platform use, then the 0_DMA_control project."""
    cands = [
        os.path.join(SCRIPT_DIR, "project.hex"),
        os.path.join(SCRIPT_DIR, "Objects", "project.hex"),
        os.path.normpath(
            os.path.join(
                SCRIPT_DIR,
                "..",
                "0_DMA_control",
                "0_fan_hs_p_show_sw_P_test",
                "Objects",
                "project.hex",
            )
        ),
        os.path.join(os.getcwd(), "Objects", "project.hex"),
        os.path.join(os.getcwd(), "project.hex"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def resolve_hex_path(arg):
    """Resolve the HEX path: use the default firmware when omitted, otherwise try multiple representations for SSH escaping compatibility."""
    if not arg:
        p = default_hex_path()
        if p is None:
            raise FileNotFoundError(
                "default firmware not found: 0_DMA_control/.../Objects/project.hex"
            )
        return p
    for cand in path_candidates(arg):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"hex file not found: {arg}")


def parse_intel_hex(path):
    """Parse Intel HEX (:LLAAAATT[DD..]CC) and return contiguous [(absolute_address, bytes)] segments."""
    segments, base = [], 0
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            if not line.startswith(":"):
                raise ValueError(f"line {lineno}: not an Intel HEX record")
            try:
                rec = bytes.fromhex(line[1:])
            except ValueError:
                raise ValueError(f"line {lineno}: invalid hex characters")
            if len(rec) < 5 or sum(rec) & 0xFF:
                raise ValueError(f"line {lineno}: bad record or checksum")
            ln, addr, typ = rec[0], (rec[1] << 8) | rec[2], rec[3]
            data = rec[4:-1]
            if len(data) != ln:
                raise ValueError(f"line {lineno}: length mismatch")
            if typ == 0x00:
                segments.append([base + addr, bytearray(data)])
            elif typ == 0x01:  # EOF
                break
            elif typ == 0x02:  # Extended segment address (<< 4).
                base = ((data[0] << 8) | data[1]) << 4
            elif typ == 0x04:  # Extended linear address (<< 16).
                base = ((data[0] << 8) | data[1]) << 16
            # Ignore 0x03/0x05 start-address records because they do not affect flashing.
    segments.sort(key=lambda s: s[0])
    merged = []
    for addr, data in segments:
        if merged and merged[-1][0] + len(merged[-1][1]) == addr:
            merged[-1][1] += data
        else:
            merged.append([addr, data])
    if not merged:
        raise ValueError("no data records in hex file")
    return [(a, bytes(d)) for a, d in merged]


class Stm32Flash:
    """STM32F1 system bootloader protocol (ST AN3155): 8E1 at 115200 baud with bytewise-XOR checksums."""

    ACK, NACK = 0x79, 0x1F

    def __init__(self, port, timeout=1.0):
        self.ser = serial.Serial(
            port,
            FLASH_BAUD,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_EVEN,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            write_timeout=timeout,
        )
        # pyserial 3.5 does not accept dtr/rts constructor arguments; set them after opening:
        # Do not reset or drive BOOT0 to avoid interfering with the manual-jumper procedure.
        self.ser.dtr = False
        self.ser.rts = False
        self._base_timeout = timeout
        self.extended_erase = False  # Whether extended erase command 0x44 is supported, as detected by probing.

    def close(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass

    def _ack(self, desc, timeout=None):
        if timeout is not None:
            self.ser.timeout = timeout
        try:
            b = self.ser.read(1)
        finally:
            if timeout is not None:
                self.ser.timeout = self._base_timeout
        if not b:
            raise RuntimeError(f"{desc}: no reply")
        if b[0] == self.ACK:
            return
        if b[0] == self.NACK:
            raise RuntimeError(f"{desc}: NACK")
        raise RuntimeError(f"{desc}: reply 0x{b[0]:02X}")

    def _cmd(self, code, desc):
        self.ser.write(bytes((code, 0xFF ^ code)))
        self._ack(desc)

    @staticmethod
    def _addr_frame(addr):
        a = addr.to_bytes(4, "big")
        return a + bytes((a[0] ^ a[1] ^ a[2] ^ a[3],))

    def sync(self, stop_evt):
        """Send 0x7F repeatedly and detect the bootloader after the user sets BOOT0=1 and resets the device."""
        deadline = time.time() + FLASH_SYNC_WAIT
        while time.time() < deadline:
            if stop_evt.is_set():
                return False
            try:
                self.ser.reset_input_buffer()
                self.ser.write(b"\x7f")
                b = self.ser.read(1)
                if b and b[0] in (self.ACK, self.NACK):
                    return True
            except (serial.SerialException, OSError):
                return False
            time.sleep(0.05)
        return False

    def get_id(self):
        """Read the chip ID (F103C8T6 = 0x0410) to confirm bootloader mode."""
        self._cmd(0x02, "GET_ID")
        n = self.ser.read(1)
        if not n:
            raise RuntimeError("GET_ID: no reply")
        id_bytes = self.ser.read(n[0] + 1)
        if len(id_bytes) != n[0] + 1:
            raise RuntimeError("GET_ID: short reply")
        self._ack("GET_ID end")
        return int.from_bytes(id_bytes, "big")

    def probe(self):
        """Use GET (0x00) to read the bootloader version and detect support for extended erase command 0x44."""
        self._cmd(0x00, "GET")
        n = self.ser.read(1)
        if not n:
            raise RuntimeError("GET: no reply")
        version = self.ser.read(1)
        cmds = self.ser.read(n[0])
        if len(version) != 1 or len(cmds) != n[0]:
            raise RuntimeError("GET: short reply")
        self._ack("GET end")
        self.extended_erase = 0x44 in set(cmds)
        return version[0], set(cmds)

    def erase(self):
        """Mass erase: prefer extended erase 0x44 and fall back to classic erase 0x43 when the F103 bootloader does not support it."""
        if self.extended_erase:
            self._cmd(0x44, "EXTENDED_ERASE")
            self.ser.write(
                b"\xff\xff\x00"
            )  # Page count 0xFFFF requests the full chip, with checksum 0x00.
        else:
            self._cmd(0x43, "ERASE")
            self.ser.write(
                b"\xff\x00"
            )  # N=0xFF requests a mass erase, with checksum 0x00.
        self._ack("mass erase", timeout=FLASH_ERASE_TIMEOUT)

    def write_memory(self, addr, data):
        """Write one data block of at most 256 bytes, padding to a 4-byte boundary with 0xFF."""
        if len(data) % 4:
            data = data + b"\xff" * (4 - len(data) % 4)
        n = len(data)
        checksum = n - 1
        for b in data:
            checksum ^= b
        self._cmd(0x31, "WRITE_MEMORY")
        self.ser.write(self._addr_frame(addr))
        self._ack("write address")
        self.ser.write(bytes((n - 1,)) + data + bytes((checksum,)))
        self._ack("write data")

    def read_memory(self, addr, length):
        """Read back at most 256 bytes of memory for post-flash verification."""
        self._cmd(0x11, "READ_MEMORY")
        self.ser.write(self._addr_frame(addr))
        self._ack("read address")
        n = length - 1
        self.ser.write(bytes((n, n ^ 0xFF)))
        self._ack("read length")
        data = self.ser.read(length)
        if len(data) != length:
            raise RuntimeError("READ_MEMORY: short reply")
        return bytes(data)

    def go(self, addr=FLASH_APP_START):
        """Jump to and run the application firmware."""
        self._cmd(0x21, "GO")
        self.ser.write(self._addr_frame(addr))
        self._ack("GO address")


def _flash_chunks(addr, data):
    """Split data into 256-byte blocks, padding the final block to 4-byte alignment for both writing and verification."""
    off = 0
    while off < len(data):
        chunk = data[off : off + 256]
        if len(chunk) % 4:
            chunk = chunk + b"\xff" * (4 - len(chunk) % 4)
        yield addr + off, chunk
        off += len(chunk)


class FlashThread(threading.Thread):
    """Background flashing thread that leaves terminal refresh unblocked and writes progress to the log panel."""

    def __init__(self, state, port, hex_path, old_link):
        super().__init__(daemon=True)
        self.state = state
        self.port = port
        self.hex_path = hex_path
        self.old_link = old_link
        self._stop_evt = threading.Event()

    def abort(self):
        self._stop_evt.set()

    def run(self):
        s = self.state
        flasher = None
        try:
            if self.old_link:
                self.old_link.stop()
                try:
                    self.old_link.join(timeout=2.0)
                except Exception:
                    pass
            segments = parse_intel_hex(self.hex_path)
            total = sum(len(d) for _, d in segments)
            s.log(
                f"Flash {os.path.basename(self.hex_path)}: {len(segments)} segment(s), {total} B"
            )
            s.log(
                "Set BOOT0=1, press reset (or power cycle); auto-detecting bootloader..."
            )
            for _ in range(
                40
            ):  # Wait for the previous serial thread to release the port completely.
                try:
                    flasher = Stm32Flash(self.port)
                    break
                except (serial.SerialException, OSError):
                    time.sleep(0.25)
            if flasher is None:
                raise RuntimeError(
                    f"cannot open {self.port} (maybe busy by another program)"
                )
            try:
                if not flasher.sync(self._stop_evt):
                    raise RuntimeError(
                        "bootloader not responding: check BOOT0=1 and press reset"
                    )
                version, _cmds = flasher.probe()
                chip = flasher.get_id()
                erase_mode = "extended" if flasher.extended_erase else "classic"
                s.log(
                    f"Bootloader OK v{version}, chip ID 0x{chip:04X}, erase={erase_mode}"
                )
                flasher.erase()
                s.log("Mass erase OK, writing...")
                for addr, data in segments:
                    for a, chunk in _flash_chunks(addr, data):
                        if self._stop_evt.is_set():
                            raise RuntimeError("aborted by user")
                        flasher.write_memory(a, chunk)
                s.log("Writing done, verifying...")
                for addr, data in segments:
                    for a, chunk in _flash_chunks(addr, data):
                        if flasher.read_memory(a, len(chunk)) != chunk:
                            raise RuntimeError(f"verify mismatch at 0x{a:08X}")
                flasher.go()
                s.log(
                    f"Flash OK: {total} B verified, firmware running (BOOT0 back to 0 for next boot)"
                )
            finally:
                if flasher:
                    flasher.close()
        except Exception as e:
            s.log(f"Flash FAILED: {e}")
            s.log(
                "Hint: check BOOT0=1, pressed reset, port not occupied; then retry flash"
            )
        finally:
            with s.lock:
                s.flash_thread = None
                s.flash_new_serial = SerialLink(
                    s
                )  # Restore serial acquisition regardless of flashing success.
                s.flash_active = False
                s.flash_hex = ""
            s.flash_new_serial.start()


# ------------------------------------------------------------------ Interactive commands
def run_command(text, state, pid, serial_link, gl_link, player):
    quit_now = False
    text = text.strip()
    if not text:
        return serial_link, gl_link, player, quit_now
    parts = shlex.split(text, posix=False)
    cmd, args = parts[0].lower(), parts[1:]

    def need(n, usage):
        if len(args) < n:
            state.log(f"Usage: {usage}")
            return False
        return True

    if state.flash_active and cmd not in (
        "flash",
        "help",
        "?",
        "ports",
        "q",
        "quit",
        "exit",
        "record",
        "gains",
        "t_aim",
        "import",
    ):
        state.log(
            f"Flashing in progress: '{cmd}' disabled (type 'flash abort' to cancel)"
        )
        return serial_link, gl_link, player, quit_now

    if cmd in ("q", "quit", "exit"):
        quit_now = True
    elif cmd in ("help", "?"):
        state.log("help: pid on/off | t_aim X | gains Kp Ki | heater CH W | fan R1 R2")
        state.log("      import FILE | start [cooling] | stop | zero | record on/off")
        state.log("      connect | disconnect | port <port|auto> | ports | q")
        state.log(
            "      flash [hex] - flash STM32F1 via UART (default firmware if no arg)"
        )
        state.log("      cooling - send cooling cmd (heaters 0, fans 100%)")
    elif cmd == "ports":
        ports = serial.tools.list_ports.comports()
        if not ports:
            state.log("No COM ports found")
        for p in ports:
            state.log(f"  {p.device} - {p.description}  [{p.hwid}]")
    elif cmd == "port":
        if need(1, "port <port|auto> [baud] (e.g. COM5 or /dev/ttyUSB0)"):
            new_port = "AUTO" if args[0].upper() == "AUTO" else args[0]
            new_baud = int(args[1]) if len(args) > 1 else state.baud
            if serial_link:
                serial_link.stop()
            state.port, state.baud = new_port, new_baud
            serial_link = SerialLink(state)
            serial_link.start()
            state.log(f"Serial port switched to {new_port.upper()} @ {new_baud}")
    elif cmd == "connect":
        if serial_link is None or not serial_link.is_alive():
            serial_link = SerialLink(state)
            serial_link.start()
        if gl_link is None or not gl_link.is_alive():
            gl_link = ModbusLink(state)
            gl_link.start()
    elif cmd == "disconnect":
        if serial_link:
            serial_link.stop()
        if gl_link:
            gl_link.stop()
        serial_link = gl_link = None
        state.log("Serial / GL860A disconnected")
    elif cmd == "flash":
        if player and player.is_alive():
            state.log("A command is running: stop it first, then flash")
        elif state.flash_active:
            if args and args[0].lower() in ("abort", "cancel"):
                if state.flash_thread:
                    state.flash_thread.abort()
                    state.log("Flash abort requested...")
            else:
                state.log("Flashing in progress, please wait (flash abort to cancel)")
        else:
            try:
                path = resolve_hex_path(args[0] if args else None)
                port = (
                    state.port
                    if state.port and state.port.upper() != "AUTO"
                    else find_ch340_port()
                )
                if port is None:
                    state.log("CH340 not found: check USB connection and retry")
                    return serial_link, gl_link, player, quit_now
                state.flash_active = True
                state.flash_hex = os.path.basename(path)
                state.flash_thread = FlashThread(state, port, path, serial_link)
                state.flash_thread.start()
                state.log(f"Flashing started: {path}")
            except Exception as e:
                state.flash_active = False
                state.flash_hex = ""
                state.log(f"Flash error: {e}")
    elif cmd == "pid":
        if args and args[0].lower() in ("on", "1", "true"):
            state.t_aim = PID_START_T_AIM
            pid.set_t_aim(PID_START_T_AIM)
            state.pid_enabled = True
            pid.enable()
            state.log(f"PID enabled, T_aim = {state.t_aim:.1f} C")
        else:
            state.pid_enabled = False
            pid.disable()
            state.log("PID disabled")
    elif cmd == "t_aim":
        if need(1, "t_aim <target temp C>"):
            v = float(args[0])
            state.t_aim = v
            pid.set_t_aim(v)
            state.log(f"T_aim = {v:.1f} C")
    elif cmd == "gains":
        if need(2, "gains <Kp> <Ki>"):
            pid.set_pid_gains(float(args[0]), float(args[1]))
            state.log(f"PID gains: Kp = {args[0]}, Ki = {args[1]}")
    elif cmd in ("heater", "h"):
        if need(2, "heater <CH 1-12> <power W>"):
            ch, w = int(args[0]), float(args[1])
            if 1 <= ch <= 12:
                state.heater_targets[ch] = w
                send_mhf(state, serial_link)
                state.log(f"CH{ch} heater -> {w:.3f} W")
            else:
                state.log("CH must be 1-12")
    elif cmd in ("fan", "f"):
        if need(2, "fan <Fan1 RPM> <Fan2 RPM>"):
            state.fan_targets[1] = clamp_fan(args[0])
            state.fan_targets[2] = clamp_fan(args[1])
            send_mhf(state, serial_link)
            state.log(
                f"Fans -> {state.fan_targets[1]:.0f} / {state.fan_targets[2]:.0f} RPM"
            )
    elif cmd == "zero":
        zero_output(state, serial_link)
        state.log("All outputs zeroed")
    elif cmd == "record":
        if not args:
            state.log(
                f"Recording {'ON' if state.recording else 'OFF'}: {rec_summary(state)}"
            )
        elif args[0].lower() in ("on", "1", "true"):
            if not state.recording:
                rec_start(state, state.data_dir)
                state.recording = True
                state.log(
                    f"Recording started (3 files): "
                    f"{os.path.basename(state.rec_base)}_*"
                )
            else:
                state.log("Already recording")
        elif args[0].lower() in ("off", "0", "false"):
            state.recording = False
            state.log(f"Recording stopped: {rec_summary(state)}")
        else:
            # Retain the legacy record <seconds> argument; event-driven sampling no longer uses the interval.
            if not state.recording:
                rec_start(state, state.data_dir)
                state.recording = True
                state.log(
                    f"Recording started (3 files): "
                    f"{os.path.basename(state.rec_base)}_*"
                )
            else:
                state.log("Already recording")
    elif cmd == "import":
        if not args:
            state.log("Usage: import <command.csv>")
        else:
            try:
                rows, used = load_command_file(args[0])
                state.cmd_rows = rows
                state.cmd_path = used
                state.cmd_name = os.path.basename(used)
                state.log(f"Imported {used}: {len(rows)} steps, {rows[-1]['t']:.1f}s")
            except Exception as e:
                state.log(f"Import error: {e}")
                state.log(
                    "Hint: use relative path import Control_command.csv, or cd to program dir"
                )
    elif cmd == "start":
        want_cooling = bool(args) and args[0].lower() == "cooling"
        if player and player.is_alive():
            state.log("Command already running")
        elif want_cooling:
            # start cooling sends one cooling command without replaying commands; PID then controls the fans when enabled.
            send_cooling(state, serial_link)
        elif serial_link is None:
            state.log("Serial not connected")
        elif not state.cmd_rows:
            state.log("No command loaded (import FILE first)")
        else:
            if not state.recording:
                folder = (
                    os.path.dirname(os.path.abspath(state.cmd_path))
                    if state.cmd_path
                    else state.data_dir
                )
                rec_start(state, folder)
                state.recording = True
                state.auto_record = True
                state.log(
                    f"Auto recording started (3 files): "
                    f"{os.path.basename(state.rec_base)}_*"
                )
            player = CommandPlayer(
                state, serial_link, pid, list(state.cmd_rows), state.cmd_name
            )
            player.start()
    elif cmd == "cooling":
        if args and args[0].lower() in ("stop", "off"):
            state.log("Cooling is one-shot; fans follow PID from next cycle")
        else:
            send_cooling(state, serial_link)
    elif cmd == "stop":
        stopped = False
        if player and player.is_alive():
            player.stop()
            player.join(timeout=3)
            player = None
            stopped = True
        if not stopped:
            state.log("No command running")
    else:
        state.log(f"Unknown command: {cmd} (type help)")
    return serial_link, gl_link, player, quit_now


# ------------------------------------------------------------------ Main workflow
def parse_args():
    p = argparse.ArgumentParser(description="PID thermal control terminal")
    p.add_argument("command", nargs="?", help="command CSV to load at startup")
    p.add_argument(
        "--port",
        default="AUTO",
        help="STM32 serial port; AUTO = auto-find CH340 (default)",
    )
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--ip", default="192.168.0.2", help="GL860A IP")
    p.add_argument("--tcp-port", type=int, default=502)
    p.add_argument("--t-aim", type=float, default=40.0, help="target temperature C")
    p.add_argument(
        "--refresh", type=float, default=0.5, help="dashboard refresh seconds"
    )
    p.add_argument(
        "--pid-interval",
        type=float,
        default=0.1,
        help="PID compute interval seconds (default 0.1 = same as v3)",
    )
    p.add_argument(
        "--record",
        type=float,
        default=0.0,
        help="start data recording every N seconds (0=off)",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="data output directory (default: <script>/data)",
    )
    p.add_argument("--kp", type=float, default=None)
    p.add_argument("--ki", type=float, default=None)
    return p.parse_args()


def enable_ansi():
    if os.name == "nt":
        import ctypes

        k32 = ctypes.windll.kernel32
        k32.SetConsoleMode(k32.GetStdHandle(-11), 7)


def main():
    args = parse_args()
    enable_ansi()
    logging.getLogger("pymodbus").setLevel(
        logging.CRITICAL
    )  # Silence pymodbus console logs to keep the interface clean.
    state = SharedState(args.port, args.baud, args.ip, args.tcp_port, args.t_aim)
    if args.data_dir:
        state.data_dir = os.path.abspath(args.data_dir)
    pid = PIDFanController(t_aim=args.t_aim)
    state.pid = pid
    if args.kp is not None:
        pid.set_kp(args.kp)
    if args.ki is not None:
        pid.ki = args.ki

    serial_link = SerialLink(state)
    gl_link = ModbusLink(state)
    serial_link.start()
    gl_link.start()
    player = None
    ports = [p.device for p in serial.tools.list_ports.comports()]
    state.log("Available COM ports: " + ("; ".join(ports) if ports else "none"))
    state.log(f"Working dir: {os.getcwd()}")
    if state.port.upper() == "AUTO":
        state.log(
            "Serial -> AUTO (auto-finding CH340), GL860A -> {0}:{1}".format(
                state.ip, state.tcp_port
            )
        )
    else:
        state.log(
            f"Serial -> {state.port} @ {state.baud}, GL860A -> {state.ip}:{state.tcp_port}"
        )

    if args.command:
        try:
            state.cmd_rows, used = load_command_file(args.command)
            state.cmd_path = used
            state.cmd_name = os.path.basename(used)
            state.log(f"Command loaded: {state.cmd_name} ({len(state.cmd_rows)} steps)")
        except Exception as e:
            state.log(f"Import error: {e}")

    if args.record > 0:
        state.record_interval = max(0.1, args.record)
        rec_start(state, state.data_dir)
        state.recording = True
        state.log(f"Recording started (3 files): {os.path.basename(state.rec_base)}_*")

    term_old = None
    if sys.stdin.isatty():
        if msvcrt is not None:
            input_mode, buf = "msvcrt", []
        else:
            # Linux terminal: cbreak mode for character-by-character input without echo,
            # Preserve output-side \n-to-\r\n conversion to prevent staircase misalignment over SSH.
            import termios
            import tty

            fd = sys.stdin.fileno()
            term_old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            input_mode, buf = "posix", []
    else:
        # SSH/pipe input: poll in the main loop without a separate thread.
        input_mode, buf = "pipe", []

    quit_now = False
    last_pid = 0.0
    last_draw = 0.0
    try:
        while not quit_now:
            now = time.time()
            if input_mode == "msvcrt":
                for line in poll_input(buf):
                    if line == "__QUIT__":
                        quit_now = True
                    else:
                        try:
                            serial_link, gl_link, player, qn = run_command(
                                line, state, pid, serial_link, gl_link, player
                            )
                            quit_now = (
                                quit_now or qn
                            )  # Blank lines and later commands must not overwrite the exit state.
                        except (ValueError, IndexError) as e:
                            state.log(f"Command error: {e}")
            elif input_mode in ("posix", "pipe"):
                for line in poll_pipe(buf):
                    if line == "__QUIT__":
                        quit_now = True
                    else:
                        try:
                            serial_link, gl_link, player, qn = run_command(
                                line, state, pid, serial_link, gl_link, player
                            )
                            quit_now = (
                                quit_now or qn
                            )  # Blank lines and later commands must not overwrite the exit state.
                        except (ValueError, IndexError) as e:
                            state.log(f"Command error: {e}")

            with state.lock:  # Serial thread used to resume acquisition after flashing.
                if state.flash_new_serial is not None:
                    serial_link = state.flash_new_serial
                    state.flash_new_serial = None
            if state.pid_enabled and now - last_pid >= args.pid_interval:
                run_pid(state, pid, serial_link)
                last_pid = now
            if state.recording:  # Event-driven sampling for three output files.
                sample_temp(state)
                sample_heater(state)
            if now - last_draw >= args.refresh:
                draw(state, "".join(buf))
                last_draw = now
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        if term_old is not None:
            try:
                import termios

                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, term_old)
            except Exception:
                pass
        zero_output(state, serial_link)
        if player:
            player.stop()
            player.join(timeout=2)
        if serial_link:
            serial_link.stop()
            serial_link.join(timeout=2)
        if gl_link:
            gl_link.stop()
            gl_link.join(timeout=2)
        sys.stdout.write("\x1b[?25h\nExited.\n")


if __name__ == "__main__":
    main()
