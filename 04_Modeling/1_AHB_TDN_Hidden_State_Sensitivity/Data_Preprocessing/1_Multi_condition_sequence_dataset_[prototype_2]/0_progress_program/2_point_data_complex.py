import os
import pickle
import re

import numpy as np
import pandas as pd

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Database/1_Multi_condition_sequence_dataset_[prototype_2]/AT_40'))
SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Database/0_Dataset/Data_AT40_D2_05'))
SOURCE_CSV = 'dataset_aligned_0.5s.csv'
SAMPLE_INTERVAL = 0.5

TEMP_MAX, TEMP_MIN = 100, 20
FLUX_MAX, FLUX_MIN = 0.03, 0
COOLING_MAX, COOLING_MIN = 6300, 0


def command_sort_key(path):
    name = os.path.basename(path)
    match = re.search(r'Command_(\d+)$', name, flags=re.IGNORECASE)
    if match:
        return (0, int(match.group(1)))
    return (1, name)


def command_number(path):
    match = re.search(r'Command_(\d+)$', os.path.basename(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f'无法从目录名解析 Command 编号: {path}')
    return int(match.group(1))


def collect_jobs():
    jobs = []
    for name in os.listdir(BASE_PATH):
        cmd_path = os.path.join(BASE_PATH, name)
        if not os.path.isdir(cmd_path):
            continue
        if not re.match(r'Command_\d+$', name, flags=re.IGNORECASE):
            continue

        csv_path = os.path.join(cmd_path, SOURCE_CSV)
        if not os.path.isfile(csv_path):
            print(f'[跳过] {name}: 未找到 {SOURCE_CSV}')
            continue

        jobs.append(
            {
                'command': name,
                'command_no': command_number(cmd_path),
                'cmd_path': cmd_path,
                'csv_path': csv_path,
            }
        )

    jobs = sorted(jobs, key=lambda job: command_sort_key(job['cmd_path']))
    if not jobs:
        raise RuntimeError(f'未找到可处理的 {SOURCE_CSV}')
    return jobs


def ceil_to_interval(value, interval):
    return np.ceil(value / interval) * interval


def floor_to_interval(value, interval):
    return np.floor(value / interval) * interval


def build_time_grid(t_min, t_max, interval):
    n_points = int(round((t_max - t_min) / interval)) + 1
    return np.round(t_min + np.arange(n_points) * interval, 10)


def attach_time_bounds(jobs):
    for job in jobs:
        time_s = pd.read_csv(job['csv_path'], usecols=['Time(s)'])['Time(s)']
        job['time_min'] = time_s.min()
        job['time_max'] = time_s.max()
        job['raw_length'] = len(time_s)


def build_shared_time_grid(jobs):
    starts = [ceil_to_interval(job['time_min'], SAMPLE_INTERVAL) for job in jobs]
    ends = [floor_to_interval(job['time_max'], SAMPLE_INTERVAL) for job in jobs]
    shared_start = max(starts)
    shared_end = min(ends)

    if shared_start > shared_end:
        raise ValueError(
            '所有文件没有共同覆盖的时间范围: '
            f'latest start={shared_start:.3f}, earliest end={shared_end:.3f}'
        )

    return build_time_grid(shared_start, shared_end, SAMPLE_INTERVAL)


def align_to_grid(df, target_time_grid):
    df = df.copy()
    df.set_index('Time(s)', inplace=True)
    df.sort_index(inplace=True)

    combined_index = df.index.union(target_time_grid)
    df_aligned = (
        df.reindex(combined_index)
        .sort_index()
        .interpolate(method='index', limit_direction='both')
        .loc[target_time_grid]
        .copy()
    )
    df_aligned.reset_index(inplace=True)
    df_aligned.rename(columns={'index': 'Time(s)'}, inplace=True)

    if len(df_aligned) != len(target_time_grid):
        raise RuntimeError(f'对齐后长度异常: {len(df_aligned)} != {len(target_time_grid)}')
    if df_aligned.isna().any().any():
        raise RuntimeError('对齐后仍存在 NaN')
    return df_aligned


def require_columns(df, job):
    required_cols = ['Time(s)', 'Ambient_Temp(C)']
    required_cols += [f'CH{i}_P(W)' for i in range(1, 13)]
    required_cols += [f'Fan{i}_RPM' for i in range(1, 3)]
    required_cols += [f'CH{i}_Temp(C)' for i in range(1, 13)]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"{job['command']} 缺少列: {missing}")


def build_pkl_data(df):
    flux_scales = np.array([30, 10, 6, 6, 6, 6, 3, 3, 3, 3, 3, 3], dtype=np.float32)
    heater_areas = np.array(
        [
            50 * 50,
            30 * 30,
            20 * 20,
            20 * 20,
            20 * 20,
            20 * 20,
            10 * 10,
            10 * 10,
            10 * 10,
            10 * 10,
            10 * 10,
            10 * 10,
        ],
        dtype=np.float32,
    )

    power_cols = [f'CH{i}_P(W)' for i in range(1, 13)]
    cooling_cols = [f'Fan{i}_RPM' for i in range(1, 3)]
    temp_cols = [f'CH{i}_Temp(C)' for i in range(1, 13)]

    time_list = df['Time(s)'].values.astype(np.float32)
    envtemp = df['Ambient_Temp(C)'].values.astype(np.float32)
    power = df[power_cols].values.astype(np.float32)
    cooling = df[cooling_cols].values.astype(np.float32)
    temp = df[temp_cols].values.astype(np.float32)

    flux = flux_scales * power / (100 * heater_areas)
    flux_list = (flux - FLUX_MIN) / (FLUX_MAX - FLUX_MIN)
    cooling_list = (cooling - COOLING_MIN) / (COOLING_MAX - COOLING_MIN)
    envtemp_list = (envtemp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
    temp_list = (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)

    n_steps = len(time_list)
    lengths = {
        'time_list': len(time_list),
        'flux_list': len(flux_list),
        'cooling_list': len(cooling_list),
        'envtemp_list': len(envtemp_list),
        'temp_list': len(temp_list),
    }
    if set(lengths.values()) != {n_steps}:
        raise RuntimeError(f'pkl 字段长度不一致: {lengths}')

    return {
        'time_list': np.float32(time_list),
        'flux_list': np.float32(flux_list),
        'cooling_list': np.float32(cooling_list),
        'envtemp_list': np.float32(envtemp_list),
        'temp_steady_list': np.float32([]),
        'temp_list': np.float32(temp_list),
    }


def main():
    jobs = collect_jobs()
    attach_time_bounds(jobs)
    target_time_grid = build_shared_time_grid(jobs)
    expected_length = len(target_time_grid)

    os.makedirs(SAVE_PATH, exist_ok=True)
    print(
        f'共享 pkl 时间轴: {target_time_grid[0]:.1f} ~ {target_time_grid[-1]:.1f} s, '
        f'{expected_length} 行'
    )

    for job in jobs:
        df_data = pd.read_csv(job['csv_path'])
        require_columns(df_data, job)
        df_data = align_to_grid(df_data, target_time_grid)
        data = build_pkl_data(df_data)

        if len(data['time_list']) != expected_length:
            raise RuntimeError(
                f"{job['command']} pkl 长度异常: {len(data['time_list'])} != {expected_length}"
            )
        if not np.allclose(data['time_list'], target_time_grid.astype(np.float32)):
            raise RuntimeError(f"{job['command']} pkl 时间轴未对齐")

        save_file = os.path.join(SAVE_PATH, f"{job['command_no']}.pkl")
        with open(save_file, 'wb') as f:
            pickle.dump(data, f)

        print(
            f"{job['command']} -> {os.path.basename(save_file)}, "
            f"length={len(data['time_list'])}, "
            f"time={data['time_list'][0]:.1f}~{data['time_list'][-1]:.1f}s"
        )


if __name__ == '__main__':
    main()
