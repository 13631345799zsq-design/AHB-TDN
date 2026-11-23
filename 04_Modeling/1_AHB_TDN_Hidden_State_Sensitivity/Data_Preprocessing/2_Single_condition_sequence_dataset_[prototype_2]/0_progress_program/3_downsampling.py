"""
Downsample aligned data from 0.1-second intervals to larger intervals and visualize the results.
"""
import os
import re

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def downsample(
    df: pd.DataFrame, interval: float, target_grid: np.ndarray = None
) -> pd.DataFrame:
    """
    Downsample a DataFrame to the specified interval in seconds.

    Linear interpolation on a uniform grid ensures a strictly regular time axis.
    """
    df = df.copy()
    df.set_index('Time(s)', inplace=True)
    df.sort_index(inplace=True)

    if target_grid is None:
        t_min = ceil_to_interval(df.index.min(), interval)
        t_max = floor_to_interval(df.index.max(), interval)
        target_grid = build_time_grid(t_min, t_max, interval)

    target_grid = np.asarray(target_grid)

    # Merge the original index with the target grid, interpolate, and select target points.
    combined_index = df.index.union(target_grid)
    df_reindexed = df.reindex(combined_index).sort_index()
    df_interp = df_reindexed.interpolate(method='index', limit_direction='both')
    df_result = df_interp.loc[target_grid].copy()

    df_result.reset_index(inplace=True)
    df_result.rename(columns={'index': 'Time(s)'}, inplace=True)

    expected_length = len(target_grid)
    if len(df_result) != expected_length:
        raise RuntimeError(
            f'下采样长度不一致: {len(df_result)} != {expected_length}, interval={interval}'
        )
    if df_result.isna().any().any():
        raise RuntimeError(f'下采样后仍存在 NaN, interval={interval}')

    return df_result


def plot_comparison(
    df_orig: pd.DataFrame, df_down: pd.DataFrame, interval: float, save_path: str
):
    """
    Plot a 2 x 2 comparison of 12-channel power and temperature before and after downsampling.
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    cmap = plt.get_cmap('tab20')

    for i in range(1, 13):
        power_col = f'CH{i}_P(W)'
        temp_col = f'CH{i}_Temp(C)'
        color = cmap(i - 1)

        # Top left: original heater power.
        axs[0, 0].plot(
            df_orig['Time(s)'],
            df_orig[power_col],
            label=f'CH{i}',
            color=color,
            alpha=0.7,
            linewidth=0.6,
        )
        # Top right: downsampled heater power.
        axs[0, 1].plot(
            df_down['Time(s)'],
            df_down[power_col],
            label=f'CH{i}',
            color=color,
            alpha=0.8,
            linewidth=0.8,
        )
        # Bottom left: original temperature.
        axs[1, 0].plot(
            df_orig['Time(s)'],
            df_orig[temp_col],
            label=f'CH{i}',
            color=color,
            alpha=0.7,
            linewidth=0.6,
        )
        # Bottom right: downsampled temperature.
        axs[1, 1].plot(
            df_down['Time(s)'],
            df_down[temp_col],
            label=f'CH{i}',
            color=color,
            alpha=0.8,
            linewidth=0.8,
        )

    # Ambient temperature.
    env_col = 'Ambient_Temp(C)'
    axs[1, 0].plot(
        df_orig['Time(s)'],
        df_orig[env_col],
        label='Ambient',
        color='black',
        alpha=0.8,
        linewidth=0.8,
    )
    axs[1, 1].plot(
        df_down['Time(s)'],
        df_down[env_col],
        label='Ambient',
        color='black',
        alpha=0.8,
        linewidth=0.8,
    )

    titles = [
        'Original Heater Power (0.1 s)',
        f'Downsampled Heater Power ({interval} s)',
        'Original Temperature (0.1 s)',
        f'Downsampled Temperature ({interval} s)',
    ]
    ylabels = ['Power (W)', 'Power (W)', 'Temperature (°C)', 'Temperature (°C)']

    for idx, ax in enumerate(axs.flat):
        ax.set_title(titles[idx], fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabels[idx])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper left', fontsize='x-small', ncol=3)

    plt.suptitle(
        f'Downsampling Comparison: 0.1 s → {interval} s',
        fontsize=14,
        fontweight='bold',
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  [绘图] 已保存: {os.path.basename(save_path)}')


def plot_all_intervals(df_orig: pd.DataFrame, results: dict, save_path: str):
    """
    Plot all downsampling intervals together, using CH1 for comparison.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    ch = 1
    power_col = f'CH{ch}_P(W)'
    temp_col = f'CH{ch}_Temp(C)'

    # Power comparison.
    axes[0].plot(
        df_orig['Time(s)'],
        df_orig[power_col],
        label='Original (0.1 s)',
        color='gray',
        alpha=0.5,
        linewidth=0.6,
    )
    # Temperature comparison.
    axes[1].plot(
        df_orig['Time(s)'],
        df_orig[temp_col],
        label='Original (0.1 s)',
        color='gray',
        alpha=0.5,
        linewidth=0.6,
    )

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for (interval, df_down), c in zip(results.items(), colors):
        axes[0].plot(
            df_down['Time(s)'],
            df_down[power_col],
            label=f'{interval} s',
            color=c,
            alpha=0.8,
            linewidth=0.8,
        )
        axes[1].plot(
            df_down['Time(s)'],
            df_down[temp_col],
            label=f'{interval} s',
            color=c,
            alpha=0.8,
            linewidth=0.8,
        )

    axes[0].set_title(f'CH{ch} Power — All Downsample Intervals', fontweight='bold')
    axes[0].set_ylabel('Power (W)')
    axes[0].legend(loc='upper left', fontsize='small')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].set_title(
        f'CH{ch} Temperature — All Downsample Intervals', fontweight='bold'
    )
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].legend(loc='upper left', fontsize='small')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  [绘图] 已保存: {os.path.basename(save_path)}')


def ceil_to_interval(value: float, interval: float) -> float:
    return np.ceil(value / interval) * interval


def floor_to_interval(value: float, interval: float) -> float:
    return np.floor(value / interval) * interval


def build_time_grid(t_min: float, t_max: float, interval: float) -> np.ndarray:
    n_points = int(round((t_max - t_min) / interval)) + 1
    return np.round(t_min + np.arange(n_points) * interval, 10)


def command_sort_key(name: str):
    match = re.search(r'Command_(\d+)$', name, flags=re.IGNORECASE)
    if match:
        return (0, int(match.group(1)))
    return (1, name)


def list_command_dirs(base_at_path: str):
    return sorted(
        [
            name
            for name in os.listdir(base_at_path)
            if os.path.isdir(os.path.join(base_at_path, name))
            and re.match(r'Command_\d+$', name, flags=re.IGNORECASE)
        ],
        key=command_sort_key,
    )


def collect_jobs():
    jobs = []
    for at_folder in AT_FOLDERS:
        base_at_path = os.path.join(BASE_DIR, at_folder)
        if not os.path.isdir(base_at_path):
            print(f'跳过: {at_folder} 文件夹不存在')
            continue

        for cmd_dir in list_command_dirs(base_at_path):
            cmd_path = os.path.join(base_at_path, cmd_dir)
            aligned_csv = os.path.join(cmd_path, 'dataset_aligned.csv')
            if not os.path.isfile(aligned_csv):
                print(f'  [跳过] {at_folder}/{cmd_dir}: 未找到 dataset_aligned.csv')
                continue

            jobs.append(
                {
                    'at_folder': at_folder,
                    'cmd_dir': cmd_dir,
                    'cmd_path': cmd_path,
                    'aligned_csv': aligned_csv,
                }
            )

    if not jobs:
        raise RuntimeError('未找到可处理的 dataset_aligned.csv')
    return jobs


def attach_time_bounds(jobs):
    for job in jobs:
        time_s = pd.read_csv(job['aligned_csv'], usecols=['Time(s)'])['Time(s)']
        job['raw_length'] = len(time_s)
        job['time_min'] = time_s.min()
        job['time_max'] = time_s.max()


def build_shared_target_grids(jobs):
    target_grids = {}
    for interval in DOWNSAMPLE_INTERVALS:
        starts = [ceil_to_interval(job['time_min'], interval) for job in jobs]
        ends = [floor_to_interval(job['time_max'], interval) for job in jobs]
        shared_start = max(starts)
        shared_end = min(ends)

        if shared_start > shared_end:
            raise ValueError(
                f'{interval}s 下采样没有所有文件共同覆盖的时间范围: '
                f'latest start={shared_start:.3f}, earliest end={shared_end:.3f}'
            )

        target_grids[interval] = build_time_grid(shared_start, shared_end, interval)
    return target_grids


# ==================== Configuration ====================
# Target downsampling intervals in seconds; adjust as needed.
DOWNSAMPLE_INTERVALS = [0.5, 1.0, 5.0]
AT_FOLDERS = ['AT_35', 'AT_40', 'AT_45']
# ================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Database/2_Single_condition_sequence_dataset_[prototype_2]'))


def main():
    jobs = collect_jobs()
    attach_time_bounds(jobs)
    target_grids = build_shared_target_grids(jobs)

    print(f'共找到 {len(jobs)} 个 dataset_aligned.csv')
    for interval, target_grid in target_grids.items():
        print(
            f'共享下采样网格 {interval}s: '
            f'{target_grid[0]:.1f} ~ {target_grid[-1]:.1f} s, '
            f'{len(target_grid)} 行'
        )

    for job in jobs:
        at_folder = job['at_folder']
        cmd_dir = job['cmd_dir']
        cmd_path = job['cmd_path']
        print(f'\n{"="*60}')
        print(f'处理: {at_folder}/{cmd_dir}')
        print(f'{"="*60}')

        df_orig = pd.read_csv(job['aligned_csv'])
        print(
            f'  原始数据: {len(df_orig)} 行, 时间间隔 0.1 s, '
            f'时间范围 {df_orig["Time(s)"].min():.1f} ~ {df_orig["Time(s)"].max():.1f} s'
        )

        results = {}

        for interval in DOWNSAMPLE_INTERVALS:
            print(f'\n--- 下采样到 {interval} s ---')
            df_down = downsample(df_orig, interval, target_grids[interval])
            results[interval] = df_down

            # Save the CSV file in the corresponding Command folder.
            fname = f'dataset_aligned_{interval}s.csv'
            save_csv = os.path.join(cmd_path, fname)
            df_down.to_csv(save_csv, index=False)
            print(
                f'  [保存] {fname}  ({len(df_down)} 行, 压缩比 {len(df_orig)/len(df_down):.1f}x)'
            )

            # Plot the comparison for this interval.
            plot_name = f'compare_{interval}s.png'
            plot_comparison(
                df_orig, df_down, interval, os.path.join(cmd_path, plot_name)
            )

        # Plot the combined comparison for all intervals.
        print('\n--- 绘制汇总对比图 ---')
        plot_all_intervals(
            df_orig, results, os.path.join(cmd_path, 'compare_all_intervals.png')
        )

        # Print summary statistics.
        print('\n========== 下采样汇总 ==========')
        print(f'{"间隔 (s)":<12} {"行数":<10} {"压缩比":<10}')
        print(f'{"0.1 (原始)":<12} {len(df_orig):<10} {"1.0x":<10}')
        for interval, df_down in results.items():
            ratio = len(df_orig) / len(df_down)
            print(f'{interval:<12} {len(df_down):<10} {ratio:.1f}x')
        print('================================')

    print('\n全部下采样处理完成！')


if __name__ == '__main__':
    main()
