"""
Analyze repeatability across ambient-temperature conditions.

For each Command under AT_35, AT_40, and AT_45, load dataset_aligned.csv,
subtract ambient temperature from the heat-source temperatures, plot the
temperature changes, and compare temperature evolution between Commands
using R-squared.
"""
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ==================== Configuration ====================
AT_FOLDERS = ['AT_35', 'AT_40', 'AT_45']
NUM_CHANNELS = 12
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Database/1_Multi_condition_sequence_dataset_[prototype_1]'))
# ==================================================


def load_command_data(at_dir: str) -> dict:
    """
    Load dataset_aligned.csv from every Command_X folder under an AT folder.

    Return {command_name: df} with ambient temperature subtracted from each
    heat-source temperature.
    """
    commands = {}
    if not os.path.isdir(at_dir):
        return commands

    for sub in sorted(os.listdir(at_dir)):
        sub_path = os.path.join(at_dir, sub)
        if not os.path.isdir(sub_path) or not sub.startswith('Command'):
            continue
        csv_path = os.path.join(sub_path, 'dataset_aligned.csv')
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)

        # Subtract ambient temperature to obtain delta T.
        ambient_col = 'Ambient_Temp(C)'
        for ch in range(1, NUM_CHANNELS + 1):
            temp_col = f'CH{ch}_Temp(C)'
            if temp_col in df.columns and ambient_col in df.columns:
                df[temp_col] = df[temp_col] - df[ambient_col]

        commands[sub] = df
    return commands


def compute_r2_between_commands(cmds: dict, channel: int) -> pd.DataFrame:
    """
    Compute the R-squared value for one channel between every pair of Commands.

    Align the time axes before computing each value.
    """
    names = list(cmds.keys())
    n = len(names)
    r2_matrix = pd.DataFrame(np.nan, index=names, columns=names)

    for i, j in itertools.combinations(range(n), 2):
        name_a, name_b = names[i], names[j]
        df_a, df_b = cmds[name_a], cmds[name_b]
        col = f'CH{channel}_Temp(C)'

        # Interpolate both curves over their shared time range.
        t_min = max(df_a['Time(s)'].min(), df_b['Time(s)'].min())
        t_max = min(df_a['Time(s)'].max(), df_b['Time(s)'].max())
        # Use the larger time step of the two curves.
        dt_a = df_a['Time(s)'].diff().median()
        dt_b = df_b['Time(s)'].diff().median()
        dt = max(dt_a, dt_b)
        t_grid = np.arange(t_min, t_max + dt / 2, dt)

        y_a = np.interp(t_grid, df_a['Time(s)'].values, df_a[col].values)
        y_b = np.interp(t_grid, df_b['Time(s)'].values, df_b[col].values)

        r2 = r2_score(y_a, y_b)
        r2_matrix.loc[name_a, name_b] = r2
        r2_matrix.loc[name_b, name_a] = r2

    # Set the diagonal to 1.
    for name in names:
        r2_matrix.loc[name, name] = 1.0
    return r2_matrix


def plot_all_channels_overview(all_at_data: dict, cmd_name: str, save_path: str):
    """
    Plot all 12 channels for one Command in a 4 x 3 overview.

    Each subplot overlays the delta-T curves for AT_35, AT_40, and AT_45.
    all_at_data: {at_name: {command_name: df}}
    """
    fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True)
    cmap = plt.get_cmap('tab10')

    for ch in range(1, NUM_CHANNELS + 1):
        row = (ch - 1) // 3
        col = (ch - 1) % 3
        ax = axes[row][col]

        for at_idx, (at_name, commands) in enumerate(all_at_data.items()):
            if cmd_name not in commands:
                continue
            df = commands[cmd_name]
            color = cmap(at_idx % 10)
            temp_col = f'CH{ch}_Temp(C)'
            if temp_col in df.columns:
                ax.plot(
                    df['Time(s)'],
                    df[temp_col],
                    label=at_name,
                    color=color,
                    alpha=0.85,
                    linewidth=0.7,
                )

        ax.set_title(f'CH{ch}', fontsize=10, fontweight='bold')
        ax.set_ylabel('ΔT (°C)', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper left', fontsize=6)

    for col in range(3):
        axes[-1, col].set_xlabel('Time (s)')

    fig.suptitle(
        f'{cmd_name} — All Channels ΔTemperature (T - T_ambient)',
        fontsize=14,
        fontweight='bold',
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==================== Main workflow ====================
if __name__ == '__main__':
    # --- Step 1: collect data from all AT folders. ---
    all_at_data = {}  # {at_name: {command_name: df}}

    for at_folder in AT_FOLDERS:
        at_dir = os.path.join(BASE_DIR, at_folder)
        print(f'\n{"="*60}')
        print(f'加载: {at_folder}')
        print(f'{"="*60}')

        commands = load_command_data(at_dir)
        if not commands:
            print(f'  [跳过] {at_folder} 下没有有效的 Command 数据')
            continue

        all_at_data[at_folder] = commands
        print(f'  找到 {len(commands)} 个 Command: {list(commands.keys())}')
        for name, df in commands.items():
            print(
                f'    {name}: {len(df)} 行, '
                f'时间 {df["Time(s)"].min():.1f} ~ {df["Time(s)"].max():.1f} s'
            )

    if not all_at_data:
        print('没有找到任何有效数据，退出。')
    else:
        # --- Step 2: plot a 4 x 3 overview for each Command. ---
        all_commands = set()
        for commands in all_at_data.values():
            all_commands.update(commands.keys())

        for cmd_name in sorted(all_commands):
            overview_path = os.path.join(BASE_DIR, f'{cmd_name}_channels_overview.png')
            plot_all_channels_overview(all_at_data, cmd_name, overview_path)
            print(f'[绘图] 已保存: {overview_path}')

    print('\n全部处理完成！')
