import os
import re

import numpy as np
import pandas as pd

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Database/2_Single_condition_sequence_dataset_[prototype_2]'))
AT_FOLDERS = ["AT_35", "AT_40", "AT_45"]
SAMPLE_INTERVAL = 0.1
OUTPUT_FILE = "dataset_aligned.csv"


def command_sort_key(path):
    name = os.path.basename(path)
    match = re.search(r"Command_(\d+)$", name, flags=re.IGNORECASE)
    if match:
        return (0, int(match.group(1)))
    return (1, name)


def list_command_dirs(at_path):
    command_dirs = []
    for name in os.listdir(at_path):
        full_path = os.path.join(at_path, name)
        if os.path.isdir(full_path) and re.match(
            r"Command_\d+$", name, flags=re.IGNORECASE
        ):
            command_dirs.append(full_path)
    return sorted(command_dirs, key=command_sort_key)


def find_source_csv(cmd_path, keyword):
    keyword = keyword.lower()
    matches = sorted(
        os.path.join(cmd_path, name)
        for name in os.listdir(cmd_path)
        if name.lower().endswith(".csv") and keyword in name.lower()
    )
    if not matches:
        raise FileNotFoundError(f"No *{keyword}*.csv found in {cmd_path}")
    if len(matches) > 1:
        print(
            f"  Warning: multiple *{keyword}*.csv files in {cmd_path}, using {matches[0]}"
        )
    return matches[0]


def collect_jobs():
    jobs = []
    for at_folder in AT_FOLDERS:
        at_path = os.path.join(BASE_PATH, at_folder)
        if not os.path.isdir(at_path):
            print(f"Skip missing folder: {at_folder}")
            continue

        for cmd_path in list_command_dirs(at_path):
            jobs.append(
                {
                    "at_folder": at_folder,
                    "command": os.path.basename(cmd_path),
                    "cmd_path": cmd_path,
                    "heater_csv": find_source_csv(cmd_path, "heater"),
                    "temp_csv": find_source_csv(cmd_path, "temp"),
                }
            )

    if not jobs:
        raise RuntimeError("No Command_* folders found.")
    return jobs


def ceil_to_interval(value, interval):
    return np.ceil(value / interval) * interval


def floor_to_interval(value, interval):
    return np.floor(value / interval) * interval


def build_time_grid(t_min, t_max, interval):
    n_points = int(round((t_max - t_min) / interval)) + 1
    return np.round(t_min + np.arange(n_points) * interval, 10)


def read_time_bounds(job):
    heater_time = pd.read_csv(job["heater_csv"], usecols=["Time(s)"])
    temp_time = pd.read_csv(job["temp_csv"], usecols=["Time(s)"])

    raw_min = min(heater_time["Time(s)"].min(), temp_time["Time(s)"].min())
    raw_max = max(heater_time["Time(s)"].max(), temp_time["Time(s)"].max())
    return (
        ceil_to_interval(raw_min, SAMPLE_INTERVAL),
        floor_to_interval(raw_max, SAMPLE_INTERVAL),
    )


def build_shared_time_grid(jobs):
    starts = []
    ends = []

    for job in jobs:
        start, end = read_time_bounds(job)
        job["aligned_start"] = start
        job["aligned_end"] = end
        starts.append(start)
        ends.append(end)

    shared_start = max(starts)
    shared_end = min(ends)
    if shared_start > shared_end:
        raise ValueError(
            "No shared time range across all files: "
            f"latest start={shared_start:.3f}, earliest end={shared_end:.3f}"
        )

    return build_time_grid(shared_start, shared_end, SAMPLE_INTERVAL)


def load_and_align(job, target_time_grid):
    df_heater = pd.read_csv(job["heater_csv"])
    df_temp = pd.read_csv(job["temp_csv"])

    df_heater.set_index("Time(s)", inplace=True)
    df_temp.set_index("Time(s)", inplace=True)

    df_merged = pd.concat([df_heater, df_temp], axis=1)
    df_merged.sort_index(inplace=True)
    df_interpolated = df_merged.interpolate(method="index", limit_direction="both")

    combined_index = df_interpolated.index.union(target_time_grid)
    df_final = (
        df_interpolated.reindex(combined_index)
        .sort_index()
        .interpolate(method="index", limit_direction="both")
        .loc[target_time_grid]
        .copy()
    )

    df_final.reset_index(inplace=True)
    df_final.rename(columns={"index": "Time(s)"}, inplace=True)

    expected_length = len(target_time_grid)
    if len(df_final) != expected_length:
        raise RuntimeError(
            f"{job['at_folder']}/{job['command']} length mismatch: "
            f"{len(df_final)} != {expected_length}"
        )
    if df_final.isna().any().any():
        raise RuntimeError(
            f"{job['at_folder']}/{job['command']} still contains NaN after alignment."
        )

    return df_final


def main():
    jobs = collect_jobs()
    target_time_grid = build_shared_time_grid(jobs)
    expected_length = len(target_time_grid)

    print(
        "Shared target grid: "
        f"{target_time_grid[0]:.1f}s -> {target_time_grid[-1]:.1f}s, "
        f"interval={SAMPLE_INTERVAL}s, rows={expected_length}"
    )

    for job in jobs:
        print(f"Processing: {job['at_folder']}/{job['command']}")
        df_final = load_and_align(job, target_time_grid)
        df_final.to_csv(os.path.join(job["cmd_path"], OUTPUT_FILE), index=False)
        print(f"  Saved {OUTPUT_FILE}, rows={len(df_final)}")

    print(f"All files aligned with the same sequence length: {expected_length}")


if __name__ == "__main__":
    main()
