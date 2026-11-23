"""Generate randomized steady or transient operating commands with LHS.

The script combines the two command-generation procedures used in the study.
Set ``GENERATION_MODE`` to:

* ``"transient"`` for complex, independently timed step combinations; or
* ``"steady"`` for one operating point held over the entire experiment.

Each command contains 12 heater-power channels and two fan-speed channels.
Latin hypercube sampling (LHS) distributes the sampled values over each
channel's admissible range.  Generated commands are compared with existing
commands in the selected output directory before they are saved.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

# Main switch: use "transient" for complex combinations or "steady" for a
# constant operating point.
GENERATION_MODE = "transient"

NUM_COMMANDS = 9
TOTAL_TIME = 1800.0                 # Experiment duration [s].
DT = 0.2                            # Transient-command sampling interval [s].
MAX_SEGMENT_RANGE = (400.0, 1000.0) # LHS range of maximum segment length [s].
SEGMENT_DURATION_RATIO = 2.0        # maximum duration / minimum duration.
SIMILARITY_THRESHOLD = 0.985
TEMPORAL_BINS = 200
MAX_ATTEMPTS = 1000

ROOT = Path(__file__).resolve().parent
OUTPUT_DIRS = {
    "transient": ROOT / "transient_commands",
    "steady": ROOT / "steady_commands",
}

CHANNELS = [f"CH{i}" for i in range(1, 13)] + ["FAN1", "FAN2"]
CSV_HEADERS = ["Time", *CHANNELS]

# Channels in one part share the same switching times. Different parts use
# independently generated switching times to create asynchronous combinations.
PARTS = (
    {
        "channels": ("CH1",),
        "lower": (0,), "upper": (30,),
        "title": "Part 1: CH1 (heater 1)",
        "ylabel": "Power (W)", "ylim": (-1, 33),
    },
    {
        "channels": ("CH2",),
        "lower": (0,), "upper": (10,),
        "title": "Part 2: CH2 (heater 2)",
        "ylabel": "Power (W)", "ylim": (-0.5, 11),
    },
    {
        "channels": ("CH3", "CH4", "CH5", "CH6"),
        "lower": (0, 0, 0, 0), "upper": (6, 6, 6, 6),
        "title": "Part 3: CH3-CH6 (heaters 3-6)",
        "ylabel": "Power (W)", "ylim": (-0.3, 7),
    },
    {
        "channels": ("CH7", "CH8", "CH9", "CH10", "CH11", "CH12"),
        "lower": (0, 0, 0, 0, 0, 0), "upper": (3, 3, 3, 3, 3, 3),
        "title": "Part 4: CH7-CH12 (heaters 7-12)",
        "ylabel": "Power (W)", "ylim": (-0.2, 3.5),
    },
    {
        "channels": ("FAN1", "FAN2"),
        "lower": (0, 0), "upper": (6300, 6300),
        "title": "Part 5: FAN1-FAN2 ({0} or [1000, 6300] RPM)",
        "ylabel": "Speed (RPM)", "ylim": (-500, 7000),
    },
)

CHANNEL_SCALES = np.array(
    [30, 10, *([6] * 4), *([3] * 6), 6300, 6300], dtype=float
)
COLORS = (
    "royalblue", "darkorange", "forestgreen", "crimson",
    "mediumorchid", "teal", "sienna", "slategray",
)


def lhs(n_samples, lower, upper):
    """Return an LHS design scaled from the unit cube to physical limits."""
    unit_samples = qmc.LatinHypercube(d=len(lower)).random(n=n_samples)
    return qmc.scale(unit_samples, lower, upper)


def segment_duration_design(n_commands):
    """Assign every transient command an LHS-sampled duration range."""
    maximum = lhs(n_commands, [MAX_SEGMENT_RANGE[0]],
                  [MAX_SEGMENT_RANGE[1]]).ravel()
    maximum = np.round(maximum, 2)
    return [(round(value / SEGMENT_DURATION_RATIO, 2), value)
            for value in maximum]


def segment_boundaries(minimum, maximum):
    """Create segment boundaries that exactly cover [0, TOTAL_TIME]."""
    boundaries = [0.0]
    while TOTAL_TIME - boundaries[-1] > maximum:
        remaining = TOTAL_TIME - boundaries[-1]
        duration = np.random.uniform(minimum, min(maximum, remaining - minimum))
        boundaries.append(round(boundaries[-1] + duration, 2))
    boundaries.append(TOTAL_TIME)
    return boundaries


def format_samples(samples, channels):
    """Apply the resolution and physical dead band used by the actuators."""
    if channels[0].startswith("FAN"):
        samples = np.round(samples)
        samples[(samples > 0) & (samples < 1000)] = 0
    else:
        samples = np.round(samples, 2)
    return samples


def build_step_signal(time, boundaries, values):
    """Map one value per segment onto the uniformly sampled time vector."""
    # searchsorted returns the segment immediately to the left of each time.
    # Clipping assigns the final endpoint to the final segment.
    indices = np.searchsorted(boundaries[1:], time, side="right")
    return values[np.minimum(indices, len(values) - 1)]


def generate_command(mode, duration_range=None):
    """Generate one command table for the requested operating mode."""
    if mode == "steady":
        time = np.array([0.0, TOTAL_TIME])
    else:
        time = np.round(np.arange(0.0, TOTAL_TIME + DT / 2, DT), 1)

    data = {"Time": time}
    for part in PARTS:
        if mode == "steady":
            boundaries = [0.0, TOTAL_TIME]
        else:
            boundaries = segment_boundaries(*duration_range)

        n_segments = len(boundaries) - 1
        samples = lhs(n_segments, part["lower"], part["upper"])
        samples = format_samples(samples, part["channels"])

        for column, channel in enumerate(part["channels"]):
            if mode == "steady":
                data[channel] = np.repeat(samples[0, column], len(time))
            else:
                data[channel] = build_step_signal(
                    time, boundaries, samples[:, column]
                )

    return pd.DataFrame(data, columns=CSV_HEADERS)


def normalized_values(command):
    """Normalize all actuator values by their physical upper limits."""
    return command[CHANNELS].to_numpy(dtype=float) / CHANNEL_SCALES


def command_feature(command, mode):
    """Represent a command for cosine-similarity duplicate detection.

    A steady command is represented by its 14 normalized actuator values.
    A transient command is first averaged into temporal bins so that both its
    operating values and switching sequence contribute to the comparison.
    """
    values = normalized_values(command)
    if mode == "steady":
        return values.mean(axis=0)

    bin_edges = np.linspace(0, len(values), TEMPORAL_BINS + 1, dtype=int)
    binned = [values[start:end].mean(axis=0)
              for start, end in zip(bin_edges[:-1], bin_edges[1:])]
    return np.asarray(binned, dtype=np.float32).ravel()


def command_files(output_dir):
    """Return existing command CSV files ordered by their numeric suffix."""
    pattern = re.compile(r"command_(\d+)\.csv$")
    indexed = []
    for path in output_dir.glob("command_*.csv"):
        match = pattern.fullmatch(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    return sorted(indexed)


def load_features(output_dir, mode):
    """Load existing commands once and cache their comparison features."""
    features = []
    for number, path in command_files(output_dir):
        feature = command_feature(pd.read_csv(path), mode)
        features.append((number, feature, np.linalg.norm(feature)))
    return features


def similarity_result(command, existing_features, mode):
    """Return the closest command and whether the threshold is exceeded."""
    feature = command_feature(command, mode)
    norm = np.linalg.norm(feature)
    similarities = [
        (float(np.dot(feature, old_feature) / (norm * old_norm)), number)
        for number, old_feature, old_norm in existing_features
    ]
    maximum, closest = max(similarities, default=(-1.0, None))
    return maximum > SIMILARITY_THRESHOLD, closest, maximum, feature, norm


def plot_detail(command, path, mode):
    """Save a five-panel plot showing every actuator channel."""
    time = command["Time"].to_numpy()
    figure, axes = plt.subplots(5, 1, figsize=(8, 15), sharex=True)

    for index, (axis, part) in enumerate(zip(axes, PARTS)):
        for column, channel in enumerate(part["channels"]):
            axis.step(time, command[channel], where="post", label=channel,
                      color=COLORS[column % len(COLORS)], linewidth=1.2)
        if len(part["channels"]) > 1:
            axis.legend(loc="upper right", ncol=len(part["channels"]), fontsize=9)
        axis.set(title=part["title"], ylabel=part["ylabel"], ylim=part["ylim"])
        axis.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(f"Command Profile ({mode.capitalize()} LHS)",
                    fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_overview(command, path, mode):
    """Save a compact overview separating heater power and fan speed."""
    time = command["Time"].to_numpy()
    figure, (power_axis, fan_axis) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True
    )

    for index, channel in enumerate(CHANNELS[:12]):
        power_axis.step(time, command[channel], where="post", label=channel,
                        color=COLORS[index % len(COLORS)], linewidth=1.0)
    power_axis.set(title="All Power Channels", ylabel="Power (W)")
    power_axis.legend(loc="upper right", ncol=6, fontsize=8)
    power_axis.grid(alpha=0.3)

    for channel, color, style in (("FAN1", "red", "-"),
                                  ("FAN2", "firebrick", "--")):
        fan_axis.step(time, command[channel], where="post", label=channel,
                      color=color, linestyle=style, linewidth=1.5)
    fan_axis.set(title="Fan Channels", xlabel="Time (s)", ylabel="Speed (RPM)",
                 ylim=(-500, 7000))
    fan_axis.legend(loc="upper right")
    fan_axis.grid(alpha=0.3)

    figure.suptitle(f"Command Overview ({mode.capitalize()} LHS)",
                    fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    """Generate, screen, plot, and save a batch of unique commands."""
    mode = GENERATION_MODE.lower()
    if mode not in OUTPUT_DIRS:
        raise ValueError('GENERATION_MODE must be "transient" or "steady".')

    output_dir = OUTPUT_DIRS[mode]
    output_dir.mkdir(exist_ok=True)
    existing_files = command_files(output_dir)
    existing_features = load_features(output_dir, mode)
    next_number = existing_files[-1][0] + 1 if existing_files else 1
    duration_design = (segment_duration_design(NUM_COMMANDS)
                       if mode == "transient" else [None] * NUM_COMMANDS)

    print(f"Mode: {mode}")
    print(f"Loaded {len(existing_features)} existing commands from {output_dir}.")
    print(f"Generating {NUM_COMMANDS} new commands, starting at {next_number:04d}.\n")

    generated = 0
    for offset, duration_range in enumerate(duration_design):
        number = next_number + offset
        for attempt in range(1, MAX_ATTEMPTS + 1):
            command = generate_command(mode, duration_range)
            duplicate, closest, similarity, feature, norm = similarity_result(
                command, existing_features, mode
            )
            print(f"[command_{number:04d}] attempt {attempt}: "
                  f"maximum similarity = {similarity:.6f}")
            if duplicate:
                print(f"  Too similar to command_{closest:04d}; regenerating.")
                continue

            stem = output_dir / f"command_{number:04d}"
            command.to_csv(stem.with_suffix(".csv"), index=False)
            plot_detail(command, stem.with_name(stem.name + "_detail.png"), mode)
            plot_overview(command, stem.with_name(stem.name + "_overview.png"), mode)
            existing_features.append((number, feature, norm))
            generated += 1
            break
        else:
            print(f"[command_{number:04d}] skipped after {MAX_ATTEMPTS} attempts.")

    print(f"\nFinished: {generated} commands generated in {output_dir}.")


if __name__ == "__main__":
    main()
