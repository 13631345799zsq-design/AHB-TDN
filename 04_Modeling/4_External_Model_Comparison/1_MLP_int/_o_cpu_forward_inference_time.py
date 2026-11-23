import argparse
import csv
import platform
import statistics
import time
from pathlib import Path

import torch

from _0_MLP_Model import Transient_Model


SCRIPT_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark one-step MLP forward inference on CPU."
    )
    parser.add_argument("--warmup-runs", type=int, default=1000)
    parser.add_argument("--timed-runs", type=int, default=100000)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    return parser.parse_args()


def build_inputs(model, batch_size):
    torch.manual_seed(0)
    return (
        torch.rand(batch_size, 1, dtype=torch.float32),
        torch.rand(batch_size, model.flux_dim, dtype=torch.float32),
        torch.rand(batch_size, model.cooling_dim, dtype=torch.float32),
        torch.rand(batch_size, 1, dtype=torch.float32),
        torch.rand(batch_size, model.theta_dim, dtype=torch.float32),
    )


def main():
    args = parse_args()
    if args.warmup_runs < 0 or args.timed_runs < 2:
        raise ValueError("warmup-runs must be non-negative and timed-runs at least 2.")
    if args.num_threads <= 0:
        raise ValueError("num-threads must be positive.")

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(1)

    device = torch.device("cpu")
    model = Transient_Model().to(device)
    checkpoint = args.model_path
    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        model.load_state_dict(
            torch.load(str(checkpoint), map_location=device, weights_only=True)
        )
    model.eval()

    inputs = build_inputs(model, BATCH_SIZE)

    elapsed_ms = []
    with torch.inference_mode():
        for _ in range(args.warmup_runs):
            model(*inputs)

        for _ in range(args.timed_runs):
            start_ns = time.perf_counter_ns()
            model(*inputs)
            end_ns = time.perf_counter_ns()
            elapsed_ms.append((end_ns - start_ns) / 1_000_000.0)

    mean_ms = statistics.mean(elapsed_ms)
    std_ms = statistics.stdev(elapsed_ms)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.output_dir / "cpu_forward_inference_times.csv"
    summary_csv = args.output_dir / "cpu_forward_inference_summary.csv"

    with raw_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["run", "elapsed_ms"],
        )
        writer.writeheader()
        for run, elapsed in enumerate(elapsed_ms, start=1):
            writer.writerow(
                {
                    "run": run,
                    "elapsed_ms": f"{elapsed:.9f}",
                }
            )

    with summary_csv.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "device",
            "cpu",
            "checkpoint",
            "batch_size",
            "num_threads",
            "warmup_runs",
            "timed_runs",
            "mean_ms",
            "std_ms",
            "mean ± std (ms)",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "device": "cpu",
                "cpu": platform.processor(),
                "checkpoint": str(checkpoint)
                if checkpoint is not None
                else "random initialization",
                "batch_size": BATCH_SIZE,
                "num_threads": args.num_threads,
                "warmup_runs": args.warmup_runs,
                "timed_runs": args.timed_runs,
                "mean_ms": f"{mean_ms:.9f}",
                "std_ms": f"{std_ms:.9f}",
                "mean ± std (ms)": f"{mean_ms:.6f} ± {std_ms:.6f}",
            }
        )

    print(f"CPU single-sample forward latency: {mean_ms:.6f} ± {std_ms:.6f} ms")
    print(f"Raw timings: {raw_csv}")
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()
