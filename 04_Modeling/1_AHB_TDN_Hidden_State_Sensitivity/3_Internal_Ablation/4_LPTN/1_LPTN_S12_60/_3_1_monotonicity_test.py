# One-step input Jacobians for the dimensionless S12 model, using NODE state sampling.

import argparse
import csv
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from _0_RC_Model import Transient_Model


TEST_SEED = 42
DEFAULT_PKL_INDICES = list(range(80, 100))
DEFAULT_START_EXCLUSION_S = 5.0  # 排除每个 pkl 开头的数据


def fold_checkpoint(base_dir, fold):
    model_dir = base_dir / f"fold_{fold}_bs_2" / "models_1"
    paths = list(model_dir.glob("model_epoch_*.pth"))
    if not paths:
        raise FileNotFoundError(f"No model_epoch_*.pth under {model_dir}")
    return max(paths, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))


def load_model(checkpoint, device, dtype=torch.float32):
    model = Transient_Model().to(device=device, dtype=dtype)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def reconstruct_states(
    model, data_path, indices, max_points, start_exclusion, device
):
    files = sorted(data_path.glob("*.pkl"), key=lambda path: int(path.stem))
    selected = [files[index] for index in indices]
    parts = [[] for _ in range(6)]
    locations = []
    dtype = next(model.parameters()).dtype
    print("data files: " + ", ".join(path.name for path in selected), flush=True)

    with torch.no_grad():
        for path in selected:
            with path.open("rb") as file:
                data = pickle.load(file)
            time = torch.as_tensor(
                data["time_list"], dtype=dtype, device=device
            ).view(-1, 1)
            flux = torch.as_tensor(
                data["flux_list"], dtype=dtype, device=device
            ).reshape(-1, model.flux_dim)
            cooling = torch.as_tensor(
                data["cooling_list"], dtype=dtype, device=device
            ).reshape(-1, model.cooling_dim)
            env = torch.as_tensor(
                data["envtemp_list"], dtype=dtype, device=device
            ).view(-1, 1)
            temperature = torch.as_tensor(
                data["temp_list"], dtype=dtype, device=device
            ).reshape(-1, model.theta_dim)
            length = min(len(time), len(flux), len(cooling), len(env), len(temperature))

            v_curr = temperature[0:1]
            sink_curr = model.init_sink_state(v_curr)
            for step in range(length - 1):
                dt = ((time[step + 1] - time[step]) / model.time_scale).view(1, 1)
                inputs = (
                    dt,
                    flux[step:step + 1],
                    cooling[step:step + 1],
                    env[step:step + 1],
                    v_curr,
                    sink_curr,
                )
                if time[step] >= time[0] + start_exclusion:
                    for target, value in zip(parts, inputs):
                        target.append(value)
                    locations.append({
                        "file": path.name, "step": step,
                        "time_s": float(time[step]),
                        "elapsed_s": float(time[step] - time[0]),
                    })
                v_curr, sink_curr = model(*inputs)
            print(f"  reconstructed {path.name}: {length - 1} steps", flush=True)

    if not parts[0]:
        raise RuntimeError("No states remain after start exclusion")
    tensors = [torch.cat(values) for values in parts]
    finite = torch.stack([
        torch.isfinite(value).all(dim=1) for value in tensors
    ]).all(dim=0)
    tensors = [value[finite] for value in tensors]
    locations = [location for location, keep in zip(locations, finite.tolist()) if keep]
    if not tensors[0].shape[0]:
        raise RuntimeError("No finite reconstructed states")
    reconstructed_state_count = tensors[0].shape[0]
    if max_points and tensors[0].shape[0] > max_points:
        take = torch.linspace(
            0, tensors[0].shape[0] - 1, max_points, device=device
        ).long()
        tensors = [value.index_select(0, take) for value in tensors]
        locations = [locations[index] for index in take.tolist()]
    return tensors, reconstructed_state_count, locations


def jacobians(output, inputs):
    rows = [[] for _ in inputs]
    for index in range(output.shape[1]):
        gradients = torch.autograd.grad(
            output[:, index].sum(), inputs,
            retain_graph=index < output.shape[1] - 1,
        )
        for row, gradient in zip(rows, gradients):
            row.append(gradient.unsqueeze(1))
    return [torch.cat(row, dim=1) for row in rows]


def strict_summary(name, expected, magnitude, checked_states, total_states, physical_requirement=True):
    violating_states = (magnitude.detach() > 0).flatten(start_dim=1).any(dim=1)
    magnitude = magnitude.detach().reshape(-1)
    violating = magnitude > 0
    return {
        "criterion": name,
        "expected": expected,
        "physical_requirement": physical_requirement,
        "checked_state_count": checked_states,
        "excluded_state_count": total_states - checked_states,
        "checked_count": int(magnitude.numel()),
        "violation_count": int(violating.sum()),
        "violation_rate_percent": (
            100.0 * int(violating.sum()) / magnitude.numel() if magnitude.numel() else 0.0
        ),
        "violating_state_count": int(violating_states.sum()),
        "max_violation": (
            float(magnitude.max()) if magnitude.numel() else 0.0
        ),
        "passed": (not bool(violating.any())) if magnitude.numel() else None,
    }


def evaluate(model, time_step, flux, cooling, env, v_curr, sink_curr):
    flux = flux.detach().clone().requires_grad_(True)
    cooling = cooling.detach().clone().requires_grad_(True)
    temp_next, sink_next = model(
        time_step, flux, cooling, env, v_curr, sink_curr
    )
    jac_flux, jac_cooling = jacobians(temp_next, (flux, cooling))
    if not all(torch.isfinite(value).all() for value in (temp_next, sink_next, jac_flux, jac_cooling)):
        raise RuntimeError("Non-finite predictions or Jacobians")
    # The implicit step exchanges heat with air at sink_next, not sink_curr.
    # Cooling is guaranteed only when every next-step sink node is at/above ambient.
    hot_sink = (sink_next.detach() >= env).all(dim=1)
    hot_heaters = (v_curr >= env).all(dim=1)
    total_states = flux.shape[0]
    results = [
        strict_summary("flux_all", ">= 0", torch.relu(-jac_flux), total_states, total_states),
        strict_summary(
            "cooling_all", "<= 0", torch.relu(jac_cooling),
            total_states, total_states, physical_requirement=False,
        ),
        strict_summary(
            "cooling_hot_sink", "<= 0", torch.relu(jac_cooling[hot_sink]),
            int(hot_sink.sum()), total_states,
        ),
        strict_summary(
            "cooling_hot_heaters", "<= 0", torch.relu(jac_cooling[hot_heaters]),
            int(hot_heaters.sum()), total_states, physical_requirement=False,
        ),
    ]
    violations = []
    for name, jac, mask in (
        ("flux_all", jac_flux, jac_flux < 0),
        ("cooling_all", jac_cooling, jac_cooling > 0),
    ):
        for sample, heater, channel in mask.nonzero().tolist():
            violations.append({
                "sample_index": sample, "criterion": name,
                "heater": heater + 1, "input_channel": channel + 1,
                "derivative_normalized": float(jac[sample, heater, channel].detach()),
                "input_value_normalized": float((flux if name == "flux_all" else cooling)[sample, channel].detach()),
                "min_heater_minus_env_C": float(((v_curr[sample] - env[sample]) * model.temp_scale).min()),
                "min_sink_curr_minus_env_C": float(((sink_curr[sample] - env[sample]) * model.temp_scale).min()),
                "min_sink_next_minus_env_C": float(((sink_next[sample].detach() - env[sample]) * model.temp_scale).min()),
                "all_heaters_at_or_above_env": bool(hot_heaters[sample]),
                "all_next_sinks_at_or_above_env": bool(hot_sink[sample]),
            })
    return results, violations


def main():
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    torch.set_num_threads(1)
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Strict one-step RC input-Jacobian test at reconstructed states (not a full-rollout Jacobian)."
    )
    parser.add_argument("--folds", type=int, nargs="+", help="Default: all folds with saved checkpoints")
    parser.add_argument(
        "--data-path", type=Path,
        default=Path(__file__).resolve().parents[4] / "Database" / "0_Dataset" / "Data_AT40_D1_05",
    )
    parser.add_argument(
        "--indices", type=int, nargs="+", default=DEFAULT_PKL_INDICES,
        help="Zero-based indices in numerically sorted pkl files; default 80..99 (81.pkl..100.pkl)",
    )
    parser.add_argument("--max-points", type=int, default=512, help="States tested per fold; default 512, 0 checks all")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--start-exclusion", type=float, default=DEFAULT_START_EXCLUSION_S
    )
    parser.add_argument(
        "--output", type=Path,
        default=base_dir / "rc_monotonicity_all_folds.csv",
    )
    args = parser.parse_args()
    if args.max_points < 0 or args.start_exclusion < 0:
        parser.error("--max-points and --start-exclusion must be nonnegative")
    if args.folds is None:
        args.folds = sorted(
            int(path.name.split("_")[1])
            for path in base_dir.glob("fold_*_bs_2")
            if any((path / "models_1").glob("model_epoch_*.pth"))
        )
    if not args.folds:
        raise FileNotFoundError(f"No trained fold checkpoints under {base_dir}")
    print(f"folds: {args.folds}; device: CPU; dtype: {args.dtype}; strict sign check (tolerance=0)")
    print("State reconstruction: full recursive rollout, initialized only at the first frame.")
    print(f"NODE sampling: exclude elapsed time < {args.start_exclusion:g} s, keep finite states, concatenate files, then linspace indices (max_points={args.max_points}).", flush=True)
    print("Check dT_next/dflux >= 0 and dT_next/dcooling <= 0 with current states fixed.")
    print("cooling_all reports EVERY sampled state; no temperature-based filtering.", flush=True)
    print("cooling_hot_sink separately checks the next-step sink >= ambient physical condition.", flush=True)
    print("cooling_hot_heaters separately checks current heaters >= ambient; this alone does not guarantee cooling.", flush=True)
    print("Derivatives use normalized temperatures/inputs; positive normalization scales preserve their signs.", flush=True)

    device = torch.device("cpu")
    dtype = getattr(torch, args.dtype)
    all_results = []
    all_violations = []
    for fold in args.folds:
        checkpoint = fold_checkpoint(base_dir, fold)
        model = load_model(checkpoint, device, dtype)
        print(f"\nfold={fold}, sink_nodes={model.sink_dim}, C/R dimensionless, time_scale={model.time_scale:g} s, power_reference={model.power_reference:g} W", flush=True)
        states, reconstructed_state_count, locations = reconstruct_states(
            model, args.data_path, args.indices, args.max_points,
            args.start_exclusion, device,
        )
        results, violations = evaluate(model, *states)
        for row in violations:
            row.update(locations[row["sample_index"]])
            row.update({"fold": fold, "checkpoint": str(checkpoint), "dtype": args.dtype})
            all_violations.append(row)
        print(
            f"\nfold={fold}, checkpoint={checkpoint.name}, "
            f"reconstructed_states={reconstructed_state_count}, "
            f"tested={states[0].shape[0]}"
        )
        print("criterion         expected       violations   max_violation  result")
        for row in results:
            row.update({
                "fold": fold,
                "checkpoint": str(checkpoint),
                "reconstructed_state_count": reconstructed_state_count,
                "tested_state_count": states[0].shape[0],
                "start_exclusion_s": args.start_exclusion,
                "dtype": args.dtype,
                "parameter_units": model.get_extra_state()["parameter_units"],
            })
            all_results.append(row)
            print(
                f"{row['criterion']:<17} {row['expected']:<14} "
                f"{row['violation_count']:>7}/{row['checked_count']:<7} "
                f"{row['max_violation']:<14.7g} "
                f"{'N/A' if row['passed'] is None else 'PASS' if row['passed'] else 'FAIL'} "
                f"({row['violation_rate_percent']:.6g}%)"
            )
            print(f"  checked states={row['checked_state_count']}, excluded={row['excluded_state_count']}, violating states={row['violating_state_count']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "fold", "checkpoint", "reconstructed_state_count",
            "tested_state_count", "start_exclusion_s", "criterion", "expected",
            "dtype", "parameter_units", "physical_requirement",
            "checked_state_count", "excluded_state_count",
            "checked_count", "violation_count", "violation_rate_percent",
            "violating_state_count", "max_violation", "passed",
        ])
        writer.writeheader()
        writer.writerows(all_results)
    details_path = args.output.with_name(args.output.stem + "_violations.csv")
    with details_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "fold", "checkpoint", "dtype", "sample_index", "file", "step",
            "time_s", "elapsed_s", "criterion", "heater", "input_channel",
            "derivative_normalized", "input_value_normalized",
            "min_heater_minus_env_C", "min_sink_curr_minus_env_C",
            "min_sink_next_minus_env_C", "all_heaters_at_or_above_env",
            "all_next_sinks_at_or_above_env",
        ])
        writer.writeheader()
        writer.writerows(all_violations)
    overall = (
        "FAIL" if any(row["passed"] is False for row in all_results)
        else "INCOMPLETE" if any(row["passed"] is None for row in all_results)
        else "PASS"
    )
    physical_results = [row for row in all_results if row["physical_requirement"]]
    physical_overall = (
        "FAIL" if any(row["passed"] is False for row in physical_results)
        else "INCOMPLETE" if any(row["passed"] is None for row in physical_results)
        else "PASS"
    )
    print(f"\noverall sign checks (including unconditional cooling): {overall}")
    print(f"physical criteria (flux_all + cooling_hot_sink): {physical_overall}")
    print(f"saved: {args.output}")
    print(f"violation details: {details_path}")


if __name__ == "__main__":
    main()
