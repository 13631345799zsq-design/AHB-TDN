"""Strict five-criterion monotonicity test for NODE-Core-MLP."""

import argparse
import csv
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from _0_RNODE_Model import Transient_Model

TEST_SEED = 42
DEFAULT_FOLDS = list(range(1, 5 + 1))
DEFAULT_PKL_INDICES = list(range(80, 100))


def fold_checkpoint(base_dir, fold):
    model_dir = base_dir / f"fold_{fold}_bs_2" / "models_1"
    paths = list(model_dir.glob("model_epoch_*.pth"))
    if not paths:
        raise FileNotFoundError(f"No model_epoch_*.pth under {model_dir}")
    return max(paths, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))


def load_model(checkpoint, device):
    model = Transient_Model().to(device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def reconstruct_states(model, data_path, indices, max_points, device):
    files = sorted(data_path.glob("*.pkl"), key=lambda path: int(path.stem))
    selected = [files[index] for index in indices]
    parts = [[] for _ in range(5)]

    with torch.no_grad():
        for path in selected:
            with path.open("rb") as file:
                data = pickle.load(file)
            time = torch.as_tensor(
                data["time_list"], dtype=torch.float32, device=device
            ).view(-1, 1)
            flux = torch.as_tensor(
                data["flux_list"], dtype=torch.float32, device=device
            ).reshape(-1, model.flux_dim)
            cooling = torch.as_tensor(
                data["cooling_list"], dtype=torch.float32, device=device
            ).reshape(-1, model.cooling_dim)
            env = torch.as_tensor(
                data["envtemp_list"], dtype=torch.float32, device=device
            ).view(-1, 1)
            temperature = torch.as_tensor(
                data["temp_list"], dtype=torch.float32, device=device
            ).reshape(-1, model.theta_dim)
            length = min(len(time), len(flux), len(cooling), len(env), len(temperature))

            v_curr, h_old = temperature[0:1], None
            for step in range(length - 1):
                dt = ((time[step + 1] - time[step]) / 5.0).view(1, 1)
                hidden = (
                    torch.zeros(1, model.augment_dim, device=device)
                    if h_old is None
                    else h_old
                )
                theta = v_curr - env[step : step + 1]
                inputs = (
                    dt,
                    flux[step : step + 1],
                    cooling[step : step + 1],
                    theta,
                    hidden,
                )
                for target, value in zip(parts, inputs):
                    target.append(value)
                v_curr, h_old = model(
                    dt,
                    inputs[1],
                    inputs[2],
                    env[step : step + 1],
                    v_curr,
                    hidden,
                )

    tensors = [torch.cat(values) for values in parts]
    finite = torch.stack([torch.isfinite(value).all(dim=1) for value in tensors]).all(
        dim=0
    )
    tensors = [value[finite] for value in tensors]
    all_positive = (tensors[3] > 0).all(dim=1)
    tensors = [value[all_positive] for value in tensors]
    if not tensors[0].shape[0]:
        raise RuntimeError("No finite states with all 12 theta channels > 0")
    positive_state_count = tensors[0].shape[0]
    if max_points and tensors[0].shape[0] > max_points:
        take = torch.linspace(
            0, tensors[0].shape[0] - 1, max_points, device=device
        ).long()
        tensors = [value.index_select(0, take) for value in tensors]
    return tensors, positive_state_count


def jacobians(output, inputs):
    rows = [[] for _ in inputs]
    for index in range(output.shape[1]):
        gradients = torch.autograd.grad(
            output[:, index].sum(), inputs, retain_graph=True, allow_unused=True
        )
        for row, gradient, reference in zip(rows, gradients, inputs):
            row.append(
                (
                    torch.zeros_like(reference) if gradient is None else gradient
                ).unsqueeze(1)
            )
    return [torch.cat(row, dim=1) for row in rows]


def strict_summary(name, expected, magnitude):
    magnitude = magnitude.detach().reshape(-1)
    violating = magnitude > 0
    return {
        "criterion": name,
        "expected": expected,
        "checked_count": int(magnitude.numel()),
        "violation_count": int(violating.sum().item()),
        "max_violation": float(magnitude.max().item()) if magnitude.numel() else 0.0,
    }


def evaluate(model, time_step, flux, cooling, theta, hidden):
    flux, cooling, theta, hidden = [
        value.detach().clone().requires_grad_(True)
        for value in (flux, cooling, theta, hidden)
    ]
    zero_env = theta.new_zeros(theta.shape[0], 1)
    theta_next, _ = model(time_step, flux, cooling, zero_env, theta, hidden)
    jac_flux, jac_cooling, _, _ = jacobians(theta_next, (flux, cooling, theta, hidden))

    theta_diff = theta.detach().clone().requires_grad_(True)
    diffusion = model.ode_func_core.net_theta_2(theta_diff, hidden.detach())
    jac_diffusion = jacobians(diffusion, (theta_diff,))[0]

    diagonal = (
        torch.eye(model.theta_dim, dtype=torch.bool, device=theta.device)
        .unsqueeze(0)
        .expand(theta.shape[0], -1, -1)
    )

    return [
        strict_summary("flux_self", ">= 0", torch.relu(-jac_flux[diagonal])),
        strict_summary("flux_cross", "= 0", jac_flux[~diagonal].abs()),
        strict_summary(
            "cooling_all",
            "<= 0 (all theta > 0)",
            torch.relu(jac_cooling),
        ),
        strict_summary(
            "diffusion_self",
            "<= 0",
            torch.relu(jac_diffusion[diagonal]),
        ),
        strict_summary(
            "diffusion_cross",
            ">= 0",
            torch.relu(-jac_diffusion[~diagonal]),
        ),
    ]


def main():
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=DEFAULT_FOLDS)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "Database" / "0_Dataset" / "Data_AT40_D1_05",
    )
    parser.add_argument("--indices", type=int, nargs="+", default=DEFAULT_PKL_INDICES)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "strict_monotonicity_all_folds.csv",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    all_results = []
    for fold in args.folds:
        checkpoint = fold_checkpoint(base_dir, fold)
        model = load_model(checkpoint, device)
        states, positive_state_count = reconstruct_states(
            model, args.data_path, args.indices, args.max_points, device
        )
        results = evaluate(model, *states)
        print(
            f"\nfold={fold}, checkpoint={checkpoint.name}, "
            f"positive_states={positive_state_count}, "
            f"tested={states[0].shape[0]}"
        )
        print(
            "criterion         expected                    violations   max_violation  result"
        )
        for row in results:
            row.update(
                {
                    "fold": fold,
                    "checkpoint": str(checkpoint),
                    "positive_state_count": positive_state_count,
                    "tested_state_count": states[0].shape[0],
                    "passed": row["violation_count"] == 0,
                }
            )
            all_results.append(row)
            print(
                f"{row['criterion']:<17} {row['expected']:<27} "
                f"{row['violation_count']:>7}/{row['checked_count']:<7} "
                f"{row['max_violation']:<14.7g} {'PASS' if row['passed'] else 'FAIL'}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "fold",
                "checkpoint",
                "positive_state_count",
                "tested_state_count",
                "criterion",
                "expected",
                "checked_count",
                "violation_count",
                "max_violation",
                "passed",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)
    print(
        f"\noverall: {'PASS' if all(row['passed'] for row in all_results) else 'FAIL'}"
    )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
