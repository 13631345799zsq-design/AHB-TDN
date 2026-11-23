import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
import time
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from _0_RNODE_Model import Transient_Model


GT_V_CURR_INTERVAL = 0
TF = 1  # 0:transient test, 1: steady test
DF = 2  # 1:Device 1, 2:Device 2
AT = 40  # 40:Data_AT40, 35:Data_AT35, 45:Data_AT45 (only used when TF=0)

time_D_min, time_D_max = 0, 5
temp_min, temp_max = 20, 100
n_folds = 5
time_data = '05'


def format_duration(seconds):
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def print_progress(prefix, current, total, start_time, file_loss=None, force=False):
    update_interval = 10
    bar_width = 30
    if not force and current < total and current % update_interval != 0:
        return

    total = max(total, 1)
    current = min(current, total)
    progress = current / total
    elapsed = time.perf_counter() - start_time
    eta = elapsed * (total - current) / current if current > 0 else 0.0
    filled = int(bar_width * progress)
    bar = '=' * filled + '-' * (bar_width - filled)
    loss_text = ''
    if file_loss is not None and current > 0:
        loss_text = f", avg_loss={file_loss / current:.6f}"
    line = (
        f"\r{prefix} [{bar}] {current}/{total} "
        f"({progress * 100:5.1f}%) elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta)}{loss_text}"
    )
    print(line, end='\n' if current >= total or force else '', flush=True)


def freeze_model_weights(model):
    for param in model.parameters():
        param.requires_grad_(False)


def should_use_gt_v_curr(step_idx):
    return (
        GT_V_CURR_INTERVAL is not None
        and GT_V_CURR_INTERVAL > 0
        and step_idx > 0
        and step_idx % GT_V_CURR_INTERVAL == 0
    )


# ==========================================
# 3. Dataset with full-sequence loading.
# ==========================================
class FullSequenceDataset(Dataset):
    def __init__(self, datapath, file_ids):
        # 1. Collect all pickle files.
        all_files = {
            int(os.path.splitext(f)[0]): f
            for f in os.listdir(datapath)
            if f.endswith('.pkl')
        }
        missing_file_ids = [file_id for file_id in file_ids if file_id not in all_files]
        if missing_file_ids:
            raise FileNotFoundError(f"数据集 {datapath} 缺少文件编号: {missing_file_ids}")
        # 2. Select by actual file number to avoid positional-index errors across AT datasets of different sizes.
        self.file_ids = list(file_ids)
        self.files = [all_files[file_id] for file_id in self.file_ids]
        self.datapath = datapath
        print(f"全序列数据集加载: {len(self.files)} 个文件 -> {self.files}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        with open(os.path.join(self.datapath, file_name), 'rb') as f:
            data = pickle.load(f)
        # Preprocess all maps to 64 x 64.
        data['time_list'] = torch.from_numpy(data['time_list'].astype(np.float32)).view(
            -1, 1
        )
        data['flux_list'] = torch.from_numpy(data['flux_list'].astype(np.float32))
        data['cooling_list'] = torch.from_numpy(data['cooling_list'].astype(np.float32))
        data['envtemp_list'] = torch.from_numpy(
            data['envtemp_list'].astype(np.float32)
        ).view(-1, 1)
        data['temp_steady_list'] = torch.from_numpy(
            data['temp_steady_list'].astype(np.float32)
        )
        data['temp_list'] = torch.from_numpy(data['temp_list'].astype(np.float32))
        return data


# 1. Define the static coordinates.
chip_loc = [
    [65, 55, 50, 50],
    [15, 90, 30, 30],
    [142, 13.5, 20, 20],
    [142, 43.5, 20, 20],
    [142, 73.5, 20, 20],
    [142, 103.5, 20, 20],
    [115, 20, 10, 10],
    [90, 20, 10, 10],
    [65, 20, 10, 10],
    [40, 20, 10, 10],
    [22, 40, 10, 10],
    [22, 65, 10, 10],
]
cooling_loc = [[50 - 25, 63.5 - 25, 50, 50], [130 - 25, 63.5 - 25, 50, 50]]

chip_centers = torch.tensor(
    [[l[0] + l[2] / 2, l[1] + l[3] / 2] for l in chip_loc], dtype=torch.float32
)
fan_centers = torch.tensor(
    [[l[0] + l[2] / 2, l[1] + l[3] / 2] for l in cooling_loc], dtype=torch.float32
)
# ==========================================
# 4. Main training loop.
# ==========================================
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Database/0_Dataset'))
savepath = os.path.dirname(__file__)
point_labels = [
    "chip1",
    "chip2",
    "chip3",
    "chip4",
    "chip5",
    "chip6",
    "chip7",
    "chip8",
    "chip9",
    "chip10",
    "chip11",
    "chip12",
]

# Load the data.
device = torch.device('cpu')

if TF == 0:  # transient test
    if AT == 40:
        if DF == 1:
            dataset_name = f'_AT{AT}_D1_{time_data}'
        elif DF == 2:
            dataset_name = f'_AT{AT}_D2_{time_data}'
    elif AT in (35, 45):
        dataset_name = f'_AT{AT}_D1_{time_data}'
elif TF == 1:  # steady test
    if DF == 1:
        dataset_name = f'_T2S_D1_{time_data}'
    elif DF == 2:
        dataset_name = f'_T2S_D2_{time_data}'
datapath = os.path.join(path, 'Data' + dataset_name)

case_error_means = []
case_error_means_norm = []
fold_error_stats = []
fold_error_means = []
fold_error_means_norm = []
mean_ae_statistics_by_fold = []
max_ae_statistics_by_fold = []
mean_ae_per_case_all = []
max_ae_per_case_all = []
for n_fold in range(n_folds):
    fold_start_time = time.perf_counter()
    fold_case_error_means = []
    fold_case_error_means_norm = []
    absolute_error_all = []
    if TF == 0:  # transient test
        test_file_ids = list(range(81, 101))
    else:  # steady test
        test_file_ids = list(range(1, 21))
    test_dataset = FullSequenceDataset(datapath, test_file_ids)
    test_dataloader = DataLoader(
        test_dataset, batch_size=len(test_file_ids), shuffle=False
    )

    foldpath = os.path.join(savepath, f'fold_{n_fold+1}_bs_2')
    modelfile = os.path.join(foldpath, 'models_1')
    resultfile = os.path.join(foldpath, 'results' + dataset_name)

    os.makedirs(resultfile, exist_ok=True)

    model = Transient_Model().to(device)
    model_name = max(
        [
            name
            for name in os.listdir(modelfile)
            if name.startswith('model_epoch_') and name.endswith('.pth')
        ],
        key=lambda name: int(name.split('_')[-1].split('.')[0]),
    )
    model.load_state_dict(
        torch.load(os.path.join(modelfile, model_name), map_location=device)
    )
    freeze_model_weights(model)
    model.eval()
    print(
        f"Fold {n_fold+1}/{n_folds} started. "
        f"Model: {os.path.join(modelfile, model_name)}, "
        f"Data: {datapath}, "
        f"GT v_curr interval: {GT_V_CURR_INTERVAL}, "
        f"result dir: {resultfile}"
    )
    # test_part======================================================================== #
    with torch.no_grad():
        model.eval()  # Switch to evaluation mode.
        test_losses = []
        # Iterate over all files; each file is one batch with size 1.
        for batch_idx, data in enumerate(test_dataloader):

            batch_start_time = time.perf_counter()
            time_seq = data['time_list'].to(device)  # [T, 1]
            flux_seq = data['flux_list'].to(device)
            cool_seq = data['cooling_list'].to(device)
            envtemp_seq = data['envtemp_list'].to(device)
            gt_temp = data['temp_list'].to(device)
            B = time_seq.shape[0]
            n = 1
            tst = 0
            ted = int(time_seq.shape[1])
            time_seq = time_seq[:, tst:ted]
            flux_seq = flux_seq[:, tst:ted]
            cool_seq = cool_seq[:, tst:ted]
            envtemp_seq = envtemp_seq[:, tst:ted]
            gt_temp = gt_temp[:, tst:ted]

            # Initialize the state.
            pred_temp = []
            v_curr = gt_temp[:, 0]
            h_old = None
            # === Recursive time step ===
            file_loss = 0.0
            steps = time_seq.shape[1] - 1
            for t in range(steps):
                if should_use_gt_v_curr(t):
                    v_curr = gt_temp[:, t]
                dt = time_seq[:, t + 1] - time_seq[:, t]
                dt_norm = (dt - time_D_min) / (time_D_max - time_D_min)
                time_D_v = dt_norm.view(B, 1)
                flux_v = flux_seq[:, t]
                cooling_v = cool_seq[:, t]
                envtemp_v = envtemp_seq[:, t].view(B, 1)
                v_gt_next = gt_temp[:, t + 1]

                # 2. Forward pass using all required physical quantities.
                v_next_pred, h_curr = model(
                    time_D_v, flux_v, cooling_v, envtemp_v, v_curr, h_old
                )
                v_curr = v_next_pred.detach()
                h_old = h_curr.detach()
                pred_temp.append(v_next_pred.detach())
                file_loss += F.mse_loss(v_next_pred, v_gt_next, reduction='mean').item()
                print_progress(
                    f"Fold {n_fold+1}/{n_folds} Batch {batch_idx+1}/{len(test_dataloader)}",
                    t + 1,
                    steps,
                    batch_start_time,
                    file_loss,
                )
            avg_file_loss = file_loss / steps
            test_losses.append(avg_file_loss * B)
            print(
                f"Batch {batch_idx+1}/{len(test_dataloader)}, Avg MSE Loss: {avg_file_loss:.6f}"
            )

            # Compute each test case's mean absolute temperature error over all prediction times and 12 sensors.
            # The prediction sequence corresponds to gt_temp[:, 1:], so their time dimensions are aligned.
            pred_temp_tensor = torch.stack(pred_temp, dim=1)
            error_temp_norm = torch.abs(pred_temp_tensor - gt_temp[:, 1:])
            batch_case_error_means_norm = (
                error_temp_norm.mean(dim=(1, 2)).cpu().tolist()
            )
            batch_case_error_means = [
                error_mean * (temp_max - temp_min)
                for error_mean in batch_case_error_means_norm
            ]
            case_error_means.extend(batch_case_error_means)
            case_error_means_norm.extend(batch_case_error_means_norm)
            fold_case_error_means.extend(batch_case_error_means)
            fold_case_error_means_norm.extend(batch_case_error_means_norm)

            # Save absolute temperature errors for every sample, time step, and sensor in this fold [degC].
            error_temp_actual = error_temp_norm.detach().cpu().numpy() * (
                temp_max - temp_min
            )
            absolute_error_all.append(error_temp_actual)

            # Export temperatures using the same column format as temperature_data_1.csv.
            # Prediction starts at time_seq[1], so time, GT, and PD are naturally aligned.
            time_seq_np_all = time_seq.detach().cpu().numpy()
            gt_temp_np_all = gt_temp.detach().cpu().numpy()
            pred_temp_np_all = pred_temp_tensor.detach().cpu().numpy()

            for b in range(B):
                global_file_idx = test_dataset.file_ids[batch_idx * B + b]
                time_np = time_seq_np_all[b, 1:, 0]
                gt_temp_actual = (
                    temp_min + (temp_max - temp_min) * gt_temp_np_all[b, 1:, :]
                )
                pred_temp_actual = (
                    temp_min + (temp_max - temp_min) * pred_temp_np_all[b]
                )
                error_temp_actual = np.abs(pred_temp_actual - gt_temp_actual)
                csv_data = np.column_stack(
                    (
                        time_np,
                        gt_temp_actual,
                        pred_temp_actual,
                        error_temp_actual,
                    )
                )
                csv_header = ','.join(
                    ['time']
                    + [f'GT_{i+1}' for i in range(12)]
                    + [f'PD_{i+1}' for i in range(12)]
                    + [f'ER_{i+1}' for i in range(12)]
                )
                np.savetxt(
                    os.path.join(resultfile, f'temperature_data_{global_file_idx}.csv'),
                    csv_data,
                    delimiter=',',
                    header=csv_header,
                    comments='',
                    fmt='%.6f',
                )

            # # -------------------------------------------------------------- (plotting)
            # print(f"Saving figures for Batch {batch_idx+1}/{len(test_dataloader)} ...")
            # figure_start_time = time.perf_counter()
            # pred_temp = torch.stack(pred_temp, dim=1)

            # time_seq_np_all = time_seq.detach().cpu().numpy()
            # gt_temp_np_all = gt_temp.detach().cpu().numpy()
            # pred_temp_np_all = pred_temp.detach().cpu().numpy()

            # for b in range(B):
            #     dataset_position = batch_idx * len(test_file_ids) + b
            #     global_file_idx = test_dataset.file_ids[dataset_position]

            #     x_axis_data = time_seq_np_all[b].squeeze()
            #     gt_temp_np = gt_temp_np_all[b, 1:, :]
            #     pred_temp_np = pred_temp_np_all[b]
            #     length = min(len(gt_temp_np), len(pred_temp_np))

            #     # Save one CSV per test case. Predictions start from time_seq[1],
            #     # so time, GT and PD use the same aligned sequence length.
            #     time_np = x_axis_data[1:length+1]
            #     gt_temp_actual = temp_min + (temp_max - temp_min) * gt_temp_np[:length]
            #     pred_temp_actual = temp_min + (temp_max - temp_min) * pred_temp_np[:length]
            #     error_temp_actual = np.abs(pred_temp_actual - gt_temp_actual)
            #     case_error_mean = float(np.mean(error_temp_actual))
            #     case_error_means.append(case_error_mean)
            #     fold_case_error_means.append(case_error_mean)
            #     csv_data = np.column_stack((
            #         time_np,
            #         gt_temp_actual,
            #         pred_temp_actual,
            #         error_temp_actual,
            #     ))
            #     csv_header = ','.join(
            #         ['time']
            #         + [f'GT_{i+1}' for i in range(12)]
            #         + [f'PD_{i+1}' for i in range(12)]
            #         + [f'ER_{i+1}' for i in range(12)]
            #     )
            #     np.savetxt(
            #         os.path.join(resultfile, f'temperature_data_{global_file_idx}.csv'),
            #         csv_data,
            #         delimiter=',',
            #         header=csv_header,
            #         comments='',
            #         fmt='%.6f',
            #     )

            #     # print('gt_temp', np.max(gt_temp_np), np.min(gt_temp_np))
            #     # print('pred_temp', np.max(pred_temp_np), np.min(pred_temp_np))

            #     fig, axes = plt.subplots(4, 3, figsize=(16, 8))
            #     axes = axes.flatten()

            #     for i in range(12):
            #         ax = axes[i]
            #         ln1 = ax.plot(x_axis_data[1:length+1], gt_temp_np[:length, i], 'k-', label='GT', alpha=0.6, linewidth=1.5)
            #         ln2 = ax.plot(x_axis_data[1:length+1], pred_temp_np[:length, i], 'r--', label='Prediction', alpha=0.6, linewidth=1.5)
            #         ax.set_ylabel('Normalized Temp', fontsize=9)
            #         ax.set_ylim(0, 1)
            #         ax.grid(True, alpha=0.3)

            #         ax_err = ax.twinx()
            #         ln3 = ax_err.plot(x_axis_data[1:length+1], error_temp_actual[:, i], 'b--', label='Error', alpha=0.6, linewidth=1.5)
            #         ax_err.set_ylim(0, 1)
            #         ax_err.tick_params(axis='y', labelcolor='blue')

            #         ax.set_title(f"{point_labels[i]}", fontsize=10, fontweight='bold')
            #         ax.axvline(x=800, color='blue', linestyle='--', linewidth=1, alpha=0.4, zorder=1)
            #         ax.axvline(x=1600, color='blue', linestyle='--', linewidth=1, alpha=0.4, zorder=1)

            #         if i == 0:
            #             lns = ln1 + ln2 + ln3
            #             labs = [l.get_label() for l in lns]
            #             ax.legend(lns, labs, loc='upper right', fontsize=8)

            #     plt.tight_layout()
            #     plt.savefig(os.path.join(resultfile, f'temperature_curves_{global_file_idx}.png'), dpi=150)
            #     plt.close()
            #     print_progress(
            #         f"Saving figures Batch {batch_idx+1}/{len(test_dataloader)}",
            #         b + 1,
            #         B,
            #         figure_start_time,
            #     )
            # --------------------------------------------------------------

        # Print the overall mean loss.
        overall_avg_loss = sum(test_losses) / len(test_dataset)
        print(f"Test completed. Overall Avg MSE Loss: {overall_avg_loss:.6f}")
        print(f"Fold runtime: {format_duration(time.perf_counter() - fold_start_time)}")
        print(f"{'='*50}\n")

    # Aggregate all time steps and 12 sensors for each test sample, then compute fold-level mean and standard deviation.
    absolute_error_values = np.concatenate(absolute_error_all, axis=0)
    mean_ae_per_case = absolute_error_values.mean(axis=(1, 2))
    max_ae_per_case = absolute_error_values.max(axis=(1, 2))
    mean_ae_per_case_all.extend(mean_ae_per_case.tolist())
    max_ae_per_case_all.extend(max_ae_per_case.tolist())
    mean_ae_statistics_by_fold.append(
        (
            f'fold_{n_fold+1}',
            float(np.mean(mean_ae_per_case)),
            float(np.std(mean_ae_per_case, ddof=1)),
            len(mean_ae_per_case),
        )
    )
    max_ae_statistics_by_fold.append(
        (
            f'fold_{n_fold+1}',
            float(np.mean(max_ae_per_case)),
            float(np.std(max_ae_per_case, ddof=1)),
            len(max_ae_per_case),
        )
    )

    fold_error_mean = float(np.mean(fold_case_error_means))
    fold_error_std = float(np.std(fold_case_error_means, ddof=1))
    fold_error_mean_norm = float(np.mean(fold_case_error_means_norm))
    fold_error_std_norm = float(np.std(fold_case_error_means_norm, ddof=1))
    fold_error_stats.append(
        (
            f'fold_{n_fold+1}',
            fold_error_mean_norm,
            fold_error_std_norm,
            fold_error_mean,
            fold_error_std,
        )
    )
    fold_error_means.append(fold_error_mean)
    fold_error_means_norm.append(fold_error_mean_norm)

summary_stats = [
    (
        'case-level',
        float(np.mean(case_error_means_norm)),
        float(np.std(case_error_means_norm, ddof=1)),
        float(np.mean(case_error_means)),
        float(np.std(case_error_means, ddof=1)),
    ),
    (
        'fold-level',
        float(np.mean(fold_error_means_norm)),
        float(np.std(fold_error_means_norm, ddof=1)),
        float(np.mean(fold_error_means)),
        float(np.std(fold_error_means, ddof=1)),
    ),
]
summary_file = os.path.join(
    savepath,
    f'{os.path.basename(resultfile)}_error_statistics.csv',
)
with open(summary_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            'scope',
            'mean_normalized',
            'std_normalized',
            'mean_temperature_C',
            'std_temperature_C',
        ]
    )
    for fold_name, norm_mean, norm_std, temp_mean, temp_std in fold_error_stats:
        writer.writerow(
            [
                fold_name,
                f'{norm_mean:.6f}',
                f'{norm_std:.6f}',
                f'{temp_mean:.6f}',
                f'{temp_std:.6f}',
            ]
        )
    for level_name, norm_mean, norm_std, temp_mean, temp_std in summary_stats:
        writer.writerow(
            [
                level_name,
                f'{norm_mean:.6f}',
                f'{norm_std:.6f}',
                f'{temp_mean:.6f}',
                f'{temp_std:.6f}',
            ]
        )

print(f"Error statistics saved to: {summary_file}")

# Save the per-fold statistics together in the script directory.
for statistic_name, statistics_by_fold in [
    ('MeanAE', mean_ae_statistics_by_fold),
    ('MaxAE', max_ae_statistics_by_fold),
]:
    if statistic_name == 'MeanAE':
        case_level_values = mean_ae_per_case_all
    else:
        case_level_values = max_ae_per_case_all
    statistic_file = os.path.join(
        savepath,
        f'{os.path.basename(resultfile)}_{statistic_name}_statistics.csv',
    )
    with open(statistic_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                'scope',
                f'mean_{statistic_name}_temperature_C',
                f'std_{statistic_name}_temperature_C',
                f'{statistic_name}_temperature_C_mean±std',
                'case_count',
            ]
        )
        for fold_name, statistic_mean, statistic_std, case_count in statistics_by_fold:
            writer.writerow(
                [
                    fold_name,
                    f'{statistic_mean:.6f}',
                    f'{statistic_std:.6f}',
                    f'{statistic_mean:.6f} ± {statistic_std:.6f}',
                    case_count,
                ]
            )
        writer.writerow(
            [
                'case-level',
                f'{np.mean(case_level_values):.6f}',
                f'{np.std(case_level_values, ddof=1):.6f}',
                f'{np.mean(case_level_values):.6f} ± {np.std(case_level_values, ddof=1):.6f}',
                len(case_level_values),
            ]
        )
        fold_means = [statistic_mean for _, statistic_mean, _, _ in statistics_by_fold]
        writer.writerow(
            [
                'fold-level',
                f'{np.mean(fold_means):.6f}',
                f'{np.std(fold_means, ddof=1):.6f}',
                f'{np.mean(fold_means):.6f} ± {np.std(fold_means, ddof=1):.6f}',
                len(fold_means),
            ]
        )
    print(f"{statistic_name} statistics saved to: {statistic_file}")
