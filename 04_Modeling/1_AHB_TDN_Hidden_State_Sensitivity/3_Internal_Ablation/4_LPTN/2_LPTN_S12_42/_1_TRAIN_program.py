import torch, math
import torch.nn.functional as F
import numpy as np
import os
import pickle
import random
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from _0_RC_Model import Transient_Model


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================ #
# 5. 增强版 Loss Map 绘制 (Professional Loss Visualization)
# ============================================================ #
def plot_loss_map(loss_train_history, loss_validation_history,  save_dir, filename):
    loss_train_np = np.array(loss_train_history)
    loss_validation_np = np.array(loss_validation_history)
    fig, ax = plt.subplots(figsize=(8, 6)) 
    ax.plot(loss_train_np, label='Training Loss', color='#1f77b4', linewidth=2)
    ax.plot(loss_validation_np, label='Validation Loss', color='#d62728', linewidth=2)
    ax.set_yscale('log')
    ax.set_title('Training Loss History (Log Scale)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss (Log Scale)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend()
    ax.set_ylim(1e-6, 1e-2)
    # 保存
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300) # 300 dpi 高清保存
    plt.close()
    print(f"Loss Map 已保存至: {save_path}")
# ==========================================
# 3. 数据集 (全序列加载)
# ==========================================
class FullSequenceDataset(Dataset):
    def __init__(self, datapath, indices):
        # 1. 获取所有 pkl 文件
        self.files = [f for f in os.listdir(datapath) if f.endswith('.pkl')]
        # 2. 必须按照数字大小排序 (1, 2, 3...)
        self.files.sort(key=lambda x: int(x.split('.')[0]))
        self.files = [self.files[i] for i in indices]
        self.datapath = datapath
        print(f"全序列数据集加载: {len(self.files)} 个文件 -> {self.files}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        with open(os.path.join(self.datapath, file_name), 'rb') as f:
            data = pickle.load(f)
        # 预处理所有 Map 到 64x64
        data['time_list'] = torch.from_numpy(data['time_list'].astype(np.float32)).view(-1, 1)
        data['flux_list'] = torch.from_numpy(data['flux_list'].astype(np.float32))
        data['cooling_list'] = torch.from_numpy(data['cooling_list'].astype(np.float32))
        data['envtemp_list'] = torch.from_numpy(data['envtemp_list'].astype(np.float32)).view(-1, 1)
        data['temp_steady_list'] = torch.from_numpy(data['temp_steady_list'].astype(np.float32))
        data['temp_list'] = torch.from_numpy(data['temp_list'].astype(np.float32))
        return data
    
# 1. 静态坐标定义
chip_loc = [[65, 55, 50, 50], [15, 90, 30, 30], [142, 13.5, 20, 20], [142, 43.5, 20, 20],
            [142, 73.5, 20, 20], [142, 103.5, 20, 20], [115, 20, 10, 10], [90, 20, 10, 10], 
            [65, 20, 10, 10], [40, 20, 10, 10], [22, 40, 10, 10], [22, 65, 10, 10]]
cooling_loc = [[50-25, 63.5-25, 50, 50], [130-25, 63.5-25, 50, 50]]

chip_centers = torch.tensor([[l[0] + l[2]/2, l[1] + l[3]/2] for l in chip_loc], dtype=torch.float32)
fan_centers = torch.tensor([[l[0] + l[2]/2, l[1] + l[3]/2] for l in cooling_loc], dtype=torch.float32)
# ==========================================
# 4. 主训练逻辑
# ==========================================
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Database/0_Dataset'))
savepath = os.path.dirname(__file__)

point_labels = ["chip1", "chip2", "chip3", "chip4", "chip5", "chip6", 
                "chip7", "chip8", "chip9", "chip10", "chip11", "chip12"]

# 数据加载
batch_size = 2
time_D_min, time_D_max = 0, 5
epochs = 300
data_volume = 60
n = 2
training_data_volume = round(data_volume / math.sqrt(2) ** (n-1))
training_epochs = round(math.sqrt(2) ** (n-1) * epochs)
n_folds = 5
window_size = 80
time_data = '05'
device = torch.device('cpu')
random_seed = 42
init_log_bound = 1.0
power_reference = 30.0

datapath = os.path.join(path, f'Data_AT40_D1_{time_data}')

for n_fold in range(n_folds):
    fold_seed = random_seed + n_fold
    set_random_seed(fold_seed)
    dataloader_generator = torch.Generator().manual_seed(fold_seed)
    print(f"Fold {n_fold+1}: training_seed={fold_seed}, init_seed={fold_seed}, log_init=U(-{init_log_bound}, {init_log_bound})")

    fold_start_time = time.perf_counter()
    val_indices = list(range(60, 80))
    train_indices = list(range(training_data_volume))
    val_dataset = FullSequenceDataset(datapath, val_indices)
    train_dataset = FullSequenceDataset(datapath, train_indices)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=dataloader_generator,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_indices), shuffle=False)

    foldpath = os.path.join(savepath, f'fold_{n_fold+1}_bs_{batch_size}')
    modelfile = os.path.join(foldpath, 'models_1')
    lossfile = os.path.join(foldpath, 'losses_1')
    resultfile = os.path.join(foldpath, 'test_result')
    os.makedirs(modelfile, exist_ok=True)
    os.makedirs(lossfile, exist_ok=True)
    os.makedirs(resultfile, exist_ok=True)

    # 预训练模型路径 (请确保这里是你已经训练好的路径)
    model = Transient_Model(
        init_seed=fold_seed, init_log_bound=init_log_bound, power_reference=power_reference
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5,patience=10,min_lr=1e-5)

    print("开始递归微调...")

    best_score = None
    train_losses_G = []
    val_losses_G = []
    for epoch in range(training_epochs):
        epoch_start_time = time.perf_counter()
        # train_part======================================================================== # 
        model.train() 
        train_loss_G = 0
        # 遍历所有文件 (每个文件是一个 Batch, size=1)
        for file_idx, data in enumerate(train_dataloader):
            time_seq  = data['time_list'].to(device)         # [T, 1]
            flux_seq  = data['flux_list'].to(device)         # [T, 64, 64]
            cool_seq  = data['cooling_list'].to(device)      # [T, 64, 64]
            envtemp_seq = data['envtemp_list'].to(device)   # [T, 64, 64]
            gt_temp   = data['temp_list'].to(device)  # [T, 64, 64]
            B = time_seq.shape[0]
            # 每条序列只在起点使用一次真值初始化；窗口之间连续演化，
            # 仅在窗口边界校正温度并截断梯度图。
            v_curr = gt_temp[:, 0]
            sink_curr = model.init_sink_state(v_curr)
            # 文件级累积器
            file_loss = 0
            file_step_count = 0
            # === 随机周期的连续窗口 ===
            # 每个 epoch 的划分不同，但窗口按时间顺序首尾相接，
            # 完整覆盖整条序列，不重叠、不跳步。
            steps = time_seq.shape[1] - 1
            window_size_min = 60
            window_size_max = 100
            windows = []
            start_t = 0
            while start_t < steps:
                current_window_size = np.random.randint(window_size_min, window_size_max + 1)
                end_t = min(start_t + current_window_size, steps)
                windows.append((start_t, end_t))
                start_t = end_t
            for win_idx, (start_t, end_t) in enumerate(windows):
                batch_loss = 0.0
                batch_step = 0
                optimizer.zero_grad()
                for t in range(start_t, end_t):
                    dt = time_seq[:, t+1] - time_seq[:,t]
                    dt_norm = (dt - time_D_min) / (time_D_max - time_D_min)
                    time_D_v = dt_norm.view(B, 1)
                    flux_v = flux_seq[:, t]
                    cooling_v = cool_seq[:, t]
                    envtemp_v = envtemp_seq[:, t].view(B, 1)
                    v_gt_next = gt_temp[:, t+1]
                    # 2. 前向传播 (传入所有需要的物理量)
                    v_next_pred, sink_curr = model(
                        time_D_v, flux_v, cooling_v, envtemp_v, v_curr, sink_curr
                    )
                    loss_step = F.mse_loss(v_next_pred, v_gt_next)
                    batch_loss = batch_loss + loss_step
                    batch_step += 1
                    file_step_count += 1
                    file_loss += loss_step.item()
                    v_curr = v_next_pred
                batch_loss = batch_loss / batch_step
                batch_loss.backward()
                optimizer.step()

                # Truncated BPTT：在窗口边界用真实温度校正状态，
                # 并切断跨窗口计算图。
                v_curr = v_gt_next.detach()
                sink_curr = sink_curr.detach()

            # 计算该文件的平均单步 Loss
            train_loss_G += file_loss / file_step_count
            print(f"Epoch {epoch+1}, File {file_idx+1}/{len(train_dataloader)}, Loss: {file_loss / file_step_count:.6f}")
        train_losses_G.append(train_loss_G / len(train_dataloader))
        scheduler.step(train_losses_G[-1])
        print(f"==> Epoch {epoch+1} Finished.")
        print(f"    Total: {train_losses_G[-1]:.6f}")

        # validation_part======================================================================== # 
        with torch.inference_mode():
            model.eval()  # 切换到评估模式
            val_loss_G = 0.0
            # 遍历所有文件 (每个文件是一个 Batch, size=1)
            for batch_idx, data in enumerate(val_dataloader):
                time_seq  = data['time_list'].to(device)         # [T, 1]
                flux_seq  = data['flux_list'].to(device)         # [T, 64, 64]
                cool_seq  = data['cooling_list'].to(device)      # [T, 64, 64]
                envtemp_seq = data['envtemp_list'].to(device)   # [T, 64, 64]
                gt_temp   = data['temp_list'].to(device)  # [T, 64, 64]
                B = time_seq.shape[0]
                # 初始化
                pred_temp = []
                v_curr = gt_temp[:, 0]   # 初始状态 (只给第0帧真值) -> [1, 1, 64, 64]
                sink_curr = model.init_sink_state(v_curr)
                # 文件级累积器
                file_loss = 0.0
                # === 递归时间步 ===
                steps = time_seq.shape[1] - 1
                for t in range(steps):
                    dt = time_seq[:, t+1] - time_seq[:,t]
                    dt_norm = (dt - time_D_min) / (time_D_max - time_D_min)
                    time_D_v = dt_norm.view(B, 1)
                    flux_v = flux_seq[:, t]
                    cooling_v = cool_seq[:, t]
                    envtemp_v = envtemp_seq[:, t].view(B, 1)
                    v_gt_next = gt_temp[:, t+1]
                    # 2. 前向传播 (传入所有需要的物理量)
                    v_next_pred, sink_curr = model(
                        time_D_v, flux_v, cooling_v, envtemp_v, v_curr, sink_curr
                    )
                    loss_transient = F.mse_loss(v_next_pred, v_gt_next, reduction='mean')
                    v_curr = v_next_pred
                    pred_temp.append(v_next_pred)
                    # 3. 计算 Loss
                    loss_step = loss_transient
                    # 累积统计数据
                    file_loss += loss_step.item()
                # 计算该文件的平均单步 Loss
                val_loss_G += file_loss / steps

                # # -------------------------------------------------------------- (plotting)
                # pred_temp = torch.stack(pred_temp, dim=1)

                # time_seq_np_all = time_seq.detach().cpu().numpy()
                # gt_temp_np_all = gt_temp.detach().cpu().numpy()
                # pred_temp_np_all = pred_temp.detach().cpu().numpy()

                # for b in range(B):
                #     global_file_idx = batch_idx * len(val_indexes) + b + 1

                #     x_axis_data = time_seq_np_all[b].squeeze()
                #     gt_temp_np = gt_temp_np_all[b, 1:, :]
                #     pred_temp_np = pred_temp_np_all[b]
                #     length = min(len(gt_temp_np), len(pred_temp_np))

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
                #         ln3 = ax_err.plot(x_axis_data[1:length+1], (110 - 20) * np.abs(pred_temp_np[:length, i] - gt_temp_np[:length, i]), 'b--', label='Error', alpha=0.6, linewidth=1.5)
                #         ax_err.set_ylim(0, 5)
                #         ax_err.tick_params(axis='y', labelcolor='blue')

                #         ax.set_title(f"{point_labels[i]}", fontsize=10, fontweight='bold')
                #         ax.axvline(x=800, color='blue', linestyle='--', linewidth=1, alpha=0.4, zorder=1)
                #         ax.axvline(x=1600, color='blue', linestyle='--', linewidth=1, alpha=0.4, zorder=1)

                #         if i == 0:
                #             lns = ln1 + ln2 + ln3
                #             labs = [l.get_label() for l in lns]
                #             ax.legend(lns, labs, loc='upper right', fontsize=8)

                #     plt.tight_layout()
                #     plt.savefig(os.path.join(resultfile, f'temperature_curves_{epoch+1}_{global_file_idx}.png'), dpi=150)
                #     plt.close()
                # # --------------------------------------------------------------

            val_losses_G.append(val_loss_G / len(val_dataloader))

            # 保存模型
            if best_score is None or math.isnan(float(best_score)):
                best_score = val_losses_G[-1]
                print('val: ', val_losses_G[-1], '; ', 'best: ', best_score)
            else:
                print('val: ', val_losses_G[-1], '; ', 'best: ', best_score)
                if val_losses_G[-1] < best_score:
                    print('clear')
                    torch.save(model.state_dict(), os.path.join(modelfile, f'model_epoch_{epoch+1}.pth'))
                    best_score = val_losses_G[-1]
                    for file_name in os.listdir(resultfile):
                        if file_name.startswith("temperature_curves_") and file_name.endswith(".png"):
                            current_prefix = f"temperature_curves_{epoch + 1}_"
                            if not file_name.startswith(current_prefix):
                                os.remove(os.path.join(resultfile, file_name))
                    for file_name in os.listdir(modelfile):
                        if file_name.startswith("model_epoch_") and file_name.endswith(".pth"):
                            current_model_name = f"model_epoch_{epoch + 1}.pth"
                            if file_name != current_model_name:
                                os.remove(os.path.join(modelfile, file_name))


        # loss存储为numpy文件
        if (epoch+1) % 10 == 0: 
            np.save(os.path.join(lossfile, 'Glosses_train_%d.npy' % (epoch+1)), train_losses_G)
            np.save(os.path.join(lossfile, 'Glosses_val_%d.npy' % (epoch+1)), val_losses_G)

        epoch_elapsed = time.perf_counter() - epoch_start_time
        print(f"    Epoch Time: {epoch_elapsed:.2f} s")
        estimated_total_time = epoch_elapsed * training_epochs
        estimated_remaining_time = estimated_total_time - epoch_elapsed * (epoch + 1)
        print(f"    Estimated Total Training Time: {estimated_total_time / 3600:.2f} h")
        print(f"    Estimated Remaining Time: {max(0.0, estimated_remaining_time) / 3600:.2f} h")

    # 保存loss曲线图  
    plot_loss_map(train_losses_G, val_losses_G, foldpath, f'transient_loss_{epoch+1}.png')
    fold_elapsed = time.perf_counter() - fold_start_time
    print(f"Fold {n_fold+1} Total Time: {fold_elapsed:.2f} s")

# os.system("/usr/bin/shutdown")
