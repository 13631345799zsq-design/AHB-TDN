import numpy as np
import os
import matplotlib.pyplot as plt

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
    ax.set_ylim(1e-5, 1e0)
    # 保存
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300) # 300 dpi 高清保存
    plt.close()
    print(f"Loss Map 已保存至: {save_path}")

n_folds = 1              # 对应的 fold 编号
epoch_to_load = 50      # 想要读取哪一个 epoch 保存的 npy 文件

for n_fold in range(n_folds):
    # 自动获取当前脚本路径
    path = os.path.dirname(__file__) if os.path.dirname(__file__) else "."
    foldpath = os.path.join(path, f'fold_{n_fold+1}_bs_2')
    lossfile = os.path.join(foldpath, 'losses_1')
    resultfile = os.path.join(foldpath, 'test_result')
    filename = f"Loss_Curve_Analysis.png"

    # 确保结果文件夹存在
    os.makedirs(resultfile, exist_ok=True)

    # 定义文件路径
    train_npy_path = os.path.join(lossfile, 'Glosses_train_%d.npy' % epoch_to_load)
    val_npy_path = os.path.join(lossfile, 'Glosses_val_%d.npy' % epoch_to_load)

    # 3. 数据读取与处理
    print(f"正在读取数据: {train_npy_path} ...")
    loss_train_history = np.load(train_npy_path)
    loss_validation_history = np.load(val_npy_path)
    # 转换为 numpy 数组确保万无一失
    loss_train_np = np.array(loss_train_history)
    loss_validation_np = np.array(loss_validation_history)

    # 4. 绘图 (完全遵循你提供的视觉规范)
    plot_loss_map(loss_train_np, loss_validation_np,  foldpath, filename)

    print("-" * 30)
    print(f"处理完成！")