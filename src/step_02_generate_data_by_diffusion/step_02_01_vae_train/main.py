"""
训练条件变分自编码器 (Conditional VAE) 的主模块
"""

# 标准库模块
import os

# 第三方库模块
import numpy as np
import torch

# 本地模块
import vae_model
import data_loader
import plot_utils


if __name__ == '__main__':
    # 文件路径
    DATA_CSV_PATH = 'D:/OneDrive/Project/Tertiary_Edu/Bachelor\'s/Culmination Design/Codes/' \
                 'steps/step_01_data_conversion/data/processed/battery_aging_cycle_data.csv'

    SAVE_DIR = 'D:/OneDrive/Project/Tertiary_Edu/Bachelor\'s/Culmination Design/Codes/' \
                 'steps/step_02_generate_data_by_diffusion/step_02_01_vae_train/plots'
    
    
    # 超参数
    """
    输入特征维度: 对于LSTM, 输入特征维度是指每个时间步长中包含的特征数量
    电池老化序列(容量)是单变量时间序列, 因此输入特征维度为1
    """
    INPUT_DIM = 1
    
    """
    输出特征维度: 对于LSTM, 输出特征维度是指每个时间步长中包含的特征数量
    电池老化序列(容量)是单变量时间序列, 因此输出特征维度为1
    """
    OUTPUT_DIM = 1
    
    """
    环境条件维度: 对于电池老化序列序列为5
    分别为充电率, 放电率, 温度, 压强, DOD
    """
    CONDITION_DIM = 5
    
    """
    隐藏层维度: 对于LSTM, 隐藏层维度是指LSTM权重矩阵的大小
    或者, 在实际的实现中, 可以解释为多个并行的LSTM子单元
    每个子单元具有权重矩阵的一部分, 每个子单元仅负责一个分量
    """
    HIDDEN_DIM = 512
    
    """
    潜空间维度: 潜空间是一个n维的空间, VAE通过训练给出了两个分布
    一个是n维空间上的多元正态分布, 即Q(z|x), 由VAE Encoder给出, 作为对真实后验P(z|x)的变分近似
    Q(z|x)之被建模为正态分布, 是因为我们假设先验分布P(z)是正态的
    一个是数据空间上的似然分布P(x|z), 由VAE Decoder给出
    P(x|z)的具体形式由数据集的本身性质决定
    """
    LATENT_DIM = 32

    NUM_EPOCHS = 10000    # 训练轮次
    BATCH_SIZE = 8      # 训练批次大小
    DEVICE = torch.device("cuda")   # 在GPU上训练

    CONDITION_BOUNDS = [
        [0.0, 2.0],     # 充电率
        [0.0, 2.0],     # 放电率
        [10.0+273.15, 65.0+273.15], # 温度 (°C)
        [0.0, 500.0],   # 压强
        [0.0, 100.0]    # DOD
    ]
    EXCLUDE_BATTERY_IDS = [2, 3, 4, 5, 17]
    
    
    battery_data_loader = data_loader.BatteryDataLoader(
        csv_path=DATA_CSV_PATH,
        cond_bounds=CONDITION_BOUNDS,
        exclude_battery_ids=EXCLUDE_BATTERY_IDS,
        shuffle=False,  # 不打乱顺序以保持一致性
        batch_size=BATCH_SIZE
    )

    # 获取原始和标准化序列
    processor = battery_data_loader.processor
    original_sequences = processor.processed_data['raw_sequences']  # 原始序列
    standardized_sequences = processor.processed_data['sequences']  # 标准化序列
    battery_ids = processor.processed_data['battery_ids']
    lengths = processor.processed_data['lengths']
    target_mean = processor.processed_data['target_mean'].item()
    target_std = processor.processed_data['target_std'].item()

    # 可视化
    visualize_dir = os.path.join(SAVE_DIR, 'data_visualization')
    plot_utils.visualize_sequences(
        original_sequences=original_sequences,
        standardized_sequences=standardized_sequences,
        battery_ids=battery_ids,
        lengths=lengths,
        output_dir=visualize_dir,
        target_mean=target_mean,
        target_std=target_std
    )

    battery_data_loader = data_loader.BatteryDataLoader(
        csv_path=DATA_CSV_PATH,
        cond_bounds=CONDITION_BOUNDS,
        exclude_battery_ids=EXCLUDE_BATTERY_IDS,
        shuffle=True,
        batch_size=BATCH_SIZE)
    dataloader = battery_data_loader.create_loader()

    # Initialize the VAE
    vae = vae_model.ConditionalVAE(
        INPUT_DIM, HIDDEN_DIM, LATENT_DIM, CONDITION_DIM, OUTPUT_DIM).to(DEVICE)

    vae_model.train_vae(vae, dataloader, NUM_EPOCHS, DEVICE)

    # 绘图
    # Inference
    vae.eval()
    inference_loader = data_loader.BatteryDataLoader(
        csv_path=DATA_CSV_PATH,
        cond_bounds=CONDITION_BOUNDS,
        exclude_battery_ids=EXCLUDE_BATTERY_IDS,
        shuffle=False,
        batch_size=1
    )

    # Get raw data from the same processor used in training
    original_sequences = inference_loader.processor.processed_data['sequences']
    original_lengths = inference_loader.processor.processed_data['lengths']
    original_battery_ids = inference_loader.processor.processed_data['battery_ids']

    reconstructed_sequences = []
    recon_battery_ids = []
    recon_lengths = []

    # 推理部分（替换原文件的第 177 行到第 206 行的代码）
    try:
        with torch.no_grad():
            for batch in inference_loader.create_loader():
                sequences, conditions, lengths, battery_ids = batch
                sequences = sequences.to(DEVICE)
                conditions = conditions.to(DEVICE)
                lengths = lengths.to(DEVICE)

                try:
                    recon_seqs, _, _ = vae(sequences, lengths, conditions, lengths)
                    recon_seqs = recon_seqs.cpu().numpy()

                    for i in range(recon_seqs.shape[0]):
                        seq_len = lengths[i].item()
                        recon_seq = recon_seqs[i, :seq_len, 0]
                        # 反标准化
                        target_std_np = inference_loader.target_std.cpu().numpy()  # 转换为 NumPy
                        target_mean_np = inference_loader.target_mean.cpu().numpy()  # 转换为 NumPy
                        recon_seq = recon_seq * target_std_np + target_mean_np
                        reconstructed_sequences.append(recon_seq)
                        recon_lengths.append(seq_len)
                        recon_battery_ids.append(battery_ids[i].item())

                except RuntimeError as e:
                    print(f"Error processing batch for battery {battery_ids}: {e}")
                    continue

        # 验证数据一致性
        if len(original_battery_ids) != len(recon_battery_ids):
            print(
                f"Warning: Mismatch in number of batteries: original={len(original_battery_ids)}, reconstructed={len(recon_battery_ids)}")

        # 将原始序列转换为 NumPy 并反标准化
        original_sequences_np = []
        target_std_np = inference_loader.target_std.cpu().numpy()  # 转换为 NumPy
        target_mean_np = inference_loader.target_mean.cpu().numpy()  # 转换为 NumPy
        for seq in original_sequences:
            seq_np = seq.numpy() * target_std_np + target_mean_np
            original_sequences_np.append(seq_np)

        # 绘图并获取 MSE 统计
        mse_dict = plot_utils.plot_battery_sequences(
            original_sequences_np,
            reconstructed_sequences,
            recon_battery_ids,
            recon_lengths,
            SAVE_DIR
        )
        print(f"Plots and MSE summary saved to {SAVE_DIR}")
        print(f"Average MSE across batteries: {np.mean(list(mse_dict.values())):.4f}")

    except Exception as e:
        print(f"Error during inference: {e}")
        raise
