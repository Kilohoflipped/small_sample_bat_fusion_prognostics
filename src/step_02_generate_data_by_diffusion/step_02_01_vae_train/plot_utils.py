import os

import numpy as np
import matplotlib.pyplot as plt

def plot_battery_sequences(original_sequences, reconstructed_sequences, battery_ids, lengths, output_dir):
    """Plot original and reconstructed sequences for each battery with MSE calculation"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unique_batteries = sorted(set(battery_ids))
    mse_dict = {}  # Store MSE for each battery

    for battery_id in unique_batteries:
        indices = [i for i, bid in enumerate(battery_ids) if bid == battery_id]
        if not indices:
            continue
        idx = indices[0]
        orig_seq = original_sequences[idx]
        recon_seq = reconstructed_sequences[idx]
        seq_len = lengths[idx]

        # Validate lengths
        if len(orig_seq) != seq_len or len(recon_seq) != seq_len:
            print(
                f"Warning: Length mismatch for battery {battery_id}: orig={len(orig_seq)}, recon={len(recon_seq)}, expected={seq_len}")
            seq_len = min(len(orig_seq), len(recon_seq), seq_len)
            orig_seq = orig_seq[:seq_len]
            recon_seq = recon_seq[:seq_len]

        # Calculate MSE
        mse = np.mean((orig_seq - recon_seq) ** 2)
        mse_dict[battery_id] = mse

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(orig_seq, label='Original', color='blue', linestyle='-')
        plt.plot(recon_seq, label=f'Reconstructed (MSE: {mse:.4f})', color='red', linestyle='--')
        plt.title(f'Battery ID {battery_id} - Capacity Over Cycles (Length: {seq_len})')
        plt.xlabel('Cycle Index')
        plt.ylabel('Capacity')
        plt.legend()
        plt.grid(True)
        output_path = os.path.join(output_dir, f'battery_{battery_id}.png')
        plt.savefig(output_path)
        plt.close()

    # Save MSE summary
    mse_summary_path = os.path.join(output_dir, 'mse_summary.txt')
    with open(mse_summary_path, 'w') as f:
        for bid, mse in mse_dict.items():
            f.write(f'Battery {bid}: MSE = {mse:.4f}\n')
        avg_mse = np.mean(list(mse_dict.values()))
        f.write(f'Average MSE: {avg_mse:.4f}\n')

    return mse_dict

def visualize_sequences(original_sequences, standardized_sequences, battery_ids, lengths, output_dir, target_mean=None, target_std=None):
    """Visualize original and standardized sequences for each battery to check for anomalies."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unique_batteries = sorted(set(battery_ids))

    for battery_id in unique_batteries:
        indices = [i for i, bid in enumerate(battery_ids) if bid == battery_id]
        if not indices:
            continue
        idx = indices[0]  # 假设每个电池ID只有一个序列
        orig_seq = original_sequences[idx]
        std_seq = standardized_sequences[idx]
        seq_len = lengths[idx]

        # 确保序列长度一致
        if len(orig_seq) != seq_len or len(std_seq) != seq_len:
            print(f"Warning: Length mismatch for battery {battery_id}: orig={len(orig_seq)}, std={len(std_seq)}, expected={seq_len}")
            seq_len = min(len(orig_seq), len(std_seq), seq_len)
            orig_seq = orig_seq[:seq_len]
            std_seq = std_seq[:seq_len]

        # 绘制原始序列和标准化序列
        plt.figure(figsize=(12, 8))

        # 子图1：原始序列
        plt.subplot(2, 1, 1)
        plt.plot(orig_seq, label='Original Sequence', color='blue')
        plt.title(f'Battery ID {battery_id} - Original Capacity Sequence (Length: {seq_len})')
        plt.xlabel('Cycle Index')
        plt.ylabel('Capacity')
        plt.legend()
        plt.grid(True)

        # 子图2：标准化序列
        plt.subplot(2, 1, 2)
        plt.plot(std_seq, label='Standardized Sequence', color='green')
        plt.title(f'Battery ID {battery_id} - Standardized Sequence (Mean: {target_mean:.4f}, Std: {target_std:.4f})')
        plt.xlabel('Cycle Index')
        plt.ylabel('Normalized Capacity')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'visualize_battery_{battery_id}.png')
        plt.savefig(output_path)
        plt.close()

    print(f"Visualization plots saved to {output_dir}")
