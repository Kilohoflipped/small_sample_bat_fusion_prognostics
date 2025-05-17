import logging
import os
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config_loader import find_config_dir, find_project_root_dir, load_and_merge_configs
from config.config_setter import setup_logging
from src.models.diffusion.conditional_denoising_model import ConditionalDenoisingModel
from src.models.diffusion.data_lodaer import LatentDataset
from src.models.diffusion.diffusion_process import q_sample
from src.models.diffusion.schedulers import get_scheduler

logger = logging.getLogger(__name__)

# 定义用于查找项目根目录的标记文件
PROJECT_ROOT_MARKER: str = "pyproject.toml"
# 定义配置目录在项目根目录下的相对路径
CONFIG_DIR_RELATIVE_PATH: str = "config"

# 定义配置文件名
LOG_CONFIG_PATH: str = "logging.yaml"
DATA_IO_PATH_CONFIG_PATH: str = "data_io_paths.yaml"
DIFFUSION_CONFIG_PATH: str = "model/diffusion.yaml"


# --- 主训练函数 ---
def train(config_dict: Dict[str, Any], project_root_dir: str):
    """
    训练 Conditional DDPM 模型的主函数。

    Args:
        config_dict: 配置字典
        project_root_dir: 项目根目录路径
    """
    # --- 加载配置 ---

    # 从配置中获取不同部分的参数
    train_config = config_dict.get("diffusion_model", {}).get("training", {})  # 训练相关的参数
    model_config = config_dict.get("diffusion_model", {}).get("model", {})  # 扩散模型整体配置
    scheduler_config = model_config.get("scheduler", {})

    data_paths_interim = config_dict.get("output_dirs", {}).get("interim_data", "")
    data_paths_interim = os.path.join(project_root_dir, data_paths_interim)

    print("--- 加载配置 ---")

    # --- 设置设备 ---
    # 优先使用 GPU (cuda)，如果不可用则使用 CPU
    device_using = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device_using}")

    # --- 设置数据加载器 ---
    print("\n--- 设置数据加载器 ---")
    # 从配置中获取潜在数据和条件文件的路径
    latent_data_file = os.path.join(data_paths_interim, "vae_latent", "vae_latent_data.pt")
    condition_data_file = os.path.join(data_paths_interim, "vae_latent", "vae_latent_condition.pt")

    if not latent_data_file or not condition_data_file:
        raise ValueError(
            "配置文件中必须指定 'data.processed.vae_latent_data' "
            "和 'data.processed.vae_latent_conditions' 的路径。"
        )

    # 实例化 LatentDataset
    dataset = LatentDataset(latent_data_file, condition_data_file)

    # 实例化 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.get("batch_size", 6),
        shuffle=train_config.get("shuffle_data", True),
        num_workers=train_config.get("num_workers", 0),
        pin_memory=(train_config.get("pin_memory", True) if device_using.type == "cuda" else False),
    )
    print(f"DataLoader 初始化完成，总共 {len(dataloader)} 批数据。")

    # --- 实例化模型 ---
    print("\n--- 实例化模型 ---")
    # 从 dataset 中获取潜在维度和条件维度，确保模型参数匹配
    model = ConditionalDenoisingModel(
        latent_dim=dataset.latent_dim,
        condition_input_dim=dataset.condition_dim,
        time_embedding_dim=model_config.get("network_params", {}).get("time_embedding_dim", 128),
        condition_embedding_dim=model_config.get("network_params", {}).get(
            "condition_embedding_dim", 128
        ),
        model_channels=model_config.get("network_params", {}).get("model_channels", 256),
        num_mlp_blocks=model_config.get("network_params", {}).get("num_mlp_blocks", 3),
    )
    model.to(device_using)  # 将模型发送到指定设备
    print("ConditionalDenoisingModel 初始化完成。")

    # --- 实例化调度器 ---
    print("\n--- 实例化调度器 ---")
    # 使用之前实现的 get_scheduler 函数
    scheduler = get_scheduler(scheduler_config)
    # 调度器本身不需要发送到设备，它主要处理参数和计算逻辑
    # 但是，我们需要从调度器获取 alpha_bars 并将其发送到设备
    if not hasattr(scheduler, "alphas_cumprod"):
        raise AttributeError("所选的调度器对象没有 'alphas_cumprod' 属性，无法进行前向扩散。")
    alpha_bars = scheduler.alphas_cumprod.to(device_using)
    total_timesteps = scheduler.config.num_train_timesteps  # 总的扩散时间步 T

    print(f"总扩散时间步 (T): {total_timesteps}")
    print(f"alpha_bars 已准备在 {device_using} 上。")

    # --- 设置优化器 ---
    print("\n--- 设置优化器 ---")
    optimizer = optim.AdamW(  # AdamW 是处理权重衰减的常用选择
        model.parameters(),
        lr=train_config.get("learning_rate", 1e-4),
        weight_decay=train_config.get("weight_decay", 0.0),  # 从配置获取权重衰减
    )
    print(f"优化器初始化完成: {type(optimizer).__name__}, 学习率: {optimizer.defaults['lr']}")

    # --- 训练循环 ---
    print("\n--- 开始训练 ---")
    num_epochs = train_config.get("num_epochs", 100)  # 从配置获取总训练 epoch 数
    save_interval = train_config.get("save_interval", 10)  # 从配置获取保存模型间隔
    checkpoint_dir = config_dict.get("output_dirs", {}).get("diffusion_model", "")
    checkpoint_dir = os.path.join(project_root_dir, checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)  # 确保检查点目录存在

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        total_epoch_loss = 0  # 记录当前 epoch 的总损失

        # 使用 tqdm 包装 dataloader 以显示进度条和当前 loss
        dataloader_tqdm = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for batch_idx, (batch_z_0, batch_y) in enumerate(dataloader_tqdm):
            # 将数据和条件发送到设备
            batch_z_0 = batch_z_0.to(device_using)
            batch_y = batch_y.to(device_using)

            # --- 前向扩散步骤 ---
            # 为批次中的每个样本随机采样一个时间步 t (0 到 T-1)
            # Diffusers 通常使用 0 到 T-1，Scheduler 在 step 中处理映射
            # 或者，根据调度器需要，采样 1 到 T
            # 这里我们采样 0 到 T-1，假设调度器内部或模型能处理
            # 简单起见，采样 [0, total_timesteps-1]
            t = torch.randint(0, total_timesteps, (batch_z_0.shape[0],), device=device_using).long()

            # 生成与原始潜在向量形状相同的随机噪声
            noise = torch.randn_like(batch_z_0)

            # 应用前向扩散过程，从 z_0 和噪声在时间步 t 计算 z_t
            # 推荐使用调度器的 add_noise 方法
            # 注意：并非所有 Diffusers 调度器都有 add_noise 方法，DDPMScheduler 和 DDIMScheduler 有
            # 如果您使用的是这些调度器，优先使用它们的方法
            if hasattr(scheduler, "add_noise"):
                z_t = scheduler.add_noise(batch_z_0, noise, t)
            else:
                # 如果调度器没有 add_noise，则使用我们实现的 q_sample 函数
                # 这要求 scheduler 已经计算并提供了 alpha_bars
                z_t = q_sample(batch_z_0, t, noise, alpha_bars)

            # --- 模型预测 ---
            # 将带噪声的潜在向量 z_t, 时间步 t, 条件 y 输入到 Conditional 去噪模型
            # 模型预测添加到 z_t 中的噪声
            predicted_noise = model(z_t, t, batch_y)

            # --- 计算损失 ---
            # 损失通常是模型预测的噪声与实际添加到数据中的噪声之间的均方误差 (MSE)
            loss = F.mse_loss(predicted_noise, noise)

            # --- 反向传播和优化器步骤 ---
            optimizer.zero_grad()  # 清除之前的梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新模型参数

            total_epoch_loss += loss.item()  # 累加损失

            # 更新 tqdm 进度条的显示信息
            dataloader_tqdm.set_postfix(loss=loss.item())

        # 计算当前 epoch 的平均损失
        avg_epoch_loss = total_epoch_loss / len(dataloader)
        # 在进度条结束后打印 epoch 总结信息
        print(f"Epoch {epoch+1}/{num_epochs} 结束。平均损失: {avg_epoch_loss:.6f}")

        # --- 检查点保存 ---
        # 每隔 save_interval 个 epoch 或在最后一个 epoch 保存模型权重
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"conditional_diffusion_model_epoch_{epoch+1}.pth"
            )
            # 保存模型的 state_dict
            torch.save(model.state_dict(), checkpoint_path)
            print(f"检查点已保存到 {checkpoint_path}")
            # 可选：同时保存优化器的状态、学习率调度器的状态等，以便后续恢复训练
        if (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"conditional_diffusion_model_epoch_final.pth"
            )
            # 保存模型的 state_dict
            torch.save(model.state_dict(), checkpoint_path)
            print(f"检查点已保存到 {checkpoint_path}")
            # 可选：同时保存优化器的状态、学习率调度器的状态等，以便后续恢复训练

    print("\n--- 训练完成 ---")


# --- 脚本入口点 ---
if __name__ == "__main__":
    project_root_dir: str = find_project_root_dir(marker_file=PROJECT_ROOT_MARKER)
    config_dir: str = find_config_dir(
        project_root_dir=project_root_dir, config_dir_relative_path=CONFIG_DIR_RELATIVE_PATH
    )
    log_config_path: str = os.path.join(config_dir, LOG_CONFIG_PATH)
    data_io_path_config_path: str = os.path.join(config_dir, DATA_IO_PATH_CONFIG_PATH)
    diffusion_config_path: str = os.path.join(config_dir, DIFFUSION_CONFIG_PATH)

    setup_logging(log_config_path)

    all_config_files: List[str] = [
        data_io_path_config_path,
        diffusion_config_path,
    ]
    config = load_and_merge_configs(all_config_files)

    train(config, project_root_dir)
