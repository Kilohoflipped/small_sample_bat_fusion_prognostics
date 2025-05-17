import logging
import os
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from config.config_loader import find_config_dir, find_project_root_dir, load_and_merge_configs
from config.config_setter import setup_logging
from src.models.vae.data_loader import BatteryDataLoader
from src.models.vae.vae import ConditionalVAE

logger = logging.getLogger(__name__)

# 定义用于查找项目根目录的标记文件
PROJECT_ROOT_MARKER: str = "pyproject.toml"
# 定义配置目录在项目根目录下的相对路径
CONFIG_DIR_RELATIVE_PATH: str = "config"

# 定义配置文件名
LOG_CONFIG_PATH: str = "logging.yaml"
DATA_IO_PATH_CONFIG_PATH: str = "data_io_paths.yaml"
VAE_MODEL_CONFIG_PATH: str = "model/vae.yaml"

# 环境条件边界 (用于数据加载和归一化)
CONDITION_BOUNDS: List[List[float]] = [
    [0.0, 2.0],  # 充电率
    [0.0, 2.0],  # 放电率
    [10.0 + 273.15, 65.0 + 273.15],  # 温度 (°C), 转换为开尔文
    [0.0, 500.0],  # 压强
    [0.0, 100.0],  # DOD
]

# 排除的电池 ID 列表.
EXCLUDE_BATTERY_IDS: List[Any] = [1, 2, 3, 4, 5, 17]


# --- 主函数：准备潜在空间数据 ---
def prepare_latent_data(config_dict: Dict[str, Any], project_root_dir: str):
    """
    使用训练好的 CVAE Encoder 将预处理数据编码到潜在空间，并保存潜在向量和对应的条件。

    Args:
        config_dict: 配置字典
        project_root_dir: 项目根目录路径
    """

    # 从配置中获取各个部分的参数和路径
    preprocessed_data_dir = config_dict.get("output_dirs", {}).get("preprocessed_data", "")
    preprocessed_data_dir = os.path.join(project_root_dir, preprocessed_data_dir)
    standardized_data_path = os.path.join(
        preprocessed_data_dir, "step4_battery_aging_cycle_data_standardized.csv"
    )
    vae_checkpoint_path = config_dict.get("output_dirs", {}).get("vae_model", "")
    vae_checkpoint_path = os.path.join(project_root_dir, vae_checkpoint_path, "vae_model_final.pth")

    vae_config = config_dict.get("vae_model", {})

    # 获取用于保存潜在空间数据的中间目录的相对路径
    interim_data_relative_dir = config_dict.get("output_dirs", {}).get("interim_data", "")

    # --- 构建输出文件的完整路径 ---
    # 首先构建潜在空间数据的输出目录
    output_latent_dir = os.path.join(
        project_root_dir,
        interim_data_relative_dir,
        "vae_latent",  # 在中间数据目录下创建一个名为 vae_latent 的子目录
    )
    # 构建潜在向量保存文件的完整路径
    output_latent_file = os.path.join(
        output_latent_dir,
        "vae_latent_data.pt",
    )
    # 构建条件信息保存文件的完整路径
    output_condition_file = os.path.join(
        output_latent_dir,  # 使用相同的输出目录
        "vae_latent_condition.pt",
    )

    # --- 校验必要配置 ---
    if not standardized_data_path:
        raise ValueError("配置文件中必须指定 'data.processed.standardized_data' 的路径。")
    if not vae_checkpoint_path:
        raise ValueError(
            "配置文件中必须指定 'model.vae.checkpoint_path' (训练好的 VAE 权重) 的路径。"
        )
    if not output_latent_file or not output_condition_file:
        raise ValueError(
            "配置文件中必须指定 'data.processed.vae_latent_data' 和 'data.processed.vae_latent_conditions' 的保存路径。"
        )

    # --- 设置设备 ---
    device_using = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device_using}")

    # --- 设置数据加载器 (用于预处理数据) ---
    print("\n--- 加载预处理数据 ---")
    battery_dataloader_instance = BatteryDataLoader(
        csv_path=standardized_data_path,
        cond_bounds=CONDITION_BOUNDS,
        exclude_battery_ids=EXCLUDE_BATTERY_IDS,
        shuffle=False,
        batch_size=vae_config.get("batch_size", 4),
    )
    dataloader = battery_dataloader_instance.create_loader()
    if (
        not hasattr(battery_dataloader_instance.dataset, "sequences")
        or not battery_dataloader_instance.dataset.sequences
    ):
        raise ValueError(
            "BatteryDataLoader/Dataset 未成功加载序列数据，请检查 BatteryDataLoader 初始化。"
        )

    # --- 加载训练好的 CVAE 模型 (Encoder 部分) ---
    print("\n--- 加载训练好的 CVAE 模型 ---")
    # 实例化完整的 CVAE 模型（因为权重文件通常保存的是整个模型的 state_dict）
    cvae_model = ConditionalVAE(
        input_dim=vae_config.get("input_dim", 1),
        hidden_dim=vae_config.get("hidden_dim", 128),
        condition_dim=vae_config.get("condition_dim", 5),
        latent_dim=vae_config.get("latent_dim", 64),
        output_dim=vae_config.get("output_dim", 1),
    )

    # 加载训练好的模型权重
    if not os.path.exists(vae_checkpoint_path):
        raise FileNotFoundError(f"CVAE 检查点文件未找到: {vae_checkpoint_path}")

    try:
        cvae_model.load_state_dict(torch.load(vae_checkpoint_path, map_location=device_using))
        cvae_model.to(device_using)
        print("CVAE 模型权重加载成功。")
    except Exception as e:
        print(f"加载或应用 CVAE 模型权重时出错: {e}")
        raise

    # 获取 Encoder 部分
    if not hasattr(cvae_model, "encoder"):
        raise AttributeError("您的 CVAEModel 类中没有名为 'encoder' 的属性。请检查模型定义。")
    encoder = cvae_model.encoder
    print("成功获取 CVAE Encoder。")

    # --- 设置模型为评估模式并禁用梯度计算 ---
    encoder.eval()  # 设置为评估模式，影响 BatchNorm 和 Dropout 等层
    print("Encoder 设置为评估模式。")

    # --- 编码数据到潜在空间 ---
    print("\n--- 开始编码数据 ---")
    all_latents = []  # 用于收集所有潜在向量
    all_conditions = []  # 用于收集所有条件信息

    with torch.no_grad():  # 在推理（编码）过程中禁用梯度计算
        # collate_fn 返回 (padded_seqs, conditions, lengths, battery_ids)
        # 其中 padded_seqs 形状是 (batch_size, max_len, 1)
        # conditions 形状是 (batch_size, condition_dim)
        # lengths 形状是 (batch_size,)
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="编码数据到潜在空间", unit="batch")
        ):
            # 解包批次数据
            batch_x0_padded, batch_y, batch_length, batch_ids = batch

            # 将数据和条件发送到设备
            batch_x0_padded = batch_x0_padded.to(device_using)
            batch_y = batch_y.to(device_using)
            batch_length = batch_length.to(device_using)  # Encoder forward 需要 lengths

            # --- 通过 Encoder 前向传播 ---
            # 您的 VaeEncoder forward 是 encoder(input_sequences, input_sequences_lengths, conditions)
            # 所以将填充后的序列、实际长度和条件送入 Encoder
            latent_mean, latent_logvar = encoder(
                batch_x0_padded, batch_length, batch_y
            )  # 根据您的 Encoder 调用调整

            # 收集潜在空间的均值 (作为 z_0) 和对应的条件信息
            all_latents.append(latent_mean.cpu())  # 移回 CPU
            all_conditions.append(batch_y.cpu())  # 收集条件信息，也移回 CPU

    # 将所有批次的潜在向量和条件信息连接起来
    all_latents = torch.cat(all_latents, dim=0)
    all_conditions = torch.cat(all_conditions, dim=0)

    print(f"\n数据编码完成。总共编码了 {all_latents.shape[0]} 个潜在向量。")
    print(f"所有潜在向量的总形状: {all_latents.shape}")
    print(f"所有条件信息的总形状: {all_conditions.shape}")

    # --- 保存潜在空间数据和条件 ---
    print("\n--- 保存潜在空间数据和条件 ---")
    # 确保保存文件的目录存在
    output_dir = os.path.dirname(output_latent_file)
    os.makedirs(output_dir, exist_ok=True)

    # 保存潜在向量和条件信息张量到 .pt 文件
    try:
        torch.save(all_latents, output_latent_file)
        torch.save(all_conditions, output_condition_file)
        print(f"潜在数据保存到: {output_latent_file}")
        print(f"条件数据保存到: {output_condition_file}")
        print("\n潜在空间数据和条件准备完成，可用于训练 Conditional DDPM。")
    except Exception as e:
        print(f"保存潜在数据或条件数据时出错: {e}")
        raise


# --- 脚本入口点 ---
if __name__ == "__main__":
    project_root_dir: str = find_project_root_dir(marker_file=PROJECT_ROOT_MARKER)
    config_dir: str = find_config_dir(
        project_root_dir=project_root_dir, config_dir_relative_path=CONFIG_DIR_RELATIVE_PATH
    )
    log_config_path: str = os.path.join(config_dir, LOG_CONFIG_PATH)
    data_io_path_config_path: str = os.path.join(config_dir, DATA_IO_PATH_CONFIG_PATH)
    vae_model_config_path: str = os.path.join(config_dir, VAE_MODEL_CONFIG_PATH)

    setup_logging(log_config_path)

    all_config_files: List[str] = [
        data_io_path_config_path,
        vae_model_config_path,
    ]
    config = load_and_merge_configs(all_config_files)

    prepare_latent_data(config, project_root_dir)
