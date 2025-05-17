"""
训练条件变分自编码器 (Conditional VAE) 的主模块.
"""

import logging
import os
from typing import Any, List

import torch

from config.config_loader import (
    find_config_dir,
    find_project_root_dir,
    load_and_merge_configs,
)
from config.config_setter import setup_logging
from src.models.vae import data_loader
from src.models.vae.vae import ConditionalVAE

logger = logging.getLogger(__name__)

# 原始数据名称
DATA_FILENAME = "step4_battery_aging_cycle_data_standardized.csv"
# 模型保存文件路径.
MODEL_FILENAME = "vae_model_final.pth"


# --- 超参数配置 ---
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

# 定义用于查找项目根目录的标记文件
PROJECT_ROOT_MARKER = "pyproject.toml"
# 定义配置目录在项目根目录下的相对路径
CONFIG_DIR_RELATIVE_PATH = "config"

# 定义日志配置文件名
LOG_CONFIG_PATH = "logging.yaml"
# 定义文件路径配置文件名
DATA_IO_PATH_CONFIG_PATH = "data_io_paths.yaml"
# 定义VAE模型配置文件名
VAE_MODEL_CONFIG_PATH = "model/vae.yaml"


def main():
    """
    训练脚本的主函数, 负责配置加载, 数据处理, 模型训练和保存.
    """
    logger.info("开始加载和处理配置.")

    # 从当前脚本执行的目录开始向上查找项目根目录
    project_root_dir = find_project_root_dir(marker_file=PROJECT_ROOT_MARKER)
    logger.info("检测到的项目根目录: %s", project_root_dir)

    config_dir = find_config_dir(
        project_root_dir=project_root_dir, config_dir_relative_path=CONFIG_DIR_RELATIVE_PATH
    )
    logger.info("配置目录路径: %s", config_dir)

    # 构建重要配置路径
    log_config_path = os.path.join(config_dir, LOG_CONFIG_PATH)
    data_io_path_config_path = os.path.join(config_dir, DATA_IO_PATH_CONFIG_PATH)
    vae_model_config_path = os.path.join(config_dir, VAE_MODEL_CONFIG_PATH)

    # 配置日志
    setup_logging(log_config_path)
    all_config_files: List[str] = []
    all_config_files.append(data_io_path_config_path)
    all_config_files.append(vae_model_config_path)
    config = load_and_merge_configs(all_config_files)

    data_csv_path = config.get("output_dirs", {}).get("preprocessed_data", "")
    data_csv_path = os.path.join(project_root_dir, data_csv_path)
    data_csv_path = os.path.join(data_csv_path, DATA_FILENAME)

    model_save_path = config.get("output_dirs", {}).get("vae_model", "")
    model_save_path = os.path.join(project_root_dir, model_save_path)
    model_save_path = os.path.join(model_save_path, MODEL_FILENAME)

    input_dim = config.get("vae_model", {}).get("input_dim", 1)
    output_dim = config.get("vae_model", {}).get("output_dim", 1)
    condition_dim = config.get("vae_model", {}).get("condition_dim", 5)
    hidden_dim = config.get("vae_model", {}).get("hidden_dim", 16)
    latent_dim = config.get("vae_model", {}).get("latent_dim", 8)
    num_epochs = config.get("vae_model", {}).get("num_epochs", 10)
    batch_size = config.get("vae_model", {}).get("batch_size", 32)
    device_using = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("正在使用设备: %s", device_using)

    # --- 数据加载和处理 ---
    logger.info("开始加载和处理训练数据.")
    # 初始化数据加载器用于训练 (shuffle=True)
    train_data_loader_instance = data_loader.BatteryDataLoader(
        csv_path=data_csv_path,
        cond_bounds=CONDITION_BOUNDS,
        exclude_battery_ids=EXCLUDE_BATTERY_IDS,
        shuffle=True,
        batch_size=batch_size,
    )
    dataloader = train_data_loader_instance.create_loader()
    logger.info("训练数据加载器创建成功.")

    # --- 模型初始化 ---
    logger.info("初始化 VAE 模型.")
    vae: ConditionalVAE = ConditionalVAE(
        input_dim, hidden_dim, latent_dim, condition_dim, output_dim
    ).to(device_using)
    logger.info("VAE 模型初始化完成.")

    # --- 模型加载 (如果存在) ---
    if os.path.exists(model_save_path):
        logger.info("检测到模型文件: %s, 正在加载模型参数.", model_save_path)
        try:
            # 加载模型的 state_dict, 并映射到当前设备
            vae.load_state_dict(torch.load(model_save_path, map_location=device_using))
            logger.info("模型参数加载成功, 将基于此继续训练.")
        except FileNotFoundError as e:
            logger.error("模型文件不存在: %s", e)
            logger.warning("模型加载失败, 将从头开始训练.")
            # 如果加载失败, 重新初始化模型以确保是随机权重 (虽然已经初始化过一次)
            vae = ConditionalVAE(input_dim, hidden_dim, latent_dim, condition_dim, output_dim).to(
                device_using
            )

        except Exception as e:
            logger.error("加载模型参数时发生错误: %s", e)
            logger.warning("模型加载失败, 将从头开始训练.")
            raise

    else:
        logger.info("未检测到模型文件: %s, 将从头开始训练.", model_save_path)

    # --- 模型训练 ---
    logger.info("开始训练 VAE 模型.")
    vae.train_vae(vae, dataloader, num_epochs, device_using, enable_teacher_forcing=True)
    logger.info("VAE 模型训练完成.")

    # --- 模型保存 ---
    logger.info("开始保存训练好的模型.")
    try:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(vae.state_dict(), model_save_path)
        logger.info("训练好的模型已保存到: %s", model_save_path)
    except Exception as e:
        logger.error("保存模型时发生错误: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
