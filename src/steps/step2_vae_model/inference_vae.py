"""
使用训练好的条件变分自编码器 (Conditional VAE) 进行推理生成的模块.
根据 train_vae.py 和 vae_model.py 的内容进行修改, 以统一配置加载和模型处理逻辑.
"""

import logging
import math
import os
import sys
from typing import Any, List

import numpy as np
import torch

import plot_utils  # 假设 plot_utils 模块可用
from config.config_loader import (
    find_config_dir,
    find_project_root_dir,
    load_and_merge_configs,
)
from config.config_setter import setup_logging
from src.models import data_loader, vae_model  # 假设 src.models 路径正确

logger = logging.getLogger(__name__)

# 原始数据名称
DATA_FILENAME = "step4_battery_aging_cycle_data_standardized.csv"
# 模型保存文件路径.
MODEL_FILENAME = "vae_model_final.pth"

# --- 超参数配置 ---
# 环境条件边界 (用于数据加载和归一化) - 从 train_vae.py 复制
CONDITION_BOUNDS: List[List[float]] = [
    [0.0, 2.0],  # 充电率
    [0.0, 2.0],  # 放电率
    [10.0 + 273.15, 65.0 + 273.15],  # 温度 (°C), 转换为开尔文
    [0.0, 500.0],  # 压强
    [0.0, 100.0],  # DOD
]

# 排除的电池 ID 列表. - 从 train_vae.py 复制
EXCLUDE_BATTERY_IDS: List[Any] = [17]  # 根据需要调整此列表

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
    推理脚本的主函数, 负责配置加载, 模型加载, 进行纯自馈生成和结果绘图.
    """
    logger.info("开始加载和处理配置 (推理脚本).")

    try:
        # 从当前脚本执行的目录开始向上查找项目根目录
        project_root_dir = find_project_root_dir(marker_file=PROJECT_ROOT_MARKER)
        logger.info(f"检测到的项目根目录: {project_root_dir}")

        config_dir = find_config_dir(
            project_root_dir=project_root_dir, config_dir_relative_path=CONFIG_DIR_RELATIVE_PATH
        )
        logger.info(f"配置目录路径: {config_dir}")

        # 构建重要配置路径
        log_config_path = os.path.join(config_dir, LOG_CONFIG_PATH)
        data_io_path_config_path = os.path.join(config_dir, DATA_IO_PATH_CONFIG_PATH)
        vae_model_config_path = os.path.join(config_dir, VAE_MODEL_CONFIG_PATH)

    except FileNotFoundError as e:
        logger.error(f"查找项目根目录或配置目录失败: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"查找项目路径时发生未知错误: {e}")
        sys.exit(1)

    # 配置日志
    setup_logging(log_config_path)

    # 加载和合并配置文件
    all_config_files: List[str] = []
    all_config_files.append(data_io_path_config_path)
    all_config_files.append(vae_model_config_path)
    try:
        config = load_and_merge_configs(all_config_files)
    except Exception as e:
        logger.error(f"加载和合并配置文件失败: {e}")
        sys.exit(1)

    # 从配置中获取文件路径和模型参数
    data_csv_path = config.get("output_dirs", {}).get("preprocessed_data", "")
    data_csv_path = os.path.join(project_root_dir, data_csv_path)
    data_csv_path = os.path.join(data_csv_path, DATA_FILENAME)  # 使用与训练脚本相同的数据文件名

    model_save_dir = config.get("output_dirs", {}).get("vae_model", "")
    model_save_dir = os.path.join(project_root_dir, model_save_dir)
    model_path = os.path.join(model_save_dir, MODEL_FILENAME)

    plot_output_dir = config.get("output_dirs", {}).get("plots", "")
    plot_output_dir = os.path.join(project_root_dir, plot_output_dir)
    plot_output_dir = os.path.join(plot_output_dir, "inference_vae")

    # 从配置中获取模型超参数
    input_dim = config.get("vae_model", {}).get("input_dim", 1)
    output_dim = config.get("vae_model", {}).get("output_dim", 1)
    condition_dim = config.get("vae_model", {}).get("condition_dim", 5)
    hidden_dim = config.get("vae_model", {}).get("hidden_dim", 16)  # 与训练脚本保持一致
    latent_dim = config.get("vae_model", {}).get("latent_dim", 8)  # 与训练脚本保持一致
    inference_batch_size = config.get("vae_model", {}).get("batch_size", 32)  # 从配置中读取批次大小

    device_using = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"正在使用设备: {device_using}")

    # --- 模型初始化 ---
    logger.info("初始化 VAE 模型用于推理.")
    vae: vae_model.ConditionalVAE = vae_model.ConditionalVAE(
        input_dim, hidden_dim, latent_dim, condition_dim, output_dim
    ).to(device_using)
    logger.info("VAE 模型初始化完成.")

    # --- 模型加载 (必须存在用于推理) ---
    if not os.path.exists(model_path):
        logger.error(f"未检测到模型文件: {model_path}. 推理需要预训练的模型.")
        sys.exit(1)

    logger.info(f"检测到模型文件: {model_path}, 正在加载模型参数...")
    try:
        # 加载模型的 state_dict, 并映射到当前设备
        vae.load_state_dict(torch.load(model_path, map_location=device_using))
        logger.info("模型参数加载成功.")
    except Exception as e:
        logger.error(f"加载模型参数时发生错误: {e}", exc_info=True)
        logger.error("模型加载失败, 无法进行推理.")
        sys.exit(1)

    # --- 模型推理 (纯自馈生成) ---
    logger.info("开始进行纯自馈生成.")

    # 设置模型为评估模式
    vae.eval()
    # 确保关闭梯度计算
    with torch.no_grad():
        # 初始化数据加载器用于获取推理数据 (shuffle=False)
        # 这样可以按原始顺序逐个生成序列, 方便与原始数据对比绘图
        logger.info("初始化推理数据加载器.")
        try:
            inference_loader_instance = data_loader.BatteryDataLoader(
                csv_path=data_csv_path,
                cond_bounds=CONDITION_BOUNDS,
                exclude_battery_ids=EXCLUDE_BATTERY_IDS,
                shuffle=False,  # 不打乱以匹配原始数据顺序
                batch_size=inference_batch_size,  # 使用从配置加载的批次大小
            )
            inference_dataloader = inference_loader_instance.create_loader()
            logger.info("推理数据加载器创建成功.")
        except Exception as e:
            logger.error(f"创建推理数据加载器失败: {e}", exc_info=True)
            # 推理加载器失败会阻止后续生成和绘图
            inference_dataloader = None  # 阻止后续循环执行

        # 获取用于对比的原始未标准化序列数据 (从 processor 获取 raw_sequences)
        # 获取长度和电池 ID
        original_sequences_unnormalized: List[np.ndarray] = []
        original_lengths: List[int] = []
        original_battery_ids: List[str] = []
        inference_target_mean: float = 0.0
        inference_target_std: float = 1.0

        if (
            inference_loader_instance and inference_dataloader
        ):  # 只有当 inference_loader 和 dataloader 创建成功时才获取数据
            processor_inf = inference_loader_instance.processor
            if "raw_sequences" in processor_inf.processed_data:
                original_sequences_unnormalized = processor_inf.processed_data["raw_sequences"]
            else:
                logger.warning(
                    "processor.processed_data 中未找到 'raw_sequences'. 原始数据将不可用."
                )
                original_sequences_unnormalized = []  # 确保列表为空如果键不存在

            if "lengths" in processor_inf.processed_data:
                original_lengths = processor_inf.processed_data["lengths"]
            else:
                logger.warning("processor.processed_data 中未找到 'lengths'. 原始长度将不可用.")
                original_lengths = []  # 确保列表为空如果键不存在

            if "battery_ids" in processor_inf.processed_data:
                original_battery_ids = processor_inf.processed_data["battery_ids"]
            else:
                logger.warning(
                    "processor.processed_data 中未找到 'battery_ids'. 原始电池 ID 将不可用."
                )
                original_battery_ids = []  # 确保列表为空如果键不存在

            # 获取用于反标准化的参数
            # 检查键是否存在再访问
            inference_target_mean = processor_inf.processed_data.get(
                "target_mean", torch.tensor(0.0)
            ).item()
            inference_target_std = processor_inf.processed_data.get(
                "target_std", torch.tensor(1.0)
            ).item()

            # 检查获取的数据长度是否匹配
            if len(original_sequences_unnormalized) != len(original_battery_ids):
                logger.warning(
                    f"原始数据获取长度不一致: 序列={len(original_sequences_unnormalized)}, "
                    f"电池ID={len(original_battery_ids)}. 这可能导致绘图问题."
                )
        else:
            logger.warning("推理数据加载器未成功创建, 跳过原始数据获取.")

        generated_sequences: List[np.ndarray] = []
        gen_battery_ids: List[str] = []
        gen_lengths: List[int] = []

        if inference_dataloader:  # 仅在推理加载器成功创建时执行生成循环
            logger.info("开始按批次生成序列.")
            try:
                # 遍历数据集，逐个获取原始输入, 条件, 长度和电池 ID
                for i, batch in enumerate(inference_dataloader):
                    # batch 包含: sequences (标准化的输入序列), conditions (标准化条件), lengths (原始长度), battery_ids (List[str])
                    if len(batch) < 4:
                        logger.warning(
                            f"警告: 批次 {i+1} 数据格式不正确, 预期至少包含序列, 条件, 长度和电池ID. 跳过此批次."
                        )
                        continue

                    sequences, conditions, lengths, battery_ids_batch = batch

                    # 过滤掉长度为 0 的序列并按长度降序排序 (packed_padded_sequence 要求)
                    # 将长度张量转换为 Python list 以便排序和过滤
                    lengths_list: List[int] = lengths.tolist()
                    valid_indices = [j for j, length in enumerate(lengths_list) if length > 0]

                    if not valid_indices:
                        # 如果批次中没有有效序列, 跳过此批次
                        continue

                    sequences = sequences[valid_indices].to(device_using)
                    conditions = conditions[valid_indices].to(device_using)
                    lengths = lengths[valid_indices].to(
                        device_using
                    )  # 确保 lengths 也是在 device 上
                    battery_ids_batch = [
                        battery_ids_batch[j] for j in valid_indices
                    ]  # 过滤电池 ID 列表

                    # 按长度降序排序
                    lengths, sorted_indices = torch.sort(lengths, descending=True)
                    sequences = sequences[sorted_indices]
                    conditions = conditions[sorted_indices]
                    battery_ids_batch = [battery_ids_batch[j] for j in sorted_indices.tolist()]

                    # --- 纯自馈生成步骤 ---
                    # 1. 使用 Encoder 编码输入序列和条件，得到潜空间的均值和方差 Q(z|x, c)
                    # 注意: 编码器期望长度张量在 CPU
                    lengths_on_cpu = lengths.cpu()
                    mu, logvar = vae.encoder(sequences, lengths_on_cpu, conditions)

                    # 2. 从编码器分布 Q(z|x, c) 中采样一个潜向量 z
                    z = vae.reparameterize(mu, logvar)

                    # 3. 使用 Decoder 进行纯自馈生成
                    #    通过设置 target_sequences=None 启用纯自馈模式
                    #    解码器期望目标长度张量在设备上
                    generated_seqs = vae.decoder(
                        z,  # 潜向量
                        conditions,  # 标准化条件
                        lengths.to(device_using),  # 目标生成长度 (使用原始序列长度), 确保在设备上
                        target_sequences=None,  # <<< 触发纯自馈生成
                    )
                    # generated_seqs shape: (batch_size, max_length, output_dim)

                    # 处理生成的序列批次
                    # 将生成的序列移回 CPU 并转换为 NumPy
                    generated_seqs_np = generated_seqs.cpu().numpy()

                    # 遍历批次中的每个序列
                    for j in range(generated_seqs_np.shape[0]):
                        # 获取该序列的实际有效长度
                        seq_len = lengths[j].item()
                        # 提取有效长度部分的生成序列 (特征维度为 1)
                        gen_seq = generated_seqs_np[j, :seq_len, 0]  # 提取 numpy array

                        # 对生成的序列进行反标准化，恢复到原始容量范围
                        # 使用 inference_loader 的标准化参数进行反标准化
                        # 这些参数是 float, 可以直接用于 numpy array
                        gen_seq_unnormalized = (
                            gen_seq * inference_target_std + inference_target_mean
                        )

                        # 将反标准化后的生成序列, 长度和电池 ID 存储起来
                        generated_sequences.append(gen_seq_unnormalized)
                        gen_lengths.append(seq_len)
                        gen_battery_ids.append(battery_ids_batch[j])

                logger.info("序列生成完成.")
                # 验证生成数据数量与原始数据数量是否一致
                if original_sequences_unnormalized and len(generated_sequences) != len(
                    original_sequences_unnormalized
                ):
                    logger.warning(
                        f"生成序列数量 ({len(generated_sequences)}) 与原始序列数量 ({len(original_sequences_unnormalized)}) 不匹配."
                    )

            except Exception as e:
                logger.error(f"纯自馈生成过程中发生错误: {e}", exc_info=True)
                generated_sequences = []  # 清空部分生成的数据, 防止绘图时出错
                gen_battery_ids = []
                gen_lengths = []

        # --- 绘图: 比较原始序列与纯自馈生成的序列 ---
        logger.info("开始绘制纯自馈生成结果.")
        # 创建绘图目录
        os.makedirs(plot_output_dir, exist_ok=True)
        logger.info(f"绘制原始序列与纯自馈生成的序列, 保存至 {plot_output_dir}...")

        if (
            generated_sequences
            and original_sequences_unnormalized
            and len(generated_sequences) == len(original_sequences_unnormalized)
            and len(gen_battery_ids) == len(original_battery_ids)  # 确保电池ID数量也匹配
        ):
            try:
                # 调用绘图工具函数
                # original_sequences_unnormalized 已准备好
                mse_dict_gen = plot_utils.plot_battery_sequences(
                    original_sequences_unnormalized,  # 原始未标准化序列
                    generated_sequences,  # 纯自馈生成的未标准化序列
                    gen_battery_ids,  # 电池 ID 列表 (使用生成的, 理论上与原始顺序一致)
                    gen_lengths,  # 序列长度列表 (使用生成的, 理论上与原始顺序一致)
                    plot_output_dir,  # 绘图保存目录
                )

                # 打印生成结果的 MSE 统计
                logger.info(f"纯自馈生成图和 MSE 摘要已保存至 {plot_output_dir}")
                # 计算并打印平均 MSE (注意：在生成任务中 MSE 仅供参考)
                if mse_dict_gen:
                    # 确保 mse_dict_gen 不为空且包含数值
                    valid_mses = [
                        v
                        for v in mse_dict_gen.values()
                        if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)
                    ]
                    if valid_mses:
                        average_mse = np.mean(valid_mses)
                        logger.info(f"电池平均 MSE (原始 vs 生成): {average_mse:.4f}")
                    else:
                        logger.warning("MSE 字典中的数值无效或为空, 无法计算平均 MSE.")

                else:
                    logger.warning("MSE 字典为空, 无法计算平均 MSE.")

            except Exception as e:
                logger.error(f"绘制生成结果图时发生错误: {e}", exc_info=True)
        else:
            logger.warning(
                "跳过生成结果绘图, 原因: 没有生成数据 或 生成数据数量与原始数据不匹配 或 电池ID数量不匹配."
            )


# 如果脚本作为主程序运行, 执行 main 函数
if __name__ == "__main__":
    main()
