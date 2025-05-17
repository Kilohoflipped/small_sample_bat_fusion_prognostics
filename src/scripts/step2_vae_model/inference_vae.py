"""
使用训练好的条件变分自编码器 (Conditional VAE) 进行推理生成的模块.
根据 train_vae.py 和 vae.py 的内容进行修改, 以统一配置加载和模型处理逻辑.
"""

import logging
import math
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch

from config.config_loader import (
    find_config_dir,
    find_project_root_dir,
    load_and_merge_configs,
    )
from config.config_setter import setup_logging
from src.models.vae import data_loader
from src.models.vae.vae import ConditionalVAE
from src.modules.visualization.plotter.vae import StyleSetter, VAEPlotter

# 获取 logger 实例
logger = logging.getLogger(__name__)

# 原始数据文件名
DATA_FILENAME: str = "step4_battery_aging_cycle_data_standardized.csv"
# 模型保存文件名
MODEL_FILENAME: str = "vae_model_final.pth"
# Style Setter 配置文件名


# --- 常量定义 (考虑移至配置文件以提高一致性) ---
# 环境条件边界 (用于数据加载和归一化)
# 考虑从配置中加载这些边界, 以保持与训练脚本的一致性
CONDITION_BOUNDS: List[List[float]] = [
    [0.0, 2.0],  # 充电率 (C-rate)
    [0.0, 2.0],  # 放电率 (C-rate)
    [10.0 + 273.15, 65.0 + 273.15],  # 温度 (°C), 转换为开尔文
    [0.0, 500.0],  # 压强 (kPa)
    [0.0, 100.0],  # DOD (%)
]

# 排除的电池 ID 列表.
# 考虑从配置中加载此列表
EXCLUDE_BATTERY_IDS: List[Any] = [1, 2, 3, 4, 5, 17]

# 定义用于查找项目根目录的标记文件
PROJECT_ROOT_MARKER: str = "pyproject.toml"
# 定义配置目录在项目根目录下的相对路径
CONFIG_DIR_RELATIVE_PATH: str = "config"

# 定义配置文件名
LOG_CONFIG_PATH: str = "logging.yaml"
DATA_IO_PATH_CONFIG_PATH: str = "data_io_paths.yaml"
VAE_MODEL_CONFIG_PATH: str = "model/vae.yaml"
PLOT_STYLE_CONFIG_FILENAME: str = "plot/plot_style.yaml"


def main() -> None:
    """
    推理脚本的主函数, 负责配置加载, 模型加载, 进行纯自馈生成和结果绘图.
    """
    logger.info("开始加载和处理配置 (推理脚本).")

    # --- 配置加载和路径设置 ---
    try:
        # 从当前脚本执行的目录开始向上查找项目根目录
        project_root_dir: str = find_project_root_dir(marker_file=PROJECT_ROOT_MARKER)
        logger.info(f"检测到的项目根目录: {project_root_dir}")

        config_dir: str = find_config_dir(
            project_root_dir=project_root_dir, config_dir_relative_path=CONFIG_DIR_RELATIVE_PATH
        )
        logger.info(f"配置目录路径: {config_dir}")

        # 构建重要配置文件的完整路径
        log_config_path: str = os.path.join(config_dir, LOG_CONFIG_PATH)
        data_io_path_config_path: str = os.path.join(config_dir, DATA_IO_PATH_CONFIG_PATH)
        vae_model_config_path: str = os.path.join(config_dir, VAE_MODEL_CONFIG_PATH)
        style_config_path: str = os.path.join(
            config_dir, PLOT_STYLE_CONFIG_FILENAME
        )  # Style Setter 配置路径

    except FileNotFoundError as e:
        logger.error(f"查找项目根目录或配置目录失败: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"查找项目路径时发生未知错误: {e}", exc_info=True)
        sys.exit(1)

    # 配置日志系统
    setup_logging(log_config_path)

    # 加载和合并所有必需的配置文件
    # 将 style_config_path 添加到需要加载的配置文件列表中
    all_config_files: List[str] = [
        data_io_path_config_path,
        vae_model_config_path,
        style_config_path,
    ]
    try:
        config: Dict[str, Any] = load_and_merge_configs(all_config_files)
        logger.info("配置文件加载和合并成功.")
    except Exception as e:
        logger.error(f"加载和合并配置文件失败: {e}", exc_info=True)
        sys.exit(1)

    # --- 从配置中获取文件路径和模型参数 ---
    # 数据文件路径
    data_base_dir: str = config.get("output_dirs", {}).get("preprocessed_data", "")
    if not data_base_dir:
        logger.error("配置中未指定 'output_dirs.preprocessed_data'.")
        sys.exit(1)
    data_csv_path: str = os.path.join(project_root_dir, data_base_dir, DATA_FILENAME)
    logger.info(f"预期的数据文件路径: {data_csv_path}")

    # 模型文件路径
    model_base_dir: str = config.get("output_dirs", {}).get("vae_model", "")
    if not model_base_dir:
        logger.error("配置中未指定 'output_dirs.vae_model'.")
        sys.exit(1)
    model_path: str = os.path.join(project_root_dir, model_base_dir, MODEL_FILENAME)
    logger.info(f"预期的模型文件路径: {model_path}")

    # 绘图输出目录
    plot_base_dir: str = config.get("output_dirs", {}).get("plots", "")
    plot_output_dir: str = ""  # 初始化为空字符串
    if not plot_base_dir:
        logger.warning("配置中未指定 'output_dirs.plots', 将跳过绘图保存.")
    else:
        plot_output_dir_base: str = os.path.join(project_root_dir, plot_base_dir)
        # 在绘图目录下创建推理专用的子目录
        plot_output_dir = os.path.join(plot_output_dir_base, "inference_vae")
        logger.info(f"绘图输出目录: {plot_output_dir}")
        # 尝试创建绘图目录 (由 VAEPlotter 的方法负责创建，这里只记录路径)

    # 从配置中获取模型超参数
    model_config: Dict[str, Any] = config.get("vae_model", {})
    if not model_config:
        logger.error("配置中未找到 'vae_model' 部分.")
        sys.exit(1)

    input_dim: int = model_config.get("input_dim", 1)
    output_dim: int = model_config.get("output_dim", 1)
    condition_dim: int = model_config.get("condition_dim", 5)
    hidden_dim: int = model_config.get("hidden_dim", 16)
    latent_dim: int = model_config.get("latent_dim", 8)
    inference_batch_size: int = model_config.get("batch_size", 32)  # 从配置中读取批次大小

    # 设置设备 (CUDA 或 CPU)
    device_using: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"正在使用设备: {device_using}")

    # --- 模型初始化 ---
    logger.info("初始化 VAE 模型用于推理.")
    vae: ConditionalVAE = ConditionalVAE(
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

    # --- 数据加载用于获取原始数据和标准化参数 ---
    logger.info("初始化推理数据加载器以获取原始数据和标准化参数.")
    inference_loader_instance: data_loader.BatteryDataLoader | None = None
    inference_dataloader: torch.utils.data.DataLoader | None = None
    try:
        # 使用相同的参数初始化数据加载器, shuffle=False 以保持原始顺序
        # 保留 shuffle=False 是为了在绘图时可以更容易地根据原始顺序进行对齐 (通过 battery_ids)
        inference_loader_instance = data_loader.BatteryDataLoader(
            csv_path=data_csv_path,
            cond_bounds=CONDITION_BOUNDS,
            exclude_battery_ids=EXCLUDE_BATTERY_IDS,
            shuffle=False,
            batch_size=inference_batch_size,
        )
        inference_dataloader = inference_loader_instance.create_loader()
        logger.info("推理数据加载器创建成功.")
    except FileNotFoundError as e:
        logger.error(f"创建推理数据加载器失败, 数据文件未找到: {e}", exc_info=True)
        # 数据加载失败会阻止后续生成和绘图
        sys.exit(1)
    except Exception as e:
        logger.error(f"创建推理数据加载器失败: {e}", exc_info=True)
        sys.exit(1)

    # 获取用于对比的原始未标准化序列数据, 长度和电池 ID
    # 以及用于反标准化的参数. 这些是原始数据加载时的全局列表/参数.
    original_sequences_unnormalized: List[np.ndarray] = []
    original_lengths: List[int] = []
    original_battery_ids: List[str] = []  # 这个列表的顺序应该与上面两个列表的顺序一致
    inference_target_mean: float = 0.0
    inference_target_std: float = 1.0

    # 检查 inference_loader_instance 是否成功创建
    if inference_loader_instance:
        # 访问 processor 的 processed_data
        processor_inf = inference_loader_instance.processor
        processed_data = processor_inf.processed_data

        # 获取原始数据 (未标准化), 长度和电池 ID
        if "raw_sequences" in processed_data:
            original_sequences_unnormalized = processed_data["raw_sequences"]
            logger.info(f"成功获取 {len(original_sequences_unnormalized)} 条原始未标准化序列.")
        else:
            logger.error(
                "processor.processed_data 中未找到 'raw_sequences'. 原始数据将不可用, 无法进行绘图对比."
            )
            # 无法获取原始数据则退出
            sys.exit(1)

        if "lengths" in processed_data:
            original_lengths = processed_data["lengths"]
            logger.info(f"成功获取 {len(original_lengths)} 条原始长度.")
        else:
            logger.error("processor.processed_data 中未找到 'lengths'. 原始长度将不可用.")
            sys.exit(1)

        if "battery_ids" in processed_data:
            original_battery_ids = processed_data["battery_ids"]
            logger.info(f"成功获取 {len(original_battery_ids)} 个原始电池 ID.")
        else:
            logger.error("processor.processed_data 中未找到 'battery_ids'. 原始电池 ID 将不可用.")
            sys.exit(1)

        # 检查获取的原始数据长度是否匹配
        if not (
            len(original_sequences_unnormalized)
            == len(original_battery_ids)
            == len(original_lengths)
        ):
            logger.error(
                f"原始数据获取长度不一致: 序列={len(original_sequences_unnormalized)}, "
                f"电池ID={len(original_battery_ids)}, 长度={len(original_lengths)}. 无法进行后续处理."
            )
            sys.exit(1)

        # 获取用于反标准化的参数
        # 使用 get 并提供默认值以防止 KeyError
        inference_target_mean_tensor = processed_data.get("target_mean", torch.tensor(0.0))
        inference_target_std_tensor = processed_data.get("target_std", torch.tensor(1.0))

        # 确保是 tensor 并且是标量, 然后转换为 float
        if (
            isinstance(inference_target_mean_tensor, torch.Tensor)
            and inference_target_mean_tensor.ndim == 0
        ):
            inference_target_mean = inference_target_mean_tensor.item()
        else:
            logger.warning(
                "从 processor 获取的 'target_mean' 不是期望的标量 tensor. 将使用默认值 0.0."
            )
            inference_target_mean = 0.0

        if (
            isinstance(inference_target_std_tensor, torch.Tensor)
            and inference_target_std_tensor.ndim == 0
        ):
            # 避免标准差为 0 导致除以零
            if inference_target_std_tensor.item() > 1e-6:  # 使用一个小阈值判断是否接近于零
                inference_target_std = inference_target_std_tensor.item()
            else:
                logger.warning(
                    f"从 processor 获取的 'target_std' 接近或等于零 ({inference_target_std_tensor.item()}). 将使用默认值 1.0 进行反标准化."
                )
                inference_target_std = 1.0
        else:
            logger.warning(
                "从 processor 获取的 'target_std' 不是期望的标量 tensor. 将使用默认值 1.0."
            )
            inference_target_std = 1.0

    else:
        # 如果 inference_loader_instance 未成功创建, 在前面已经 exit 了, 这里只是理论上
        logger.error("推理数据加载器未成功创建, 无法获取原始数据和标准化参数.")
        sys.exit(1)

    # --- 模型推理 (纯自馈生成) ---
    logger.info("开始进行纯自馈生成.")

    # 设置模型为评估模式
    vae.eval()
    # 确保关闭梯度计算
    with torch.no_grad():
        # 使用字典存储每个电池 ID 对应的生成序列和长度
        # 键为电池 ID (str), 值为包含生成序列(numpy)和长度(int)的字典
        generated_data_by_id: Dict[str, Dict[str, Any]] = {}

        # 仅在推理加载器和原始电池ID成功获取时执行生成循环
        if inference_dataloader and original_battery_ids:
            logger.info(
                f"共有 {len(original_battery_ids)} 个电池的序列需要生成, 分为 {len(inference_dataloader)} 个批次."
            )
            try:
                # 遍历数据集，逐个获取批次数据
                for i, batch in enumerate(inference_dataloader):
                    if len(batch) < 4:
                        logger.warning(
                            f"警告: 批次 {i+1}/{len(inference_dataloader)} 数据格式不正确, 预期至少包含序列, 条件, 长度和电池ID. 跳过此批次."
                        )
                        continue

                    sequences, conditions, lengths, battery_ids_batch = batch

                    # 过滤掉长度为 0 的序列并按长度降序排序 (packed_padded_sequence 要求)
                    lengths_list: List[int] = lengths.tolist()
                    valid_indices = [j for j, length in enumerate(lengths_list) if length > 0]

                    if not valid_indices:
                        logger.info(
                            f"批次 {i+1}/{len(inference_dataloader)} 中没有有效序列 (长度 > 0). 跳过此批次."
                        )
                        continue

                    # 应用有效索引过滤
                    sequences_filtered = sequences[valid_indices].to(device_using)
                    conditions_filtered = conditions[valid_indices].to(device_using)
                    lengths_filtered = lengths[
                        valid_indices
                    ]  # 暂时不移动到设备, 等待编码器使用 CPU 长度
                    battery_ids_filtered = [battery_ids_batch[j] for j in valid_indices]

                    # 按过滤后的长度降序排序
                    lengths_sorted, sorted_indices = torch.sort(lengths_filtered, descending=True)
                    sequences_sorted = sequences_filtered[sorted_indices]
                    conditions_sorted = conditions_filtered[sorted_indices]
                    # battery_ids_sorted 与排序后的数据顺序一致
                    battery_ids_sorted = [battery_ids_filtered[j] for j in sorted_indices.tolist()]

                    # --- 纯自馈生成步骤 ---
                    # 使用 Encoder 编码
                    lengths_on_cpu = lengths_sorted.cpu()  # 编码器期望长度在 CPU
                    mu, logvar = vae.encoder(sequences_sorted, lengths_on_cpu, conditions_sorted)
                    # 在推理时通常直接使用 mu 作为潜向量
                    z = mu  # 或者 vae.reparameterize(mu, logvar) 如果需要随机性进行多样生成

                    # 使用 Decoder 进行纯自馈生成
                    lengths_on_device = lengths_sorted.to(
                        device_using
                    )  # 解码器期望目标长度在设备上
                    generated_seqs_batch = vae.decoder(
                        z,
                        conditions_sorted,
                        lengths_on_device,
                        target_sequences=None,  # 触发纯自馈生成
                    )
                    # generated_seqs_batch shape: (batch_size_in_batch, max_length_in_batch, output_dim)

                    # 处理生成的序列批次并按电池 ID 存储到字典
                    generated_seqs_np = generated_seqs_batch.cpu().numpy()

                    # 遍历批次中的每个序列 (按排序后的顺序)
                    for j in range(generated_seqs_np.shape[0]):
                        # 获取该序列的实际有效长度 (来自排序后的长度)
                        seq_len = lengths_sorted[j].item()
                        # 提取有效长度部分的生成序列 (特征维度为 1)
                        # 假设输出维度 output_dim 为 1, 否则需要指定或循环处理
                        # 注意: 即使 output_dim > 1, 这里也只取了第一维进行绘图，如果需要绘制多维，需要修改绘图逻辑
                        if output_dim > 1:
                            # 如果 output_dim > 1, 这里只取第一个特征维度的序列
                            gen_seq_padded = generated_seqs_np[j, :seq_len, 0]
                            logger.warning(
                                f"VAE 输出维度 > 1 ({output_dim}), 绘图时仅使用第一个特征维度进行对比."
                            )
                        else:
                            gen_seq_padded = generated_seqs_np[j, :seq_len, 0]

                        # 对生成的序列进行反标准化
                        # 使用 inference_loader 获取的标准化参数进行反标准化
                        gen_seq_unnormalized = (
                            gen_seq_padded * inference_target_std + inference_target_mean
                        )

                        current_battery_id = battery_ids_sorted[j]

                        # 将生成的序列及其长度存储到字典中, 以电池 ID 为键
                        # 如果同一个电池 ID 出现在不同的批次 (不常见), 这里会覆盖
                        # 如果期望每个 ID 只处理一次, 可以在加载器或处理器中处理
                        if current_battery_id in generated_data_by_id:
                            # 记录警告, 但继续处理当前批次的结果
                            logger.warning(
                                f"警告: 电池ID {current_battery_id} 在生成过程中重复出现. 覆盖之前的结果."
                            )

                        generated_data_by_id[current_battery_id] = {
                            "generated_sequence": gen_seq_unnormalized,
                            "length": seq_len,
                        }

                logger.info(f"序列生成完成. 已为 {len(generated_data_by_id)} 个电池存储生成结果.")

            except Exception as e:
                logger.error(f"纯自馈生成过程中发生错误: {e}", exc_info=True)
                # 清空部分生成的数据字典
                generated_data_by_id = {}
                logger.warning("生成过程因错误中断, 清空已生成数据.")

        else:
            logger.error("无法进行序列生成, 原因: 数据加载器未创建成功 或 未获取到原始电池ID.")

    # --- 绘图: 比较原始序列与纯自馈生成的序列 ---
    # 只有当指定了绘图目录, 且成功获取了原始数据和生成了数据时才进行绘图
    if plot_output_dir and original_battery_ids and generated_data_by_id:
        logger.info("开始绘制纯自馈生成结果.")
        logger.info(f"绘制原始序列与纯自馈生成的序列, 保存至 {plot_output_dir}...")

        # 准备用于绘图的对齐列表
        # 根据原始电池 ID 的顺序, 从字典中提取对应的生成序列和长度
        aligned_original_sequences: List[np.ndarray] = []
        aligned_generated_sequences: List[np.ndarray] = []
        aligned_battery_ids: List[str] = []
        aligned_lengths: List[int] = []

        # 遍历原始电池 ID 列表 (这个列表的顺序应该与 original_sequences_unnormalized 一致)
        for original_id_index, original_id in enumerate(original_battery_ids):
            if original_id in generated_data_by_id:
                try:
                    # 获取对应的原始序列
                    original_seq = original_sequences_unnormalized[original_id_index]

                    # 获取对应的生成序列和长度 (从字典中)
                    generated_item = generated_data_by_id[original_id]
                    generated_seq = generated_item["generated_sequence"]
                    gen_len = generated_item["length"]

                    # 将对齐后的数据添加到列表中
                    aligned_original_sequences.append(original_seq)
                    aligned_generated_sequences.append(generated_seq)
                    aligned_battery_ids.append(original_id)  # 使用原始ID作为对齐后的ID
                    aligned_lengths.append(gen_len)

                except IndexError:
                    # 理论上不应该发生, 除非 original_sequences_unnormalized 长度与 original_battery_ids 不一致
                    logger.warning(
                        f"警告: 原始序列列表长度与原始电池ID列表不匹配, 无法获取ID {original_id} 对应的原始序列. 跳过此ID的绘图."
                    )
                except Exception as e:
                    logger.warning(
                        f"警告: 处理电池ID {original_id} 的对齐数据时发生未知错误: {e}. 跳过此ID的绘图."
                    )

            else:
                logger.warning(
                    f"警告: 未在生成结果中找到电池ID {original_id} 的数据. 跳过此ID的绘图."
                )

        logger.info(f"已成功对齐 {len(aligned_battery_ids)} 条序列用于绘图.")

        # 确保对齐后的数据不为空且数量一致
        if (
            aligned_generated_sequences
            and aligned_original_sequences
            and len(aligned_generated_sequences) == len(aligned_original_sequences)
            and len(aligned_battery_ids) == len(aligned_original_sequences)
            and len(aligned_lengths) == len(aligned_original_sequences)
        ):
            try:
                # --- 初始化 StyleSetter 和 VAEPlotter ---
                # 假设 StyleSetter 可以通过 style_config_path 进行初始化
                try:
                    style_setter_instance = StyleSetter(config.get("plot_style", {}))
                    vae_plotter_instance = VAEPlotter(style_setter_instance)
                    logger.info("VAEPlotter 初始化成功.")
                except Exception as e:
                    logger.error(f"初始化 StyleSetter 或 VAEPlotter 失败: {e}", exc_info=True)
                    logger.warning("跳过生成结果绘图.")
                    vae_plotter_instance = None  # 置为 None 以阻止后续调用

                if vae_plotter_instance:
                    # --- 调用 VAEPlotter 的绘图方法 ---
                    mse_dict_gen: Dict[str, float] | None = (
                        vae_plotter_instance.plot_generated_vs_original(
                            aligned_original_sequences,  # 原始序列 (已对齐)
                            aligned_generated_sequences,  # 生成序列 (已对齐)
                            aligned_battery_ids,  # 电池 ID 列表 (已对齐)
                            aligned_lengths,  # 序列长度列表 (已对齐)
                            plot_output_dir,  # 绘图保存目录
                        )
                    )

                    # 打印生成结果的 MSE 统计
                    logger.info(f"纯自馈生成图和 MSE 摘要已尝试保存至 {plot_output_dir}")
                    # 计算并打印平均 MSE (注意：在生成任务中 MSE 仅供参考)
                    if mse_dict_gen:
                        # 确保 mse_dict_gen 不为空且包含数值
                        valid_mses = [
                            v
                            for v in mse_dict_gen.values()
                            if isinstance(v, (int, float))
                            and not math.isnan(v)
                            and not math.isinf(v)
                        ]
                        if valid_mses:
                            average_mse = float(np.mean(valid_mses))  # 确保转换为 float
                            logger.info(f"电池平均 MSE (原始 vs 生成): {average_mse:.4f}")
                        else:
                            logger.warning("MSE 字典中的数值无效或为空, 无法计算平均 MSE.")

                    else:
                        logger.warning("plot_generated_vs_original 函数未返回有效的 MSE 字典.")
                else:
                    logger.warning("VAEPlotter 实例未创建成功, 跳过绘图.")

            except Exception as e:
                logger.error(f"绘制生成结果图时发生错误: {e}", exc_info=True)
                logger.warning("生成结果绘图过程失败.")

        else:
            logger.warning("跳过生成结果绘图, 原因: 没有对齐的数据 或 对齐数据数量不一致.")
            logger.warning(
                f"对齐原始序列数量: {len(aligned_original_sequences)}, 对齐生成序列数量: {len(aligned_generated_sequences)}, "
                f"对齐电池ID数量: {len(aligned_battery_ids)}, 对齐长度数量: {len(aligned_lengths)}"
            )
    else:
        logger.warning("跳过生成结果绘图, 原因: 未指定有效的绘图输出目录 或 原始数据/生成数据缺失.")
        logger.warning(
            f"绘图目录有效: {bool(plot_output_dir)}, 原始电池ID数量: {len(original_battery_ids) if original_battery_ids else 0}, "
            f"生成数据ID数量: {len(generated_data_by_id) if generated_data_by_id else 0}"
        )


# 如果脚本作为主程序运行, 执行 main 函数
if __name__ == "__main__":
    main()
