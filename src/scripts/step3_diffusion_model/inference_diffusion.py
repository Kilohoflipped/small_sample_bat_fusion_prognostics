import logging
import os
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from config.config_loader import find_config_dir, find_project_root_dir, load_and_merge_configs
from config.config_setter import setup_logging
from src.models.diffusion.conditional_denoising_model import ConditionalDenoisingModel
from src.models.diffusion.sampling import sample_diffusion
from src.models.diffusion.schedulers import get_scheduler
from src.models.vae.data_loader import BatteryDataLoader
from src.models.vae.vae import ConditionalVAE
from src.modules.visualization.plotter.diffusion import (
    DiffusionPlotter,
)  # 导入新建的 DiffusionPlotter
from src.modules.visualization.style_setter import StyleSetter  # 导入 StyleSetter

logger = logging.getLogger(__name__)

# 配置相关的常量 (根据您的项目实际情况调整)
PROJECT_ROOT_MARKER: str = "pyproject.toml"
CONFIG_DIR_RELATIVE_PATH: str = "config"
LOG_CONFIG_PATH: str = "logging.yaml"
DATA_IO_PATH_CONFIG_PATH: str = "data_io_paths.yaml"
DIFFUSION_CONFIG_PATH: str = "model/diffusion.yaml"
VAE_CONFIG_PATH: str = "model/vae.yaml"
PLOT_STYLE_CONFIG_FILENAME: str = "plot/plot_style.yaml"  # 确保这个文件存在于你的 config 目录

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


# --- 主可视化函数 ---
def visualize_generation(config_dict: Dict[str, Any], project_root_dir: str):
    """
    加载训练好的 Conditional DDPM 和 CVAE Decoder，生成合成数据并与真实数据进行可视化对比。
    使用 DiffusionPlotter 处理绘图逻辑.

    Args:
        config_dict: 合并后的配置字典。
        project_root_dir: 项目根目录路径。
    """
    print("--- 开始可视化生成数据 ---")

    # --- 加载配置 ---
    # 获取必要的配置参数
    diffusion_config = config_dict.get("diffusion_model", {})
    vae_config = config_dict.get("vae_model", {})

    model_config_diff = diffusion_config.get("model", {})
    # 采样配置通常放在 diffusion_model 下的一个子节，例如 'sampling'
    sampling_config = diffusion_config.get("sampling", {})
    # 获取绘图风格配置
    plot_style_config = config_dict.get("plot_style", {})

    # 获取模型检查点路径 (假设在各自的模型配置节下)
    vae_checkpoint_path_rel = config_dict.get("output_dirs", {}).get("vae_model", "")
    vae_checkpoint_path = os.path.join(
        project_root_dir, vae_checkpoint_path_rel, "vae_model_final.pth"
    )
    diffusion_checkpoint_path_rel = config_dict.get("output_dirs", {}).get("diffusion_model", "")
    diffusion_checkpoint_path = os.path.join(
        project_root_dir,
        diffusion_checkpoint_path_rel,
        "conditional_diffusion_model_epoch_final.pth",
    )

    # 获取预处理数据路径和 BatteryDataLoader 参数
    preprocessed_csv_path_rel = config_dict.get("output_dirs", {}).get("interim_data", "")
    preprocessed_csv_path = os.path.join(
        project_root_dir,
        preprocessed_csv_path_rel,
        "preprocessed/step4_battery_aging_cycle_data_standardized.csv",
    )

    # 假设图表目录在 config_dict 的 'plot_dirs.diffusion_generation' 节下
    plot_output_dir_rel = config_dict.get("output_dirs", {}).get("plots", "")
    plot_output_dir = os.path.join(project_root_dir, plot_output_dir_rel, "inference_diffusion")
    os.makedirs(plot_output_dir, exist_ok=True)  # 确保图表目录存在

    # --- 初始化 StyleSetter 和 DiffusionPlotter ---
    style_setter = StyleSetter(plot_style_config)
    diffusion_plotter = DiffusionPlotter(style_setter)
    print("已初始化 StyleSetter 和 DiffusionPlotter。")

    # --- 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 加载训练好的模型 ---
    print("\n--- 加载训练好的模型 ---")

    # 加载 Conditional Diffusion Model
    print("加载 Conditional Diffusion Model...")
    # 模型实例化参数从配置中获取
    diff_model = ConditionalDenoisingModel(
        latent_dim=sampling_config.get("latent_dim"),
        condition_input_dim=sampling_config.get("condition_dim"),
        time_embedding_dim=model_config_diff.get("network_params", {}).get(
            "time_embedding_dim", 128
        ),
        condition_embedding_dim=model_config_diff.get("network_params", {}).get(
            "condition_embedding_dim", 128
        ),
        model_channels=model_config_diff.get("network_params", {}).get("model_channels", 256),
        num_mlp_blocks=model_config_diff.get("network_params", {}).get("num_mlp_blocks", 3),
    )
    # 校验模型检查点路径
    if not diffusion_checkpoint_path or not os.path.exists(diffusion_checkpoint_path):
        raise FileNotFoundError(
            f"Diffusion 模型检查点未找到: {diffusion_checkpoint_path}. "
            f"请在配置中指定 'diffusion_model.checkpoint_path'"
        )
    # 加载模型权重并发送到设备，设置为评估模式
    diff_model.load_state_dict(torch.load(diffusion_checkpoint_path, map_location=device))
    diff_model.to(device)
    diff_model.eval()  # 设置为评估模式
    print("Conditional Diffusion Model 加载成功。")

    # 加载 CVAE Decoder (作为 ConditionalVAE 模型的一部分)
    print("加载 CVAE Decoder...")
    # VAE 模型实例化参数从配置中获取
    vae_input_dim = vae_config.get("input_dim")
    vae_condition_dim = vae_config.get("condition_dim")
    vae_latent_dim = vae_config.get("latent_dim")
    # 校验 VAE 模型实例化参数
    if vae_input_dim is None or vae_condition_dim is None or vae_latent_dim is None:
        raise ValueError("VAE 配置中必须指定 'input_dim', 'condition_dim', 'latent_dim'。")
    # 校验 VAE 检查点路径
    if not vae_checkpoint_path or not os.path.exists(vae_checkpoint_path):
        raise FileNotFoundError(
            f"VAE 模型检查点未找到: {vae_checkpoint_path}. 请在配置中指定 'vae_model.checkpoint_path'"
        )

    # 实例化完整的 ConditionalVAE 模型
    cvae_model = ConditionalVAE(
        input_dim=vae_input_dim,
        condition_dim=vae_condition_dim,
        latent_dim=vae_latent_dim,
        hidden_dim=vae_config.get("hidden_dim"),
        output_dim=vae_config.get("output_dim"),
    )
    # 加载 VAE 模型权重
    cvae_model.load_state_dict(torch.load(vae_checkpoint_path, map_location=device))
    cvae_model.to(device)
    # 获取 Decoder 部分并设置为评估模式
    if not hasattr(cvae_model, "decoder"):
        raise AttributeError("ConditionalVAE 类中没有名为 'decoder' 的属性。请检查模型定义。")
    decoder = cvae_model.decoder
    decoder.eval()  # 设置为评估模式
    print("CVAE Decoder 加载成功。")

    # 加载扩散模型使用的调度器
    # 采样函数需要调度器的 step 逻辑
    scheduler_config = (
        config_dict.get("diffusion_model", {}).get("sampling", {}).get("scheduler", {})
    )
    if not scheduler_config:
        raise ValueError("配置文件中必须指定 'diffusion_model.model.scheduler' 配置。")
    # 使用我们之前实现的 get_scheduler 函数

    diffusion_scheduler = get_scheduler(scheduler_config)
    # 调度器本身不需要发送到设备，它主要处理参数和计算逻辑

    # --- 加载真实数据用于对比 ---
    print("\n--- 加载真实数据用于对比 ---")
    # 实例化 BatteryDataLoader，编码时 shuffle 必须为 False
    if not preprocessed_csv_path:
        raise ValueError("配置文件中必须指定 'data.processed.standardized_data' 的路径。")

    # 可视化时，通常不需要批量加载所有数据，只需加载数据集实例以便按索引获取样本
    # 如果 BatteryDataLoader 创建 Dataset 后就存储起来了，可以直接使用 BatteryDataLoader 实例获取 dataset
    # 或者，实例化 BatteryDataLoader 并只获取 dataset，不创建完整的 DataLoader
    real_data_processor_instance = (
        BatteryDataLoader(  # 使用 DataLoader 是为了方便获取 processor 和 dataset
            csv_path=preprocessed_csv_path,
            cond_bounds=CONDITION_BOUNDS,
            exclude_battery_ids=EXCLUDE_BATTERY_IDS,
            shuffle=False,
            batch_size=1,
        )
    )
    real_dataset = real_data_processor_instance.dataset  # 获取 BatteryDataset 实例

    print(f"已加载包含 {len(real_dataset)} 个样本的真实数据集。")

    # --- 选择样本进行可视化 ---
    # 选择要可视化对比的真实样本数量和索引
    num_samples_to_viz = sampling_config.get(
        "num_samples_to_viz", 100
    )  # 从配置获取要可视化的样本数量

    if num_samples_to_viz > len(real_dataset):
        print(
            f"警告: 要可视化的样本数量 ({num_samples_to_viz}) 大于数据集大小 ({len(real_dataset)})。将可视化所有样本。"
        )
        sample_indices = list(range(len(real_dataset)))
    else:
        # 选择前 N 个样本进行可视化
        sample_indices = list(range(num_samples_to_viz))
        # 或者：随机选择 N 个样本进行可视化 (需要导入 random 并设置随机种子以保证复现性)
        # import random
        # random.seed(sampling_config.get('viz_random_seed', 42)) # 从配置获取随机种子
        # sample_indices = random.sample(range(len(real_dataset)), num_samples_to_viz)

    print(f"已选择 {len(sample_indices)} 个真实样本进行可视化对比。")

    # --- 生成并可视化 ---
    print("\n--- 生成并可视化 ---")

    # 设置为无梯度模式进行推理和解码
    with torch.no_grad():
        for i, idx in enumerate(tqdm(sample_indices, desc="生成并可视化样本", unit="sample")):
            # 从真实数据集中获取原始样本的序列、条件、长度和ID
            # BatteryDataset __getitem__ 返回 (seq, cond, length, battery_id)
            real_seq, real_cond, real_length, battery_id = real_dataset[idx]

            # --- 准备生成所需的条件 ---
            # 将条件张量增加一个批次维度并发送到设备
            cond_for_generation = real_cond.unsqueeze(0).to(device)  # 形状: (1, condition_dim)
            latent_dim = vae_config.get("latent_dim")  # 获取潜在维度用于采样

            logger.info(f"处理样本 {i+1}/{len(sample_indices)} (Battery ID: {battery_id})")
            logger.info(f"  真实条件形状: {cond_for_generation.shape}")

            # --- 准备无条件条件 (用于 CFG) ---
            # 如果配置文件中 guidance_scale > 0，就需要创建无条件条件。
            # 假设无条件条件是形状与条件张量相同（1, condition_dim）的全零张量。
            unconditional_condition = None  # 初始化为 None
            current_guidance_scale = sampling_config.get(
                "guidance_scale", 0.0
            )  # 从配置获取当前的 guidance_scale

            if current_guidance_scale > 0:
                # 创建与 conditional_condition 形状相同、数据类型相同的全零张量作为无条件条件
                # 使用 vae_condition_dim 来确保维度正确
                unconditional_condition = torch.zeros(
                    cond_for_generation.shape[0],  # 批次大小 (这里是 1)
                    vae_condition_dim,  # 条件维度
                    dtype=cond_for_generation.dtype,  # 数据类型与条件一致
                    device=device,
                )
                # 如果您的无条件条件在 VAE 训练时是其他值（例如特殊索引），这里需要相应修改。
                # 全零是最常见的处理连续条件的方式。

            logger.info(f"\n处理样本 {i+1}/{len(sample_indices)} (Battery ID: {battery_id})")
            logger.info(f"  真实条件形状: {cond_for_generation.shape}")
            if unconditional_condition is not None:
                logger.info(f"  无条件条件形状: {unconditional_condition.shape}")

            # --- 使用 Conditional DDPM 进行采样生成潜在向量 ---
            # 调用 src/models/diffusion/sampling.py 中的采样函数
            # 这个函数应该接收 diff_model, diffusion_scheduler, latent_dim, condition 等参数
            generated_latent = sample_diffusion(
                model=diff_model,
                scheduler=diffusion_scheduler,
                latent_dim=latent_dim,
                condition=cond_for_generation,  # 使用当前真实样本的条件
                num_sampling_steps=sampling_config.get(
                    "num_sampling_steps", 50
                ),  # 从配置获取采样步数
                guidance_scale=current_guidance_scale,  # 使用获取的 CFG 引导强度
                unconditional_condition=unconditional_condition,
                device=device,
                # 根据您的 sample_diffusion 函数签名，传入其他必要的参数
            )  # 采样函数返回的张量形状通常是 (num_generated_samples, latent_dim)
            # 如果只生成一个样本，形状为 (1, latent_dim)

            # --- 使用 CVAE Decoder 解码潜在向量 ---
            # Decoder forward 方法需要 latent_vector, conditions, output_lengths 等参数
            # 这里的 conditions 仍然是用于生成的条件 (cond_for_generation)
            # output_lengths 需要指定解码器生成的目标长度。为了与原始序列对比，通常设为原始序列的实际长度。
            output_length_for_decoding = torch.tensor(
                [real_length], device=device
            ).long()  # 形状: (1,) 批次的长度张量

            # 调用 Decoder 进行解码
            # Decoder forward: decoder(latent_vector, conditions, output_lengths, target_sequences=None, sampling_probability=0.0)
            generated_sequence_padded = decoder(
                latent_vector=generated_latent,  # 生成的潜在向量
                conditions=cond_for_generation,  # 用于生成的条件
                output_lengths=output_length_for_decoding,  # 目标输出长度
                target_sequences=None,  # 纯生成，不使用教师强制
                sampling_probability=0.0,  # 纯生成，不使用教师强制
            )  # Decoder 返回填充后的序列，形状可能是 (1, max_decode_len, 1)

            # --- 准备用于绘图的数据 ---
            # 将真实序列和生成的序列（去除填充和特征维度）移到 CPU 并转换为 numpy 数组
            # 真实序列 real_seq 的形状可能是 (seq_len, 1) 或 (seq_len,)
            real_seq_plot = (
                real_seq[:, 0].cpu().numpy()
                if real_seq.ndim == 2 and real_seq.shape[1] == 1
                else real_seq.cpu().numpy()
            )

            # 生成序列 generated_sequence_padded 形状是 (1, max_decode_len, 1)
            # 我们只取实际长度的部分，并去除批次和特征维度
            generated_sequence = generated_sequence_padded[0, :real_length, 0].cpu().numpy()

            # --- 使用 DiffusionPlotter 绘图 ---
            diffusion_plotter.plot_generated_vs_real(
                real_sequence=real_seq_plot,
                generated_sequence=generated_sequence,
                battery_id=battery_id,
                conditions=real_cond,  # 传入原始条件张量
                output_dir=plot_output_dir,
                sample_index=i,  # 传入样本索引用于文件名
            )

    print("\n--- 可视化生成完成 ---")


# --- 脚本入口点 ---
if __name__ == "__main__":
    # --- 加载所有必要的配置文件 ---
    project_root_dir: str = find_project_root_dir(marker_file=PROJECT_ROOT_MARKER)
    config_dir: str = find_config_dir(
        project_root_dir=project_root_dir, config_dir_relative_path=CONFIG_DIR_RELATIVE_PATH
    )
    # 构建各个配置文件的完整路径
    log_config_path: str = os.path.join(config_dir, LOG_CONFIG_PATH)
    data_io_path_config_path: str = os.path.join(config_dir, DATA_IO_PATH_CONFIG_PATH)
    diffusion_config_path: str = os.path.join(config_dir, DIFFUSION_CONFIG_PATH)
    vae_config_path: str = os.path.join(config_dir, VAE_CONFIG_PATH)
    plot_style_config_path: str = os.path.join(config_dir, PLOT_STYLE_CONFIG_FILENAME)

    # 设置日志记录
    setup_logging(log_config_path)

    # 加载并合并所有配置
    all_config_files: List[str] = [
        data_io_path_config_path,
        diffusion_config_path,
        vae_config_path,
        plot_style_config_path,  # 包含 plot_style 配置
    ]
    config = load_and_merge_configs(all_config_files)

    # 添加对可视化所需关键配置项的检查
    # (确保 config_dict 中存在这些嵌套的键)
    required_viz_configs = [
        ("vae_model", "input_dim"),  # VAE 模型参数
        ("vae_model", "condition_dim"),  # VAE 模型参数
        ("vae_model", "latent_dim"),  # VAE 模型参数
        ("diffusion_model", "model", "network_params"),  # Diffusion 模型参数
        ("diffusion_model", "sampling", "scheduler"),  # Diffusion 调度器配置
        ("diffusion_model", "sampling", "num_sampling_steps"),  # Diffusion 采样步数
        ("diffusion_model", "sampling", "guidance_scale"),  # CFG 参数
        ("plot_style",),  # 确保 plot_style 存在于配置中
    ]

    print("--- 检查关键配置项 ---")
    for keys in required_viz_configs:
        current = config
        path = []
        for i, key in enumerate(keys):
            path.append(key)
            if key not in current:
                raise ValueError(f"配置中缺少关键路径: {' -> '.join(path)}. 请检查您的 YAML 文件。")
            current = current[key]
        print(f"  √ 配置项 '{' -> '.join(keys)}' 已找到。")

    # 调用主可视化函数
    visualize_generation(config, project_root_dir)
