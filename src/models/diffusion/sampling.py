import torch
import torch as th  # 常用的 torch 别名
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from tqdm import tqdm  # 用于在采样过程中显示进度条

# 导入您需要的模块
from src.models.diffusion.conditional_denoising_model import (
    ConditionalDenoisingModel,
    )


# from diffusers import DDPMScheduler # 或者导入特定调度器


def sample_diffusion(
    model: ConditionalDenoisingModel,  # 训练好的 ConditionalDenoisingModel 实例
    scheduler: SchedulerMixin,  # Hugging Face Diffusers 调度器实例 (已通过 get_scheduler 初始化)
    latent_dim: int,  # 潜在空间的维度
    condition: torch.Tensor,  # 用于控制生成过程的条件信息，形状为 (batch_size, condition_dim)
    num_sampling_steps: int,  # 实际用于推理的步数，通常远小于训练时的总时间步 T (例如 50 或 100)
    guidance_scale: float = 0.0,  # 无分类器引导 (Classifier-Free Guidance, CFG) 的强度。0 表示不使用 CFG。
    unconditional_condition: (
        torch.Tensor | None
    ) = None,  # 如果使用 CFG (guidance_scale > 0)，则需要提供无条件条件张量
    device: torch.device = torch.device("cpu"),  # 执行采样的设备
    # 其他可选参数，例如用于控制随机性的 generator，用于 DDIM 的 eta 等
    generator: (
        torch.Generator | None
    ) = None,  # torch.Generator 实例，用于控制初始噪声生成的可复现性
    eta: float = 0.0,  # 用于 DDIM 调度器，eta=0.0 是 DDIM 采样，eta=1.0 接近 DDPM 采样
    # ... 可以根据需要添加其他参数
) -> torch.Tensor:
    """
    使用训练好的 Conditional Diffusion Model 和给定的条件，通过逆向扩散过程生成潜在向量。

    Args:
        model: 训练好的 ConditionalDenoisingModel 实例。
        scheduler: Hugging Face Diffusers 调度器实例 (已初始化)。它负责计算每一步的去噪。
        latent_dim: 潜在空间的维度。
        condition: 用于生成数据的条件张量，形状为 (batch_size, condition_dim)。
        num_sampling_steps: 反向去噪（采样）过程实际执行的步数。
        guidance_scale: 无分类器引导 (CFG) 的强度。0 表示无引导。
        unconditional_condition: 用于 CFG 的无条件条件张量，形状与 condition 相同。如果 guidance_scale > 0，则必须提供。
        device: 执行采样的设备。
        generator: 可选的 torch.Generator 实例，用于控制初始噪声的可复现性。
        eta: 用于 DDIM 调度器的 eta 值。
        ... 其他参数

    Returns:
        生成的潜在向量 z_0，形状为 (batch_size, latent_dim)。
    """
    # 确保模型和条件张量在正确的设备上，并设置模型为评估模式
    model.to(device)
    model.eval()  # 设置为评估模式，影响 BatchNorm, Dropout 等层
    condition = condition.to(device)
    if guidance_scale > 0.0:
        if unconditional_condition is None:
            raise ValueError("使用 CFG (guidance_scale > 0) 时，必须提供 unconditional_condition。")
        unconditional_condition = unconditional_condition.to(device)

    # 使用调度器设置用于采样的总步数和对应的时间步序列
    # 调度器内部会根据总时间步 T 和 num_sampling_steps 生成一个稀疏的时间步序列 (例如 [999, 990, ..., 0])
    scheduler.set_timesteps(
        num_sampling_steps, device=device
    )  # 将 scheduler 的 timesteps 也放到设备上

    # --- 获取初始噪声 (z_T) ---
    # 采样过程从一个标准正态分布的噪声开始
    batch_size = condition.shape[0]  # 批次大小由输入的条件决定
    # Shape: (batch_size, latent_dim)
    if generator is not None:
        # 如果提供了 generator，使用它来生成可复现的噪声
        latent_zt = th.randn(batch_size, latent_dim, generator=generator, device=device)
    else:
        # 直接在目标设备上生成噪声
        latent_zt = th.randn(batch_size, latent_dim, device=device)

    # --- 逆向扩散（去噪）循环 ---
    # 遍历调度器生成的时间步序列 (通常是从大到小)
    print(f"开始生成 {batch_size} 个样本，使用 {num_sampling_steps} 步采样...")
    # 使用 tqdm 包装时间步迭代，显示进度条
    for t in tqdm(scheduler.timesteps, desc="Diffusion Sampling", unit="step"):

        # 将当前时间步 t 转换为适用于模型的张量格式 (batch_size,)
        # Diffusers 的模型通常期望一个批次大小的时间步张量
        t_batch = torch.tensor([t] * batch_size, device=device).long()

        # --- 模型预测 (预测噪声) ---
        # 使用训练好的模型预测添加到 z_t 中的噪声
        # 模型输入: (z_t, t, condition)
        # 模型输出: 预测的噪声 (epsilon_theta)

        # 处理无分类器引导 (CFG)
        if guidance_scale > 0.0:
            # 如果使用 CFG，需要同时计算有条件预测和无条件预测
            # 将 z_t 和 condition/unconditional_condition 拼接，进行一次前向传播
            # 模型输入张量的形状将是 (batch_size * 2, ...)
            model_input = torch.cat([latent_zt, latent_zt], dim=0)
            condition_input = torch.cat([condition, unconditional_condition], dim=0)
            t_input = torch.cat([t_batch, t_batch], dim=0)  # 时间步也复制一份

            # 获取预测结果 (同时包含有条件和无条件)
            noise_pred_both = model(model_input, t_input, condition_input)
            # 将结果分回有条件和无条件两部分
            noise_pred_conditional, noise_pred_unconditional = noise_pred_both.chunk(
                2
            )  # 使用 batch_size 来分割

            # 应用 CFG 公式计算最终用于去噪的预测噪声
            # predicted_noise = epsilon_unconditional + guidance_scale * (epsilon_conditional - epsilon_unconditional)
            # 注意：如果模型预测的是 V 预测 (prediction_type='v_prediction') 或样本 (prediction_type='sample')，公式会略有不同
            # 假设您的模型预测的是噪声 (prediction_type='epsilon')，这是最常见的情况
            noise_pred = noise_pred_unconditional + guidance_scale * (
                noise_pred_conditional - noise_pred_unconditional
            )

        else:
            # 不使用 CFG，直接进行有条件预测
            noise_pred = model(latent_zt, t_batch, condition)  # 形状: (batch_size, latent_dim)

        # --- 调度器步骤 (一步去噪) ---
        # 使用调度器根据当前样本 z_t、预测的噪声 noise_pred 和当前时间步 t，计算下一步的样本 z_{t-1}
        # 调度器 step 方法的签名通常是 step(model_output, timestep, sample, **kwargs)
        # model_output 是模型的预测结果 (noise_pred)
        # timestep 是当前的时间步 t (通常是标量 int)
        # sample 是当前的样本 z_t
        # **kwargs 可能包含 eta 等参数 (对于 DDIM)
        # step 方法返回一个 SchedulerOutput 对象，其 .prev_sample 就是计算出的 z_{t-1}
        scheduler_output = scheduler.step(
            model_output=noise_pred,
            timestep=t,  # 传递标量时间步 t
            sample=latent_zt,
            eta=eta,  # 传递 eta 参数给调度器 (如果需要)
            # 其他 kwargs...
        )

        # 更新当前样本 z_t 为计算出的 z_{t-1}，用于下一次循环
        latent_zt = scheduler_output.prev_sample

    # --- 采样完成 ---
    # 循环结束后， latent_zt 就是最终生成的潜在向量 z_0
    generated_latent_z0 = latent_zt

    print("扩散模型采样完成。")

    return generated_latent_z0
