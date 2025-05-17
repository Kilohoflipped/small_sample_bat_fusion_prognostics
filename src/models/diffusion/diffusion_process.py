import torch


def q_sample(
    z_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alpha_bars: torch.Tensor
) -> torch.Tensor:
    """
    根据前向扩散过程，在时间步 t 上对潜在向量 z_0 进行加噪。
    公式: z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * noise

    Args:
        z_0: 原始的无噪声潜在向量，形状为 (batch_size, latent_dim)。
        t: 当前的时间步张量，形状为 (batch_size,)，通常包含整数时间步索引。
        noise: 与 z_0 形状相同的随机噪声张量，形状为 (batch_size, latent_dim)。
        alpha_bars: 累积的 alpha 参数 (alpha_bar_t = alpha_1 * ... * alpha_t)，
                    这是一个 tensor，形状为 (timesteps,)，其中 timesteps 是扩散总步数 T。
                    这个参数通常从噪声调度器中获取。

    Returns:
        在时间步 t 上的带噪声潜在向量 z_t，形状为 (batch_size, latent_dim)。

    Raises:
        ValueError: 如果时间步 t 的值超出了 alpha_bars 的范围。
    """
    # 检查 t 的范围是否有效
    if not (t.min() >= 0 and t.max() < len(alpha_bars)):
        raise ValueError(
            f"时间步 t 的值范围错误。最小值为 {t.min().item()}, 最大值为 {t.max().item()}，"
            f"alpha_bars 的长度（总时间步）为 {len(alpha_bars)}。"
        )

    # 从 alpha_bars 中根据时间步 t 获取 sqrt(alpha_bar_t) 和 sqrt(1 - alpha_bar_t)
    # 使用广播 ([:, None]) 将其形状从 (batch_size,) 扩展到 (batch_size, 1)，
    # 以便与 z_0 和 noise 进行逐元素的乘法
    sqrt_alpha_bar_t = torch.sqrt(alpha_bars[t])[:, None]
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bars[t])[:, None]

    # 应用前向扩散公式
    z_t = sqrt_alpha_bar_t * z_0 + sqrt_one_minus_alpha_bar_t * noise

    return z_t
