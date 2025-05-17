import math

import torch
import torch.nn as nn


# 辅助模块：时间步嵌入层 (将时间步转换为高维向量)
# 使用标准的正弦位置嵌入 + MLP
class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        """
        Args:
            dim: 嵌入后的维度。通常设置为模型内部通道数的倍数。
        """
        super().__init__()
        self.dim = dim

        # 计算正弦嵌入所需的分母项 1 / (10000^(2i/dim))
        div_term = torch.exp(torch.arange(0.0, dim, 2) * -(math.log(10000.0) / dim))
        # 将 div_term 注册为 buffer，它不是模型参数，但需要在 state_dict 中保存
        self.register_buffer("div_term", div_term)

        # MLP 部分，将初始的正弦嵌入进一步转换
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),  # 投射到更高维度
            nn.GELU(),  # GELU 激活函数
            nn.Linear(dim * 4, dim * 4),  # 再次投射，通常输出维度会用于调制主干网络的层
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: 当前时间步张量，形状为 (batch_size,)，通常是整数。

        Returns:
            时间步的嵌入向量，形状为 (batch_size, dim * 4)。
        """
        # 确保时间步是浮点数以便进行计算
        timesteps = timesteps.float()

        # 计算正弦和余弦嵌入
        # timesteps[:, None]: (batch_size, 1)
        # self.div_term[None, :]: (1, dim/2)
        # 相乘后得到 (batch_size, dim/2)
        sin_emb = torch.sin(timesteps[:, None] * self.div_term[None, :])
        cos_emb = torch.cos(timesteps[:, None] * self.div_term[None, :])

        # 拼接正弦和余弦嵌入，得到初始的正弦位置嵌入 (batch_size, dim)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)

        # 通过 MLP 进行最终的嵌入转换
        return self.mlp(emb)  # 输出形状: (batch_size, dim * 4)


# 辅助模块：条件信息嵌入层 (将原始条件信息转换为高维向量)
# 假设条件信息是固定维度的向量或可以被嵌入为固定维度
class ConditionEmbedding(nn.Module):
    def __init__(self, condition_input_dim: int, condition_embedding_dim: int):
        """
        Args:
            condition_input_dim: 原始条件信息的维度。
            condition_embedding_dim: 嵌入后的维度。通常设置为与时间嵌入输出维度匹配或相关。
        """
        super().__init__()
        # 一个简单的 MLP 来嵌入条件信息
        self.mlp = nn.Sequential(
            nn.Linear(condition_input_dim, condition_embedding_dim * 4),  # 可以先投射到更高维
            nn.GELU(),
            nn.Linear(condition_embedding_dim * 4, condition_embedding_dim),  # 最终输出维度
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            condition: 条件信息张量，形状为 (batch_size, condition_input_dim)。

        Returns:
            条件信息的嵌入向量，形状为 (batch_size, condition_embedding_dim)。
        """
        # 假设 condition 已经是适合 MLP 的浮点数张量
        return self.mlp(condition)  # 输出形状: (batch_size, condition_embedding_dim)


# 核心模块：一个带有条件和时间注入的 MLP 块
# 使用 ResNet-like 残差连接可以帮助训练深层网络
class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, time_emb_dim: int, cond_emb_dim: int):
        """
        Args:
            dim: 块的输入/输出特征维度。
            time_emb_dim: 时间步嵌入的维度 (来自 TimestepEmbedding 的输出维度)。
            cond_emb_dim: 条件信息嵌入的维度 (来自 ConditionEmbedding 的输出维度)。
        """
        super().__init__()
        # 第一个线性层
        self.fc1 = nn.Linear(dim, dim * 4)
        # 归一化层 (对于 MLP，LayerNorm 或 BatchNorm1d 比较常见)
        # 如果潜在向量是 (Batch, LatentDim)，LayerNorm(dim) 比较合适
        self.norm = nn.LayerNorm(dim * 4)
        # 激活函数
        self.act = nn.GELU()
        # 第二个线性层 (降维回原来的 dim)
        self.fc2 = nn.Linear(dim * 4, dim)

        # 用于将时间嵌入和条件嵌入投射到适合注入到主干网络维度的线性层
        # 这里我们将投射后的嵌入加到归一化和激活之前/之后
        self.time_proj = nn.Linear(time_emb_dim, dim * 4)  # 投射到 fc1 的输出维度
        self.cond_proj = nn.Linear(cond_emb_dim, dim * 4)  # 投射到 fc1 的输出维度

    def forward(
        self, x: torch.Tensor, time_emb: torch.Tensor, cond_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: 块的输入特征，形状为 (batch_size, dim)。
            time_emb: 时间步嵌入，形状为 (batch_size, time_emb_dim)。
            cond_emb: 条件信息嵌入，形状为 (batch_size, cond_emb_dim)。

        Returns:
            块的输出特征，形状为 (batch_size, dim)。
        """
        # 复制输入用于残差连接
        residual = x

        # 第一个线性层
        h = self.fc1(x)  # (batch_size, dim * 4)

        # --- 条件和时间嵌入注入 ---
        # 将时间嵌入和条件嵌入投射到匹配 h 的维度，并加到 h 上
        # 这种注入方式 (加到线性层输出，归一化和激活之前) 是一种常见模式
        h = h + self.time_proj(time_emb)  # (batch_size, dim * 4) + (batch_size, dim * 4)
        h = h + self.cond_proj(cond_emb)  # (batch_size, dim * 4) + (batch_size, dim * 4)

        # 归一化和激活
        h = self.norm(h)  # (batch_size, dim * 4) - LayerNorm 在最后一个维度上归一化
        h = self.act(h)  # (batch_size, dim * 4)

        # 第二个线性层
        h = self.fc2(h)  # (batch_size, dim)

        # --- 残差连接 ---
        # 残差连接，输入和输出维度相同
        return h + residual  # (batch_size, dim) + (batch_size, dim)


# 主模型：基于 MLP 的 Conditional 去噪模型
class ConditionalDenoisingModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,  # 潜在空间的维度 (来自 CVAE)
        condition_input_dim: int,  # 原始条件信息的维度
        time_embedding_dim: int = 128,  # 时间步嵌入的维度
        condition_embedding_dim: int = 128,  # 条件信息嵌入的维度
        model_channels: int = 256,  # 模型内部特征的基本通道数/维度
        num_mlp_blocks: int = 3,  # ResMLP 块的数量
    ):
        """
        基于 MLP 的 Conditional 去噪模型。

        Args:
            latent_dim: 潜在空间的维度。
            condition_input_dim: 原始条件信息的维度。
            time_embedding_dim: 时间步嵌入的维度。
            condition_embedding_dim: 条件信息嵌入的维度。
            model_channels: 模型内部 ResMLP 块的特征维度。
            num_mlp_blocks: 模型包含的 ResMLP 块的数量。
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.condition_input_dim = condition_input_dim
        self.model_channels = model_channels  # 模型的“宽度”或特征维度
        self.num_mlp_blocks = num_mlp_blocks

        # 确保时间嵌入和条件嵌入的输出维度与 MLP 块的注入维度一致
        # TimestepEmbedding 输出 dim * 4
        # ConditionEmbedding 输出 dim
        # ResMLPBlock 注入维度需要 dim * 4 (来自 TimeEmb) 和 dim (来自 CondEmb 最终输出)
        # 需要调整 TimestepEmbedding 的输出维度或 ConditionEmbedding 的最终输出维度，
        # 或者在 ResMLPBlock 中调整 proj 层的输入维度。
        # 我们调整 ConditionEmbedding 使其最终输出维度也为 time_embedding_dim * 4，方便融合注入
        # 或者，调整 ResMLPBlock 的 proj 层以匹配 TimeEmb 和 CondEmb 的真实输出维度

        # 方案：让 TimeEmb 输出 time_embedding_dim * 4， CondEmb 输出 condition_embedding_dim
        # ResMLPBlock 的 proj 层根据 TimeEmb 和 CondEmb 的真实输出维度来定义
        # 这样更灵活。修改 ConditionEmbedding 和 ResMLPBlock 的 proj 层定义。

        # 修改 ConditionEmbedding 模块
        class FlexibleConditionEmbedding(nn.Module):
            def __init__(self, condition_input_dim: int, embedding_dim: int):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(condition_input_dim, embedding_dim * 4),
                    nn.GELU(),
                    nn.Linear(embedding_dim * 4, embedding_dim),  # 最终输出维度为 embedding_dim
                )

            def forward(self, condition: torch.Tensor) -> torch.Tensor:
                return self.mlp(condition)  # 输出形状: (batch_size, embedding_dim)

        # 修改 ResMLPBlock 模块，使其接受 TimeEmb 和 CondEmb 的实际输出维度
        class FlexibleResMLPBlock(nn.Module):
            def __init__(
                self, dim: int, actual_time_emb_out_dim: int, actual_cond_emb_out_dim: int
            ):
                super().__init__()
                self.fc1 = nn.Linear(dim, dim * 4)
                self.norm = nn.LayerNorm(dim * 4)
                self.act = nn.GELU()
                self.fc2 = nn.Linear(dim * 4, dim)

                # Projections for time and condition embeddings - input dims match actual outputs
                self.time_proj = nn.Linear(
                    actual_time_emb_out_dim, dim * 4
                )  # 投射到 fc1 的输出维度
                self.cond_proj = nn.Linear(
                    actual_cond_emb_out_dim, dim * 4
                )  # 投射到 fc1 的输出维度

            def forward(
                self, x: torch.Tensor, time_emb: torch.Tensor, cond_emb: torch.Tensor
            ) -> torch.Tensor:
                residual = x
                h = self.fc1(x)
                # 注入在归一化前
                h = h + self.time_proj(time_emb)
                h = h + self.cond_proj(cond_emb)
                h = self.norm(h)
                h = self.act(h)
                h = self.fc2(h)
                return h + residual

        # --- Main Model __init__ continues with Flexible modules ---

        # 1. Embedding layers
        # TimestepEmbedding 输出 time_embedding_dim * 4
        self.time_embed = TimestepEmbedding(time_embedding_dim)
        # ConditionEmbedding 输出 condition_embedding_dim
        self.condition_embed = FlexibleConditionEmbedding(
            condition_input_dim, condition_embedding_dim
        )

        # 实际的时间嵌入输出维度
        actual_time_emb_out_dim = time_embedding_dim * 4
        # 实际的条件嵌入输出维度
        actual_cond_emb_out_dim = condition_embedding_dim

        # 2. MLP 块和投射层
        layers = []
        # 输入潜在向量到模型内部维度的初始投射
        layers.append(nn.Linear(latent_dim, model_channels))
        layers.append(nn.GELU())  # 初始激活

        # 添加 ResMLP 块
        for _ in range(num_mlp_blocks):
            layers.append(
                FlexibleResMLPBlock(
                    dim=model_channels,  # 块的输入/输出维度就是 model_channels
                    actual_time_emb_out_dim=actual_time_emb_out_dim,
                    actual_cond_emb_out_dim=actual_cond_emb_out_dim,
                )
            )
            # 如果 ResMLPBlock 内部没有最后的激活，可以在这里添加
            # layers.append(nn.GELU()) # 例如

        # 最终输出层：将模型内部维度映射回潜在向量维度 (预测噪声)
        layers.append(nn.Linear(model_channels, latent_dim))

        # 使用 ModuleList 来组织层，方便在 forward 中迭代并传递 embeddings
        # 如果使用 Sequential，则需要在 Sequential 中的每个模块里处理 embeddings 注入
        # ModuleList 更灵活
        self.network = nn.ModuleList(layers)

        # --- Refined forward pass to manually pass embeddings to blocks ---

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Conditional Denoising Model 的前向传播。

        Args:
            z_t: 带噪声的潜在向量，形状为 (batch_size, latent_dim)。
            t: 时间步，形状为 (batch_size,)。
            y: 条件信息，形状为 (batch_size, condition_input_dim)。

        Returns:
            预测的噪声，形状为 (batch_size, latent_dim)。
        """
        # 1. 嵌入时间步和条件信息
        time_emb = self.time_embed(t)  # (batch_size, time_embedding_dim * 4)
        condition_emb = self.condition_embed(y)  # (batch_size, condition_embedding_dim)

        # 2. 逐层/逐块通过网络，并注入嵌入
        x = z_t
        # 第一层 (输入投射和初始激活)
        x = self.network[0](x)  # nn.Linear(latent_dim, model_channels)
        x = self.network[1](x)  # nn.GELU()

        # 通过 ResMLP 块并注入嵌入
        # ResMLP 块从索引 2 开始
        for i in range(2, 2 + self.num_mlp_blocks):
            # Note: the ResMLPBlock expects time_emb and cond_emb
            # Here, we are passing the full outputs from the embedding layers directly
            x = self.network[i](x, time_emb, condition_emb)

        # 最终输出层 (在所有 ResMLP 块之后)
        # 索引是 2 + num_mlp_blocks
        predicted_noise = self.network[2 + self.num_mlp_blocks](x)

        return predicted_noise
