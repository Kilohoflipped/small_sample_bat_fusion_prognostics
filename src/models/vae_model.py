import math
import random
from typing import List, Tuple

import torch
from torch import nn
from torch.nn.utils import rnn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class VaeEncoder(nn.Module):
    """
    VAE编码器, 将变长序列和条件编码为潜空间分布的参数(均值和对数方差).
    使用LSTM处理序列数据.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        condition_dim: int,
    ):
        """
        初始化编码器.

        Args:
            input_dim (int): 输入序列特征的维度.
            hidden_dim (int): LSTM隐藏状态的维度.
            latent_dim (int): 潜空间的维度.
            condition_dim (int): 条件特征的维度.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        # LSTM输入维度是序列特征维度加上条件特征维度
        self.lstm = nn.LSTM(input_dim + condition_dim, hidden_dim, batch_first=True)

        # 将LSTM最终隐藏状态映射到潜空间均值
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # 将LSTM最终隐藏状态映射到潜空间对数方差
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        input_sequences: torch.Tensor,
        input_sequences_lengths: torch.Tensor,
        conditions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        编码器前向传播.

        Args:
            input_sequences (Tensor): 填充后的输入序列, (batch_size, max_seq_len, input_dim).
            input_sequences_lengths (Tensor): 每个序列的实际长度, (batch_size,).
            conditions (Tensor): 环境条件, (batch_size, condition_dim).

        Returns:
            tuple: (mu, logvar) - 潜空间分布的均值和对数方差, 形状均为 (batch_size, latent_dim).
        """
        batch_size, max_seq_len, _ = input_sequences.size()

        # 在时间维度上扩展条件, 与序列拼接
        conditions_expanded = conditions.unsqueeze(1).repeat(1, max_seq_len, 1)
        lstm_input = torch.cat([input_sequences, conditions_expanded], dim=-1)

        # 打包填充序列以高效处理变长序列
        packed_sequences = rnn.pack_padded_sequence(
            lstm_input,
            input_sequences_lengths.cpu(),  # pack_padded_sequence 需要cpu tensor for lengths
            batch_first=True,
            enforce_sorted=True,
        )

        # 通过LSTM处理打包序列, 只取最终隐藏状态 (h_n)
        # lstm返回 (output, (h_n, c_n))
        _, (h_n, _) = self.lstm(packed_sequences)

        # 对于单层LSTM, h_n 的形状是 (1, batch_size, hidden_dim). 移除第一个维度.
        h_n = h_n.squeeze(0)

        # 将最终隐藏状态映射到均值和对数方差
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)

        return mu, logvar


class VaeDecoder(nn.Module):
    """
    VAE解码器, 将潜空间向量和环境条件解码为序列数据.
    使用LSTM逐时间步生成序列. 支持教师强制.
    """

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        """
        初始化解码器.

        Args:
            latent_dim (int): 潜空间的维度.
            condition_dim (int): 条件特征的维度.
            hidden_dim (int): LSTM隐藏状态的维度.
            output_dim (int): 输出序列特征的维度.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 从潜向量初始化LSTM的隐藏状态
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.latent_to_cell = nn.Linear(latent_dim, hidden_dim)

        # LSTM输入维度是前一时间步输出维度加上条件特征维度
        self.lstm = nn.LSTM(output_dim + condition_dim, hidden_dim, batch_first=True)

        # 将LSTM隐藏状态映射到当前时间步输出
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        latent_vector: torch.Tensor,
        conditions: torch.Tensor,
        output_lengths: torch.Tensor,
        target_sequences: torch.Tensor | None = None,
        sampling_probability: float = 1.0,
    ) -> torch.Tensor:
        """
        解码器前向传播进行序列生成或重建.

        Args:
            latent_vector (Tensor): 潜空间向量, (batch_size, latent_dim).
            conditions (Tensor): 环境条件, (batch_size, condition_dim).
            output_lengths (Tensor): 要生成的每个序列的目标长度, (batch_size,).
            target_sequences (Tensor, optional): 用于教师强制的原始输入序列.
                                                 形状为 (batch_size, max_seq_len, output_dim).
                                                 若为 None 则进行纯生成. 默认为 None.
            sampling_probability (float, optional): 使用教师强制的概率.
                                                    只在 target_sequences 不为 None 时有效.
                                                    默认为 1.0 (纯教师强制).

        Returns:
            Tensor: 从潜空间重建或生成的序列, 形状为 (batch_size, max_len, output_dim).
        """
        batch_size = latent_vector.size(0)
        max_length = output_lengths.max().item()
        device = latent_vector.device

        # 初始化LSTM的隐藏状态和细胞状态
        hidden_state_init = self.latent_to_hidden(latent_vector).unsqueeze(0)
        cell_state_init = self.latent_to_cell(latent_vector).unsqueeze(0)
        hidden = (hidden_state_init, cell_state_init)

        reconstructed_outputs = []

        # 强制设置第一个时间步的输出为初始 SOH 值
        initial_soh = torch.ones(batch_size, 1, self.output_dim, device=device)
        reconstructed_outputs.append(initial_soh)

        # 确定是否处于纯生成模式
        is_generating = target_sequences is None

        # LSTM的第一个输入 (用于预测时间步1的值) 使用初始 SOH 值
        current_lstm_input_value = initial_soh

        # 在时间维度上扩展条件张量
        conditions_expanded = conditions.unsqueeze(1).repeat(1, max_length, 1)

        # LSTM循环生成后续时间步 (从 t=1 到 max_length-1)
        for t in range(1, max_length):
            # 组合当前输入值和条件作为LSTM输入
            lstm_input = torch.cat(
                [current_lstm_input_value, conditions_expanded[:, t, :].unsqueeze(1)],
                dim=-1,
            )

            # 经过LSTM一步
            output, hidden = self.lstm(lstm_input, hidden)

            # 预测时间步 t 的值
            predicted_value_at_t = self.hidden_to_output(output)
            reconstructed_outputs.append(predicted_value_at_t)

            # 为下一个时间步 (t+1) 准备输入
            if t < max_length - 1:
                if is_generating:
                    # 纯生成模式: 下一个输入使用时间步 t 的预测值 (自馈)
                    next_lstm_input_value = predicted_value_at_t.detach()
                else:
                    # 训练/重建模式: 根据采样概率决定下一个输入
                    if random.random() < sampling_probability:
                        # 教师强制: 下一个输入使用时间步 t 的真实值
                        next_lstm_input_value = target_sequences[:, t, :].unsqueeze(1)
                    else:
                        # 自馈: 下一个输入使用时间步 t 的预测值
                        next_lstm_input_value = predicted_value_at_t.detach()

                current_lstm_input_value = next_lstm_input_value

        # 拼接所有时间步的输出
        reconstructed_sequences = torch.cat(reconstructed_outputs, dim=1)

        return reconstructed_sequences


class ConditionalVAE(nn.Module):
    """
    条件变分自编码器, 结合环境条件进行序列的编码、潜空间采样和序列解码.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        condition_dim: int,
        output_dim: int,
    ):
        """
        初始化条件VAE模型.

        Args:
            input_dim (int): 输入序列特征的维度.
            hidden_dim (int): LSTM隐藏状态的维度.
            latent_dim (int): 潜空间的维度.
            condition_dim (int): 条件特征的维度.
            output_dim (int): 输出序列特征的维度 (通常与 input_dim 相同用于重建).
        """
        super().__init__()
        self.encoder = VaeEncoder(input_dim, hidden_dim, latent_dim, condition_dim)
        self.decoder = VaeDecoder(latent_dim, condition_dim, hidden_dim, output_dim)

    def forward(
        self,
        input_sequences: torch.Tensor,
        input_sequences_lengths: torch.Tensor,
        conditions: torch.Tensor,
        output_lengths: torch.Tensor,  # 解码器需要知道目标长度
        sampling_probability: float = 1.0,
        enable_teacher_forcing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        条件VAE前向传播.

        Args:
            input_sequences (Tensor): 填充后的输入序列, (batch_size, max_seq_len, input_dim).
            input_sequences_lengths (Tensor): 每个序列的实际长度 (用于编码器), (batch_size,).
            conditions (Tensor): 环境条件, (batch_size, condition_dim).
            output_lengths (Tensor): 要生成的每个序列的目标长度 (用于解码器), (batch_size,).
            sampling_probability (float, optional): 教师强制的概率 (用于解码器). 默认为 1.0.
            enable_teacher_forcing (bool, optional): 是否启用教师强制. 默认为 False.

        Returns:
            tuple: (reconstructed_sequences, mu, logvar) -
                   从潜空间重建的序列 (batch_size, max_output_len, output_dim),
                   以及潜空间分布的均值 (batch_size, latent_dim) 和对数方差 (batch_size, latent_dim).
        """
        # 编码器将序列和条件编码为潜空间分布参数 (mu, logvar)
        mu, logvar = self.encoder(input_sequences, input_sequences_lengths, conditions)
        # 使用重参数化技巧从潜空间分布中采样一个潜向量 z
        latent_vector = self.reparameterize(mu, logvar)

        # 解码器根据潜向量、条件和目标长度重建序列
        # 训练/重建时, 传入原始输入序列作为 target_sequences 用于计划采样
        if enable_teacher_forcing:
            target_sequences = input_sequences
        else:
            target_sequences = None
        reconstructed_sequences = self.decoder(
            latent_vector,
            conditions,
            output_lengths,
            target_sequences=target_sequences,  # 在训练时使用原始序列作为教师信号
            sampling_probability=sampling_probability,
        )

        return reconstructed_sequences, mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        使用重参数化技巧从潜在分布 N(mu, exp(logvar)) 中采样.

        Args:
            mu (Tensor): 潜空间分布的均值, (batch_size, latent_dim).
            logvar (Tensor): 潜空间分布的对数方差, (batch_size, latent_dim).

        Returns:
            Tensor: 从潜在分布中采样的样本, (batch_size, latent_dim).
        """
        # 确保 logvar 数值稳定, 避免 exp(logvar) 过大或过小
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 从标准正态分布采样 epsilon
        sample = mu + eps * std  # 计算采样值
        return sample


def vae_loss(
    reconstructed_sequences: torch.Tensor,
    input_sequences: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    lengths: torch.Tensor,
    monotonicity_feature_dim: int = 0,  # 指定需要单调性的特征维度 (默认为第一个)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算VAE损失: 重构损失 (使用MSE), KL散度 和 单调性损失.
    重构损失和单调性损失通过掩码处理变长序列.

    Args:
        reconstructed_sequences (Tensor): 从潜空间重建的序列, (batch_size, max_seq_len, output_dim).
        input_sequences (Tensor): 填充后的输入序列, (batch_size, max_seq_len, input_dim).
        mu (Tensor): 潜空间分布的均值向量, (batch_size, latent_dim).
        logvar (Tensor): 潜空间分布的对数方差向量, (batch_size, latent_dim).
        lengths (Tensor): 每个序列的实际长度, (batch_size,). 用于损失的掩码.
        monotonicity_feature_dim (int): 需要检查单调性的特征维度索引.

    Returns:
        tuple: (recon_loss, kl_loss, mono_loss) -
               平均重构损失, 平均KL散度损失, 平均单调性损失 (均平均到每个样本或有效元素).
    """
    batch_size = input_sequences.size(0)
    max_seq_len = input_sequences.size(1)
    device = input_sequences.device

    # --- 重构损失 (Reconstruction Loss) ---
    # 计算整个填充张量上的均方误差 (element-wise)
    mse_per_element = (reconstructed_sequences - input_sequences) ** 2

    # 创建一个掩码, 形状为 (batch_size, max_seq_len), 值是 True/False
    # 如果 time_step < actual_length, 则为 True (有效), 否则为 False (填充)
    mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)

    # 将掩码扩展到与 MSE 相同的维度以便逐元素相乘
    mask_recon = mask.unsqueeze(-1).expand_as(mse_per_element)

    # 将掩码转换为浮点类型, 以便与误差相乘
    mask_recon = mask_recon.float()

    # 应用掩码, 使得填充位置的误差为 0
    masked_mse = mse_per_element * mask_recon

    # 计算所有有效元素的总误差. 先在时间步和特征维度求和, 再在批次维度求和
    total_recon_error = torch.sum(masked_mse)

    # 总有效元素数量
    num_valid_elements = torch.sum(lengths) * input_sequences.size(-1)
    # 将总误差平均到每个有效元素
    # 避免除以零
    if num_valid_elements == 0:
        recon_loss = torch.zeros(1, device=device)
    else:
        recon_loss = total_recon_error / num_valid_elements

    # --- KL 散度损失 (KL Divergence Loss) ---
    # 计算 KL 散度: D_KL(N(mu, exp(logvar)) || N(0, 1))
    # 公式为 -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # 确保 logvar 数值稳定
    logvar = torch.clamp(logvar, min=-10, max=10)
    # KL 散度计算针对每个样本, 在 latent_dim 上求和
    kl_loss_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    # 平均 KL 散度损失到整个批次
    kl_loss = torch.mean(kl_loss_per_sample)

    # --- 单调性损失 (Monotonicity Loss) ---
    mono_loss = torch.zeros(1, device=device)  # 初始化单调性损失

    if reconstructed_sequences.size(2) > monotonicity_feature_dim:
        # 提取需要检查单调性的特征序列
        reconstructed_feature = reconstructed_sequences[
            :, :, monotonicity_feature_dim
        ]  # 形状: (batch_size, max_seq_len)

        # 计算相邻时间步的差值 (diff_t = value_t+1 - value_t)
        # 形状: (batch_size, max_seq_len - 1)
        differences = reconstructed_feature[:, 1:] - reconstructed_feature[:, :-1]

        # 创建一个掩码, 形状为 (batch_size, max_seq_len - 1)
        # 对于长度为 L 的序列, 我们检查到 L-2 索引处 (对应 max_seq_len - 1 差值)
        # 掩码值为 True 表示是有效的时间步差值 (t 到 t+1, 且 t+1 < length)
        mask_mono = torch.arange(max_seq_len - 1, device=device).unsqueeze(0) < (
            lengths.unsqueeze(1) - 1
        )
        mask_mono = mask_mono.float()  # 转换为浮点类型

        # 只考虑违反单调性 (上升) 的差值, 并应用掩码
        # max(0, diff) 用于只惩罚正的差值 (value_t+1 > value_t)
        violation_diffs = torch.relu(differences)  # Relu(diff) = max(0, diff)

        # 应用掩码
        masked_violation_diffs = violation_diffs * mask_mono

        # 计算每个序列的单调性惩罚总和
        sum_violation_diffs_per_sample = torch.sum(
            masked_violation_diffs, dim=-1
        )  # 形状: (batch_size,)

        # 将损失平均到整个批次
        # 也可以选择平均到每个序列的有效时间步差值数量 (lengths - 1)
        # 这里我们平均到整个批次大小
        mono_loss = torch.mean(sum_violation_diffs_per_sample)

    return recon_loss, kl_loss, mono_loss


def train_one_epoch(
    model: ConditionalVAE,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    num_epochs: int,
    device: torch.device,
    enable_teacher_forcing: bool,
    sampling_probability: float,
    beta: float,
    monotonicity_weight: float = 1.0,
) -> Tuple[float, float, float, float, int]:
    """
    在一个 epoch 中训练模型
    Args:
        model (ConditionalVAE): 要训练的模型实例
        dataloader (torch.utils.data.DataLoader): 训练数据的数据加载器
        optimizer (torch.optim.Optimizer): 模型的优化器
        epoch (int): 当前的 epoch 数 (从 0 开始)
        num_epochs (int): 训练的总 epoch 数
        device (torch.device): 训练设备
        enable_teacher_forcing (bool): 是否启用教师forcing
        sampling_probability (float): 当前 epoch 的计划采样概率
        beta (float): 当前 epoch 的 KL 散度权重
        monotonicity_weight (float): 单调性损失权重
    Returns:
        Tuple: 平均总损失, 平均重构损失, 平均 KL 损失, 平均单调性损失, 处理的批次数量
    """
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_mono = 0
    num_batches = 0

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Sampling Probability: {sampling_probability:.4f}, Current Beta: {beta:.4f}, Monotonicity Weight: {monotonicity_weight:.4f}"
    )

    for batch in dataloader:
        # 假定 batch 包含 (sequences, conditions, lengths, others...)
        # sequences 的形状应为 [批次大小, 最大序列长度, 特征维度]
        # lengths 的形状应为 [批次大小], 包含每个序列的实际长度
        if len(batch) < 3:
            print(
                f"警告: Epoch {epoch + 1}, 批次数据格式不正确, 预期至少包含序列, 条件和长度. 跳过此批次."
            )
            continue

        sequences, conditions, lengths = batch[:3]  # 取前三个元素

        # 过滤掉长度为 0 的序列并按长度降序排序 (packed_padded_sequence 要求)
        # 将长度张量转换为 Python list 以便排序和过滤
        lengths_list: List[int] = lengths.tolist()
        valid_indices = [i for i, length in enumerate(lengths_list) if length > 0]

        if not valid_indices:
            # 如果批次中没有有效序列, 跳过此批次
            continue

        sequences = sequences[valid_indices].to(device)
        conditions = conditions[valid_indices].to(device)
        lengths = lengths[valid_indices].to(device)  # 确保 lengths 也是在 device 上

        # 按长度降序排序
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sequences = sequences[sorted_indices]
        conditions = conditions[sorted_indices]

        # 确保 sequences 的形状是 [批次大小, 最大序列长度, 特征维度]
        # 假设 dataloader 已经处理好 padding
        if sequences.ndim != 3:
            print(
                f"警告: Epoch {epoch + 1}, Batch {num_batches + 1}, 序列张量形状不正确: {sequences.shape}. 预期 [批次大小, 最大序列长度, 特征维度]. 跳过此批次."
            )
            continue

        optimizer.zero_grad()

        # 前向传播, 传入计划采样概率
        reconstructed_sequences, mu, logvar = model(
            sequences,
            lengths,  # 编码器使用实际长度
            conditions,
            lengths,  # 解码器在训练时也使用实际长度作为目标长度
            sampling_probability=sampling_probability,
            enable_teacher_forcing=enable_teacher_forcing,
        )

        # 计算重构损失和 KL 散度损失
        # 假设 vae_loss 函数能够处理 padded 序列和对应的长度
        recon_loss, kl_loss, mono_loss = vae_loss(  # 调用 vae_loss 并接收 mono_loss
            reconstructed_sequences,
            sequences,
            mu,
            logvar,
            lengths,
        )

        # 计算总损失: 重构损失 + beta * KL散度损失
        loss = recon_loss + beta * kl_loss + mono_loss * monotonicity_weight

        # 检查损失是否有效 (非 NaN/Inf)
        if not torch.isfinite(loss):
            print(
                f"警告: Epoch {epoch + 1}, Batch {num_batches + 1} 跳过由于非有限损失: {loss.item()}"
            )
            continue

        # 反向传播和优化
        loss.backward()
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_mono += mono_loss.item()
        num_batches += 1

    # 计算并返回平均损失
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches
        avg_mono = total_mono / num_batches
        print(
            f"Avg Loss: {avg_loss:.4f}, "
            f"Avg Recon: {avg_recon:.4f}, "
            f"Avg KL: {avg_kl:.4f}, "
            f"Avg Mono: {avg_mono:.4f}"
        )
        return avg_loss, avg_recon, avg_kl, avg_mono, num_batches
    else:
        return 0.0, 0.0, 0.0, 0.0, 0  # 如果没有有效批次, 返回 0


def train_vae(
    model: ConditionalVAE,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    device: torch.device,
    enable_teacher_forcing: bool = False,
):
    """
    训练 Conditional VAE 模型.

    Args:
        model (ConditionalVAE): 要训练的模型实例.
        dataloader (torch.utils.data.DataLoader): 训练数据的数据加载器.
        num_epochs (int): 训练的总 epoch 数.
        device (torch.device): 训练设备 (例如 'cpu' 或 'cuda').
        enable_teacher_forcing (bool): 是否启用教师forcing, 默认为 False.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)

    # 学习率调度器, 监控平均总损失并在损失停止改善时降低学习率
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)

    # KL 散度权重 (beta) 的 sigmoid warm-up 计划参数
    beta_max = 0.1  # KL 权重的最大值
    beta_sigmoid_center_ratio = 0.05  # sigmoid 中心点占总 epoch 的比例 # 0.25优
    beta_sigmoid_steepness = 0.1  # sigmoid 陡峭程度 (数值越大越陡峭)

    for epoch in range(num_epochs):
        # 获取当前 epoch 的计划采样概率和 beta 值
        current_sampling_probability = get_sampling_probability(
            epoch, num_epochs, end_epoch_ratio=0.8, power=60.0  # 0.8, 50优
        )
        current_beta = get_beta_sigmoid(
            epoch,
            num_epochs,
            beta_max=beta_max,
            center_ratio=beta_sigmoid_center_ratio,
            steepness=beta_sigmoid_steepness,
        )

        current_monotonicity_weight = get_linear_warmup_schedule(
            epoch,
            num_epochs,
            0,
            1,
            0.015,  # 传入 Warm-up 比例
        )

        # 训练一个 epoch
        avg_loss, avg_recon, avg_kl, avg_mono, num_batches = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            epoch=epoch,
            num_epochs=num_epochs,
            device=device,
            enable_teacher_forcing=enable_teacher_forcing,
            sampling_probability=current_sampling_probability,
            beta=current_beta,
            monotonicity_weight=current_monotonicity_weight,
        )

        # 根据平均总损失更新学习率
        if num_batches > 0:
            scheduler.step(avg_loss)
        else:
            print(f"警告: Epoch {epoch + 1}, 没有处理任何有效批次. 跳过学习率调度更新.")


def get_sampling_probability(
    epoch: int, num_epochs: int, end_epoch_ratio: float = 0.2, power: float = 3.0
) -> float:
    """
    计算当前 epoch 的计划采样概率.
    概率从 1.0 衰减到 0.0. end_epoch_ratio 控制衰减结束在总 epoch 的比例. power 控制衰减曲线形状.
    Args:
        epoch (int): 当前的 epoch 数 (从 0 开始).
        num_epochs (int): 训练的总 epoch 数.
        end_epoch_ratio (float): 衰减结束的 epoch 比例.
        power (float): 控制衰减曲线的幂次.
    Returns:
        float: 当前 epoch 的采样概率.
    """
    end_epoch = num_epochs * end_epoch_ratio
    if end_epoch <= 0 or epoch >= end_epoch:
        return 0.0

    base = 1.0 - epoch / end_epoch
    prob = math.pow(max(0.0, base), power)

    return max(0.0, min(1.0, prob))


def get_beta_sigmoid(
    epoch: int, num_epochs: int, beta_max: float, center_ratio: float, steepness: float
) -> float:
    """
    计算当前 epoch 的 KL 散度权重 (beta) 使用 sigmoid warm-up.
    beta 从接近 0 平滑过渡到 beta_max.
    Args:
        epoch (int): 当前的 epoch 数 (从 0 开始).
        num_epochs (int): 训练的总 epoch 数.
        beta_max (float): KL 权重的最大值.
        center_ratio (float): sigmoid 中心点占总 epoch 的比例.
        steepness (float): sigmoid 的陡峭程度.
    Returns:
        float: 当前 epoch 的 beta 值.
    """
    center = num_epochs * center_ratio
    arg = steepness * (epoch - center)
    # 限制 arg 范围以避免 math.exp 溢出或下溢
    safe_arg = max(-70.0, min(70.0, arg))  # 使用更稳健的范围
    sigmoid_val = 1 / (1 + math.exp(-safe_arg))
    return beta_max * sigmoid_val


def get_linear_warmup_schedule(
    epoch: int,
    num_epochs: int,
    start_weight: float,
    end_weight: float,
    warmup_ratio: float,  # Warm-up 阶段占总 epoch 的比例
) -> float:
    """
    计算当前 epoch 的带有 Warm-up 的线性调度权重.
    在 Warm-up 阶段 (前 warmup_ratio * num_epochs), 权重保持 start_weight.
    之后线性增加到 end_weight.
    Args:
        epoch (int): 当前的 epoch 数 (从 0 开始).
        num_epochs (int): 训练的总 epoch 数.
        start_weight (float): 调度的起始权重 (Warm-up 阶段的权重).
        end_weight (float): 调度的最终权重.
        warmup_ratio (float): Warm-up 阶段占总 epoch 的比例 (0.0 到 1.0).
    Returns:
        float: 当前 epoch 的权重.
    """
    # 计算 Warm-up 阶段的 epoch 数量
    warmup_epochs_count = int(num_epochs * warmup_ratio)

    # 如果当前 epoch 在 Warm-up 阶段之前, 权重等于起始权重
    if epoch < warmup_epochs_count:
        return start_weight

    # 计算线性增加阶段的总 epoch 数量
    # 线性阶段从 warmup_epochs_count 开始, 到 num_epochs - 1 结束
    epochs_in_linear_phase = num_epochs - warmup_epochs_count

    # 如果线性阶段没有或只有一个 epoch, 且当前 epoch >= Warm-up 结束点, 直接返回最终权重
    # 或者 Warm-up 持续到总 epoch 数之后, 始终返回 start_weight
    if epochs_in_linear_phase <= 1:
        if warmup_epochs_count >= num_epochs:  # Warm-up 持续到或超过总 epoch 数
            return start_weight
        else:  # 线性阶段只有 0 或 1 个 epoch
            return end_weight if epoch >= warmup_epochs_count else start_weight

    # 计算当前 epoch 在线性增加阶段中的位置 (从 0 开始计数)
    current_epoch_in_linear_phase = epoch - warmup_epochs_count

    # 计算当前 epoch 在线性阶段中完成的比例 (0.0 到 1.0)
    # 使用 epochs_in_linear_phase - 1 来确保在线性阶段的最后一个 epoch 达到 end_weight
    ratio = float(current_epoch_in_linear_phase) / (epochs_in_linear_phase - 1)

    # Clamp 比例在 0 到 1 之间 (理论上不需要, 但更安全)
    ratio = max(0.0, min(1.0, ratio))

    # 在起始权重和最终权重之间进行线性插值
    current_weight = start_weight + (end_weight - start_weight) * ratio

    # 确保权重在起始和最终值之间
    return max(start_weight, min(end_weight, current_weight))
