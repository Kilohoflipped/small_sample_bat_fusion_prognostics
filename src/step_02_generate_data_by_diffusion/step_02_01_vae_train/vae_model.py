"""
基于LSTM模型的条件变分自编码器 (CVAE) 实现
"""

import torch
from torch import nn
from torch.nn.utils import rnn

class VaeEncoder(nn.Module):
    """
    VAE编码器, 将变长序列编码为潜空间分布

    Args:
        input_dim (int): 输入特征维度
        hidden_dim (int): LSTM隐藏层维度
        latent_dim (int): VAE潜空间维度
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)


        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        """
        将LSTM输出的隐藏状态通过全连接层映射为潜空间分布均值
        """
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        """
        将LSTM输出的隐藏状态通过全连接层映射为潜空间分布的对数方差(log(sigma^2))
        """

    def forward(self, input_sequences, input_sequences_lengths):
        """
        Args:
            input_sequences (Tensor): 填充后的输入序列, 形状为 (batch_size, max_seq_len, input_dim)
            input_sequences_lengths (Tensor): 每个序列的实际长度, 形状为 (batch_size,)

            batch_size-训练批次大小, 即一次处理的序列数量
            max_seq_len-训练批次中最长序列的长度, 短序列用填充值(如: 0)补齐
            input_dim-每个时间步的特征维度

        Returns:
            tuple: (mu, logvar) - 潜空间分布的均值和对数方差, 两者形状皆为 (batch_size, latent_dim)
        """

        # 将填充的输入序列打包为紧凑格式
        packed_sequences = rnn.pack_padded_sequence(
            input_sequences, input_sequences_lengths.cpu(), batch_first=True, enforce_sorted=True)

        _, (h_n, _) = self.lstm(packed_sequences)
        """
        将紧凑的打包序列输入LSTM, 获取隐藏状态
        返回值为元组 (output, (h_n, c_n))
        output: 所有时间步的隐藏状态
        h_n: 最终输出的隐藏状态
        c_n: 最终输出的细胞状态
        丢弃output和c_n, 因为mu和log(sigma^2)只基于最终输出的隐藏状态
        """
        # 单层lstm, 去除单维度
        h_n = h_n.squeeze(0)
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

class VaeDecoder(nn.Module):
    """
    VAE解码器, 将潜空间向量和环境条件解码为序列数据

    Args:
        output_dim (int): 输出特征维度
        condition_dim (int): 环境条件维度
        
        hidden_dim (int): LSTM隐藏层维度
        latent_dim (int): VAE潜空间维度
    """
    def __init__(self, latent_dim, condition_dim, hidden_dim, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.latent_to_cell = nn.Linear(latent_dim, hidden_dim)
        self.latent_to_start = nn.Linear(latent_dim, output_dim)

        self.lstm = nn.LSTM(output_dim + condition_dim, hidden_dim, batch_first=True)

        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent_vector, conditions, output_lengths):
        """
        Args:
            latent_vector (Tensor): 潜空间向量, 形状为 (batch_size, latent_dim)
            conditions (Tensor): 环境条件, 形状为 (batch_size, condition_dim)
            output_lengths (Tensor): 要生成的每个序列的目标长度, 形状为 (batch_size,)

        Returns:
            Tensor: 从潜空间重建的序列, 形状为 (batch_size, max_seq_len, output_dim)
        """

        batch_size = latent_vector.size(0)
        max_length = output_lengths.max().item()

        # 初始化LSTM的隐藏状态
        hidden_state_init = self.latent_to_hidden(latent_vector).unsqueeze(0)
        # 初始化LSTM的细胞状态
        # cell_state_init = torch.zeros(1, batch_size, self.hidden_dim, device=latent_vector.device)
        cell_state_init = self.latent_to_cell(latent_vector).unsqueeze(0)
        hidden = (hidden_state_init, cell_state_init)

        # 从潜空间预测初始值
        start_value = self.latent_to_start(latent_vector).unsqueeze(1)

        current_input = start_value
        outputs = []
        
        conditions_expanded = conditions.unsqueeze(1)
        for _ in range(max_length):
            # 扩展条件张量的维度, 与当前输入对齐
            lstm_input = torch.cat([current_input, conditions_expanded], dim=-1)
            # 通过内部状态hidden跨时间步传递信息
            output, hidden = self.lstm(lstm_input, hidden)
            predicted_value = self.hidden_to_output(output)
            outputs.append(predicted_value)
            current_input = predicted_value

        outputs = torch.cat(outputs, dim=1)
        # 创建全零的张量, 于存储经掩码后的输出
        masked_outputs = torch.zeros(batch_size, max_length, self.output_dim, device=conditions.device)
        for i in range(batch_size):
            len_i = output_lengths[i].item()
            masked_outputs[i, :len_i, :] = outputs[i, :len_i, :]
        # 根据实际长度掩码输出
        return masked_outputs

class ConditionalVAE(nn.Module):
    """
    条件变分自编码器, 结合环境条件进行序列生成

    Args:
        input_dim (int): 输入特征维度
        hidden_dim (int): LSTM隐藏层维度
        latent_dim (int): VAE潜空间维度
        condition_dim (int): 环境条件维度
        output_dim (int): 输出特征维度
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim, output_dim):
        super().__init__()
        self.encoder = VaeEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VaeDecoder(latent_dim, condition_dim, hidden_dim, output_dim)

    def forward(self, input_sequences, input_sequences_lengths, conditions, output_lengths):
        """
        Args:
            input_sequences (Tensor): 填充后的输入序列, 形状为 (batch_size, max_seq_len, input_dim)
            input_sequences_lengths (Tensor): 每个序列的实际长度, 形状为 (batch_size,)
            batch_size-训练批次大小, 即一次处理的序列数量
            max_seq_len-训练批次中最长序列的长度, 短序列用填充值(如: 0)补齐
            input_dim-每个时间步的特征维度 (与self.input_dim一致)

            conditions (Tensor): 环境条件, 形状为 (batch_size, condition_dim)
            output_lengths (Tensor): 要生成的每个序列的目标长度, 形状为 (batch_size,)

        Returns:
            tuple: (reconstructed_sequences, mu, logvar) - 从潜空间重建的序列及潜空间分布参数
        """
        mu, logvar = self.encoder(input_sequences, input_sequences_lengths)
        latent_vector = self.reparameterize(mu, logvar)
        reconstructed_sequences = self.decoder(latent_vector, conditions, output_lengths)
        return reconstructed_sequences, mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        """
        使用重参数化技巧从潜在分布中采样
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mu + eps * std
        return sample

def vae_loss(reconstructed_sequences, input_sequences, mu, logvar, lengths):
    """
    计算VAE损失: 重构损失 + KL散度, 考虑变长序列

    Args:
        reconstructed_sequences (Tensor): 从潜空间重建的序列
        input_sequences (Tensor): 填充后的输入序列, 形状为 (batch_size, max_seq_len, input_dim)
        mu (Tensor): 潜空间分布的均值向量
        logvar (Tensor): 潜空间分布的对数方差向量
        lengths (Tensor): 序列的长度, 应与输入序列、重建序列的长度均相同
    """
    batch_size = input_sequences.size(0)
    
    recon_loss = 0
    for i in range(batch_size):
        valid_x = input_sequences[i, :lengths[i], :]
        valid_recon = reconstructed_sequences[i, :lengths[i], :]
        recon_loss += nn.functional.mse_loss(valid_recon, valid_x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    recon_loss = recon_loss / batch_size
    kl_loss = kl_loss / batch_size

    return recon_loss, kl_loss


def train_vae(model, dataloader, num_epochs, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for batch in dataloader:
            sequences, conditions, lengths, _ = batch

            sequences = sequences.to(device)
            conditions = conditions.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            reconstructed_sequences, mu, logvar = model(sequences, lengths, conditions, lengths)
            recon_loss, kl_loss = vae_loss(reconstructed_sequences, sequences, mu, logvar, lengths)

            beta = min(0.1 * (epoch + 1) / 10, 0.1)  # 线性增加到 0.1
            loss = recon_loss + beta * kl_loss

            # 计算模型梯度
            loss.backward()
            # 根据梯度更新模型参数
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Avg Loss: {avg_loss:.4f}, Avg Recon: {avg_recon:.4f}, Avg KL: {avg_kl:.4f}")
