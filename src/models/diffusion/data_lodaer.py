import os

import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """
    用于加载潜在空间数据 (来自 VAE Encoder 输出) 和对应条件的 PyTorch Dataset。

    假设潜在数据和条件信息已预先生成并保存为 PyTorch .pt 文件。
    潜在数据的形状应为 (num_samples, latent_dim)。
    条件信息的形状应为 (num_samples, condition_dim)。
    潜在文件和条件文件中的样本顺序必须一致。
    """

    def __init__(self, latent_file: str, condition_file: str):
        """
        Args:
            latent_file: 包含潜在向量 (.pt) 文件的路径。
            condition_file: 包含对应条件 (.pt) 文件的路径。
        """
        # 检查文件是否存在
        if not os.path.exists(latent_file):
            raise FileNotFoundError(f"潜在数据文件未找到: {latent_file}")
        if not os.path.exists(condition_file):
            raise FileNotFoundError(f"条件数据文件未找到: {condition_file}")

        # --- 加载数据 ---
        try:
            # 使用 torch.load 加载 .pt 文件
            self.latent_data = torch.load(latent_file)
            self.condition_data = torch.load(condition_file)
        except Exception as e:
            print(f"加载潜在数据或条件数据时出错: {e}")
            raise  # 重新抛出异常

        # --- 数据校验 ---
        # 确保潜在数据和条件数据的样本数量一致
        if self.latent_data.shape[0] != self.condition_data.shape[0]:
            raise ValueError(
                f"潜在数据和条件数据的样本数量不一致: "
                f"{self.latent_data.shape[0]} vs {self.condition_data.shape[0]}"
            )

        # 存储数据集信息
        self.num_samples = self.latent_data.shape[0]
        self.latent_dim = self.latent_data.shape[1]
        # 潜在条件可以是标量 (如类别索引)，所以需要检查维度
        self.condition_dim = self.condition_data.shape[1] if self.condition_data.ndim > 1 else 1

        print(f"数据集加载成功！")
        print(f"  总样本数量: {self.num_samples}")
        print(f"  潜在向量维度: {self.latent_dim}")
        print(f"  条件维度: {self.condition_dim}")

    def __len__(self) -> int:
        """
        返回数据集中的总样本数量。
        """
        return self.num_samples

    def __getitem__(self, idx: int):
        """
        根据索引获取单个潜在数据样本及其对应的条件。

        Args:
            idx: 样本的索引（int）。

        Returns:
            一个元组 (z_0, y)，其中 z_0 是潜在向量 (torch.Tensor)，
            y 是对应的条件 (torch.Tensor)。
        """
        # 使用 .clone().detach() 是一个好的实践，确保返回的张量是独立的，
        # 不会意外地影响原始数据或带有不相关的梯度历史。
        z_0 = self.latent_data[idx].clone().detach()
        y = self.condition_data[idx].clone().detach()

        return z_0, y
