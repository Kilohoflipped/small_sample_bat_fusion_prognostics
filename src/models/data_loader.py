"""
电池老化循环数据加载模块，包含Dataset类定义、数据预处理及标准化功能.

本模块负责从原始数据加载、过滤、解析电池序列和条件数据,
并对序列数据进行全局 Z-score 标准化. 条件数据进行全局 Min-Max 标准化.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as torch_data

# 配置 logging
logger = logging.getLogger(__name__)

# 定义一个小的 epsilon 常量，用于数值稳定性
EPSILON: float = 1e-9


class BatteryDataset(torch_data.Dataset):
    """
    电池老化循环数据集类，用于处理时间序列数据的加载与标准化.

    Attributes:
        battery_ids: 电池ID列表, 形状为 [N,].
        sequences: 标准化后的电池循环序列数据列表, 形状为 [N, seq_len].
        conditions: 标准化后的环境条件参数列表, 形状为 [N, condition_dim].
        lengths: 每个电池循环序列的实际长度列表, 形状为 [N,].
    """

    def __init__(
        self,
        battery_ids: List[str],
        sequences: List[torch.Tensor],
        conditions: List[torch.Tensor],
        lengths: List[int],
    ):
        """
        初始化 BatteryDataset.

        Args:
            battery_ids: 电池ID列表.
            sequences: 标准化后的电池循环序列数据列表.
            conditions: 标准化后的环境条件参数列表.
            lengths: 每个电池循环序列的实际长度列表.
        """
        self.battery_ids: List[str] = battery_ids
        self.sequences: List[torch.Tensor] = sequences
        self.conditions: List[torch.Tensor] = conditions
        self.lengths: List[int] = lengths

    def __len__(self) -> int:
        """返回数据集中电池序列的总数."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """
        获取指定索引的数据项
        Args:
            idx (int): 数据项的索引
        Returns:
            Tuple[torch.Tensor, torch.Tensor, int, str]: (序列数据, 条件数据, 序列长度, 电池ID)
        """
        # 获取序列数据, 转换为float32并增加一个特征维度
        seq: torch.Tensor = self.sequences[idx].to(dtype=torch.float32).unsqueeze(-1)
        # 获取条件数据, 转换为float32
        cond: torch.Tensor = self.conditions[idx].to(dtype=torch.float32)

        # 获取序列长度
        length: int = self.lengths[idx]
        # 获取电池ID
        battery_id: str = self.battery_ids[idx]

        return seq, cond, length, battery_id


class BatteryDataLoader:
    """
    电池数据加载器，封装数据预处理、标准化及DataLoader创建逻辑

    Attributes:
        batch_size: 每个批次的样本数量
        shuffle: 是否打乱数据
        dataset: 构建好的Dataset实例
        target_mean: 目标序列 Z-score 标准化的均值
        target_std: 目标序列 Z-score 标准化的标准差
        raw_sequences: 原始的未标准化序列数据 (在任何处理之前)
    """

    def __init__(
        self,
        csv_path: str,
        cond_bounds: List[List[float]],
        exclude_battery_ids: Optional[List[str]],
        shuffle: bool,
        batch_size: int = 4,
    ):
        """
        初始化 BatteryDataLoader.

        Args:
            csv_path (str): 原始CSV文件路径.
            cond_bounds (List[List[float]]): 条件变量的上下界列表，用于 Min-Max 标准化.
            exclude_battery_ids (Optional[List[str]]): 需要排除的电池ID列表.
            shuffle (bool): 是否在每个 epoch 开始时打乱数据.
            batch_size (int): 批次大小. 默认为 4.
        """
        self.csv_path: str = csv_path
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle

        # 初始化数据处理器
        self.processor: BatteryDataProcessor = BatteryDataProcessor(
            csv_path, cond_bounds, exclude_battery_ids
        )

        # 使用处理后的数据构建 Dataset
        self.dataset: BatteryDataset = BatteryDataset(
            sequences=self.processor.processed_data["sequences"],
            conditions=self.processor.processed_data["conditions"],
            lengths=self.processor.processed_data["lengths"],
            battery_ids=self.processor.processed_data["battery_ids"],
        )

        # 保存标准化参数和原始数据
        self.target_mean: torch.Tensor = self.processor.processed_data["target_mean"]
        self.target_std: torch.Tensor = self.processor.processed_data["target_std"]
        # 注意: raw_sequences 这里保存的是刚从CSV加载, 未经任何标准化的原始数据
        self.raw_sequences: List[np.ndarray] = self.processor.processed_data["raw_sequences"]

    def create_loader(self) -> torch_data.DataLoader:
        """
        创建配置好的 DataLoader.
        Returns:
            torch_data.DataLoader: 配置好的 DataLoader 实例.
        """
        # 移除冗余的 dataset 初始化检查
        return torch_data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=collate_fn,  # 指定整理函数
            num_workers=0,  # 工作进程数
        )


class BatteryDataProcessor:
    """
    电池数据处理器，负责加载、过滤、解析和标准化数据.

    对序列数据进行全局 Z-score 标准化. 对条件数据进行全局 Min-Max 标准化.
    """

    def __init__(
        self,
        csv_path: str,
        cond_bounds: List[List[float]],
        exclude_battery_ids: Optional[List[str]] = None,
    ):
        """
        初始化 BatteryDataProcessor.

        Args:
            csv_path (str): 原始CSV文件路径.
            cond_bounds (List[List[float]]): 条件变量的上下界列表，用于 Min-Max 标准化.
            exclude_battery_ids (Optional[List[str]]): 需要排除的电池ID列表. 默认为 None.
        """
        self.csv_path: str = csv_path
        self.exclude_battery_ids: List[str] = exclude_battery_ids or []
        self.cond_bounds: List[List[float]] = cond_bounds

        # 加载原始数据 (序列, 条件, 长度, ID)
        self.raw_data: Dict[str, Any] = self._load_data()

        # 处理数据 (标准化等)
        self.processed_data: Dict[str, Any] = self._process_data()

    def _load_data(self) -> Dict[str, Any]:
        """
        从CSV文件加载原始数据并按电池ID分组.

        Returns:
            Dict[str, Any]: 包含原始序列, 条件, 长度和电池ID的字典.
        Raises:
             FileNotFoundError: 如果 CSV 文件不存在.
             KeyError: 如果 DataFrame 缺少必要的列.
             Exception: 加载或过滤时发生其他错误.
        """
        logger.info("开始加载和过滤原始数据: %s", self.csv_path)
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            logger.error("文件未找到: %s", self.csv_path)
            raise FileNotFoundError(f"文件未找到: {self.csv_path}") from None
        except Exception as e:
            logger.error("加载CSV文件时发生错误: %s", e, exc_info=True)
            raise RuntimeError(f"加载CSV文件时发生错误: {e}") from e

        # 检查必要的列是否存在
        required_cols_initial = [
            "battery_id",
            "target",
            "charge_rate",
            "discharge_rate",
            "temperature",
            "pressure",
            "dod",
        ]
        if not all(col in df.columns for col in required_cols_initial):
            missing_cols = [col for col in required_cols_initial if col not in df.columns]
            logger.error("加载数据失败: 输入 DataFrame 缺少必要列 %s.", missing_cols)
            raise KeyError(f"加载数据失败: 输入 DataFrame 缺少必要列 {missing_cols}.")

        # 排除指定的电池ID
        if self.exclude_battery_ids:
            original_rows = len(df)
            df = df[~df["battery_id"].isin(self.exclude_battery_ids)].copy()
            logger.info("排除指定电池ID后, 剩余数据行数: %d (原 %d)", len(df), original_rows)

        if df.empty:
            logger.error("加载并过滤后数据为空.")
            raise ValueError("加载并过滤后数据为空.")

        # 按电池ID分组并提取数据列表
        try:
            grouped = df.groupby("battery_id")

            sequences: List[np.ndarray] = [group["target"].values for _, group in grouped]
            # 提取并解析条件数据. 假定条件对于每个电池是恒定的, 取第一个值.
            conditions: List[List[float]] = [
                [
                    self._parse_rate(group["charge_rate"].iloc[0]),
                    self._parse_rate(group["discharge_rate"].iloc[0]),
                    float(group["temperature"].iloc[0]),
                    float(group["pressure"].iloc[0]),
                    float(group["dod"].iloc[0]),
                ]
                for _, group in grouped
            ]
            # 提取序列长度
            lengths: List[int] = [len(group) for _, group in grouped]
            # 提取电池ID列表
            battery_ids: List[str] = [group["battery_id"].iloc[0] for _, group in grouped]

            if not sequences:  # 检查是否有有效数据被提取
                logger.warning("按电池ID分组后没有提取到有效序列数据.")
                # 可以选择在这里抛出错误: raise ValueError("按电池ID分组后没有提取到有效序列数据.")

        except Exception as e:
            logger.error("按电池ID分组或提取数据时发生错误: %s", e, exc_info=True)
            raise RuntimeError(f"按电池ID分组或提取数据时发生错误: {e}") from e

        raw_data: Dict[str, Any] = {
            "sequences": sequences,
            "conditions": conditions,
            "lengths": lengths,
            "battery_ids": battery_ids,
        }

        logger.info("原始数据加载和提取完成.")
        return raw_data

    def _process_data(self) -> Dict[str, Any]:
        """
        对加载的原始数据进行标准化处理.

        Returns:
            Dict[str, Any]: 包含标准化后序列、条件、原始长度、电池ID及标准化参数的字典.
        Raises:
             ValueError: 如果原始数据为空或处理失败.
             RuntimeError: 如果处理过程中发生不可恢复的错误.
        """
        logger.info("开始数据处理和标准化.")
        # 检查是否有原始数据需要处理
        if not self.raw_data or not self.raw_data.get("sequences"):
            logger.error("数据处理失败: 没有可用的原始序列数据.")
            raise ValueError("数据处理失败: 没有可用的原始序列数据.")

        sequences: List[np.ndarray] = self.raw_data["sequences"]
        conditions: List[List[float]] = self.raw_data["conditions"]

        # 标准化序列数据 (Z-score 标准化)
        # 这是您需要的统计标准化步骤, 应用于 _load_data 提供的原始序列列表
        normalized_sequences, target_mean, target_std = self._normalize_sequences(sequences)
        # 标准化条件数据 (Min-Max 标准化)
        normalized_conditions = self._normalize_conditions(conditions)

        processed_data: Dict[str, Any] = {
            "sequences": normalized_sequences,
            "conditions": normalized_conditions,
            "lengths": self.raw_data["lengths"],  # 长度保持不变
            "battery_ids": self.raw_data["battery_ids"],  # 电池ID保持不变
            "target_mean": target_mean,  # 保存序列标准化参数
            "target_std": target_std,  # 保存序列标准化参数
            "raw_sequences": self.raw_data["sequences"],  # 保存原始序列数据 (加载后的)
        }

        logger.info("数据处理和标准化完成.")
        return processed_data

    @staticmethod
    def _normalize_sequences(
        sequences: List[np.ndarray],
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        对电池循环序列数据进行 Z-score 标准化.

        Args:
            sequences (List[np.ndarray]): 原始序列数据列表.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
                - 标准化后的序列数据列表 (torch.Tensor),
                - 所有序列合并计算得到的均值 (torch.Tensor),
                - 所有序列合并计算得到的标准差 (torch.Tensor).
        Raises:
            ValueError: 如果输入序列数据为空或无法标准化.
            RuntimeError: 如果在计算均值或标准差时发生错误.
        """
        if not sequences:
            logger.warning("标准化序列失败: 输入序列数据为空.")
            # 考虑返回空列表和零张量, 或者抛出错误, 当前选择抛出错误
            raise ValueError("标准化序列失败: 输入序列数据为空.")

        # 将所有序列合并成一个张量计算全局均值和标准差
        try:
            # 使用 np.concatenate 确保即使只有一个序列也能正确处理
            all_seq_flat = np.concatenate(sequences)
            seq_tensor = torch.tensor(all_seq_flat, dtype=torch.float32)

            # 确保张量非空以计算统计量
            if seq_tensor.numel() == 0:
                logger.warning("标准化序列失败: 合并后的序列张量为空.")
                raise ValueError("标准化序列失败: 合并后的序列张量为空.")

            mean: torch.Tensor = seq_tensor.mean()
            # 计算总体标准差 (unbiased=False), 添加 EPSILON 防止除以零
            std: torch.Tensor = seq_tensor.std(unbiased=False)

        except Exception as e:
            logger.error("计算序列全局均值和标准差时发生错误: %s", e, exc_info=True)
            raise RuntimeError(f"计算序列全局均值和标准差时发生错误: {e}") from e

        # 对每个序列进行标准化
        normalized_sequences: List[torch.Tensor] = []
        # 即使全局标准差为零 (所有数据点相同), 加上 EPSILON 也能避免除以零.
        # 标准化结果将是 (x - mean) / EPSILON, 如果所有 x 都等于 mean, 结果将是 0 / EPSILON = 0.
        denominator = std + EPSILON

        for i, seq in enumerate(sequences):
            try:
                # 确保序列是数值类型
                if not np.issubdtype(seq.dtype, np.number):
                    logger.warning(
                        "序列 %d 数据类型不是数值类型 (%s), 尝试转换为 float.", i, seq.dtype
                    )
                    seq = seq.astype(float)

                seq_tensor_single = torch.tensor(seq, dtype=torch.float32)
                normalized_seq = (seq_tensor_single - mean) / denominator
                normalized_sequences.append(normalized_seq)
            except Exception as e:
                logger.error("标准化单个序列 %d 时发生错误: %s", i, e, exc_info=True)
                # 根据需求, 可以选择跳过这个序列或中断并抛出错误
                # 当前选择中断并抛出错误以指示处理问题
                raise RuntimeError(f"标准化单个序列 {i} 时发生错误: {e}") from e

        return normalized_sequences, mean, std

    def _normalize_conditions(self, conditions: List[List[float]]) -> List[torch.Tensor]:
        """
        对环境条件参数进行 Min-Max 标准化.

        Args:
            conditions (List[List[float]]): 原始条件数据列表.

        Returns:
            List[torch.Tensor]: 标准化后的条件数据列表 (torch.Tensor).
        Raises:
            ValueError: 如果输入条件数据为空或边界无效.
            RuntimeError: 如果在计算或应用标准化时发生错误.
        """
        if not conditions:
            logger.warning("标准化条件失败: 输入条件数据为空.")
            # 考虑返回空列表或抛出错误, 当前选择抛出错误
            raise ValueError("标准化条件失败: 输入条件数据为空.")

        try:
            cond_tensor = torch.tensor(conditions, dtype=torch.float32)

            # 确保条件张量维度与边界匹配
            if cond_tensor.size(1) != len(self.cond_bounds):
                logger.error(
                    "标准化条件失败: 条件维度 (%d) 与边界数量 (%d) 不匹配.",
                    cond_tensor.size(1),
                    len(self.cond_bounds),
                )
                raise ValueError(
                    f"标准化条件失败: 条件维度 ({cond_tensor.size(1)}) 与边界数量 ({len(self.cond_bounds)}) 不匹配."
                )

            # 将边界转换为张量
            cond_bound_tensor = torch.tensor(self.cond_bounds, dtype=torch.float32)

            # 提取最小值和最大值
            min_vals: torch.Tensor = cond_bound_tensor[:, 0]
            max_vals: torch.Tensor = cond_bound_tensor[:, 1]
            # 计算范围, 添加 EPSILON 防止除以零
            range_vals: torch.Tensor = max_vals - min_vals + EPSILON

            # 检查范围是否有效 (非负)
            if torch.any(range_vals < EPSILON):
                logger.error("标准化条件失败: 检测到非法的条件范围 (max <= min).")
                raise ValueError("标准化条件失败: 检测到非法的条件范围 (max <= min).")

            # 执行Min-Max标准化: (x - min) / (max - min + EPSILON)
            # 注意使用 unsqueezed min_vals 和 range_vals 进行广播
            normalized_conditions_tensor: torch.Tensor = (
                cond_tensor - min_vals.unsqueeze(0)
            ) / range_vals.unsqueeze(0)

        except Exception as e:
            logger.error("计算或应用条件 Min-Max 标准化时发生错误: %s", e, exc_info=True)
            raise RuntimeError(f"计算或应用条件 Min-Max 标准化时发生错误: {e}") from e

        # 将标准化后的张量拆分成列表
        normalized_conditions: List[torch.Tensor] = [c for c in normalized_conditions_tensor]
        return normalized_conditions

    @staticmethod
    def _parse_rate(rate: Any) -> float:
        """
        解析电池的充电或放电率.

        Args:
            rate (Any): 电池率, 可以是字符串 (例如 '1.0', '1C') 或数字 (例如 1.0).

        Returns:
            float: 解析后的浮点数率值.

        Raises:
            ValueError: 如果输入格式无法解析.
        """
        # 如果输入已经是浮点数或整数, 直接转换为浮点数返回
        if isinstance(rate, (float, int)):
            return float(rate)
        # 如果输入是字符串
        elif isinstance(rate, str):
            # 移除 "C" 并去除首尾空格
            rate_str = rate.replace("C", "").strip()
            try:
                # 尝试将处理后的字符串转换为浮点数
                return float(rate_str)
            except ValueError:
                # 如果转换失败, 抛出格式错误
                raise ValueError(f"Invalid rate format: '{rate}'")
        else:
            # 如果输入既不是数字也不是字符串, 则格式无效
            raise ValueError(f"Invalid rate format: '{rate}'")


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, str]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    整理批次数据，填充序列至最大长度，堆叠条件和长度张量.

    Args:
        batch (List[Tuple]): Dataset 返回的批次数据列表, 每个元素是 (seq, cond, length, battery_id) 元组.
                             其中 seq 形状为 (seq_len, 1), cond 形状为 (condition_dim,), length 为 int, battery_id 为 str.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
            - padded_seqs (Tensor): 填充后的序列张量, 形状 (batch_size, max_len, 1).
            - conditions (Tensor): 堆叠后的条件张量, 形状 (batch_size, condition_dim).
            - lengths (Tensor): 实际序列长度的张量, 形状 (batch_size,).
            - battery_ids (List[str]): 批次中电池ID列表, 顺序与排序后的数据对应.
    """
    # 解压批次数据
    seqs, conditions, lengths, battery_ids = zip(*batch)

    # 根据序列长度降序排序，以便进行 pad_sequence
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)

    # 按照排序后的索引重新组织数据
    battery_ids = [battery_ids[i] for i in sorted_indices]
    seqs = [seqs[i] for i in sorted_indices]
    conditions = [conditions[i] for i in sorted_indices]
    lengths = [lengths[i] for i in sorted_indices]

    # 对序列进行填充，使其长度一致
    padded_seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    # 将条件数据堆叠成一个张量
    conditions = torch.stack(conditions)
    # 将长度数据转换为张量，并指定 dtype. 设备通常由 DataLoader 自动处理或在 VAEEncoder 中移至 CPU
    lengths = torch.tensor(lengths, dtype=torch.long)

    return padded_seqs, conditions, lengths, battery_ids
