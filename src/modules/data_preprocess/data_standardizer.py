"""
使用特定方法对时间序列数据进行标准化的模块.

本模块包含 DataStandardizer 类, 专注于对时间序列数据进行标准化处理,
主要用于处理电池老化数据. 标准化结果将覆盖原始目标列.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataStandardizerConfig:
    """
    数据标准化的配置.

    包含标准化方法所需的参数，不包含目标列名或新列创建选项.
    目前该配置类没有特定的参数，但保留以便未来扩展.
    """


class DataStandardizer:
    """
    处理时间序列数据标准化的类，专注于特定方法.

    使用 (最大值 + 初始值) / 2 作为分母对时间序列数据进行标准化.
    按组（例如 battery_id）应用标准化，结果直接覆盖原始目标列.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 DataStandardizer.

        从配置字典中提取参数，并创建配置类的实例.

        Args:
            config (Dict[str, Any]): 数据标准化相关的配置字典.
                                 期望包含 'data_standardizer' 键，其值是包含标准化配置的字典.
        """
        # standardizer_config_dict = config.get("data_standardizer", {})

        # try:
        #     # 使用 dataclass 解析配置
        #     self.config: DataStandardizerConfig =
        #     DataStandardizerConfig(**standardizer_config_dict)
        # except TypeError as e:
        #     logger.error("数据标准化配置字典与 DataStandardizerConfig 定义不匹配: %s", e)
        #     raise ValueError(f"数据标准化配置字典与 DataStandardizerConfig 定义不匹配: {e}") from e
        # except Exception as e:
        #     logger.error("初始化 DataStandardizer 配置时发生未预期错误: %s", e, exc_info=True)
        #     raise RuntimeError(f"初始化 DataStandardizer 配置时发生未预期错误: {e}") from e

        logger.info("DataStandardizer 初始化完成")
        # logger.info("配置: %s", self.config)

    def _calculate_denominator(self, signal: np.ndarray) -> float:
        """
        计算标准化所需的分母: (最大值 + 初始值) / 2.

        Args:
            signal (np.ndarray): 输入时间序列.

        Returns:
            float: 计算出的分母.

        Raises:
            ValueError: 如果输入信号为空或长度不足，无法计算分母.
            RuntimeError: 如果在计算过程中发生其他错误.
        """
        if signal is None or len(signal) < 1:
            # 至少需要一个点来获取初始值和最大值
            raise ValueError("计算分母时输入信号为空或长度不足.")

        try:
            # 获取初始值 (第一个元素). 确保是浮点数以便计算
            initial_value: float = float(signal[0])

            # 获取最大值. 确保是浮点数以便计算
            max_value: float = float(np.max(signal))

            # 计算分母
            denominator: float = (max_value + initial_value) / 2.0

            # 检查分母是否接近零
            if abs(denominator) < 1e-9:
                # 如果分母接近零，记录警告，并让 ZeroDivisionError 在标准化步骤中处理
                logger.error(
                    f"计算出的分母 ({denominator}) 接近零. 标准化步骤可能会发生 ZeroDivisionError."
                )
                raise RuntimeError(f"计算出的分母 ({denominator}) 接近零.")

            return denominator

        except Exception as e:
            # 捕获计算过程中的其他潜在错误 (例如，信号包含非数值)
            logger.error("计算分母时发生错误: %s", e, exc_info=True)
            raise RuntimeError(f"计算分母时发生错误: {e}") from e

    def _standardize_signal(self, signal: np.ndarray, denominator: float) -> np.ndarray:
        """
        对单个时间序列应用标准化.

        标准化方法：除以计算出的分母.

        Args:
            signal (np.ndarray): 输入时间序列.
            denominator (float): 计算出的分母.

        Returns:
            np.ndarray: 标准化后的时间序列.

        Raises:
            ValueError: 如果输入信号无效.
            ZeroDivisionError: 如果分母为零，无法进行标准化.
            RuntimeError: 如果在标准化过程中发生其他错误.
        """
        if signal is None or len(signal) == 0:
            raise ValueError("标准化信号时输入信号为空或长度为零.")

        if denominator == 0:
            raise ZeroDivisionError("计算出的分母为零，无法进行标准化.")

        try:
            # 应用标准化
            # 确保信号是数值类型，避免除法错误
            if not np.issubdtype(signal.dtype, np.number):
                logger.warning("信号数据类型不是数值类型 (%s), 尝试转换为 float.", signal.dtype)
                try:
                    signal = signal.astype(float)
                except Exception as e:
                    raise RuntimeError(f"标准化信号时无法将信号转换为 float: {e}") from e

            standardized_signal = signal / denominator
            return standardized_signal
        except Exception as e:
            logger.error("标准化信号时发生错误: %s", e, exc_info=True)
            raise RuntimeError(f"标准化信号时发生错误: {e}") from e

    def standardize_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, str]:
        """
        按 battery_id 对指定目标列的数据进行标准化.

        使用传入的目标列名和配置中的标准化参数. 标准化结果将覆盖原始目标列.

        Args:
            df (pd.DataFrame): 输入数据，包含电池老化数据.
                               应包含 'battery_id' 和 target_column 参数指定的列.
            target_column (str): 需要标准化的目标列名.

        Returns:
            Tuple[pd.DataFrame, str]:
                - pd.DataFrame: 包含标准化后数据的 DataFrame. 原始目标列被覆盖.
                                返回的 DataFrame 始终是原始 DataFrame 的副本，并已进行修改.
                - str: 标准化处理后的目标列名 (即传入的 target_column).

        Raises:
            ValueError: 如果输入数据为空.
            KeyError: 如果输入 DataFrame 缺少标准化所需的必要列 ('battery_id', target_column).
            RuntimeError: 如果处理过程中发生不可恢复的错误 (例如，按 battery_id 分组失败，
                          或单个电池的处理中发生 Value, ZeroDivision, Runtime Error).
        """
        logger.info("开始对列 '%s' 进行数据标准化 (覆盖原始列)...", target_column)

        if df is None or df.empty:
            logger.error("数据标准化失败: 输入数据为空.")
            raise ValueError("数据标准化失败: 输入数据为空.")

        # 检查必要列是否存在
        required_cols = ["battery_id", target_column]
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            logger.error("数据标准化失败: 输入 DataFrame 缺少必要列 %s.", missing_cols)
            raise KeyError(f"数据标准化失败: 输入 DataFrame 缺少必要列 {missing_cols}.")

        # 确保 battery_id 列是字符串类型，以正确进行分组
        try:
            # 避免修改原始df，创建一个副本
            df_standardized = df.copy()
            df_standardized["battery_id"] = df_standardized["battery_id"].astype(str)
            # 过滤掉无效的 battery_id
            df_standardized = df_standardized[
                (df_standardized["battery_id"] != "nan") & (df_standardized["battery_id"] != "")
            ].copy()

        except Exception as e:
            logger.error(
                "数据标准化失败: 转换 'battery_id' 类型或过滤时发生错误: %s",
                e,
                exc_info=True,
            )
            raise RuntimeError(
                f"数据标准化失败: 转换 'battery_id' 类型或过滤时发生错误: {e}"
            ) from e

        if df_standardized.empty:
            logger.error("过滤无效 battery_id 后数据为空, 无法进行标准化.")
            # 按照其他类的风格，如果关键步骤后数据为空，抛出错误
            raise RuntimeError("数据标准化失败: 过滤无效 battery_id 后数据为空.")

        # 按 battery_id 分组标准化
        try:
            grouped = df_standardized.groupby("battery_id")
            group_keys = list(grouped.groups.keys())

            if not group_keys:
                # 如果分组结果为空，说明没有有效的 battery_id 数据用于标准化
                logger.error("数据标准化失败: 输入数据中没有有效的电池 ID 可用于分组标准化.")
                raise RuntimeError("数据标准化失败: 输入数据中没有有效的电池 ID 可用于分组标准化.")

        except Exception as e:
            logger.error("数据标准化失败: 按 'battery_id' 分组时发生错误: %s", e, exc_info=True)
            raise RuntimeError(f"数据标准化失败: 按 'battery_id' 分组时发生错误: {e}") from e

        processed_battery_count = 0  # 跟踪尝试处理的电池数量
        failed_battery_ids = []  # 跟踪处理失败的电池ID

        # 遍历每个电池组的键，然后获取对应的分组 DataFrame
        for bid in group_keys:
            processed_battery_count += 1
            # 获取该电池 ID 对应的分组 DataFrame 的索引, 用于更新原始 df_standardized
            battery_indices = grouped.get_group(bid).index

            signal_series = df_standardized.loc[battery_indices, target_column]
            signal_array = signal_series.values

            if signal_array is None or len(signal_array) == 0:
                logger.warning(f"电池 {bid} 的标准化目标信号为空或长度为零, 跳过标准化.")
                # 不计入失败, 只是跳过
                continue

            try:
                # 计算分母
                denominator = self._calculate_denominator(signal_array)

                # 执行标准化
                standardized_signal_array = self._standardize_signal(signal_array, denominator)

                # 将标准化结果应用回原始 df_standardized 中对应的行
                # 使用之前获取的 battery_indices 来确保对正确行进行更新
                df_standardized.loc[battery_indices, target_column] = standardized_signal_array
                logger.debug("电池 %s 数据标准化应用成功.", bid)

            except (ValueError, ZeroDivisionError, RuntimeError) as e:
                logger.error(
                    f"对电池 {bid} 进行数据标准化时发生错误: {e}. 跳过该电池.", exc_info=True
                )
                failed_battery_ids.append(bid)
                # 不在这里抛出错误, 而是记录失败的电池 ID, 并继续处理其他电池
                # 可以在循环结束后检查是否有失败的电池, 如果需要则抛出汇总错误

        if failed_battery_ids:
            logger.warning(
                "部分电池未能成功进行数据标准化. 失败电池数量: %d/%d. 失败电池ID: %s",
                len(failed_battery_ids),
                processed_battery_count,
                failed_battery_ids,
            )
            # 根据需求, 如果部分电池失败被认为是关键错误, 可以在这里抛出 Runtime 错误
            # raise RuntimeError(f"部分电池标准化失败: {failed_battery_ids}")
        elif processed_battery_count > 0:
            logger.info("所有尝试处理的电池都已成功进行数据标准化.")
        else:
            # 理论上不会走到这里, 因为前面检查了 group_keys
            logger.warning("没有电池被尝试进行数据标准化.")

        logger.info(f"数据标准化流程完成. 处理后的目标列: '{target_column}'.")

        # 返回修改后的 DataFrame 副本和原始目标列名
        return df_standardized, target_column
