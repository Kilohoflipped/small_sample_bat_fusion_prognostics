"""
使用小波变换对时间序列数据进行去噪的模块.

本模块包含 DataDenoiser 类, 用于对时间序列数据应用小波去噪,
主要用于处理电池老化数据中的噪声. 提供多种方式获取去噪结果.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pywt

# 配置日志记录器
logger = logging.getLogger(__name__)
# 设置日志级别，例如 DEBUG, INFO, WARNING, ERROR, CRITICAL
# 如果您在使用 root logger，请根据您的应用整体日志配置来决定是否需要在这里配置 handler 和 formatter
# logging.basicConfig(level=logging.INFO)


@dataclass
class DataDenoiserConfig:
    """
    数据去噪的配置.

    包含小波去噪所需的参数.

    Attributes:
        wavelet (str): 使用的小波基函数名称. 默认值为 "db4".
        threshold_mode (str): 小波系数阈值处理模式 ("soft", "hard", etc.). 默认值为 "soft".
    """

    wavelet: str = "db4"
    threshold_mode: str = "soft"


class DataDenoiser:
    """
    处理时间序列数据去噪的类，专注于小波去噪.

    使用小波变换对时间序列数据进行去噪，按组（例如 battery_id）应用去噪.
    提供多种方法获取去噪结果: 返回按电池分组的原始/去噪数组字典,
    或创建基于去噪结果长度的新长格式 DataFrame.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 DataDenoiser.

        从配置字典中提取参数，并创建配置类的实例.

        Args:
            config (Dict[str, Any]): 数据去噪相关的配置字典.
                                 期望包含 'data_denoiser' 键，其值是包含去噪配置的字典.
        """
        denoiser_config_dict = config.get("data_denoiser", {})

        try:
            self.config: DataDenoiserConfig = DataDenoiserConfig(**denoiser_config_dict)
        except TypeError as e:
            logger.error("数据去噪配置字典与 DataDenoiserConfig 定义不匹配: %s", e)
            raise ValueError(f"数据去噪配置字典与 DataDenoiserConfig 定义不匹配: {e}") from e

        logger.info("DataDenoiser 初始化完成")
        logger.info("配置: %s", self.config)

    def _process_signal_wavelets(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """
        对单个时间序列应用小波分解、阈值处理和重构的内部核心逻辑.

        如果处理失败或信号不适合小波处理，返回 None.
        这个方法**不**检查输出长度是否与输入一致，直接返回 pywt.wavedec 的结果.

        Args:
            signal (np.ndarray): 输入时间序列（一维 numpy 数组）.

        Returns:
            Optional[np.ndarray]: 小波重构后的时间序列. 如果处理失败，返回 None.
        """
        if signal is None or len(signal) < 2:
            logger.debug("信号无效或长度小于 2, 不适合小波处理. 返回 None.")
            return None

        original_signal_len = len(signal)
        wavelet_name = self.config.wavelet
        threshold_mode = self.config.threshold_mode

        try:
            # 使用 level=None 自动确定最大分解层数
            coeffs = pywt.wavedec(signal, wavelet_name, level=None)

            # coeffs 结构: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
            # 最细层次细节系数是 coeffs[-1]
            if not coeffs or len(coeffs) < 2 or coeffs[-1] is None or len(coeffs[-1]) == 0:
                logger.debug("小波分解结果异常或细节系数为空, 无法估计噪声标准差. 返回 None.")
                return None

            detail_coeffs_finest = coeffs[-1]
            # 检查最细层次细节系数是否包含非有限值
            if not np.all(np.isfinite(detail_coeffs_finest)):
                logger.debug("最细层次细节系数包含非有限值, 无法估计噪声标准差. 返回 None.")
                return None

            # 使用 Median Absolute Deviation 估计噪声标准差
            mad = np.median(np.abs(detail_coeffs_finest - np.median(detail_coeffs_finest)))
            sigma = mad / 0.6745

            if sigma <= 1e-9:
                logger.debug("估计的噪声标准差接近零 (<= 1e-9), 认为无需去噪. 返回原始信号.")
                # 如果噪声可忽略, 返回原始信号的副本作为“去噪”结果
                return np.copy(signal)

            # 计算阈值 (VisuShrink)
            # 确保 original_signal_len 大于 1 用于对数计算
            if original_signal_len <= 1:
                logger.debug(
                    "信号长度过短 (%d <= 1) 无法计算通用阈值. 返回 None.", original_signal_len
                )
                return None

            log_len = np.log(original_signal_len)
            if log_len <= 0:  # 长度 > 1 时通常不会发生, 但作为安全检查
                logger.debug("计算 log(信号长度) 结果非正, 阈值计算异常. 返回 None.")
                return None

            threshold = sigma * np.sqrt(2 * log_len)

            # 对细节系数 (coeffs[1:]) 应用阈值处理, 保留近似系数 coeffs[0]
            thresholded_coeffs = [coeffs[0]] + [
                pywt.threshold(c, threshold, mode=threshold_mode) for c in coeffs[1:]
            ]

            # 执行重构 - 重构结果长度可能与原始信号不同
            denoised = pywt.waverec(thresholded_coeffs, wavelet_name)

            # 检查重构结果是否包含非有限数值 (NaN 或 Inf)
            if not np.all(np.isfinite(denoised)):
                logger.error("小波重构结果包含非有限数值 (NaN 或 Inf). 返回 None.")
                return None

            logger.debug(
                "小波处理成功完成分解和重构. 原始长度 %d, 重构长度 %d.",
                original_signal_len,
                len(denoised),
            )
            # 返回重构后的信号, 不论长度是否匹配
            return denoised

        except ValueError as e:
            logger.warning(
                "小波处理失败 (可能信号长度与小波不兼容或系数问题). 错误: %s. 返回 None.", e
            )
            return None
        except Exception as e:
            logger.error("小波处理过程中发生意外错误: %s. 返回 None.", e, exc_info=True)
            return None

    def denoise_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, str]:
        """
        按 battery_id 对指定目标列的数据进行去噪，返回一个新的长格式 DataFrame.

        此方法调用 get_denoised_long_dataframe 来生成基于去噪结果长度的 DataFrame.
        如果去噪失败或处理失败, 对应电池在目标列将填充 NaN. 返回一个新的 DataFrame.

        Args:
            df (pd.DataFrame): 输入数据 (长格式). 包含 'battery_id', 'cycle_idx', target_column 等列.
            target_column (str): 需要去噪的目标列名.

        Returns:
            pd.DataFrame: 包含去噪后数据的新长格式 DataFrame. 行数可能与原始 DataFrame 不同.
                          去噪失败的电池对应 target_column 列将填充 NaN.
            str: 去噪后的目标列名

        Raises:
            ValueError: 如果输入数据为空.
            KeyError: 如果输入 DataFrame 缺少必要列.
            RuntimeError: 如果初始化处理或分组失败.
        """
        logger.info("开始对列 '%s' 进行数据去噪 (返回新的长格式 DataFrame)...", target_column)

        # 原有的前置检查仍然保留，确保输入有效
        if df is None or df.empty:
            logger.error("数据去噪失败: 输入数据为空.")
            raise ValueError("数据去噪失败: 输入数据为空.")

        required_cols = ["battery_id", "cycle_idx", target_column]
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            logger.error("数据去噪失败: 输入 DataFrame 缺少必要列 %s.", missing_cols)
            raise KeyError(f"数据去噪失败: 输入 DataFrame 缺少必要列 {missing_cols}.")

        # 直接调用并返回 get_denoised_long_dataframe 的结果
        # get_denoised_long_dataframe 内部会处理 battery_id 的类型和过滤无效 ID
        try:
            df_denoised = self.get_denoised_long_dataframe(df, target_column)
        except Exception as e:
            # 捕获 get_denoised_long_dataframe 可能抛出的异常
            logger.error("调用 get_denoised_long_dataframe 失败: %s", e, exc_info=True)
            raise RuntimeError(f"数据去噪失败: 调用 get_denoised_long_dataframe 失败: {e}") from e

        logger.info("数据去噪流程完成. 返回新的长格式 DataFrame.")

        # 返回新的长格式 DataFrame
        return df_denoised, target_column

    def get_denoised_results_by_battery(
        self, df: pd.DataFrame, target_column: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        按 battery_id 进行小波去噪，返回包含原始和去噪后信号的字典.

        即使去噪后信号长度与原始信号不匹配也返回.
        同时计算并包含其他列在各电池组内的众数作为静态特征代表值.

        Args:
            df (pd.DataFrame): 输入数据，包含电池老化数据 (长格式).
                               应包含 'battery_id', 'cycle_idx', target_column 参数指定的列,
                               以及其他静态特征列.
            target_column (str): 需要去噪的目标列名.

        Returns:
            Dict[str, Dict[str, Any]]: 字典，键为 battery_id.
                                       值为另一个字典，包含 'original_cycle_idx',
                                       'original_signal', 'denoised_signal' (可能为 None 或长度不匹配),
                                       以及 'other_features_mode' (包含其他列的众数).

        Raises:
            ValueError: 如果输入数据为空.
            KeyError: 如果输入 DataFrame 缺少必要列.
            RuntimeError: 如果初始化处理或分组失败.
        """
        logger.info("开始按 battery_id 获取去噪结果字典 (允许长度不匹配)...")

        if df is None or df.empty:
            logger.error("获取去噪结果字典失败: 输入数据为空.")
            raise ValueError("获取去噪结果字典失败: 输入数据为空.")

        required_cols = ["battery_id", "cycle_idx", target_column]
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            logger.error("获取去噪结果字典失败: 输入 DataFrame 缺少必要列 %s.", missing_cols)
            raise KeyError(f"获取去噪结果字典失败: 输入 DataFrame 缺少必要列 {missing_cols}.")

        try:
            df_processed = df.copy()
            df_processed["battery_id"] = df_processed["battery_id"].astype(str)
            df_processed = df_processed[
                (df_processed["battery_id"] != "nan") & (df_processed["battery_id"] != "")
            ].copy()
        except Exception as e:
            logger.error("获取去噪结果字典失败: 处理 'battery_id' 错误: %s", e, exc_info=True)
            raise RuntimeError(f"获取去噪结果字典失败: 处理 'battery_id' 错误: {e}") from e

        if df_processed.empty:
            logger.warning("过滤无效 battery_id 后数据为空.")
            return {}

        grouped = df_processed.groupby("battery_id")
        group_keys = list(grouped.groups.keys())

        if not group_keys:
            logger.warning("输入数据中没有有效的电池 ID.")
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        # 除 battery_id, cycle_idx, target_column 外的其他列被视为静态特征
        other_cols = [col for col in df_processed.columns if col not in required_cols]

        for bid in group_keys:
            logger.debug("处理电池获取去噪结果字典: %s", bid)
            df_group = grouped.get_group(bid)

            original_cycle_idx = df_group["cycle_idx"].values
            original_signal = df_group[target_column].values
            original_length = len(original_signal)

            # 计算其他列在组内的众数
            other_features_mode: Dict[str, Any] = {}
            for col in other_cols:
                try:
                    # 使用 mode() 获取众数, 可能返回多个或空 Series
                    mode_val = df_group[col].mode()
                    if not mode_val.empty:
                        # 如果有众数, 取第一个作为代表
                        other_features_mode[col] = mode_val.iloc[0]
                    else:
                        # 如果众数为空 (例如所有值都是 NaN), 设置为 None
                        other_features_mode[col] = None
                except Exception as e:
                    logger.warning("计算电池 %s 列 '%s' 的众数失败: %s. 设置为 None.", bid, col, e)
                    other_features_mode[col] = None

            # 调用内部处理逻辑获取原始去噪结果
            denoised_signal = self._process_signal_wavelets(original_signal)

            results[bid] = {
                "original_battery_id": bid,
                "original_cycle_idx": original_cycle_idx,
                "original_signal": original_signal,
                "denoised_signal": denoised_signal,  # 可能为 None 或长度与原始信号不同
                "other_features_mode": other_features_mode,
                "target_column_name": target_column,
                "original_length": original_length,
                "denoised_length": (len(denoised_signal) if denoised_signal is not None else 0),
            }

            # 记录去噪结果长度与原始长度不匹配或处理失败的情况
            if denoised_signal is not None and len(denoised_signal) != original_length:
                logger.warning(
                    "电池 %s 去噪结果长度 (%d) 与原始长度 (%d) 不匹配. 结果存储在字典中.",
                    bid,
                    len(denoised_signal),
                    original_length,
                )
            elif denoised_signal is None:
                logger.warning("电池 %s 的去噪处理失败. 结果在字典中为 None.", bid)
            elif np.array_equal(denoised_signal, original_signal) and original_length > 0:
                # 如果信号非空且去噪结果与原始相同，记录一下
                logger.debug("电池 %s 去噪结果与原始信号相同 (可能噪声极小或处理无效).", bid)
            elif original_length == 0:
                logger.debug("电池 %s 原始信号长度为零.", bid)

        logger.info("按 battery_id 获取去噪结果字典完成. 处理了 %d 个电池.", len(results))
        return results

    def get_denoised_long_dataframe(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        按 battery_id 进行小波去噪，返回一个新的长格式 DataFrame, 行数基于去噪结果长度.

        新的 DataFrame 包含 battery_id, 基于去噪结果长度的 cycle_idx (或原始长度填充 NaN),
        去噪后的 target 列 (或 NaN), 以及其他特征的众数.
        去噪成功的电池行数等于去噪结果长度. 去噪失败的电池行数等于原始长度并填充 NaN.

        Args:
            df (pd.DataFrame): 输入数据，包含电池老化数据 (长格式).
                               应包含 'battery_id', 'cycle_idx', target_column 参数指定的列,
                               以及其他特征列.
            target_column (str): 需要去噪的目标列名.

        Returns:
            pd.DataFrame: 新的长格式 DataFrame. 行数可能与原始 DataFrame 不同.

        Raises:
            ValueError: 如果输入数据为空.
            KeyError: 如果输入 DataFrame 缺少必要列.
            RuntimeError: 如果初始化处理或分组失败.
        """
        logger.info("开始创建基于去噪结果长度的新长格式 DataFrame...")

        if df is None or df.empty:
            logger.error("创建新长格式 DataFrame 失败: 输入数据为空.")
            raise ValueError("创建新长格式 DataFrame 失败: 输入数据为空.")

        required_cols = ["battery_id", "cycle_idx", target_column]
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            logger.error("创建新长格式 DataFrame 失败: 输入 DataFrame 缺少必要列 %s.", missing_cols)
            raise KeyError(
                f"创建新长格式 DataFrame 失败: 输入 DataFrame 缺少必要列 {missing_cols}."
            )

        try:
            df_processed = df.copy()
            df_processed["battery_id"] = df_processed["battery_id"].astype(str)
            df_processed = df_processed[
                (df_processed["battery_id"] != "nan") & (df_processed["battery_id"] != "")
            ].copy()
        except Exception as e:
            logger.error(
                "创建新长格式 DataFrame 失败: 处理 'battery_id' 错误: %s", e, exc_info=True
            )
            raise RuntimeError(f"创建新长格式 DataFrame 失败: 处理 'battery_id' 错误: {e}") from e

        if df_processed.empty:
            logger.warning("过滤无效 battery_id 后数据为空.")
            # 为空 DataFrame 定义列，包括 battery_id, cycle_idx, target_column 和其他列
            columns = ["battery_id", "cycle_idx", target_column] + [
                col for col in df.columns if col not in required_cols
            ]
            return pd.DataFrame(columns=columns)

        grouped = df_processed.groupby("battery_id")
        group_keys = list(grouped.groups.keys())

        if not group_keys:
            logger.warning("输入数据中没有有效的电池 ID.")
            # 为空 DataFrame 定义列，包括 battery_id, cycle_idx, target_column 和其他列
            columns = ["battery_id", "cycle_idx", target_column] + [
                col for col in df.columns if col not in required_cols
            ]
            return pd.DataFrame(columns=columns)

        output_rows: List[Dict[str, Any]] = []
        # 除 battery_id, cycle_idx, target_column 外的其他列被视为静态特征
        other_cols = [col for col in df_processed.columns if col not in required_cols]
        # 获取所有可能的静态特征列列表，用于最终确定列顺序
        all_static_feature_cols = sorted(list(set(other_cols)))

        for bid in group_keys:
            logger.debug("处理电池创建新长格式行: %s", bid)
            df_group = grouped.get_group(bid)
            original_signal = df_group[target_column].values
            original_length = len(original_signal)
            original_cycle_idx_values = df_group["cycle_idx"].values  # 保留原始 cycle index

            # 计算其他列在组内的众数
            other_features_mode: Dict[str, Any] = {}
            for col in other_cols:
                try:
                    # 使用 mode() 获取众数, 可能返回多个或空 Series
                    mode_val = df_group[col].mode()
                    if not mode_val.empty:
                        # 如果有众数, 取第一个作为代表
                        other_features_mode[col] = mode_val.iloc[0]
                    else:
                        # 如果众数为空 (例如所有值都是 NaN), 设置为 None
                        other_features_mode[col] = None
                except Exception as e:
                    logger.warning("计算电池 %s 列 '%s' 的众数失败: %s. 设置为 None.", bid, col, e)
                    other_features_mode[col] = None

            # 调用内部处理逻辑
            denoised_signal = self._process_signal_wavelets(original_signal)

            # 确定输出 DataFrame 中此块的数据和长度
            block_length: int
            cycle_idx_values_for_block: np.ndarray
            target_values_for_block: np.ndarray

            if denoised_signal is None:
                # 去噪处理失败 (_process_signal_wavelets 返回 None).
                # 创建一个原始长度的块, 目标列填充 NaN.
                logger.warning(
                    "电池 %s 去噪处理失败 (_process_signal_wavelets 返回 None). 在新长格式中填充 NaN.",
                    bid,
                )
                block_length = original_length
                # 使用原始 cycle index 值
                cycle_idx_values_for_block = original_cycle_idx_values
                # 目标列填充 NaN
                target_values_for_block = np.full(original_length, np.nan)
            else:
                # 去噪产生了一个序列 (长度可能不同)
                block_length = len(denoised_signal)
                target_values_for_block = denoised_signal

                # 记录去噪结果长度不同或无效 (结果与原始相同) 的情况，并确定 cycle_idx
                if block_length != original_length:
                    logger.warning(
                        "电池 %s 去噪结果长度 (%d) 与原始长度 (%d) 不匹配. 在新长格式中使用去噪长度并生成新的 cycle_idx.",
                        bid,
                        block_length,
                        original_length,
                    )
                    # 如果长度改变, 生成新的 cycle_idx 从 1 到 block_length
                    cycle_idx_values_for_block = np.arange(1, block_length + 1)
                elif np.array_equal(denoised_signal, original_signal) and original_length > 0:
                    # 尝试去噪, 但结果与原始相同. 仍使用去噪结果 (即原始信号).
                    # 此时 block_length 是 original_length. 使用原始 cycle index.
                    logger.debug(
                        "电池 %s 去噪结果与原始信号相同. 在新长格式中使用该结果和原始 cycle_idx.",
                        bid,
                    )
                    cycle_idx_values_for_block = original_cycle_idx_values
                else:
                    # 去噪成功且长度匹配且结果不同. 使用原始 cycle index.
                    logger.debug(
                        "电池 %s 去噪成功且长度匹配并有效. 在新长格式中使用该结果和原始 cycle_idx.",
                        bid,
                    )
                    cycle_idx_values_for_block = original_cycle_idx_values

            # 为该电池块生成行
            if block_length > 0:
                # 确保 cycle_idx_values_for_block 长度与 target_values_for_block 长度一致
                # 理论上根据上面的逻辑，这两者长度应该已经相等，但为了健壮性可以再次检查或切片
                effective_length = min(
                    block_length, len(cycle_idx_values_for_block), len(target_values_for_block)
                )
                if effective_length < block_length:
                    logger.warning(
                        "电池 %s 的 cycle_idx 或 target 数组长度与 block_length 不匹配. 使用最小长度 %d.",
                        bid,
                        effective_length,
                    )

                for i in range(effective_length):
                    row_dict: Dict[str, Any] = {
                        "battery_id": bid,
                        "cycle_idx": cycle_idx_values_for_block[i],  # 使用确定的 cycle_idx
                        target_column: target_values_for_block[i],  # 使用确定的 target 值
                    }
                    # 添加静态特征 (该块中所有行都相同)
                    row_dict.update(other_features_mode)
                    output_rows.append(row_dict)
            else:
                logger.warning(
                    "电池 %s 处理后序列长度为零 (%d). 不在新长格式中包含任何行.", bid, block_length
                )

        logger.info("创建基于去噪结果长度的新长格式 DataFrame 完成. 总行数: %d", len(output_rows))
        # 将行字典列表转换为最终的 DataFrame
        if output_rows:
            new_long_df = pd.DataFrame(output_rows)
            # 确保列顺序合理, 例如 id, cycle, target, 然后是其他列
            # 获取输出行中存在的所有列 (以第一行为代表)
            all_output_cols = list(output_rows[0].keys()) if output_rows else []
            # 定义期望的列顺序前缀
            desired_order_prefix = ["battery_id", "cycle_idx", target_column]
            # 获取不在前缀中的剩余列
            remaining_cols = [col for col in all_output_cols if col not in desired_order_prefix]
            # 对剩余列按字母顺序排序以保持一致性
            remaining_cols.sort()
            # 合并得到最终的期望列顺序
            final_column_order = desired_order_prefix + remaining_cols
            # 重新索引 DataFrame 以强制执行列顺序
            new_long_df = new_long_df.reindex(columns=final_column_order)
        else:
            # 如果没有生成行, 返回一个具有定义列的空 DataFrame
            columns = ["battery_id", "cycle_idx", target_column] + [
                col for col in df.columns if col not in required_cols
            ]
            new_long_df = pd.DataFrame(columns=columns)

        return new_long_df
