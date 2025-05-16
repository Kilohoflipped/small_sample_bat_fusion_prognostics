"""
使用变分模态分解（VMD）对时间序列数据进行趋势分离的模块.

本模块包含 DataDecomposer 类, 用于对时间序列数据应用 VMD 分解，
提取趋势、残差和所有模态数据，并将其作为新列添加到 DataFrame 中.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sktime.transformations.series.vmd import VmdTransformer

logger = logging.getLogger(__name__)


@dataclass
class DataDecomposerConfig:
    """
    数据分解（VMD）的配置.

    包含 VMD 算法和输出列命名所需的参数，不包含目标列名.

    Attributes:
        K (int): VMD 分解的模态（IMF）数量. 默认值为 5.
        alpha (float): VMD 的二次惩罚因子. 默认值为 1000.
        tau (float): 数据保真约束的拉格朗日乘子参数. 默认值为 0.
        init (int): 模态中心频率的初始化方法 (1: 随机, 2: 用户指定). 默认值为 1.
        tol (float): 停止准则的容忍度. 默认值为 1e-7.
        trend_modes (int): 用于构成趋势部分的低频模态数量 (从 0 开始计数). 默认值为 2.
        trend_column_suffix (str): 趋势列名使用的后缀. 默认值为 "_trend".
        residual_column_suffix (str): 残差列名使用的后缀. 默认值为 "_residual".
        mode_column_prefix_template (str): 模态列名使用的前缀模板.
                                          格式字符串，包含一个用于插入原始列名的占位符 '{}'.
                                          默认值为 "{}_mode_".
    """

    K: int = 5
    alpha: float = 1000
    tau: float = 0
    init: int = 1
    tol: float = 1e-7
    trend_modes: int = 2
    trend_column_suffix: str = "_trend"
    residual_column_suffix: str = "_residual"
    mode_column_prefix_template: str = "{}_mode_"

    def __post_init__(self):
        # 增加参数验证
        if self.K <= 0:
            raise ValueError("K (模态数量) 必须是正整数.")
        if self.alpha < 0:
            raise ValueError("alpha (二次惩罚因子) 不能为负数.")
        if (
            not 0 <= self.init <= 2
        ):  # 假设 init 只能是 0, 1, 2 (取决于 sktime 实现，这里基于常见 VMD)
            logger.warning(f"VMD init 参数 {self.init} 可能不是标准值 (0, 1, 2).")
        if self.tol < 0:
            raise ValueError("tol (容忍度) 不能为负数.")
        if self.trend_modes < 0 or self.trend_modes > self.K:
            logger.warning(
                f"trend_modes ({self.trend_modes}) 应在 [0, K] 范围内. 将在分解时进行调整."
            )


class DataDecomposer:
    """
    处理时间序列数据趋势分解的类，专注于 VMD.

    使用 VMD 对时间序列数据进行分解，提取趋势、残差和所有模态，
    并将结果作为新列添加到 DataFrame 中.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 DataDecomposer.

        从配置字典中提取参数，并创建配置类的实例.

        Args:
            config (Dict[str, Any]): 趋势分解相关的配置字典.
                                 期望包含 'data_decomposer' 键，其值是包含分解配置的字典.
        """
        decomposer_config_dict = config.get("data_decomposer", {})

        try:
            # 使用 dataclass 解析配置，如果键不存在或类型不匹配会抛出 TypeError
            self.config: DataDecomposerConfig = DataDecomposerConfig(**decomposer_config_dict)
        except TypeError as e:
            logger.error("数据分解配置字典与 DataDecomposerConfig 定义不匹配: %s", e)
            raise ValueError(f"数据分解配置字典与 DataDecomposerConfig 定义不匹配: {e}") from e
        except ValueError as e:
            logger.error("数据分解配置参数验证失败: %s", e)
            raise ValueError(f"数据分解配置参数验证失败: {e}") from e
        except Exception as e:
            logger.error("初始化 DataDecomposer 配置时发生未预期错误: %s", e, exc_info=True)
            raise RuntimeError(f"初始化 DataDecomposer 配置时发生未预期错误: {e}") from e

        logger.info("DataDecomposer 初始化完成")
        logger.info("配置: %s", self.config)

    def _decompose_signal(
        self, signal: np.ndarray
    ) -> Tuple[np.ndarray | None, np.ndarray | None, pd.DataFrame | None]:
        """
        对单个时间序列应用 VMD 分解，提取趋势并返回所有模态.

        Args:
            signal (np.ndarray): 输入时间序列.

        Returns:
            tuple: (trend, residual, modes_df)
                - trend: 低频趋势部分（前 trend_modes 个模态之和）.
                - residual: 高频残差部分.
                - modes_df: 包含所有分解模态的 DataFrame，列为 'mode_i'.
                如果分解失败或信号太短，返回 (None, None, None).
        """
        if signal is None or len(signal) == 0:
            logger.warning("信号无效或长度为零, 无法进行 VMD 分解.")
            return None, None, None

        signal_len = len(signal)

        # VMD 需要信号长度大于等于模态数 K
        if signal_len < self.config.K:
            logger.warning(
                f"信号长度 ({signal_len}) 小于模态数 K ({self.config.K}), 无法进行 VMD 分解."
            )
            return None, None, None  # 信号太短，无法分解

        # 确保 trend_modes 是有效的索引范围
        valid_trend_modes = max(
            0, min(self.config.trend_modes, self.config.K)
        )  # 确保 trend_modes 在 [0, K] 范围内

        if valid_trend_modes >= self.config.K:
            logger.warning(
                f"趋势模态数 ({self.config.trend_modes}) 调整为 {valid_trend_modes} >= 总模态数 K ({self.config.K}). "
                "趋势将是原始信号，残差为零."
            )
            # 返回原始信号作为趋势，残差为零，模态为空 DataFrame
            # 理论上 VMD 还是会生成 K 个模态，但按配置定义趋势>=K时，就返回原始信号作为趋势
            try:
                # 即使趋势>=K，也执行 VMD 获取模态数据，以便用户可选地使用它们
                series = pd.Series(signal)
                transformer = VmdTransformer(
                    K=self.config.K,
                    alpha=self.config.alpha,
                    tau=self.config.tau,
                    init=self.config.init,
                    tol=self.config.tol,
                    returned_decomp="u",
                )
                modes = transformer.fit_transform(series)
                # 重命名模态列为 'mode_i' 格式，方便外部处理
                if modes is not None and not modes.empty:
                    modes.columns = [f"mode_{i}" for i in range(modes.shape[1])]
                else:
                    modes = pd.DataFrame()  # 确保返回 DataFrame
                return signal, np.zeros_like(signal), modes
            except Exception as e:
                logger.error(
                    f"趋势模态数 >= K 时执行 VMD 提取模态失败: {e}. 返回 (原始信号, 零残差, None).",
                    exc_info=True,
                )
                return signal, np.zeros_like(signal), None  # VMD 失败，但趋势残差按定义返回

        try:
            # 转换为 pandas Series (sktime VMDTransformer 需要 Series)
            series = pd.Series(signal)

            # 初始化 VmdTransformer
            transformer = VmdTransformer(
                K=self.config.K,
                alpha=self.config.alpha,
                tau=self.config.tau,
                init=self.config.init,
                tol=self.config.tol,
                returned_decomp="u",  # 返回所有模态
            )

            # 执行分解
            modes = transformer.fit_transform(series)  # modes 是一个 DataFrame, 每列是一个模态

            # 确保分解出的模态数量有效
            if modes is None or modes.empty or modes.shape[1] == 0:
                logger.warning("VMD 分解未生成任何模态数据.")
                return None, None, None  # 未生成模态数据，分解失败

            # VMD 分解出的模态数量应该与 K 一致
            if modes.shape[1] != self.config.K:
                logger.warning(
                    f"VMD 分解出的模态数量 ({modes.shape[1]}) 与设定的 K ({self.config.K}) 不一致. "
                    "分解可能不成功或参数设置有问题. 返回 None, None, None."  # 此时视为分解失败
                )
                return None, None, None  # 模态数量不匹配 K，视为分解失败

            actual_modes_count = modes.shape[1]
            # trend_modes 的有效性已在上面检查并调整为 valid_trend_modes

            # 提取趋势：前 valid_trend_modes 个模态之和
            # 如果 valid_trend_modes 为 0，趋势为零向量
            if valid_trend_modes > 0:
                trend = modes.iloc[:, :valid_trend_modes].sum(axis=1).values
            else:
                trend = np.zeros_like(signal)

            # 计算残差
            # 这里定义残差为 原始信号 - 趋势
            residual = signal - trend

            # 重命名模态列为 'mode_i' 格式
            modes.columns = [f"mode_{i}" for i in range(actual_modes_count)]

            return trend, residual, modes

        except Exception as e:
            # 捕获 VMD 库可能抛出的异常
            logger.error(
                f"对信号进行 VMD 分解时发生错误: {e}. 返回 (None, None, None).", exc_info=True
            )
            # 发生错误时，返回 None
            return None, None, None

    def decompose_data(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        按 battery_id 对指定目标列的数据进行趋势分解，生成新列 (趋势, 残差, 模态).

        Args:
            df (pd.DataFrame): 输入数据，包含电池老化数据.
                               应包含 'battery_id', 'cycle_idx' 和 target_column 参数指定的列.
            target_column (str): 需要进行趋势分解的目标列名.

        Returns:
            Tuple[pd.DataFrame, List[str]]:
                - pd.DataFrame: 包含趋势、残差和所有模态的 DataFrame. 新生成的列已添加.
                                返回的 DataFrame 始终是原始 DataFrame 的副本，并已添加新列.
                - List[str]: 分解生成的所有新列名列表 (包括趋势、残差和模态).

        Raises:
            ValueError: 如果输入数据为空.
            KeyError: 如果输入 DataFrame 缺少分解所需的必要列 ('battery_id', 'cycle_idx', target_column).
            RuntimeError: 如果处理过程中发生不可恢复的错误 (例如，无法按 battery_id 分组或合并模态数据失败).
            Exception: 捕获并重新抛出其他未预期的错误.
        """
        logger.info("开始对列 '%s' 进行趋势分解...", target_column)

        if df is None or df.empty:
            logger.error("数据分解失败: 输入数据为空.")
            raise ValueError("数据分解失败: 输入数据为空.")

        # 检查必要列是否存在
        required_cols = ["battery_id", "cycle_idx", target_column]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"数据分解失败: 输入 DataFrame 缺少必要列 {missing}.")
            raise KeyError(f"数据分解失败: 输入 DataFrame 缺少必要列 {missing}.")

        # 确保 battery_id 和 cycle_idx 列的类型适合分组和合并
        try:
            # 避免修改原始df，创建一个副本
            df_decomposed = df.copy()
            df_decomposed["battery_id"] = df_decomposed["battery_id"].astype(str)
            df_decomposed["cycle_idx"] = df_decomposed["cycle_idx"].astype(
                int
            )  # VMD通常用于等间隔数据，确保cycle_idx为整数

            # 过滤掉无效的 battery_id
            df_decomposed = df_decomposed[
                (df_decomposed["battery_id"] != "nan") & (df_decomposed["battery_id"] != "")
            ].copy()

        except Exception as e:
            logger.error(
                "数据分解失败: 转换 'battery_id' 或 'cycle_idx' 类型或过滤时发生错误: %s",
                e,
                exc_info=True,
            )
            raise RuntimeError(
                f"数据分解失败: 转换 'battery_id' 或 'cycle_idx' 类型或过滤时发生错误: {e}"
            ) from e

        if df_decomposed.empty:
            logger.warning("过滤无效 battery_id 或 cycle_idx 后数据为空, 跳过趋势分解.")
            # 返回原始df（可能是空的）的副本和空列表
            return df.copy(), []

        # 根据实际处理的列名生成输出列名
        trend_col_name = f"{target_column}{self.config.trend_column_suffix}"
        residual_col_name = f"{target_column}{self.config.residual_column_suffix}"
        mode_column_prefix = self.config.mode_column_prefix_template.format(target_column)

        # 检查生成的输出列名是否与原始列名冲突 (除了目标列本身，理论上不会覆盖)
        # 如果 trend_col_name 或 residual_col_name 与 battery_id 或 cycle_idx 冲突则报错
        if trend_col_name in ["battery_id", "cycle_idx"] or residual_col_name in [
            "battery_id",
            "cycle_idx",
        ]:
            error_msg = f"生成的趋势或残差列名与必要列冲突: 趋势 '{trend_col_name}', 残差 '{residual_col_name}'."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        output_cols: List[str] = []  # 用于记录新生成的列名

        # 添加趋势和残差列名到输出列表 (模态列名稍后添加)
        output_cols.extend([trend_col_name, residual_col_name])

        # 初始化趋势和残差列 (如果不存在)
        if trend_col_name not in df_decomposed.columns:
            df_decomposed[trend_col_name] = np.nan
        if residual_col_name not in df_decomposed.columns:
            df_decomposed[residual_col_name] = np.nan

        all_modes_list: List[pd.DataFrame] = []  # 用于收集所有电池的模态数据
        processed_battery_ids = []  # 跟踪尝试处理的电池ID
        successfully_decomposed_battery_ids = []  # 跟踪成功分解的电池ID (至少趋势和残差成功)

        # 按 battery_id 分组分解
        try:
            grouped = df_decomposed.groupby("battery_id")
            group_keys = list(grouped.groups.keys())

            if not group_keys:
                # 这种情况应该在前面的 df_decomposed.empty 检查中捕获，但作为后备检查
                logger.error("数据分解失败: 输入数据中没有有效的电池 ID 可用于分组分解.")
                raise RuntimeError("数据分解失败: 输入数据中没有有效的电池 ID 可用于分组分解.")

        except Exception as e:
            logger.error("数据分解失败: 按 'battery_id' 分组时发生错误: %s", e, exc_info=True)
            raise RuntimeError(f"数据分解失败: 按 'battery_id' 分组时发生错误: {e}") from e

        for bid in group_keys:
            processed_battery_ids.append(bid)
            mask = df_decomposed["battery_id"] == bid
            signal = df_decomposed.loc[mask, target_column].values
            cycle_idx_series = df_decomposed.loc[mask, "cycle_idx"]  # 获取对应的 cycle_idx series

            if signal is None or len(signal) == 0:
                logger.warning(f"电池 {bid} 的分解目标信号为空或长度为零, 跳过分解.")
                continue

            # 执行分解，获取趋势、残差和模态 DataFrame
            # _decompose_signal 内部处理了信号长度 < K 的情况并返回 None, None, None
            trend, residual, modes_df = self._decompose_signal(signal)

            # 将趋势和残差添加到 df_decomposed
            if trend is not None and residual is not None:
                # 确保长度匹配
                if len(trend) == len(signal) and len(residual) == len(signal):
                    df_decomposed.loc[mask, trend_col_name] = trend
                    df_decomposed.loc[mask, residual_col_name] = residual
                    successfully_decomposed_battery_ids.append(bid)
                    logger.debug("电池 %s 趋势和残差分解应用成功.", bid)
                else:
                    logger.warning(
                        f"电池 {bid} 分解结果长度不匹配 ({len(trend)} vs {len(signal)}), 未应用趋势和残差."
                    )
            else:
                # _decompose_signal 失败或信号太短已在内部记录警告/错误
                logger.debug(f"电池 {bid} VMD 分解 (趋势/残差) 失败或信号太短, 未应用结果.")

            # 处理模态数据
            if modes_df is not None and not modes_df.empty:
                # modes_df 的列名已经在 _decompose_signal 中被重命名为 'mode_i' 格式
                # 现在添加基于 target_column 的前缀
                prefixed_mode_columns = [f"{mode_column_prefix}{col}" for col in modes_df.columns]
                modes_df.columns = prefixed_mode_columns

                # 确保模态数据的索引与原始数据对齐
                # 使用原始 DataFrame 的索引来确保 battery_id 和 cycle_idx 对齐
                modes_df["battery_id"] = bid
                modes_df["cycle_idx"] = cycle_idx_series.values  # 使用获取的 cycle_idx series

                all_modes_list.append(modes_df)

                # 记录模态列名 (只在第一次成功生成模态时添加，假设所有电池的模态列数量一致)
                # 使用一个集合来避免重复添加，或者更简单地，在合并后从最终 df 获取列名
                # 暂时不在这里添加到 output_cols，在合并后统一获取

            elif bid in successfully_decomposed_battery_ids:
                # 即使没有模态数据，如果趋势和残差成功，也标记为成功分解
                logger.debug(f"电池 {bid} 趋势和残差分解成功，但未生成模态数据.")

        # 将所有电池的模态数据合并回主 DataFrame
        if all_modes_list:
            try:
                # 合并所有模态 DataFrame
                all_modes_df_combined = pd.concat(all_modes_list, ignore_index=True)
                # 使用 left merge 将模态数据合并到 df_decomposed
                # 确保合并键一致且类型正确
                all_modes_df_combined["battery_id"] = all_modes_df_combined["battery_id"].astype(
                    str
                )
                all_modes_df_combined["cycle_idx"] = all_modes_df_combined["cycle_idx"].astype(int)

                # 检查合并列是否与现有列冲突 (除了 battery_id, cycle_idx)
                existing_cols_before_merge = set(df_decomposed.columns) - {
                    "battery_id",
                    "cycle_idx",
                }
                mode_cols_to_add = set(all_modes_df_combined.columns) - {"battery_id", "cycle_idx"}
                if existing_cols_before_merge.intersection(mode_cols_to_add):
                    conflict_cols = existing_cols_before_merge.intersection(mode_cols_to_add)
                    error_msg = f"合并模态数据时发现列名冲突: {list(conflict_cols)}. 请检查模态列前缀模板或原始列名."
                    logger.error(error_msg)
                    # 在实际应用中可能需要更复杂的冲突解决策略，这里选择报错
                    raise RuntimeError(error_msg)

                df_decomposed = pd.merge(
                    df_decomposed,
                    all_modes_df_combined,
                    on=["battery_id", "cycle_idx"],
                    how="left",
                    suffixes=(
                        "",
                        "_vmd_modes_temp",
                    ),  # 使用后缀避免与原始同名列冲突 (如果存在，尽管这里应该不会)
                )

                # 清理可能由于 suffixes 产生的临时列
                # 检查是否存在临时列并清理
                temp_cols = [
                    col for col in df_decomposed.columns if col.endswith("_vmd_modes_temp")
                ]
                if temp_cols:
                    logger.warning(f"合并后发现临时后缀列: {temp_cols}. 移除它们.")
                    df_decomposed = df_decomposed.drop(columns=temp_cols)

                logger.info(f"所有电池模态数据合并完成, 合并后形状: {df_decomposed.shape}.")

                # 收集实际添加到 DataFrame 中的模态列名
                # 模态列名应该以 mode_column_prefix 开头
                added_mode_cols = [
                    col for col in df_decomposed.columns if col.startswith(mode_column_prefix)
                ]
                output_cols.extend(added_mode_cols)

            except Exception as e:
                logger.error(
                    f"合并所有电池模态数据时发生错误: %s. 模态数据未添加到主 DataFrame.",
                    e,
                    exc_info=True,
                )
                # 合并失败是严重错误，应抛出异常
                raise RuntimeError(f"合并所有电池模态数据时发生错误: {e}") from e

        else:
            logger.warning("没有生成任何模态数据，跳过合并.")

        # 检查是否所有尝试处理的电池都至少进行了趋势和残差分解
        if len(successfully_decomposed_battery_ids) != len(processed_battery_ids):
            logger.warning(
                "并非所有尝试处理的电池都成功进行了趋势和残差分解. 成功分解电池数: %d/%d.",
                len(successfully_decomposed_battery_ids),
                len(processed_battery_ids),
            )
        else:
            logger.info("所有尝试处理的电池都已进行趋势和残差分解.")

        logger.info("趋势分解流程完成.")
        # 返回包含新生成列的 DataFrame 副本，以及新生成的列名列表
        return df_decomposed, output_cols
