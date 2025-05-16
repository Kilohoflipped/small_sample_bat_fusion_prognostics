"""
处理时间序列缺失值的模块

本模块包含 DataImputer 类, 专注于插值和填充, 并确保 cycle_idx 连续
可以根据配置选择插值方法和填充策略
"""

import logging
from typing import Any, Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataImputer:
    """
    处理时间序列缺失值的类，专注于插值和填充，并确保 cycle_idx 连续
    可以根据配置选择插值方法和填充策略
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 DataImputer.

        Args:
            config (Dict[str, Any]): 缺失值处理相关的配置字典.
                                        期望包含 'data_imputer' 键, 其值是包含插值配置的字典.
        """
        self.config = config.get("data_imputer", {})
        self.interpolation_method = self.config.get("interpolation_method", "linear")
        logger.info("DataImputer 初始化完成")

    def impute_missing_values(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[pd.DataFrame, str]:
        """
        按 battery_id 处理目标列的缺失值，并确保 cycle_idx 从 1 开始连续
        对 target_column 使用插值，其他列填充该 battery_id 下出现次数最多的值

        Args:
            df (pd.DataFrame): 输入数据，应包含 battery_id, cycle_idx 和 target_column
            target_column (str): 需要进行插值处理的主要目标列名

        Returns:
            Tuple[pd.DataFrame, str]:
                - pd.DataFrame: 经过缺失值处理后的 DataFrame
                - str: 插值后的目标列名称 (通常与输入相同)

        Raises:
            ValueError: 如果输入数据为空.
            KeyError: 如果输入 DataFrame 缺少必要的列 ('battery_id', 'cycle_idx', target_column).
            RuntimeError: 如果处理过程中发生不可恢复的错误.
            Exception: 捕获并重新抛出其他未预期的错误.
        """
        logger.info("开始处理目标列 '%s' 的缺失值...", target_column)

        if df is None or df.empty:
            logger.error("缺失值处理失败: 输入数据为空.")
            raise ValueError("缺失值处理失败: 输入数据为空.")

        required_cols = ["battery_id", "cycle_idx", target_column]
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            logger.error("缺失值处理失败: 输入 DataFrame 缺少必要列 %s.", missing_cols)
            raise KeyError(f"缺失值处理失败: 输入 DataFrame 缺少必要列 {missing_cols}.")

        processed_df = df.copy()

        try:
            processed_df["battery_id"] = processed_df["battery_id"].astype(str)
            processed_df = processed_df[
                (processed_df["battery_id"] != "nan") & (processed_df["battery_id"] != "")
            ].copy()
            try:
                processed_df["cycle_idx"] = processed_df["cycle_idx"].astype(int)
            except ValueError as e:
                logger.error("缺失值处理失败: 'cycle_idx' 列包含非整数值无法转换: %s", e)
                raise ValueError(f"缺失值处理失败: 'cycle_idx' 列包含非整数值无法转换: {e}") from e
        except Exception as e:
            logger.error(
                "缺失值处理失败: 转换 'battery_id' 或 'cycle_idx' 类型或过滤时发生错误: %s",
                e,
                exc_info=True,
            )
            raise RuntimeError(
                f"缺失值处理失败: 转换 'battery_id' 或 'cycle_idx' 类型或过滤时发生错误: {e}"
            ) from e

        if processed_df.empty:
            logger.error("缺失值列名处理后输入数据为空")
            raise RuntimeError("缺失值处理失败: 列名处理后输入数据为空.")

        if not pd.api.types.is_numeric_dtype(processed_df[target_column]):
            logger.warning("目标列 '%s' 不是数值类型，尝试转换为数值类型以便插值.", target_column)
            try:
                processed_df[target_column] = pd.to_numeric(
                    processed_df[target_column], errors="coerce"
                )
            except Exception as e:
                logger.error("目标列 '%s' 转换为数值类型失败: %s", target_column, e, exc_info=True)
                raise RuntimeError(
                    f"缺失值处理失败: 目标列 '{target_column}' 转换为数值类型失败: {e}"
                ) from e

        final_imputed_df = pd.DataFrame()

        grouped = processed_df.groupby("battery_id")
        group_keys = list(grouped.groups.keys())

        if not group_keys:
            logger.error("输入数据中没有有效的电池 ID 可用于分组缺失值处理.")
            raise RuntimeError("缺失值处理失败: 输入数据中没有有效的电池 ID 可用于分组缺失值处理.")

        for bid in group_keys:
            try:
                group_df = grouped.get_group(bid).copy()
                max_cycle = group_df["cycle_idx"].max()
                if pd.isna(max_cycle):
                    logger.error("电池 %s 的 cycle_idx 最大值为 NaN, 无法确保周期连续性.", bid)
                    raise RuntimeError(
                        f"缺失值处理失败: 电池 {bid} 的 cycle_idx 列最大值为 NaN, 无法确保周期连续性."
                    )

                try:
                    full_cycles = pd.DataFrame({"cycle_idx": range(1, int(max_cycle) + 1)})
                    full_cycles["battery_id"] = bid
                except Exception as e:
                    logger.error("电池 %s 生成完整 cycle_idx 序列失败: %s", bid, e, exc_info=True)
                    raise RuntimeError(f"电池 {bid} 生成完整 cycle_idx 序列失败: {e}") from e

                merged_df = pd.merge(
                    full_cycles, group_df, on=["battery_id", "cycle_idx"], how="left"
                )
                merged_df = merged_df.sort_values("cycle_idx").reset_index(drop=True)

                # 对 target_column 进行插值和填充
                try:
                    merged_df[target_column] = merged_df[target_column].interpolate(
                        method=self.interpolation_method,
                        axis=0,
                        limit_direction="both",
                    )
                    merged_df[target_column] = merged_df[target_column].bfill()
                    merged_df[target_column] = merged_df[target_column].ffill()
                except ValueError as e:
                    logger.error(
                        "对电池 %s 进行 '%s' 插值时发生错误: %s.",
                        bid,
                        self.interpolation_method,
                        e,
                        exc_info=True,
                    )
                    raise RuntimeError(f"电池 {bid} target 列插值失败: {e}") from e
                except Exception as e:
                    logger.error(
                        "对电池 %s 进行 '%s' 插值时发生未预期错误: %s.",
                        bid,
                        self.interpolation_method,
                        e,
                        exc_info=True,
                    )
                    raise RuntimeError(f"电池 {bid} target 列插值失败: {e}") from e

                # 填充其他列的缺失值
                other_cols = [
                    col
                    for col in merged_df.columns
                    if col not in ["battery_id", "cycle_idx", target_column]
                ]
                for col in other_cols:
                    most_frequent_value = group_df[col].mode()
                    if not most_frequent_value.empty:
                        merged_df[col] = merged_df[col].fillna(most_frequent_value.iloc[0])
                    else:
                        logger.error("电池 %s 的列 '%s' 没有众数", bid, col)
                        raise RuntimeError(
                            f"填充电池 {bid} 的列 '{col}' 的缺失值时发生错误: 列 '{col}' 没有众数"
                        )

                remaining_nan_target = merged_df[target_column].isna().sum()
                if remaining_nan_target > 0:
                    logger.error(
                        "电池 %s 处理后目标列 '%s' 仍然存在 %d 个缺失值.",
                        bid,
                        target_column,
                        remaining_nan_target,
                    )
                    raise RuntimeError(
                        f"处理失败: 电池 {bid} 目标列 '{target_column}' 仍然存在 {remaining_nan_target} 个缺失值."
                    )

                final_imputed_df = pd.concat([final_imputed_df, merged_df], ignore_index=True)
                logger.debug("电池 %s 目标列和其余列缺失值处理完成.", bid)

            except KeyError as e:
                logger.error("处理电池 %s 时缺少关键列: %s. 中断处理.", bid, e)
                raise KeyError(f"处理电池 {bid} 时缺少关键列: {e}") from e
            except RuntimeError as e:
                logger.error("处理电池 %s 时发生运行时错误: %s. 中断处理.", bid, e)
                raise
            except Exception as e:
                logger.error("处理电池 %s 时发生未预期错误: %s. 中断处理.", bid, e, exc_info=True)
                raise

        if not final_imputed_df.empty:
            final_imputed_df = final_imputed_df.sort_values(
                by=["battery_id", "cycle_idx"]
            ).reset_index(drop=True)

        logger.info("目标列和其余列缺失值处理流程完成")
        return final_imputed_df, target_column
