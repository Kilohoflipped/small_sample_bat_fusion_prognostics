import logging
import os
import re
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BatteryStaticFeatures:
    """
    电池老化数据静态特征的数据类.
    """

    battery_id: Optional[str] = None
    charge_rate: Optional[float] = np.nan
    discharge_rate: Optional[float] = np.nan
    temperature: Optional[float] = np.nan
    pressure: Optional[float] = np.nan
    dod: Optional[float] = np.nan

    raw_column_name: Optional[str] = None  # 保留原始列名以供参考.

    @classmethod
    def get_static_feature_names(cls) -> list[str]:
        """获取静态特征的字段名称列表."""
        # 排除 raw_column_name 字段, 因为它主要用于内部合并.
        return [field.name for field in fields(cls) if field.name != "raw_column_name"]


class DataConverter:
    """
    负责将原始电池老化数据从特定格式转换为标准长格式 DataFrame.
    并从列名中解析静态特征.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据转换器.

        Args:
            config (Dict[str, Any]): 数据转换相关的配置字典.
        """
        self.config = config.get("data_converter", {})
        self.cycle_col_name = self.config.get("cycle_index_column", "循环序号")
        self.target_col_name_after_convert = self.config.get("target_column", "target")
        # 正则表达式用于从列名中解析静态特征.
        # 示例: "1#-0.5C/1.0C-25°-100pa-30%dod"
        self.column_parse_pattern = self.config.get(
            "column_parse_pattern",
            r"(\d+)#-([\d.]*C)/([\d.]*C)-(-?[\d.]+)°-(-?[\d.]+)pa-(-?[\d.]+)%dod",
        )

        self._compiled_pattern = re.compile(self.column_parse_pattern)

        logger.info("DataConverter 初始化完成")

    def _parse_column_name(self, raw_column_name: str) -> BatteryStaticFeatures:
        """
        解析原始数据列名, 提取电池静态参数.

        Args:
            raw_column_name: 原始数据列名.

        Returns:
            BatteryStaticFeatures 实例, 包含解析后特征.
            如果解析失败, 对应的字段值为 None 或 np.nan.
        """
        features = BatteryStaticFeatures(raw_column_name=raw_column_name)

        if self._compiled_pattern is None:
            logger.error("正则表达式编译失败，无法解析列名.")
            raise ValueError("正则表达式编译失败")

        match = self._compiled_pattern.match(raw_column_name)

        if not match:
            logger.error("列名未匹配到预期的格式: '%s'", raw_column_name)
            raise ValueError(f"列名 '{raw_column_name}' 未匹配到预期的格式.")

        try:
            # 尝试从匹配结果中提取并赋值.
            battery_id_str = match.group(1)
            if battery_id_str:
                features.battery_id = str(battery_id_str)
            else:
                logger.warning(f"列名 '{raw_column_name}': 匹配成功但未提取到 battery_id.")

            charge_rate_str = match.group(2)
            if charge_rate_str:
                try:
                    features.charge_rate = float(charge_rate_str.strip("C"))
                except ValueError:
                    logger.warning(
                        f"列名 '{raw_column_name}': 充率 '{charge_rate_str}' 转换为浮点数失败."
                    )

            discharge_rate_str = match.group(3)
            if discharge_rate_str:
                try:
                    features.discharge_rate = float(discharge_rate_str.strip("C"))
                except ValueError:
                    logger.warning(
                        f"列名 '{raw_column_name}': 放率 '{discharge_rate_str}' 转换为浮点数失败."
                    )

            temperature_celsius_str = match.group(4)
            if temperature_celsius_str:
                try:
                    features.temperature = float(temperature_celsius_str.strip("°"))
                except ValueError:
                    logger.warning(
                        f"列名 '{raw_column_name}': 温度 '{temperature_celsius_str}' 转换为浮点数失败."
                    )

            pressure_str = match.group(5)
            if pressure_str:
                try:
                    features.pressure = float(pressure_str.strip("pa"))
                except ValueError:
                    logger.warning(
                        f"列名 '{raw_column_name}': 压力 '{pressure_str}' 转换为浮点数失败."
                    )

            dod_str = match.group(6)
            if dod_str:
                try:
                    features.dod = float(dod_str.strip("%dod"))
                except ValueError:
                    logger.warning(f"列名 '{raw_column_name}': DOD '{dod_str}' 转换为浮点数失败.")

            if features.battery_id is None:
                logger.warning(f"列名 '{raw_column_name}': 成功匹配但未能提取有效的 battery_id.")

            return features

        except (ValueError, IndexError) as e:
            logger.warning(
                "解析列名 '%s' 时发生错误: %s. 返回部分解析结果.", raw_column_name, e, exc_info=True
            )
            return features
        except Exception as e:
            logger.error(f"解析列名 '{raw_column_name}' 时发生意外错误: {e}.", exc_info=True)
            return features

    def _extract_static_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        从原始数据集中提取并创建静态特征DataFrame.

        Args:
            raw_df: 原始数据集. 假设第一列是循环序号, 后续列名包含静态特征信息.

        Returns:
            包含静态特征的 DataFrame. 如果无法提取任何静态特征, 返回包含预期列名的空 DataFrame.
        """
        logger.info("开始提取静态特征...")

        if (
            raw_df is None
            or raw_df.empty
            or raw_df.shape[1] <= 1
            or self.cycle_col_name not in raw_df.columns
        ):
            logger.error(
                "原始数据无效、列数不足或找不到循环序号列 '%s', 无法提取静态特征.",
                self.cycle_col_name,
            )
            raise ValueError(
                f"原始数据无效、列数不足或找不到循环序号列 '{self.cycle_col_name}', 无法提取静态特征."
            )

        columns_to_parse = raw_df.columns.drop(self.cycle_col_name)
        cleaned_columns_to_parse = [col.strip() for col in columns_to_parse]
        static_features_list = [self._parse_column_name(col) for col in cleaned_columns_to_parse]
        static_df = pd.DataFrame(static_features_list)

        if static_df.empty:
            logger.warning("未能从列名中提取到任何静态特征.")

        logger.info("静态特征提取完成, 形状: %s.", static_df.shape)
        return static_df

    def _validate_input_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame | None:
        """
        验证输入原始 DataFrame 的基本有效性.

        Args:
            raw_df: 待验证的原始 DataFrame.

        Returns:
            如果 DataFrame 有效则返回 DataFrame 本身, 否则返回 None.
        """
        if raw_df is None or raw_df.empty:
            logger.error("输入原始数据为空, 无法进行转换.")
            raise ValueError("输入原始数据为空, 无法进行转换.")

        if self.cycle_col_name not in raw_df.columns:
            logger.error(
                "原始数据中找不到循环序号列 '%s', 无法进行 melt 操作.",
                self.cycle_col_name,
            )
            raise ValueError(
                f"原始数据中找不到循环序号列 '{self.cycle_col_name}', 无法进行 melt 操作.",
            )

        value_vars = raw_df.columns.drop(self.cycle_col_name)
        if value_vars.empty:
            logger.error("原始数据除循环序号列外没有其他列, 无法进行 melt 操作.")
            raise ValueError("原始数据除循环序号列外没有其他列, 无法进行 melt 操作.")

        return raw_df

    def _melt_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        将原始宽格式数据转换为长格式.

        Args:
            raw_df: 验证通过的原始宽格式数据集.

        Returns:
            转换为长格式的 DataFrame. 如果 melt 失败返回空 DataFrame.
        """
        try:
            value_vars = raw_df.columns.drop(self.cycle_col_name)
            melted_df = pd.melt(
                raw_df,
                id_vars=[self.cycle_col_name],
                value_vars=value_vars,
                var_name="raw_column_name",
                value_name=self.target_col_name_after_convert,
            )
            logger.info("数据 melt 完成, 形状: %s.", melted_df.shape)
            return melted_df
        except Exception as e:
            logger.error(f"数据 melt 过程中发生错误: {e}", exc_info=True)
            return pd.DataFrame()

    @staticmethod
    def _merge_static_features(melted_df: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
        """
        将 melt 后的 DataFrame 与静态特征 DataFrame 合并.

        Args:
            melted_df: melt 后的 DataFrame.
            static_df: 静态特征 DataFrame.

        Returns:
            合并后的 DataFrame. 如果合并失败返回原始 melted_df 副本.
        """
        if static_df is None or static_df.empty:
            logger.warning("静态特征 DataFrame 为空或 None, 跳过合并静态特征.")
            return melted_df.copy()

        if "raw_column_name" not in melted_df.columns or "raw_column_name" not in static_df.columns:
            logger.error("用于合并的 'raw_column_name' 列不存在于其中一个 DataFrame 中. 跳过合并.")
            return melted_df.copy()

        try:
            # 使用 left merge 确保保留所有 melt 后的行.
            merged_df = melted_df.merge(static_df, on="raw_column_name", how="left")
            logger.info("静态特征合并完成, 形状: %s.", merged_df.shape)
            return merged_df
        except Exception as e:
            logger.error(f"合并静态特征时发生错误: {e}. 返回原始 melted_df 副本.", exc_info=True)
            return melted_df.copy()

    def _clean_and_finalize_dataframe(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        对合并后的 DataFrame 进行最终的清理和整理.

        包括移除临时列, 重命名列, 过滤无效行.

        Args:
            merged_df: 合并后的 DataFrame.

        Returns:
            清理和整理后的 DataFrame. 如果清理失败返回原始 merged_df 副本.
        """
        if merged_df is None or merged_df.empty:
            logger.warning("输入合并后的 DataFrame 为空或 None, 跳过最终清理.")
            return pd.DataFrame()

        cleaned_df = merged_df.copy()

        try:
            # 移除用于合并的临时列.
            if "raw_column_name" in cleaned_df.columns:
                cleaned_df = cleaned_df.drop(columns=["raw_column_name"])

            # 重命名循环序号列.
            if self.cycle_col_name in cleaned_df.columns:
                cleaned_df = cleaned_df.rename(columns={self.cycle_col_name: "cycle_idx"})
            else:
                logger.warning(
                    "原始循环序号列 '%s' 不存在于处理后的数据中, 无法重命名为 'cycle_idx'.",
                    self.cycle_col_name,
                )

            # 确保 battery_id 是字符串类型, 并处理可能的无效值.
            if "battery_id" in cleaned_df.columns:
                initial_rows = len(cleaned_df)
                cleaned_df["battery_id"] = cleaned_df["battery_id"].astype(str)
                # 移除 battery_id 是 'nan' 或空字符串的行.
                cleaned_df = cleaned_df[
                    (cleaned_df["battery_id"] != "nan") & (cleaned_df["battery_id"] != "")
                ].copy()
                if len(cleaned_df) < initial_rows:
                    logger.warning(
                        "已移除 battery_id 为 'nan' 或空字符串的行, 减少了 %d 行.",
                        initial_rows - len(cleaned_df),
                    )
            else:
                logger.error("'battery_id' 列不存在于处理后的数据中，无法进行基于电池的后续处理.")

            logger.info("数据清理和最终整理完成, 最终形状: %s.", cleaned_df.shape)
            return cleaned_df

        except Exception as e:
            logger.error(
                f"最终数据清理和整理过程中发生错误: {e}. 返回原始 merged_df 副本.", exc_info=True
            )
            return merged_df.copy()

    def convert_to_long_format(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        将原始宽格式数据转换为长格式, 并合并静态特征.

        Args:
            raw_df: 原始宽格式数据集.

        Returns:
            转换为长格式并合并静态特征后的 DataFrame.
            如果输入数据为空或转换失败, 返回空 DataFrame.
        """
        logger.info("开始数据格式转换和特征合并...")

        validated_df = self._validate_input_dataframe(raw_df)
        if validated_df is None:
            raise ValueError("数据有效性出现问题.")

        static_df = self._extract_static_features(validated_df)
        melted_df = self._melt_dataframe(validated_df)

        if melted_df is None or melted_df.empty:
            logger.error("数据 melt 失败或结果为空.")
            return pd.DataFrame()

        merged_df = self._merge_static_features(melted_df, static_df)
        finalized_df = self._clean_and_finalize_dataframe(merged_df)

        logger.info("数据格式转换和特征合并流程完成.")
        return finalized_df

    def load_and_convert(self, raw_data_path: str) -> pd.DataFrame:
        """
        加载原始数据文件 (.xlsx 或 .csv) 并执行格式转换和初步整理.

        Args:
            raw_data_path: 原始数据文件路径 (.xlsx 或 .csv).

        Returns:
            转换为长格式并初步整理后的 DataFrame.
            如果加载或转换失败, 返回空 DataFrame.
        """
        logger.info("开始加载和转换原始数据文件: %s...", raw_data_path)
        if not os.path.exists(raw_data_path):
            logger.error("原始数据文件不存在: %s.", raw_data_path)
            raise FileNotFoundError(f"原始数据文件不存在: {raw_data_path}")

        if raw_data_path.lower().endswith(".xlsx"):
            raw_df = pd.read_excel(raw_data_path)
        elif raw_data_path.lower().endswith(".csv"):
            raw_df = pd.read_csv(raw_data_path)
        else:
            logger.error("不支持的文件格式: %s. 只支持 .xlsx 和 .csv.", raw_data_path)
            raise ValueError(f"不支持的文件格式: {raw_data_path}")

        logger.info("原始数据加载成功, 形状: %s.", raw_df.shape)

        processed_df = self.convert_to_long_format(raw_df)

        logger.info("原始数据加载和转换流程完成.")
        return processed_df
