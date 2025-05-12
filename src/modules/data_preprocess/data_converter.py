import pandas as pd
import re
import os
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

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

    raw_column_name: Optional[str] = None  # 保留原始列名以供参考


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
        self.config = config.get('data_converter', {})
        self.cycle_col_name = self.config.get('cycle_index_column', '循环序号')
        # 可以从配置中获取更灵活的解析 pattern, 如果没有则使用默认
        self.column_parse_pattern = self.config.get(
            'column_parse_pattern',
            r"(\d+)#-([\d.]*C)/([\d.]*C)-(-?[\d.]+)°-(-?[\d.]+)pa-(-?[\d.]+)%dod"
        )
        self._compiled_pattern = re.compile(self.column_parse_pattern)

        logger.info("DataConverter 初始化完成")

    def _parse_column_name(self, raw_column_name: str) -> BatteryStaticFeatures:
        """
        解析原始数据列名, 提取电池静态参数.

        Args:
            raw_column_name: 原始数据列名, 格式示例: "1#-0.5C/1.0C-25°-100pa-30%dod"

        Returns:
            BatteryStaticFeatures 实例, 包含解析后特征.
            如果解析失败, 对应的字段值为 None 或 np.nan.
        """
        # 初始化 BatteryStaticFeatures 实例, 默认值为 None 或 np.nan
        features = BatteryStaticFeatures(raw_column_name=raw_column_name)

        match = self._compiled_pattern.match(raw_column_name)

        if not match:
            logger.warning(f"无效的列名格式, 跳过解析: '{raw_column_name}'")
            return features  # 返回带有原始列名和默认值的实例

        try:
            # 尝试从匹配结果中提取并赋值给 features 实例
            # 使用 .strip('C') 移除充放电倍率后的 'C'
            features.battery_id = str(match.group(1)) if match.group(1) else None
            features.charge_rate = float(match.group(2).strip('C')) if match.group(2) else np.nan
            features.discharge_rate = float(match.group(3).strip('C')) if match.group(3) else np.nan

            temperature_celsius_str = match.group(4)
            if temperature_celsius_str:
                # 注意: 温度单位转换应该在后续的数据处理步骤中进行, 这里只解析数值
                features.temperature = float(temperature_celsius_str)
            else:
                features.temperature = np.nan

            pressure_str = match.group(5)
            if pressure_str:
                features.pressure = float(pressure_str)
            else:
                features.pressure = np.nan

            dod_str = match.group(6)
            if dod_str:
                features.dod = float(dod_str)
            else:
                features.dod = np.nan

            return features

        except ValueError as e:
            logger.warning(f"解析列名 '{raw_column_name}' 中的数值时发生错误: {e}. 返回部分解析结果.")
            # 发生错误时, 返回当前已解析的部分特征和默认值
            return features
        except IndexError:
            logger.warning(f"解析列名 '{raw_column_name}' 时索引超出范围. 正则表达式可能不匹配. 返回默认值.")
            return features

    def _extract_static_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        从原始数据集中提取并创建静态特征DataFrame.

        Args:
            raw_df: 原始数据集: 假设第一列是循环序号, 后续列名包含静态特征信息.

        Returns:
            包含静态特征的 DataFrame. 如果无法提取, 返回包含预期列名的空 DataFrame.
        """
        logger.info("开始提取静态特征...")

        # 确保 raw_df 有多于一列
        if raw_df.shape[1] <= 1:
            logger.warning("原始数据列数不足, 无法提取静态特征. 返回包含预期列名的空 DataFrame.")
            # 确保返回的 DataFrame 包含所有预期的静态特征列和 raw_column_name 列
            return pd.DataFrame(columns=list(BatteryStaticFeatures.__annotations__.keys()))

        # 提取除循环序号列外的所有列名
        if self.cycle_col_name not in raw_df.columns:
            logger.error(f"原始数据中找不到循环序号列 '{self.cycle_col_name}'. 无法提取静态特征. 返回空 DataFrame.")
            return pd.DataFrame(columns=list(BatteryStaticFeatures.__annotations__.keys()))

        columns_to_parse = raw_df.columns.drop(self.cycle_col_name)

        # 解析所有需要解析的列名, 得到 BatteryStaticFeatures 实例列表
        static_features_list = [self._parse_column_name(col) for col in columns_to_parse]

        # 将 BatteryStaticFeatures 实例列表转换为 DataFrame
        static_df = pd.DataFrame(static_features_list)

        # 移除 battery_id 为 None 的行, 这些是列名解析完全失败的行
        initial_rows = len(static_df)
        static_df = static_df.dropna(subset=['battery_id']).copy()
        if len(static_df) < initial_rows:
            logger.warning(
                f"移除 battery_id 为 None 的静态特征行, 减少了 {
                    initial_rows - len(static_df)} 行.")

        logger.info(f"静态特征提取完成, 形状: {static_df.shape}.")
        return static_df

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
        if raw_df is None or raw_df.empty:
            logger.warning("输入原始数据为空, 跳过转换. 返回空 DataFrame.")
            return pd.DataFrame()

        # 提取静态特征
        static_df = self._extract_static_features(raw_df)

        # 转换长格式
        if self.cycle_col_name not in raw_df.columns:
            logger.error(f"原始数据中找不到循环序号列 '{self.cycle_col_name}', 无法进行 melt 操作. 返回空 DataFrame.")
            return pd.DataFrame()

        value_vars = raw_df.columns.drop(self.cycle_col_name)
        if value_vars.empty:
            logger.error("原始数据除循环序号列外没有其他列, 无法进行 melt 操作. 返回空 DataFrame.")
            return pd.DataFrame()

        try:
            melted_df = pd.melt(
                raw_df,
                id_vars=[self.cycle_col_name],
                value_vars=value_vars,
                var_name="raw_column_name",
                value_name="target"
            )
            logger.info(f"数据 melt 完成, 形状: {melted_df.shape}.")

        except Exception as e:
            logger.error(f"数据 melt 时发生错误: {e}. 返回空 DataFrame.")
            return pd.DataFrame()

        # 合并数据
        if not static_df.empty:
            # 确保用于合并的 'raw_column_name' 列在两个 DataFrame 中都存在且类型兼容 (通常是字符串)
            if 'raw_column_name' not in melted_df.columns or 'raw_column_name' not in static_df.columns:
                logger.error("用于合并的 'raw_column_name' 列不存在. 跳过合并.")
                merged_df = melted_df.copy()
            else:
                # 使用 left merge 确保保留所有 melt 后的行
                merged_df = melted_df.merge(
                    static_df, on="raw_column_name", how='left')
                logger.info(f"静态特征合并完成, 形状: {merged_df.shape}.")
        else:
            logger.warning("静态特征 DataFrame 为空, 跳过合并.")
            merged_df = melted_df.copy()

        # 数据整理和重命名
        # 移除用于合并的临时列
        if 'raw_column_name' in merged_df.columns:
            # 检查 'raw_column_name' 列是否被成功合并 (即没有 NaN)
            if merged_df['raw_column_name'].isna().any():
                logger.warning("合并后的数据中存在 'raw_column_name' 的 NaN 值, 可能有部分原始列名未被解析.")
            cleaned_df = merged_df.drop(columns=["raw_column_name"])
        else:
            cleaned_df = merged_df.copy()
            logger.warning("'raw_column_name' 列不存在于合并后的数据中, 无法移除.")

        # 重命名循环序号列
        if self.cycle_col_name in cleaned_df.columns:
            cleaned_df = cleaned_df.rename(
                columns={self.cycle_col_name: "cycle_idx"})
        else:
            logger.warning(
                f"原始循环序号列 '{self.cycle_col_name}' 不存在于处理后的数据中, 无法重命名为 'cycle_idx'.")

        # 确保 battery_id 是字符串类型, 并处理可能的 NaN 或 'nan' 字符串
        if 'battery_id' in cleaned_df.columns:
            # 将 battery_id 转换为字符串类型, NaN 会变成 'nan' 字符串
            cleaned_df['battery_id'] = cleaned_df['battery_id'].astype(str)
            # 移除 battery_id 是 'nan' 字符串的行 (这些是列名解析失败的行)
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df[cleaned_df['battery_id'] != 'nan'].copy()
            if len(cleaned_df) < initial_rows:
                logger.warning(
                    f"已移除 battery_id 为 'nan' 字符串的行, 减少了 {
                        initial_rows - len(cleaned_df)} 行.")
        else:
            logger.warning("'battery_id' 列不存在于处理后的数据中.")

        logger.info(f"格式转换和初步整理完成, 最终形状: {cleaned_df.shape}.")

        return cleaned_df

    def load_and_convert(self, raw_data_path: str) -> pd.DataFrame:
        """
        加载原始数据文件 (.xlsx 或 .csv) 并执行格式转换和初步整理.

        Args:
            raw_data_path: 原始数据文件路径 (.xlsx 或 .csv).

        Returns:
            转换为长格式并初步整理后的 DataFrame.
            如果加载或转换失败, 返回空 DataFrame.
        """
        logger.info(f"开始加载和转换原始数据文件: {raw_data_path}...")
        if not os.path.exists(raw_data_path):
            logger.error(f"原始数据文件不存在: {raw_data_path}.")
            return pd.DataFrame()

        try:
            # 根据文件扩展名选择加载方法
            if raw_data_path.lower().endswith('.xlsx'):
                raw_df = pd.read_excel(raw_data_path)
            elif raw_data_path.lower().endswith('.csv'):
                raw_df = pd.read_csv(raw_data_path)
            else:
                logger.error(f"不支持的文件格式: {raw_data_path}. 只支持 .xlsx 和 .csv.")
                return pd.DataFrame()

            logger.info(f"原始数据加载成功, 形状: {raw_df.shape}.")
        except Exception as e:
            logger.error(f"加载原始数据文件 '{raw_data_path}' 时发生错误: {e}.")
            return pd.DataFrame()

        if raw_df is None or raw_df.empty:
            logger.warning(f"加载的原始数据文件 '{raw_data_path}' 为空.")
            return pd.DataFrame()

        # 执行格式转换和初步整理
        processed_df = self.convert_to_long_format(raw_df)

        logger.info("原始数据加载和转换流程完成.")
        return processed_df
