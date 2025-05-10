"""
电池老化数据预处理模块

包含数据解析、特征工程、格式转换等功能
"""

import pandas as pd
import re


class BatteryDataPreProcessor:
    """电池老化周期数据处理器

    属性:
        raw_path (str): 原始数据文件路径
        processed_path (str): 处理结果保存路径
        static_features (list): 静态特征列表
        processed_df (DataFrame): 处理后的数据集
    """

    def __init__(self, raw_path: str, processed_path: str):
        """初始化处理器

        Args:
            raw_path: 原始数据文件路径，支持.xlsx格式
            processed_path: 处理结果保存路径，应为.csv格式
        """
        self.raw_path = raw_path
        self.save_path = processed_path
        self.static_features = [
            'battery_id', 'charge_rate', 'discharge_rate',
            'temperature', 'pressure', 'dod'
        ]
        self.processed_df = None

    @staticmethod
    def _parse_column_name(column_name: str) -> dict:
        """解析列名特征，使用正则表达式提取参数

        Args:
            column_name: 原始数据列名，格式示例："1#-0.5C/1.0C-25°-100pa-30%dod"

        Returns:
            包含解析后特征的字典

        Raises:
            ValueError: 当列名格式不匹配时抛出
        """
        pattern = r"(\d+)#-([\d.]+C)/([\d.]+C)-([\d.]+)°.*?-(\d+)pa-(\d+)%dod"
        match = re.match(pattern, column_name)
        if not match:
            raise ValueError(f"无效的列名格式: {column_name}")

        return {
            "battery_id": match.group(1),
            "charge_rate": match.group(2),
            "discharge_rate": match.group(3),
            "temperature": float(match.group(4)) + 273.15,  # 转换为开尔文温度
            "pressure": float(match.group(5)),
            "dod": float(match.group(6)),
            "variable": column_name
        }

    def _create_static_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """创建静态特征DataFrame

        Args:
            raw_df: 原始数据集

        Returns:
            包含静态特征的DataFrame
        """
        static_data = [self._parse_column_name(col) for col in raw_df.columns[1:]]
        return pd.DataFrame(static_data)

    def process(self) -> None:
        """执行完整的数据处理流程"""
        # 读取原始数据
        raw_df = pd.read_excel(self.raw_path)

        # 创建静态特征
        static_df = self._create_static_features(raw_df)

        # 转换长格式
        long_df = pd.melt(
            raw_df,
            id_vars=["循环序号"],
            value_vars=raw_df.columns[1:],
            var_name="variable",
            value_name="target"
        )

        # 合并数据
        merged_df = long_df.merge(static_df, on="variable")

        # 数据清洗
        cleaned_df = (
            merged_df
            .drop(columns=["variable"])
            .dropna(subset=['target'])
            .rename(columns={"循环序号": "cycle_idx"})
            .astype({"battery_id": str})
        )

        self.processed_df = cleaned_df

    def save(self) -> None:
        """保存处理结果"""
        if self.processed_df is not None:
            self.processed_df.to_csv(self.save_path, index=False, encoding="GBK")
        else:
            raise Exception("未找到处理后的数据，请先执行process()方法")