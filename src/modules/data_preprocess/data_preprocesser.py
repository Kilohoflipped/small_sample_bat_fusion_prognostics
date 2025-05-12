import pandas as pd
import logging
import os
from typing import Dict, Any, Tuple

from .data_converter import DataConverter
from .data_cleaner import DataCleaner
from .data_imputer import DataImputer
from .data_denoiser import DataDenoiser
from .data_decomposer import DataDecomposer

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    封装所有数据预处理步骤的类.
    协调数据加载、清洗、插值、去噪和分解等操作.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 DataPreprocessor，并根据配置初始化各个处理模块.

        Args:
            config (Dict[str, Any]): 包含所有预处理步骤配置的字典.
        """
        self.config = config
        # 初始化各个具体处理模块，将对应的配置传递给它们
        self.data_converter = DataConverter(config.get("data_converter", {}))
        self.data_cleaner = DataCleaner(config.get("data_cleaner", {}))
        self.data_imputer = DataImputer(config.get("data_imputer", {}))
        self.data_denoiser = DataDenoiser(config.get("data_denoiser", {}))
        self.data_decomposer = DataDecomposer(config.get("data_decomposer", {}))

        logger.info("DataPreprocessor 初始化完成，已加载各处理模块.")

    def load_data(self, raw_data_path: str) -> pd.DataFrame:
        """
        加载原始数据文件并转换为标准长格式

        Args:
            raw_data_path (str): 原始文件路径

        Returns:
            pd.DataFrame: 转换为长格式并初步整理后的 DataFrame
                          如果加载或转换失败, 返回空 DataFrame
        """
        logger.info("DataPreprocessor: 开始数据加载和转换...")
        converted_df = pd.DataFrame()

        if raw_data_path and os.path.exists(raw_data_path):
            converted_df = self.data_converter.load_and_convert(raw_data_path)
        else:
            logger.error("DataPreprocessor: 未找到配置的数据输入文件.")

        if converted_df.empty:
            logger.error("DataPreprocessor: 数据加载和转换后为空.")

        logger.info("DataPreprocessor: 数据加载和转换流程完成.")
        return converted_df

    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        执行数据清洗步骤 (初步过滤 + 异常检测).

        Args:
            df (pd.DataFrame): 输入原始数据.

        Returns:
            tuple: (df_cleaned, df_anomalies)
                - df_cleaned: 清洗后的数据 (移除异常点).
                - df_anomalies: 检测到的异常点数据.
        """
        logger.info("DataPreprocessor: 开始数据清洗流程...")
        if df is None or df.empty:
            logger.warning("DataPreprocessor: 输入数据为空, 跳过清洗. 返回空 DataFrame.")
            return pd.DataFrame(), pd.DataFrame()

        # 调用 DataCleaner 的 clean_data 方法
        df_cleaned, df_anomalies = self.data_cleaner.clean_data(df.copy())  # 传入副本进行清洗

        logger.info(
            f"DataPreprocessor: 数据清洗完成. 清洗后数据形状: {
                df_cleaned.shape}, 检测到异常点数量: {
                len(df_anomalies)}."
        )
        return df_cleaned, df_anomalies

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行缺失值处理步骤

        Args:
            df (pd.DataFrame): 输入数据 (通常是清洗后的数据).

        Returns:
            pd.DataFrame: 插值和填充后的数据.
        """
        logger.info("DataPreprocessor: 开始缺失值处理流程...")
        if df is None or df.empty:
            logger.warning("DataPreprocessor: 输入数据为空, 跳过缺失值处理. 返回空 DataFrame.")
            return pd.DataFrame()

        # 调用 DataImputer 的 impute_missing_values 方法
        df_imputed = self.data_imputer.impute_missing_values(df.copy())  # 传入副本

        logger.info(f"DataPreprocessor: 缺失值处理完成. 处理后数据形状: {df_imputed.shape}.")
        return df_imputed

    def denoise_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行数据去噪步骤.

        Args:
            df (pd.DataFrame): 输入数据 (通常是插值后的数据).

        Returns:
            pd.DataFrame: 包含去噪后数据的 DataFrame.
        """
        logger.info("DataPreprocessor: 开始数据去噪流程...")
        if df is None or df.empty:
            logger.warning("DataPreprocessor: 输入数据为空, 跳过数据去噪. 返回空 DataFrame.")
            return pd.DataFrame()

        # 调用 DataDenoiser 的 denoise_data 方法
        df_denoised = self.data_denoiser.denoise_data(df.copy())  # 传入副本

        logger.info(f"DataPreprocessor: 数据去噪完成. 去噪后数据形状: {df_denoised.shape}.")
        return df_denoised

    def decompose_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行趋势分解步骤.

        Args:
            df (pd.DataFrame): 输入数据 (通常是去噪后的数据).

        Returns:
            pd.DataFrame: 包含趋势、残差和所有模态的 DataFrame.
        """
        logger.info("DataPreprocessor: 开始趋势分解流程...")
        if df is None or df.empty:
            logger.warning("DataPreprocessor: 输入数据为空, 跳过趋势分解. 返回空 DataFrame.")
            return pd.DataFrame()

        # 调用 DataDecomposer 的 decompose_data 方法
        df_decomposed = self.data_decomposer.decompose_data(df.copy())  # 传入副本

        logger.info(f"DataPreprocessor: 趋势分解完成. 分解后数据形状: {df_decomposed.shape}.")
        return df_decomposed
