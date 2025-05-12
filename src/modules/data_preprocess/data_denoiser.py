import pandas as pd
import numpy as np
import pywt
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DataDenoiser:
    """
    使用小波变换对时间序列数据进行去噪的类，专注于电池老化数据.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 DataDenoiser.

        Args:
            config (Dict[str, Any]): 数据去噪相关的配置字典.
        """
        self.config = config.get('data_denoiser', {})
        self.target_column = self.config.get('target_column', 'target')
        self.denoised_column_name = self.config.get(
            'denoised_column_name', f'{self.target_column}_denoised')
        self.wavelet = self.config.get('wavelet', 'db4')
        self.threshold_mode = self.config.get('threshold_mode', 'soft')

        logger.info("DataDenoiser 初始化完成")

    def _denoise_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        对单个时间序列应用小波去噪.

        Args:
            signal (np.ndarray): 输入时间序列.

        Returns:
            np.ndarray: 去噪后的时间序列. 如果信号太短无法去噪，返回原始信号.
        """
        # 检查信号长度
        if len(signal) < 2:
            logger.warning("信号长度小于 2, 无法进行小波去噪. 返回原始信号.")
            return signal

        try:
            # 计算最大分解层次
            max_level = pywt.dwt_max_level(len(signal), self.wavelet)
            if max_level == 0:
                logger.warning(f"信号长度 {len(signal)} 对于小波 '{self.wavelet}' 太短, 无法分解. 返回原始信号.")
                return signal  # 信号太短无法分解

            # 分解信号
            coeffs = pywt.wavedec(signal, self.wavelet, level=max_level)

            # 估计噪声标准差（基于最细层次的细节系数）
            # 确保细节系数存在且非空
            if not coeffs or len(coeffs) < 2 or len(coeffs[-1]) == 0:
                logger.warning("小波分解结果异常或细节系数为空, 无法估计噪声标准差. 返回原始信号.")
                return signal

            sigma = np.median(np.abs(coeffs[-1])) / \
                0.6745 if np.median(np.abs(coeffs[-1])) > 0 else 0
            if sigma == 0:
                logger.warning("估计的噪声标准差为零, 跳过阈值处理. 返回原始信号.")
                return signal

            # 计算通用阈值
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))

            # 对细节系数应用阈值
            # 从 level 1 开始 (coeffs[1:])
            coeffs[1:] = [pywt.threshold(
                c, threshold, mode=self.threshold_mode) for c in coeffs[1:]]

            # 重构信号
            denoised = pywt.waverec(coeffs, self.wavelet)

            # 确保输出长度与输入一致（边界效应可能导致长度略有差异）
            if len(denoised) != len(signal):
                logger.warning(f"小波重构信号长度 ({len(denoised)}) 与原始信号长度 ({len(signal)}) 不一致. 进行调整.")
                if len(denoised) > len(signal):
                    denoised = denoised[:len(signal)]
                elif len(denoised) < len(signal):
                    denoised = np.pad(
                        denoised, (0, len(signal) - len(denoised)), mode='edge')

            return denoised

        except Exception as e:
            logger.error(f"对信号进行小波去噪时发生错误: {e}. 返回原始信号.")
            return signal

    def denoise_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按 battery_id 对数据进行去噪，生成去噪后的新列.

        Args:
            df (pd.DataFrame): 输入数据，包含电池老化数据. 应包含 target_column.

        Returns:
            pd.DataFrame: 包含去噪后数据的 DataFrame，新增 denoised_column_name 列.
                          如果输入数据为空或处理失败，返回原始 DataFrame.
        """
        logger.info(f"开始对列 '{self.target_column}' 进行数据去噪...")
        if df is None or df.empty:
            logger.warning("输入数据为空, 跳过数据去噪. 返回空 DataFrame.")
            return pd.DataFrame()

        # 检查目标列是否存在
        if self.target_column not in df.columns:
            logger.error(f"数据去噪失败: 输入 DataFrame 缺少目标列 '{self.target_column}'. 返回原始 DataFrame.")
            return df

        df_denoised = df.copy()
        # 初始化去噪列，对于无法去噪的序列（如太短），保持原始值
        df_denoised[self.denoised_column_name] = df_denoised[self.target_column]

        # 按 battery_id 分组去噪
        for bid in df_denoised['battery_id'].unique():
            mask = df_denoised['battery_id'] == bid
            signal = df_denoised.loc[mask, self.target_column].values

            # 执行去噪
            denoised_signal = self._denoise_signal(signal)

            # 将去噪结果应用回 DataFrame
            # 确保长度匹配，尽管 _denoise_signal 内部已处理，这里再检查一次更安全
            if len(denoised_signal) == len(signal):
                df_denoised.loc[mask, self.denoised_column_name] = denoised_signal
            else:
                logger.warning(
                    f"电池 {bid} 去噪后信号长度不匹配 ({
                        len(denoised_signal)} vs {
                        len(signal)}), 未应用去噪结果.")

        logger.info(f"数据去噪完成. 新增列: '{self.denoised_column_name}'.")
        return df_denoised
