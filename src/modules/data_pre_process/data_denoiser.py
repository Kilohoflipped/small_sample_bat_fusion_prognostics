import pandas as pd
import numpy as np
import pywt


class DataDenoiser:
    """使用小波变换对时间序列数据进行去噪的类，专注于电池老化数据。"""

    def __init__(self, target_column, wavelet='db4', threshold_mode='soft'):
        """
        初始化 DataDenoiser。

        Args:
            target_column (str): 需要去噪的目标列名（如 'target'）。
            wavelet (str): 小波类型，默认为 'db4'（Daubechies 4）。
            threshold_mode (str): 阈值处理方式，'soft' 或 'hard'，默认为 'soft'。
        """
        self.target_column = target_column
        self.wavelet = wavelet
        self.threshold_mode = threshold_mode

    def _denoise_signal(self, signal):
        """
        对单个时间序列应用小波去噪。

        Args:
            signal (np.ndarray): 输入时间序列。

        Returns:
            np.ndarray: 去噪后的时间序列。
        """
        # 检查信号长度
        if len(signal) < 2:
            return signal

        # 计算最大分解层次
        max_level = pywt.dwt_max_level(len(signal), self.wavelet)
        if max_level == 0:
            return signal  # 信号太短无法分解

        # 分解信号
        coeffs = pywt.wavedec(signal, self.wavelet, level=max_level)

        # 估计噪声标准差（基于最细层次的细节系数）
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        # 计算通用阈值
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))

        # 对细节系数应用阈值
        coeffs[1:] = [pywt.threshold(
            c, threshold, mode=self.threshold_mode) for c in coeffs[1:]]

        # 重构信号
        denoised = pywt.waverec(coeffs, self.wavelet)

        # 确保输出长度与输入一致（边界效应可能导致长度略有差异）
        if len(denoised) > len(signal):
            denoised = denoised[:len(signal)]
        elif len(denoised) < len(signal):
            denoised = np.pad(
                denoised, (0, len(signal) - len(denoised)), mode='edge')

        return denoised

    def denoise_data(self, df):
        """
        按 battery_id 对数据进行去噪，生成新列 'target_denoised'。

        Args:
            df (pd.DataFrame): 输入数据，包含电池老化数据。

        Returns:
            pd.DataFrame: 包含去噪后数据的 DataFrame，新增 'target_denoised' 列。
        """
        df_denoised = df.copy()

        # 按 battery_id 分组去噪
        for bid in df['battery_id'].unique():
            mask = df_denoised['battery_id'] == bid
            signal = df_denoised.loc[mask, self.target_column].values
            denoised_signal = self._denoise_signal(signal)
            df_denoised.loc[mask, 'target_denoised'] = denoised_signal

        return df_denoised
