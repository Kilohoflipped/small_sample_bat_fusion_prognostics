import pandas as pd
import numpy as np
from sktime.transformations.series.vmd import VmdTransformer

class DataDecomposer:
    """使用变分模态分解（VMD）对时间序列数据进行趋势分离的类，专注于电池老化数据。"""

    def __init__(self, target_column, K=5, alpha=1000, tau=0, init=1, tol=1e-7, trend_modes=2):
        """
        初始化 DataDecomposer。

        Args:
            target_column (str): 需要分解的目标列名（如 'target_denoised'）。
            K (int): 分解模态数，默认为 5。
            alpha (float): 带宽约束参数，控制模态分离，默认为 1000。
            tau (float): 噪声容忍度，默认为 0。
            init (int): 初始化方法，1 为均匀初始化，默认为 1。
            tol (float): 收敛容忍度，默认为 1e-7。
            trend_modes (int): 用作趋势的模态数量，默认为 2。
        """
        self.target_column = target_column
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.init = init
        self.tol = tol
        self.trend_modes = trend_modes

    def _decompose_signal(self, signal):
        """
        对单个时间序列应用 VMD 分解，提取趋势。

        Args:
            signal (np.ndarray): 输入时间序列。

        Returns:
            tuple: (trend, residual)
                - trend: 低频趋势部分（前 trend_modes 个模态之和）。
                - residual: 高频残差部分。
        """
        if len(signal) < self.K:
            return signal, np.zeros_like(signal)

        # 转换为 pandas Series
        series = pd.Series(signal)

        # 初始化 VmdTransformer
        transformer = VmdTransformer(
            K=self.K,
            alpha=self.alpha,
            tau=self.tau,
            init=self.init,
            tol=self.tol,
            returned_decomp="u"
        )

        # 执行分解
        modes = transformer.fit_transform(series)

        # 确保趋势模态数不超过实际模态数
        trend_modes = min(self.trend_modes, modes.shape[1])

        # 提取趋势：前 trend_modes 个模态之和
        trend = modes.iloc[:, :trend_modes].sum(axis=1).values

        # 计算残差
        residual = signal - trend

        return trend, residual

    def decompose_data(self, df):
        """
        按 battery_id 对数据进行趋势分解，生成新列 'target_trend' 和 'target_residual'。

        Args:
            df (pd.DataFrame): 输入数据，包含电池老化数据。

        Returns:
            pd.DataFrame: 包含趋势和残差的 DataFrame，新增 'target_trend' 和 'target_residual' 列。
        """
        df_decomposed = df.copy()

        for bid in df['battery_id'].unique():
            mask = df_decomposed['battery_id'] == bid
            signal = df_decomposed.loc[mask, self.target_column].values
            trend, residual = self._decompose_signal(signal)
            df_decomposed.loc[mask, 'target_trend'] = trend
            df_decomposed.loc[mask, 'target_residual'] = residual

        return df_decomposed