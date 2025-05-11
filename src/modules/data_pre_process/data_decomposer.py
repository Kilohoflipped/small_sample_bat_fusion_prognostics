import pandas as pd
import numpy as np
from sktime.transformations.series.vmd import VmdTransformer

class DataDecomposer:
    """
    使用变分模态分解（VMD）对时间序列数据进行趋势分离的类，专注于电池老化数据。
    修改后可以返回所有模态数据。
    """

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
        对单个时间序列应用 VMD 分解，提取趋势并返回所有模态。

        Args:
            signal (np.ndarray): 输入时间序列。

        Returns:
            tuple: (trend, residual, modes_df)
                - trend: 低频趋势部分（前 trend_modes 个模态之和）。
                - residual: 高频残差部分。
                - modes_df: 包含所有分解模态的 DataFrame，列名为 'target_column_mode_i'。
        """
        if len(signal) < self.K:
            # 如果信号太短，无法分解，返回原始信号作为趋势，残差为零，模态为空
            return signal, np.zeros_like(signal), pd.DataFrame()

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
        modes = transformer.fit_transform(series) # modes 是一个 DataFrame

        # 确保趋势模态数不超过实际模态数
        trend_modes = min(self.trend_modes, modes.shape[1])

        # 提取趋势：前 trend_modes 个模态之和
        trend = modes.iloc[:, :trend_modes].sum(axis=1).values

        # 计算残差
        residual = signal - trend

        # 重命名模态列
        modes_df = modes.copy()
        modes_df.columns = [f'{self.target_column}_mode_{i}' for i in range(modes_df.shape[1])]

        return trend, residual, modes_df

    def decompose_data(self, df):
        """
        按 battery_id 对数据进行趋势分解，生成新列 'target_trend', 'target_residual'
        并包含所有分解模态。

        Args:
            df (pd.DataFrame): 输入数据，包含电池老化数据。

        Returns:
            pd.DataFrame: 包含趋势、残差和所有模态的 DataFrame。
        """
        df_decomposed = df.copy()
        all_modes_list = [] # 用于收集所有电池的模态数据

        for bid in df['battery_id'].unique():
            mask = df_decomposed['battery_id'] == bid
            signal = df_decomposed.loc[mask, self.target_column].values

            # 执行分解，获取趋势、残差和模态
            trend, residual, modes_df = self._decompose_signal(signal)

            # 将趋势和残差添加到 df_decomposed
            df_decomposed.loc[mask, f'{self.target_column}_trend'] = trend
            df_decomposed.loc[mask, f'{self.target_column}_residual'] = residual

            # 将模态数据与原始数据对齐并添加到列表中
            if not modes_df.empty:
                modes_df['battery_id'] = bid
                modes_df['cycle_idx'] = df_decomposed.loc[mask, 'cycle_idx'].values # 确保 cycle_idx 对齐
                all_modes_list.append(modes_df)

        # 将所有电池的模态数据合并回主 DataFrame
        if all_modes_list:
             all_modes_df = pd.concat(all_modes_list, ignore_index=True)
             # 合并时使用 left merge 确保所有原始行都在
             df_decomposed = pd.merge(df_decomposed, all_modes_df, on=['battery_id', 'cycle_idx'], how='left')

        return df_decomposed

