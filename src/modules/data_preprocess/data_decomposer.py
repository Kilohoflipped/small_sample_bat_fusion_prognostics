import pandas as pd
import numpy as np
from sktime.transformations.series.vmd import VmdTransformer
import logging
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)


class DataDecomposer:
    """
    使用变分模态分解（VMD）对时间序列数据进行趋势分离的类，专注于电池老化数据.
    可以根据配置返回趋势、残差和所有模态数据.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 DataDecomposer.

        Args:
            config (Dict[str, Any]): 趋势分解相关的配置字典.
        """
        self.config = config.get('data_decomposer', {})
        self.target_column = self.config.get('target_column', 'target_denoised')
        self.K = self.config.get('K', 5)
        self.alpha = self.config.get('alpha', 1000)
        self.tau = self.config.get('tau', 0)
        self.init = self.config.get('init', 1)
        self.tol = self.config.get('tol', 1e-7)
        self.trend_modes = self.config.get('trend_modes', 2)
        self.trend_column_name = self.config.get('trend_column_name', f'{self.target_column}_trend')
        self.residual_column_name = self.config.get(
            'residual_column_name', f'{self.target_column}_residual')
        self.mode_column_prefix = self.config.get(
            'mode_column_prefix', f'{self.target_column}_mode_')

        logger.info("DataDecomposer 初始化完成")

    def _decompose_signal(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        对单个时间序列应用 VMD 分解，提取趋势并返回所有模态.

        Args:
            signal (np.ndarray): 输入时间序列.

        Returns:
            tuple: (trend, residual, modes_df)
                - trend: 低频趋势部分（前 trend_modes 个模态之和）.
                - residual: 高频残差部分.
                - modes_df: 包含所有分解模态的 DataFrame，列名为 'mode_column_prefix_i'.
                如果分解失败或信号太短，返回原始信号作为趋势，零残差，空 DataFrame.
        """
        # VMD 需要信号长度大于等于模态数 K
        if len(signal) < self.K:
            logger.warning(f"信号长度 ({len(signal)}) 小于模态数 K ({self.K}), 无法进行 VMD 分解. 返回原始信号作为趋势.")
            # 如果信号太短，无法分解，返回原始信号作为趋势，残差为零，模态为空
            return signal, np.zeros_like(signal), pd.DataFrame()

        # 如果趋势模态数大于等于总模态数 K，则趋势就是原始信号
        if self.trend_modes >= self.K:
            logger.warning(f"趋势模态数 ({self.trend_modes}) 大于等于总模态数 K ({self.K}). 趋势将是原始信号.")
            return signal, np.zeros_like(signal), pd.DataFrame()  # 此时没有残差和独立的模态成分需要返回

        # 确保趋势模态数至少为 1
        if self.trend_modes < 1:
            logger.warning(f"趋势模态数 ({self.trend_modes}) 小于 1. 将使用第一个模态作为趋势.")
            trend_modes = 1
        else:
            trend_modes = self.trend_modes

        try:
            # 转换为 pandas Series (sktime VMDTransformer 需要 Series)
            series = pd.Series(signal)

            # 初始化 VmdTransformer
            transformer = VmdTransformer(
                K=self.K,
                alpha=self.alpha,
                tau=self.tau,
                init=self.init,
                tol=self.tol,
                returned_decomp="u"  # 返回所有模态
            )

            # 执行分解
            modes = transformer.fit_transform(series)  # modes 是一个 DataFrame, 每列是一个模态

            # 确保分解出的模态数量与 K 一致
            if modes.shape[1] != self.K:
                logger.warning(f"VMD 分解出的模态数量 ({modes.shape[1]}) 与设定的 K ({self.K}) 不一致.")
                # 继续处理，但使用实际分解出的模态数量

            # 提取趋势：前 trend_modes 个模态之和
            # 确保 trend_modes 不超过实际分解出的模态数量
            actual_trend_modes = min(trend_modes, modes.shape[1])
            trend = modes.iloc[:, :actual_trend_modes].sum(axis=1).values

            # 计算残差
            # 残差 = 原始信号 - 趋势
            residual = signal - trend

            # 重命名模态列
            modes_df = modes.copy()
            modes_df.columns = [f'{self.mode_column_prefix}{i}' for i in range(modes_df.shape[1])]

            return trend, residual, modes_df

        except Exception as e:
            logger.error(f"对信号进行 VMD 分解时发生错误: {e}. 返回原始信号作为趋势.")
            # 发生错误时，返回原始信号作为趋势，零残差，空 DataFrame
            return signal, np.zeros_like(signal), pd.DataFrame()

    def decompose_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按 battery_id 对数据进行趋势分解，生成新列 (趋势, 残差, 模态).

        Args:
            df (pd.DataFrame): 输入数据，包含电池老化数据. 应包含 target_column.

        Returns:
            pd.DataFrame: 包含趋势、残差和所有模态的 DataFrame.
                          如果输入数据为空或处理失败，返回原始 DataFrame.
        """
        logger.info(f"开始对列 '{self.target_column}' 进行趋势分解...")
        if df is None or df.empty:
            logger.warning("输入数据为空, 跳过趋势分解. 返回空 DataFrame.")
            return pd.DataFrame()

        # 检查目标列是否存在
        if self.target_column not in df.columns:
            logger.error(f"趋势分解失败: 输入 DataFrame 缺少目标列 '{self.target_column}'. 返回原始 DataFrame.")
            return df

        df_decomposed = df.copy()
        all_modes_list: List[pd.DataFrame] = []  # 用于收集所有电池的模态数据

        # 初始化趋势和残差列，对于无法分解的序列，将原始值作为趋势，残差为零
        df_decomposed[self.trend_column_name] = df_decomposed[self.target_column]
        df_decomposed[self.residual_column_name] = 0.0  # 初始化残差为 0

        for bid in df_decomposed['battery_id'].unique():
            mask = df_decomposed['battery_id'] == bid
            signal = df_decomposed.loc[mask, self.target_column].values

            # 执行分解，获取趋势、残差和模态
            trend, residual, modes_df = self._decompose_signal(signal)

            # 将趋势和残差添加到 df_decomposed
            # 确保长度匹配
            if len(trend) == len(signal) and len(residual) == len(signal):
                df_decomposed.loc[mask, self.trend_column_name] = trend
                df_decomposed.loc[mask, self.residual_column_name] = residual
            else:
                logger.warning(f"电池 {bid} 分解结果长度不匹配, 未应用趋势和残差.")

            # 将模态数据与原始数据对齐并添加到列表中
            if not modes_df.empty:
                # 确保模态数据的索引与原始数据对齐
                modes_df['battery_id'] = bid
                modes_df['cycle_idx'] = df_decomposed.loc[mask,
                                                          'cycle_idx'].values  # 确保 cycle_idx 对齐
                all_modes_list.append(modes_df)
            elif len(signal) >= self.K:  # 如果信号长度足够但 modes_df 为空，可能是分解出错
                logger.warning(f"电池 {bid} 信号长度足够 ({len(signal)} >= {self.K}) 但未生成模态数据.")

        # 将所有电池的模态数据合并回主 DataFrame
        if all_modes_list:
            try:
                all_modes_df_combined = pd.concat(all_modes_list, ignore_index=True)
                # 合并时使用 left merge 确保所有原始行都在
                df_decomposed = pd.merge(
                    df_decomposed, all_modes_df_combined, on=[
                        'battery_id', 'cycle_idx'], how='left')
                logger.info(f"所有电池模态数据合并完成, 合并后形状: {df_decomposed.shape}.")
            except Exception as e:
                logger.error(f"合并所有电池模态数据时发生错误: {e}. 模态数据未添加到主 DataFrame.")
        else:
            logger.warning("没有生成任何模态数据，跳过合并.")

        logger.info("趋势分解完成.")
        return df_decomposed
