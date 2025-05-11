import pandas as pd
import numpy as np

class DataImputer:
    """处理时间序列缺失值的类，专注于线性插值和前向填充，并确保 cycle_idx 连续。"""

    def __init__(self, target_column, max_ffill_limit=3):
        self.target_column = target_column
        self.max_ffill_limit = max_ffill_limit

    def impute_missing_values(self, df):
        """
        按 battery_id 处理缺失值，并确保 cycle_idx 从 1 开始连续。

        Args:
            df (pd.DataFrame): 输入数据，包含电池老化数据。

        Returns:
            pd.DataFrame: 插值后的数据，cycle_idx 从 1 开始连续。
        """
        df_imputed = pd.DataFrame()

        for bid in df['battery_id'].unique():
            df_bid = df[df['battery_id'] == bid].copy()

            # 确保按 cycle_idx 排序
            df_bid = df_bid.sort_values('cycle_idx')

            # 生成完整的 cycle_idx 序列（从 1 到最大值，步长为 1）
            max_idx = df_bid['cycle_idx'].max()
            full_idx = pd.DataFrame({'cycle_idx': np.arange(1, int(max_idx) + 1)})

            # 将原始数据与完整 cycle_idx 合并
            df_bid = full_idx.merge(df_bid, on='cycle_idx', how='left')

            # 填充 battery_id
            df_bid['battery_id'] = bid

            # 填充其他列的缺失值（除了 target_column）
            for col in df_bid.columns:
                if col != self.target_column and col != 'cycle_idx' and col != 'battery_id':
                    df_bid[col] = df_bid[col].ffill().bfill()  # 用前后值填充

            # 线性插值处理 target 列的缺失值（包括开头和中间）
            df_bid[self.target_column] = df_bid[self.target_column].interpolate(method='linear', limit_direction='both')

            # 如果序列开头仍有缺失值，使用后向填充
            df_bid[self.target_column] = df_bid[self.target_column].bfill()

            df_imputed = pd.concat([df_imputed, df_bid], axis=0)

        return df_imputed
