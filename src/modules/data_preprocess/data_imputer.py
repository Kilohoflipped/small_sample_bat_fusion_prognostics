import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class DataImputer:
    """
    处理时间序列缺失值的类，专注于插值和填充，并确保 cycle_idx 连续.
    可以根据配置选择插值方法和填充策略.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 DataImputer.

        Args:
            config (Dict[str, Any]): 缺失值处理相关的配置字典.
        """
        self.config = config.get('data_imputer', {})
        self.target_column = self.config.get('target_column', 'target')
        self.max_ffill_limit = self.config.get('max_ffill_limit', None)  # None 表示无限制
        self.interpolation_method = self.config.get('interpolation_method', 'linear')
        self.other_column_fill_method = self.config.get(
            'other_column_fill_method', 'ffill_bfill')  # 例如 'mode', 'mean'

        logger.info("DataImputer 初始化完成")

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按 battery_id 处理缺失值，并确保 cycle_idx 从 1 开始连续.
        对 target_column 使用插值，对其他列使用指定填充方法.

        Args:
            df (pd.DataFrame): 输入数据，包含电池老化数据.

        Returns:
            pd.DataFrame: 插值和填充后的数据，cycle_idx 从 1 开始连续.
                          如果输入数据为空或处理失败，返回空 DataFrame.
        """
        logger.info("开始处理缺失值...")
        if df is None or df.empty:
            logger.warning("输入数据为空, 跳过缺失值处理. 返回空 DataFrame.")
            return pd.DataFrame()

        # 检查必要列是否存在
        required_cols = ['battery_id', 'cycle_idx', self.target_column]
        if not all(col in df.columns for col in required_cols):
            logger.error(f"缺失值处理失败: 输入 DataFrame 缺少必要列 {required_cols}. 返回原始 DataFrame.")
            return df  # 如果缺少关键列，不进行处理

        df_imputed = pd.DataFrame()

        for bid in df['battery_id'].unique():
            df_bid = df[df['battery_id'] == bid].copy()

            # 确保按 cycle_idx 排序
            df_bid = df_bid.sort_values('cycle_idx')

            # 生成完整的 cycle_idx 序列（从 1 到最大值，步长为 1）
            # 确保 max_idx 是整数
            max_idx = int(df_bid['cycle_idx'].max()) if not df_bid['cycle_idx'].empty else 0
            if max_idx == 0:
                logger.warning(f"电池 {bid} cycle_idx 最大值为 0 或为空, 跳过缺失值填充.")
                df_imputed = pd.concat([df_imputed, df_bid], axis=0)
                continue

            full_idx = pd.DataFrame({'cycle_idx': np.arange(1, max_idx + 1)})

            # 将原始数据与完整 cycle_idx 合并
            # 使用 left merge 保留所有完整的 cycle_idx
            df_bid = full_idx.merge(df_bid, on='cycle_idx', how='left')

            # 填充 battery_id (确保新引入的行也有 battery_id)
            df_bid['battery_id'] = bid

            # 填充其他列的缺失值（除了 target_column 和 cycle_idx）
            other_cols = [
                col for col in df_bid.columns if col not in [
                    self.target_column,
                    'cycle_idx',
                    'battery_id']]
            for col in other_cols:
                if self.other_column_fill_method == 'ffill_bfill':
                    df_bid[col] = df_bid[col].ffill(
                        limit=self.max_ffill_limit).bfill(
                        limit=self.max_ffill_limit)
                elif self.other_column_fill_method == 'mode':
                    # 计算众数，如果存在多个众数，取第一个
                    mode_val = df_bid[col].mode()
                    if not mode_val.empty:
                        df_bid[col] = df_bid[col].fillna(mode_val.iloc[0])
                elif self.other_column_fill_method == 'mean':
                    df_bid[col] = df_bid[col].fillna(df_bid[col].mean())
                # 可以根据需要添加其他填充方法

            # 插值处理 target 列的缺失值
            try:
                df_bid[self.target_column] = df_bid[self.target_column].interpolate(
                    method=self.interpolation_method,
                    limit_direction='both',  # 双向插值
                    limit=self.max_ffill_limit  # 插值限制，与 ffill_limit 共用参数
                )
                # 插值后可能仍然存在开头的 NaN (如果 limit 限制了)
                # 使用 bfill 填充开头剩余的 NaN
                df_bid[self.target_column] = df_bid[self.target_column].bfill(
                    limit=self.max_ffill_limit)
                # 使用 ffill 填充结尾剩余的 NaN
                df_bid[self.target_column] = df_bid[self.target_column].ffill(
                    limit=self.max_ffill_limit)

            except ValueError as e:
                logger.error(
                    f"对电池 {bid} 进行 '{
                        self.interpolation_method}' 插值时发生错误: {e}. 跳过该电池的 target 插值.")
                # 发生错误时，该电池的 target 列将保留 NaN

            # 记录插值后剩余的缺失值数量
            remaining_nan = df_bid[self.target_column].isna().sum()
            if remaining_nan > 0:
                logger.warning(f"电池 {bid} 插值和填充后，'{self.target_column}' 列仍有 {remaining_nan} 个缺失值.")

            df_imputed = pd.concat([df_imputed, df_bid], axis=0)

        logger.info(f"缺失值处理完成, 最终数据形状: {df_imputed.shape}.")
        return df_imputed
