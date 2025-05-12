import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class DataCleaner:
    """
    负责对电池老化数据进行清洗，包括初步过滤和异常检测.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 DataCleaner.

        Args:
            config (Dict[str, Any]): 数据清洗相关的配置字典.
        """
        self.config = config.get("data_cleaner", {})
        self.features_for_isolation_forest = self.config.get(
            "features_for_isolation_forest", ["target", "cycle_idx"]
        )
        self.contamination = self.config.get("contamination", 0.05)
        self.random_state = self.config.get("random_state", 42)
        self.anomaly_column_name = self.config.get("anomaly_column_name", "anomaly")

        # 从配置中获取初步清洗的阈值
        self.initial_target_threshold = self.config.get("initial_target_threshold", 22)
        self.initial_cycle_idx_threshold = self.config.get("initial_cycle_idx_threshold", None)

        self.scaler = StandardScaler()
        self.isolation_forest_model: IsolationForest | None = None  # 初始化为 None

        logging.info("DataCleaner 初始化完成")

    def perform_initial_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行基于阈值的初步数据过滤.

        Args:
            df (pd.DataFrame): 输入数据.

        Returns:
            pd.DataFrame: 初步过滤后的数据.
        """
        logging.info("开始执行初步数据过滤...")
        initial_rows = len(df)
        cleaned_df = df.copy()

        # 检查必要列是否存在
        required_cols = ["battery_id", "cycle_idx", "target"]
        if not all(col in cleaned_df.columns for col in required_cols):
            logging.error(
                f"初步清洗失败: 输入 DataFrame 缺少必要列 {required_cols}. 返回原始 DataFrame."
            )
            return df  # 如果缺少关键列，不进行初步清洗

        # 移除 target 或 cycle_idx 为 NaN 的行
        cleaned_df = cleaned_df.dropna(subset=["target", "cycle_idx"]).copy()
        dropped_nan_rows = initial_rows - len(cleaned_df)
        if dropped_nan_rows > 0:
            logging.info(
                f"初步清洗: 移除 target 或 cycle_idx 为 NaN 的行, 减少了 {dropped_nan_rows} 行."
            )
            initial_rows = len(cleaned_df)  # 更新基准行数

        # 应用 target 阈值过滤
        if self.initial_target_threshold is not None:
            cleaned_df = cleaned_df[cleaned_df["target"] > self.initial_target_threshold].copy()
            dropped_target_rows = initial_rows - len(cleaned_df)
            if dropped_target_rows > 0:
                logging.info(
                    f"初步清洗: 移除 target <= {
                        self.initial_target_threshold} 的行, 减少了 {dropped_target_rows} 行."
                )
                initial_rows = len(cleaned_df)  # 更新基准行数

        # 应用 cycle_idx 阈值过滤
        if self.initial_cycle_idx_threshold is not None:
            cleaned_df = cleaned_df[
                cleaned_df["cycle_idx"] > self.initial_cycle_idx_threshold
            ].copy()
            dropped_cycle_rows = initial_rows - len(cleaned_df)
            if dropped_cycle_rows > 0:
                logging.info(
                    f"初步清洗: 移除 cycle_idx <= {
                        self.initial_cycle_idx_threshold} 的行, 减少了 {dropped_cycle_rows} 行."
                )

        logging.info(f"初步数据过滤完成, 剩余数据形状: {cleaned_df.shape}.")
        return cleaned_df

    def detect_anomalies_with_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按 battery_id 使用孤立森林检测异常点.

        Args:
            df (pd.DataFrame): 输入数据，包含电池老化数据. 应已包含用于检测的特征列.

        Returns:
            pd.DataFrame: 包含原始数据和异常标记的 DataFrame (新增 anomaly_column_name 列: 1 正常, -1 异常).
                          对于无法进行异常检测的电池 (如数据点太少), 标记为正常 (1).
        """
        logging.info("开始使用孤立森林检测异常点...")
        df_with_anomaly = df.copy()
        df_with_anomaly[self.anomaly_column_name] = 1  # 默认标记为正常

        # 检查用于异常检测的特征列是否存在
        if not all(col in df_with_anomaly.columns for col in self.features_for_isolation_forest):
            missing_cols = [
                col
                for col in self.features_for_isolation_forest
                if col not in df_with_anomaly.columns
            ]
            logging.error(
                f"孤立森林异常检测失败: 输入 DataFrame 缺少特征列 {missing_cols}. 跳过异常检测."
            )
            return df_with_anomaly  # 返回默认标记为正常的 DataFrame

        for bid in df_with_anomaly["battery_id"].unique():
            # 过滤出当前电池的数据，并确保按 cycle_idx 排序
            df_bid_mask = df_with_anomaly["battery_id"] == bid
            df_bid = df_with_anomaly.loc[df_bid_mask, self.features_for_isolation_forest].copy()
            df_bid = df_bid.sort_values("cycle_idx")

            if len(df_bid) < 2:
                logging.warning(
                    f"电池 {bid} 数据点少于 2 个 ({len(df_bid)}), 无法进行孤立森林检测. 标记为正常."
                )
                # 这些电池已经默认标记为正常 (1)
                continue

            try:
                X = df_bid[self.features_for_isolation_forest]
                X_scaled = self.scaler.fit_transform(X)  # 注意: scaler 在每个电池上重新 fit

                self.isolation_forest_model = IsolationForest(
                    contamination=self.contamination,
                    random_state=self.random_state,
                    # 如果 contamination 设置为 'auto', 当数据点很少时可能出现问题.
                    # 明确指定一个小数可以避免这个问题.
                    # 但需要注意，对于非常小的数据集，固定 contamination 可能不合适。
                    # 这里的处理是对小数据集直接跳过。
                )
                # 使用 fit_predict 直接获取预测结果 (-1 for outliers, 1 for inliers)
                anomaly_predictions = self.isolation_forest_model.fit_predict(X_scaled)

                # 将预测结果应用回原始 DataFrame
                df_with_anomaly.loc[df_bid_mask, self.anomaly_column_name] = anomaly_predictions

                logging.debug(
                    f"电池 {bid} 孤立森林检测完成. 检测到异常点数量: {np.sum(anomaly_predictions == -1)}"
                )

            except Exception as e:
                logging.error(
                    f"对电池 {bid} 进行孤立森林异常检测时发生错误: {e}. 该电池数据将标记为正常."
                )
                # 发生错误时，该电池数据保持默认标记 (1)

        logging.info("孤立森林异常检测完成.")
        return df_with_anomaly

    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        执行完整的数据清洗流程: 先初步过滤，再进行孤立森林异常检测.

        Args:
            df (pd.DataFrame): 输入原始数据.

        Returns:
            tuple: (df_cleaned, df_anomalies)
                - df_cleaned: 清洗后的数据 (移除异常点).
                - df_anomalies: 检测到的异常点数据.
        """
        # Step 1: 初步过滤
        df_filtered = self.perform_initial_cleaning(df)

        # Step 2: 异常检测
        # 传入过滤后的数据进行异常检测
        df_with_anomaly = self.detect_anomalies_with_isolation_forest(df_filtered)

        # Step 3: 分离正常数据和异常数据
        # 使用配置中的异常列名
        df_cleaned = (
            df_with_anomaly[df_with_anomaly[self.anomaly_column_name] == 1]
            .drop(columns=[self.anomaly_column_name])
            .copy()
        )
        df_anomalies = (
            df_with_anomaly[df_with_anomaly[self.anomaly_column_name] == -1]
            .drop(columns=[self.anomaly_column_name])
            .copy()
        )

        logging.info(
            f"数据清洗流程完成. 清洗后数据形状: {df_cleaned.shape}, 检测到异常点数量: {len(df_anomalies)}."
        )

        return df_cleaned, df_anomalies
