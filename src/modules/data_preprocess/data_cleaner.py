"""
处理电池老化数据清洗的模块.

本模块包含 DataCleaner 类, 用于对电池老化数据执行初步过滤和基于孤立森林的异常检测.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class InitialCleaningConfig:
    """
    初步数据清洗的配置

    包含基于阈值过滤数据所需的参数

    Attributes:
        target_threshold (float): 目标变量的最小阈值. 小于等于此值的行将被移除
            默认值为 22
        cycle_idx_threshold (int): 循环次数的最小阈值. 小于等于此值的行将被移除
            默认值为 0
    """

    target_threshold: float = 22
    cycle_idx_threshold: int = 0


@dataclass
class IsolationForestConfig:
    """
    孤立森林异常检测的配置

    包含配置 sklearn IsolationForest 模型所需的参数

    Attributes:
        contamination (float): IsolationForest 模型的污染率参数, 表示数据中异常点的比例估计
            必须在 (0, 0.5] 范围内
        random_state (int): IsolationForest 模型的随机状态, 用于保证结果的可复现性
            默认值为 137
        anomaly_column_name (str): 用于标记异常值的输出列名称. 1 表示正常, -1 表示异常
            默认值为 "anomaly"
    """

    contamination: float = 0.025
    random_state: int = 137
    anomaly_column_name: str = "anomaly"

    def __post_init__(self):
        # 初始化后进行参数验证
        if not 0 < self.contamination <= 0.5:
            raise ValueError("contamination 参数必须在 (0, 0.5] 范围内.")


class DataCleaner:
    """
    处理电池老化数据清洗的模块.
    本模块包含 DataCleaner 类, 用于对电池老化数据执行初步过滤和基于孤立森林的异常检测.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 DataCleaner.

        从配置字典中提取参数, 并创建配置类的实例

        Args:
            config (Dict[str, Any]): 数据清洗相关的配置字典.
                                 期望包含 'data_cleaner' 键, 其值是包含清洗配置的字典
        """
        data_cleaner_config_dict = config.get("data_cleaner", {})

        # 从配置中获取初步清洗的阈值参数, 如果不存在则使用空字典
        initial_cleaning_config_dict = data_cleaner_config_dict.get("initial_cleaning_config", {})
        self.initial_cleaning_config: InitialCleaningConfig = InitialCleaningConfig(
            **initial_cleaning_config_dict
        )

        # 从配置中获取孤立森林的配置字典, 如果不存在则使用空字典
        isolation_forest_config_dict = data_cleaner_config_dict.get("isolation_forest_config", {})
        self.isolation_forest_config: IsolationForestConfig = IsolationForestConfig(
            **isolation_forest_config_dict
        )

        self.scaler = StandardScaler()
        # IsolationForest 模型将在 detect_anomalies_with_isolation_forest 方法中按需创建
        self.isolation_forest_model: IsolationForest | None = None

        logger.info("DataCleaner 初始化完成")

    def perform_initial_cleaning(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        执行基于阈值的初步数据过滤

        Args:
            df (pd.DataFrame): 输入数据
            target_column (str): 需要进行初步过滤的目标列名

        Returns:
            pd.DataFrame: 初步过滤后的数据

        Raises:
            KeyError: 如果输入 DataFrame 缺少初步过滤所需的必要列.
        """
        logger.info("开始执行初步数据过滤, 目标列: '%s'...", target_column)
        initial_rows = len(df)
        cleaned_df = df.copy()

        # 检查必要列是否存在 (使用传入的目标列名)
        required_cols = ["battery_id", "cycle_idx", target_column]
        if not all(col in cleaned_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in cleaned_df.columns]
            logger.error("初步清洗失败: 输入 DataFrame 缺少必要列 %s.", missing_cols)
            raise KeyError(f"初步清洗失败: 输入 DataFrame 缺少必要列 {missing_cols}.")

        # 移除 target 或 cycle_idx 为 NaN 的行
        cleaned_df = cleaned_df.dropna(subset=[target_column, "cycle_idx"]).copy()
        dropped_nan_rows = initial_rows - len(cleaned_df)
        if dropped_nan_rows > 0:
            logger.info(
                "初步清洗: 移除 '%s' 或 'cycle_idx' 为 NaN 的行, 减少了 %d 行.",
                target_column,
                dropped_nan_rows,
            )
            initial_rows = len(cleaned_df)

        # 应用 target 阈值过滤
        target_threshold = self.initial_cleaning_config.target_threshold
        if target_threshold is not None and target_column in cleaned_df.columns:  # 确保目标列存在
            cleaned_df = cleaned_df[cleaned_df[target_column] > target_threshold].copy()
            dropped_target_rows = initial_rows - len(cleaned_df)
            if dropped_target_rows > 0:
                logger.info(
                    "初步清洗: 移除 '%s' <= %.2f 的行, 减少了 %d 行.",
                    target_column,
                    target_threshold,
                    dropped_target_rows,
                )
                initial_rows = len(cleaned_df)

        # 应用 cycle_idx 阈值过滤
        cycle_idx_threshold = self.initial_cleaning_config.cycle_idx_threshold
        if cycle_idx_threshold is not None and "cycle_idx" in cleaned_df.columns:
            cleaned_df = cleaned_df[cleaned_df["cycle_idx"] > cycle_idx_threshold].copy()
            dropped_cycle_rows = initial_rows - len(cleaned_df)
            if dropped_cycle_rows > 0:
                logger.info(  # 使用 logger.info
                    "初步清洗: 移除 'cycle_idx' <= %d 的行, 减少了 %d 行.",
                    cycle_idx_threshold,
                    dropped_cycle_rows,
                )

        logger.info("初步数据过滤完成, 剩余数据形状: %s.", cleaned_df.shape)
        return cleaned_df

    def detect_anomalies_with_isolation_forest(
        self, df: pd.DataFrame, target_column: str
    ) -> pd.DataFrame:
        """
        按 battery_id 使用孤立森林检测异常点.

        Args:
            df (pd.DataFrame): 输入数据, 包含电池老化数据. 应已包含用于检测的特征列.
            target_column (str): 需要进行异常检测的目标列名.

        Returns:
            pd.DataFrame: 包含原始数据和异常标记的 DataFrame (新增 anomaly_column_name 列: 1 正常, -1 异常).
                          对于无法进行异常检测的电池 (如数据点太少), 标记为正常 (1).

        Raises:
            KeyError: 如果输入 DataFrame 缺少孤立森林所需的必要列 ('battery_id', 'cycle_idx', target_column).
        """
        logger.info("开始使用孤立森林检测异常点...")
        df_with_anomaly = df.copy()

        # 使用 isolation_forest_config 中的列名添加异常标记列, 默认标记为正常
        anomaly_column_name = self.isolation_forest_config.anomaly_column_name
        df_with_anomaly[anomaly_column_name] = 1  # 初始化异常标记列

        # 定义用于孤立森林的特征列，固定为 'cycle_idx' 和传入的 target_column
        features_for_isolation_forest = ["cycle_idx", target_column]

        # 检查必要列是否存在 ('battery_id', 'cycle_idx', target_column)
        required_cols = ["battery_id"] + features_for_isolation_forest
        if not all(col in df_with_anomaly.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df_with_anomaly.columns]
            logger.error("孤立森林异常检测失败: 输入 DataFrame 缺少必要列 %s.", missing_cols)
            raise KeyError(f"孤立森林异常检测失败: 输入 DataFrame 缺少必要列 {missing_cols}.")

        # 确保 battery_id 列是字符串类型，以正确进行分组
        df_with_anomaly["battery_id"] = df_with_anomaly["battery_id"].astype(str)
        valid_battery_ids = df_with_anomaly[df_with_anomaly["battery_id"] != "nan"][
            "battery_id"
        ].unique()

        if not valid_battery_ids.size > 0:
            logger.warning("没有有效的 battery_id 可用于孤立森林异常检测.")
            # 此时 df_with_anomaly 已经默认标记为正常，直接返回即可
            return df_with_anomaly

        for bid in valid_battery_ids:
            # 过滤出当前电池的数据
            df_bid_mask = df_with_anomaly["battery_id"] == bid
            # 仅选择需要的特征列进行处理 (即 cycle_idx 和 target_column)
            df_bid_features = df_with_anomaly.loc[df_bid_mask, features_for_isolation_forest].copy()

            # IsolationForest 至少需要两个样本
            if len(df_bid_features) < 2:
                logger.warning(  # 使用 logger.warning
                    "电池 %s 数据点少于 2 个 (%d), 无法进行孤立森林检测. 标记为正常.",
                    bid,
                    len(df_bid_features),
                )
                # 这些电池已经在异常标记列初始化时默认标记为正常 (1)
                continue

            try:
                try:
                    features_scaled = self.scaler.fit_transform(df_bid_features)
                    if features_scaled is None or features_scaled.size == 0:
                        logger.warning("电池 %s 标准化特征后为空, 跳过异常检测.", bid)
                        continue
                except Exception as scaler_e:
                    logger.error(
                        "对电池 %s 特征进行标准化时发生错误: %s. 跳过异常检测.",
                        bid,
                        scaler_e,
                        exc_info=True,
                    )
                    raise

                # 创建 IsolationForest 模型实例 (使用 isolation_forest_config 中的参数)
                # 每次循环都创建新的模型实例，以确保每个电池独立训练
                self.isolation_forest_model = IsolationForest(
                    contamination=self.isolation_forest_config.contamination,
                    random_state=self.isolation_forest_config.random_state,
                )
                # 使用 fit_predict 直接获取预测结果 (-1 for 异常, 1 for 正常)
                anomaly_predictions = self.isolation_forest_model.fit_predict(features_scaled)

                # 将预测结果应用回原始 DataFrame 中对应的行
                df_with_anomaly.loc[df_bid_mask, anomaly_column_name] = anomaly_predictions

                logger.debug(  # 使用 logger.debug
                    "电池 %s 孤立森林检测完成. 检测到异常点数量: %d.",
                    bid,
                    np.sum(anomaly_predictions == -1),
                )

            except Exception as e:
                logger.error(
                    "对电池 %s 进行孤立森林异常检测时发生错误: %s. ", bid, e, exc_info=True
                )
                raise

        logger.info("孤立森林异常检测完成.")
        return df_with_anomaly

    def clean_data(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """
        执行完整的数据清洗流程: 先初步过滤, 再进行孤立森林异常检测

        Args:
            df (pd.DataFrame): 输入原始数据
            target_column (str): 需要进行清洗的目标列名

        Returns:
            tuple: (df_cleaned, df_anomalies, output_target_column)
                - df_cleaned: 清洗后的数据 (移除异常点)
                - df_anomalies: 检测到的异常点数据
                - output_target_column: 清洗后的目标列名称

        Raises:
            ValueError: 如果输入数据为空.
            KeyError: 如果输入 DataFrame 缺少清洗所需的必要列.
            RuntimeError: 如果清洗过程中发生关键错误 (如异常标记列丢失).
            Exception: 捕获并重新抛出其他未预期的错误.
        """
        if df is None or df.empty:
            logger.error("数据清洗失败: 输入数据为空.")
            raise ValueError("数据清洗失败: 输入数据为空.")

        # 检查输入 DataFrame 是否包含必要的列
        required_base_cols = ["battery_id", "cycle_idx", target_column]
        if not all(col in df.columns for col in required_base_cols):
            missing_cols = [col for col in required_base_cols if col not in df.columns]
            logger.error("数据清洗失败: 输入 DataFrame 缺少必要列 %s.", missing_cols)
            raise KeyError(f"数据清洗失败: 输入 DataFrame 缺少必要列 {missing_cols}.")

        # --- Step 1: 初步过滤 ---
        df_filtered = self.perform_initial_cleaning(df, target_column)

        if df_filtered is None or df_filtered.empty:
            logger.error("初步过滤后数据为空")
            raise RuntimeError("数据清洗失败: 初步过滤后数据为空.")

        # --- Step 2: 异常检测 ---
        df_with_anomaly = self.detect_anomalies_with_isolation_forest(df_filtered, target_column)

        # --- Step 3: 分离正常数据和异常数据, 去除标记列 ---
        # 使用配置中的异常列名
        anomaly_column_name = self.isolation_forest_config.anomaly_column_name

        # 确保异常标记列存在于进行分离的 DataFrame 中
        if anomaly_column_name not in df_with_anomaly.columns:
            logger.error("数据清洗失败: 异常检测后未找到标记列 '%s'.", anomaly_column_name)
            raise RuntimeError(f"数据清洗失败: 异常检测后未找到标记列 '{anomaly_column_name}'.")

        # 分离正常数据 (anomaly == 1) 和异常数据 (anomaly == -1)
        df_cleaned = (
            df_with_anomaly[df_with_anomaly[anomaly_column_name] == 1]
            .drop(columns=[anomaly_column_name])
            .copy()
        )
        df_anomalies = (
            df_with_anomaly[df_with_anomaly[anomaly_column_name] == -1]
            .drop(columns=[anomaly_column_name])
            .copy()
        )

        logger.info(
            "数据清洗流程完成. 清洗后数据形状: %s, 检测到异常点数量: %d.",
            df_cleaned.shape,
            len(df_anomalies),
        )

        # 清洗操作不改变原始目标列名，因此返回输入的 target_column
        output_target_column = target_column

        return df_cleaned, df_anomalies, output_target_column
