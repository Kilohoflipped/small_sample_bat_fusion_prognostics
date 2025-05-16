import logging
import os
from typing import Any, Dict, Tuple

import pandas as pd

from src.modules.data_preprocess.data_cleaner import DataCleaner

# 导入各个独立的处理器类
# 假设这些类位于 src.modules.data_preprocess 子目录下
from src.modules.data_preprocess.data_converter import DataConverter
from src.modules.data_preprocess.data_denoiser import DataDenoiser
from src.modules.data_preprocess.data_imputer import DataImputer
from src.modules.data_preprocess.data_standardizer import DataStandardizer

# 移除绘图相关的导入


logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    数据预处理流程的管线 (无绘图功能).
    负责加载配置、创建目录、按顺序执行预处理步骤、保存中间结果.
    直接协调各个独立的处理器类.
    """

    def __init__(self, config: Dict[str, Any], project_root: str):
        """
        初始化 PreprocessingPipeline.

        根据提供的配置初始化各个独立的处理器. 不包含绘图工具的初始化.

        Args:
            config (Dict[str, Any]): 包含所有流程配置的字典.
            project_root (str): 项目根目录的路径.
        """
        self.config = config
        self.project_root = project_root

        # 初始化各个独立的处理器，将完整的 config 传递给它们
        try:
            self.converter = DataConverter(self.config)
            self.cleaner = DataCleaner(self.config)
            self.imputer = DataImputer(self.config)
            self.denoiser = DataDenoiser(self.config)
            self.standardizer = DataStandardizer(self.config)

        except Exception as e:
            logger.error("初始化数据处理器时发生错误: %s", e, exc_info=True)
            # 在初始化阶段发生错误是致命的，抛出错误
            raise RuntimeError(f"初始化数据处理器时发生错误: {e}") from e

        # 移除绘图工具的初始化

        # 获取路径配置
        self.input_data_paths = config.get("input_data_paths", {})
        self.output_dirs = config.get("output_dirs", {})
        # 获取原始数据转换后的目标列名，这是整个处理流程的起点
        # DataConverter 的 target_column 配置项现在用于指定转换后的目标列名
        self.initial_target_column = config.get("data_converter", {}).get("target_column", "target")

        # 定义输出文件目录
        self.processed_data_dir = os.path.join(
            self.project_root,
            self.output_dirs.get("preprocessed_data", "data/interim/preprocessed"),
        )
        # 移除绘图目录的定义

        # 定义具体的输出文件路径 (根据步骤命名，不依赖于具体的列名)
        self.step0_converted_data_path = os.path.join(
            self.processed_data_dir, "step0_battery_aging_cycle_data_converted.csv"
        )
        # DataCleaner 会返回异常点数据
        self.step1_cleaned_data_path = os.path.join(
            self.processed_data_dir, "step1_battery_aging_cycle_data_cleaned.csv"
        )
        self.step1_anomalies_data_path = os.path.join(
            self.processed_data_dir, "step1_battery_aging_cycle_data_anomalies.csv"
        )
        self.step2_imputed_data_path = os.path.join(
            self.processed_data_dir, "step2_battery_aging_cycle_data_imputed.csv"
        )
        self.step3_denoised_data_path = os.path.join(
            self.processed_data_dir, "step3_battery_aging_cycle_data_denoised.csv"
        )
        self.step4_standardized_data_path = os.path.join(
            self.processed_data_dir, "step4_battery_aging_cycle_data_standardized.csv"
        )
        # 移除 step5_decomposed_data_path

        # 移除所有绘图输出文件名的基础路径定义

        logger.info("PreprocessingPipeline 初始化完成 (无绘图功能).")

    def _create_required_dirs(self):
        """
        创建预处理流程所需的输出目录.
        如果目录已存在，则不做任何操作.
        """
        logger.info("创建输出目录...")
        try:
            os.makedirs(self.processed_data_dir, exist_ok=True)
            logger.info(f"创建数据输出目录: {self.processed_data_dir}")
            # 移除创建绘图目录的代码
        except OSError as e:
            logger.error(f"创建输出目录失败: {e}", exc_info=True)
            raise  # 重新抛出错误，因为无法保存结果是致命的

    def _save_dataframe(self, df: pd.DataFrame, path: str, step_name: str):
        """
        保存 DataFrame 到指定路径，并记录日志.

        Args:
            df (pd.DataFrame): 要保存的 DataFrame.
            path (str): 保存文件的完整路径.
            step_name (str): 当前处理步骤的名称，用于日志记录.
        """
        if df is None or df.empty:
            logger.warning(f"{step_name} 后 DataFrame 为空或 None, 跳过保存.")
            return

        try:
            df.to_csv(path, index=False)
            logger.info(f"{step_name} 后数据已保存至: {path}")
        except Exception as e:
            logger.error(f"保存 {step_name} 后数据失败至 {path}: {e}", exc_info=True)
            # 保存失败不中断整个流程

    def run(self) -> Tuple[pd.DataFrame | None, Dict[str, str], pd.DataFrame | None]:
        """
        执行完整的预处理流程.

        Returns:
            Tuple[pd.DataFrame | None, Dict[str, str], pd.DataFrame | None]:
                - pd.DataFrame | None: 最终处理完成的数据框. 如果任何主要步骤失败，返回 None.
                - Dict[str, str]: 记录每个处理步骤后目标列名的字典.
                                  键为步骤名称 (例如, 'cleaning', 'imputation'), 值为处理后的目标列名.
                - pd.DataFrame | None: 清洗步骤中检测到的异常点数据. 如果清洗失败或没有异常点，返回 None 或空 DataFrame.
        """
        logger.info("数据预处理流程开始执行 (无绘图功能).")

        output_column_mapping: Dict[str, str] = {}
        df_anomalies: pd.DataFrame | None = None  # 初始化异常点 DataFrame
        current_df: pd.DataFrame | None = None  # 当前处理的 DataFrame
        current_target_column: str = self.initial_target_column  # 当前处理的目标列名

        # --- 1. 创建输出目录 ---
        self._create_required_dirs()

        # --- 2. 数据加载和转换 ---
        logger.info("执行数据加载和转换...")
        raw_data_path = os.path.join(
            self.project_root, self.input_data_paths.get("raw_data_path", "")
        )
        if not raw_data_path:
            logger.error("原始数据路径未配置或为空.")
            raise ValueError("原始数据路径未配置或为空.")

        # 直接调用 DataConverter 的 load_and_convert 方法
        converted_df = self.converter.load_and_convert(raw_data_path)
        if converted_df is None or converted_df.empty:
            logger.error("数据加载或转换失败或结果为空，流程终止.")
            raise ValueError("数据加载或转换失败或结果为空.")

        current_df = converted_df.copy()  # 从转换后的数据开始处理
        current_target_column = self.initial_target_column  # 设置初始目标列名

        # 检查转换后的数据是否包含必要的列
        required_initial_cols = ["battery_id", "cycle_idx", current_target_column]
        if not all(col in current_df.columns for col in required_initial_cols):
            missing_cols = [col for col in required_initial_cols if col not in current_df.columns]
            logger.error(f"数据加载或转换后缺少必要列 {missing_cols}，流程终止.")
            raise ValueError(f"数据加载或转换后缺少必要列 {missing_cols}.")

        # 转换后的数据始终保存
        self._save_dataframe(current_df, self.step0_converted_data_path, "转换")

        # 移除所有原始数据相关的绘图代码

        # --- 3. 清洗步骤 ---
        logger.info(f"执行清洗步骤，目标列: '{current_target_column}'...")
        # 清洗前保留当前列名，用于插值步骤的原始列对比 (虽然没有绘图，但保留这个信息流可能对其他用途有益)
        col_before_cleaning = current_target_column
        try:
            # DataCleaner.clean_data 返回 (df_cleaned, df_anomalies, output_target_column)
            cleaned_result_tuple = self.cleaner.clean_data(current_df, current_target_column)
            current_df, df_anomalies, current_target_column = (
                cleaned_result_tuple  # 清洗通常不改变列名
            )

            if current_df is None or current_df.empty:
                # 如果清洗后数据为空，则后续步骤无法进行
                logger.error("预处理失败: 清洗步骤后数据为空.")
                return None, output_column_mapping, df_anomalies

            output_column_mapping["cleaning"] = current_target_column  # 记录清洗后的目标列名

        except (ValueError, KeyError, RuntimeError, Exception) as e:
            logger.error(f"执行清洗步骤时发生错误: {e}", exc_info=True)
            # 清洗失败是关键错误，终止流程
            return None, output_column_mapping, df_anomalies

        logger.info(
            f"清洗步骤完成. 当前处理列: '{current_target_column}'. 数据形状: {current_df.shape}."
        )

        # 保存清洗后的数据
        self._save_dataframe(current_df, self.step1_cleaned_data_path, "清洗")

        # 保存异常点数据
        if df_anomalies is not None and not df_anomalies.empty:
            self._save_dataframe(df_anomalies, self.step1_anomalies_data_path, "异常点")

        # --- 4. 插值步骤 ---
        logger.info(f"执行插值步骤，目标列: '{current_target_column}'...")
        # 插值前保留当前列名，用于去噪步骤的原始列对比
        try:
            # DataImputer.impute_missing_values 返回 (imputed_df, output_column)
            imputed_result_tuple = self.imputer.impute_missing_values(
                current_df, current_target_column
            )
            current_df, current_target_column = imputed_result_tuple  # 插值通常不改变列名

            if current_df is None or current_df.empty:
                logger.error("预处理失败: 插值步骤后数据为空.")
                return None, output_column_mapping, df_anomalies

            output_column_mapping["imputation"] = current_target_column  # 记录插值后的目标列名

        except (ValueError, KeyError, RuntimeError, Exception) as e:
            logger.error(f"执行插值步骤时发生错误: {e}", exc_info=True)
            return None, output_column_mapping, df_anomalies

        logger.info(
            f"插值步骤完成. 当前处理列: '{current_target_column}'. 数据形状: {current_df.shape}."
        )

        # 保存插值后的数据
        self._save_dataframe(current_df, self.step2_imputed_data_path, "插值")

        # 移除所有插值后相关的绘图代码

        # --- 5. 去噪步骤 ---
        logger.info(f"执行去噪步骤，目标列: '{current_target_column}'...")
        # 去噪前保留当前列名，用于标准化步骤的原始列对比
        try:
            # DataDenoiser.denoise_data 返回 (denoised_df, output_column)
            denoised_result_tuple = self.denoiser.denoise_data(current_df, current_target_column)
            current_df, current_target_column = denoised_result_tuple  # 去噪通常不改变列名

            if current_df is None or current_df.empty:
                logger.error("预处理失败: 去噪步骤后数据为空.")
                return None, output_column_mapping, df_anomalies

            output_column_mapping["denoising"] = current_target_column  # 记录去噪后的目标列名

        except (ValueError, KeyError, RuntimeError, Exception) as e:
            logger.error(f"执行去噪步骤时发生错误: {e}", exc_info=True)
            return None, output_column_mapping, df_anomalies

        logger.info(
            f"去噪步骤完成. 当前处理列: '{current_target_column}'. 数据形状: {current_df.shape}."
        )

        # 保存去噪后的数据
        self._save_dataframe(current_df, self.step3_denoised_data_path, "去噪")

        # 移除所有去噪后相关的绘图代码

        # --- 6. 标准化步骤 ---
        # 注意：这里是 Step 6，对应 step4_standardized_data_path
        logger.info(f"执行标准化步骤，目标列: '{current_target_column}'...")
        # 标准化前保留当前列名 (可选，如果需要标准化前后的对比图)
        col_before_standardization = current_target_column
        try:
            # DataStandardizer.standardize_data 返回 (standardized_df, output_column)
            standardized_result_tuple = self.standardizer.standardize_data(
                current_df, current_target_column
            )
            current_df, current_target_column = standardized_result_tuple  # 标准化通常不改变列名

            if current_df is None or current_df.empty:
                logger.error("预处理失败: 标准化步骤后数据为空.")
                return None, output_column_mapping, df_anomalies

            output_column_mapping["standardization"] = (
                current_target_column  # 记录标准化后的目标列名
            )

        except (ValueError, KeyError, RuntimeError, Exception) as e:
            logger.error(f"执行标准化步骤时发生错误: {e}", exc_info=True)
            return None, output_column_mapping, df_anomalies

        logger.info(
            f"标准化步骤完成. 当前处理列: '{current_target_column}'. 数据形状: {current_df.shape}."
        )

        # 保存标准化后的数据
        self._save_dataframe(current_df, self.step4_standardized_data_path, "标准化")

        logger.info("数据预处理流程所有步骤执行完成.")

        # 返回最终处理后的 DataFrame, 列名映射和异常点 DataFrame
        return current_df, output_column_mapping, df_anomalies
