import pandas as pd
import logging
import os
from typing import Dict, Any, Tuple

from src.modules.data_preprocess.data_preprocesser import DataPreprocessor
from src.modules.visualization.plotter import Plotter
from src.modules.visualization.style_setter import StyleSetter  # 导入 StyleSetter

logger = logging.getLogger(__name__)


class PreProcessingPipeline:
    """
    数据预处理流程的编排者.
    负责加载配置、创建目录、按顺序执行预处理步骤、保存中间结果和生成图表.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 PreProcessingPipeline.

        Args:
            config (Dict[str, Any]): 包含所有流程配置的字典.
        """
        self.config = config
        # 初始化 DataPreprocessor 和 Plotter，将对应的配置传递给它们
        self.preprocessor = DataPreprocessor(config)

        # 初始化 StyleSetter 和 Plotter
        style_setter = StyleSetter(config.get('plot_style', {}))
        # 临时应用全局风格，主要用于字体等，只在这里应用一次
        style_setter.apply_global_style()

        plot_params = config.get('plot_params', {})
        plot_size_cm = plot_params.get('plot_size_cm', (16, 11))
        plot_dpi = plot_params.get('dpi', 300)
        self.overall_plot_max_batteries = plot_params.get('overall_plot_max_batteries', 20)

        self.plotter = Plotter(style_setter, plot_size_cm, plot_dpi)

        # 获取路径配置
        self.data_paths = config.get('data_paths', {})
        self.output_dirs = config.get('output_dirs', {})

        # 定义输出文件路径 (从配置或默认值获取输出目录)
        self.processed_data_dir = self.output_dirs.get('processed_data', 'data/processed')
        self.plot_dir = self.output_dirs.get('plots', 'plots')

        # 定义具体的输出文件名 (这些文件名也可以放在配置中)
        self.cleaned_data_path = os.path.join(
            self.processed_data_dir,
            'step_0_battery_aging_cycle_data_cleaned.csv')
        self.anomalies_data_path = os.path.join(
            self.processed_data_dir,
            'step_0_battery_aging_cycle_data_anomalies.csv')
        self.imputed_data_path = os.path.join(
            self.processed_data_dir,
            'step_1_battery_aging_cycle_data_imputed.csv')
        self.denoised_data_path = os.path.join(
            self.processed_data_dir,
            'step_2_battery_aging_cycle_data_denoised.csv')
        self.decomposed_data_path = os.path.join(
            self.processed_data_dir,
            'step_3_battery_aging_cycle_data_decomposed.csv')

        self.overall_before_cleaning_plot_path = os.path.join(
            self.plot_dir, 'overall_before_cleaning.png')
        self.overall_after_cleaning_plot_path = os.path.join(
            self.plot_dir, 'overall_after_cleaning.png')
        self.overall_after_imputation_plot_path = os.path.join(
            self.plot_dir, 'overall_after_imputation.png')

        logger.info("PreProcessingPipeline 初始化完成.")

    def run(self) -> pd.DataFrame | None:
        """
        执行完整的预处理流程.

        Returns:
            pd.DataFrame | None: 最终处理完成的数据框，如果任何步骤失败则返回 None.
        """
        logger.info("数据预处理流程开始执行.")

        # 1. 创建输出目录
        self._create_required_dirs()

        # 2. 数据加载和转换
        raw_df = self._load_and_transform_data()
        if raw_df is None or raw_df.empty:
            logger.error("Pipeline: 数据加载或转换失败，流程终止.")
            return None

        # 3. 数据清洗 (初步过滤 + 异常检测)
        cleaned_result = self._clean_data(raw_df.copy())  # 传入副本进行清洗
        if cleaned_result is None:
            logger.error("Pipeline: 数据清洗失败，流程终止.")
            return None
        df_cleaned, df_anomalies = cleaned_result

        # 4. 缺失值处理
        df_imputed = self._impute_missing_values(df_cleaned.copy())  # 传入清洗后的数据副本
        if df_imputed is None:
            logger.error("Pipeline: 缺失值处理失败，流程终止.")
            return None

        # 5. 数据去噪
        df_denoised = self._denoise_data(df_imputed.copy())  # 传入插值后的数据副本
        if df_denoised is None:
            logger.error("Pipeline: 数据去噪失败，流程终止.")
            return None

        # 6. 趋势分解
        df_decomposed = self._decompose_data(df_denoised.copy())  # 传入去噪后的数据副本
        if df_decomposed is None:
            logger.error("Pipeline: 趋势分解失败，流程终止.")
            return None

        logger.info("数据预处理流程执行完成。")
        return df_decomposed  # 返回最终处理完成的数据框

    def _create_required_dirs(self):
        """
        根据配置创建所有必要的输出目录.
        """
        for key, dir_path in self.output_dirs.items():
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Pipeline: 已创建或确认输出目录 '{key}': {dir_path}")
            except OSError as e:
                logger.error(f"Pipeline: 创建输出目录 '{dir_path}' 时发生错误: {e}")
                # 在实际应用中，这里可能需要更强的错误处理，例如终止程序

    def _load_and_transform_data(self) -> pd.DataFrame | None:
        """
        执行数据加载和初步转换步骤.

        Returns:
            pd.DataFrame | None: 加载并转换后的数据框，如果失败则返回 None.
        """
        logger.info("Pipeline: 执行数据加载和转换...")
        try:
            raw_df = self.preprocessor.load_data(
                self.data_paths.get('raw_data_excel'),
                self.data_paths.get('initial_csv_path')
            )
            if raw_df.empty:
                logger.error("Pipeline: 数据加载失败，返回空数据框.")
                return None

            # 绘制原始数据图 (如果数据来自 Excel 并进行了转换)
            if self.data_paths.get('raw_data_excel') and os.path.exists(
                    self.data_paths.get('raw_data_excel')):
                try:
                    logger.info("Pipeline: 绘制原始数据单电池图...")
                    self.plotter.plot_per_battery_raw(raw_df, self.plot_dir)
                except Exception as e:
                    logger.error(f"Pipeline: 绘制原始数据单电池图时发生错误: {e}")

            logger.info("Pipeline: 数据加载和转换步骤完成.")
            return raw_df
        except Exception as e:
            logger.error(f"Pipeline: 数据加载和转换步骤执行失败: {e}.")
            return None

    def _clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame] | None:
        """
        执行数据清洗 (初步过滤 + 异常检测) 步骤.

        Args:
            df (pd.DataFrame): 需要清洗的原始数据框.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] | None: 清洗后的数据框和异常点数据框，如果失败则返回 None.
        """
        logger.info("Pipeline: 执行数据清洗流程...")
        try:
            df_cleaned, df_anomalies = self.preprocessor.clean_data(df)
            logger.info(
                f"Pipeline: 数据清洗完成. 清洗后数据形状: {
                    df_cleaned.shape}, 检测到异常点数量: {
                    len(df_anomalies)}.")

            # 保存清洗数据
            df_cleaned.to_csv(self.cleaned_data_path, index=False)
            logger.info(f"Pipeline: 清洗后的数据已保存至: {self.cleaned_data_path}")
            if not df_anomalies.empty:
                df_anomalies.to_csv(self.anomalies_data_path, index=False)
                logger.info(f"Pipeline: 异常数据已保存至: {self.anomalies_data_path}")
            else:
                logger.info("Pipeline: 未检测到异常数据，未保存异常数据文件.")

            # 绘制清洗相关图
            try:
                logger.info("Pipeline: 绘制清洗相关图...")
                # 为了绘制 comparison 图，需要带有 anomaly 标记的完整 DataFrame
                # 临时方案: 在这里重新运行 anomaly detection 来获取 df_with_anomaly
                # 更优方案是在 DataCleaner 中返回带有 anomaly 标记的完整 DataFrame
                df_with_anomaly_temp = self.preprocessor.data_cleaner.detect_anomalies_with_isolation_forest(
                    self.preprocessor.data_cleaner.perform_initial_cleaning(df.copy()))  # 传入原始数据的副本
                self.plotter.plot_per_battery_comparison(
                    df, df_with_anomaly_temp, self.plot_dir)  # 传入原始数据和带标记数据
                self.plotter.plot_per_battery_cleaned(df_cleaned, self.plot_dir)
                self.plotter.plot_overall_sequence(
                    df,  # 绘制清洗前使用原始数据
                    '清洗前总体序列',
                    self.overall_before_cleaning_plot_path,
                    max_batteries=self.overall_plot_max_batteries)
                self.plotter.plot_overall_sequence(
                    df_cleaned,
                    '清洗后总体序列',
                    self.overall_after_cleaning_plot_path,
                    max_batteries=self.overall_plot_max_batteries)
            except Exception as e:
                logger.error(f"Pipeline: 绘制清洗相关图时发生错误: {e}")

            return df_cleaned, df_anomalies
        except Exception as e:
            logger.error(f"Pipeline: 数据清洗流程执行失败: {e}.")
            return None

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """
        执行缺失值处理步骤.

        Args:
            df (pd.DataFrame): 需要处理缺失值的数据框 (通常是清洗后的数据).

        Returns:
            pd.DataFrame | None: 处理缺失值后的数据框，如果失败则返回 None.
        """
        logger.info("Pipeline: 执行缺失值处理流程...")
        try:
            df_imputed = self.preprocessor.impute_missing_values(df)
            logger.info(f"Pipeline: 缺失值处理完成. 处理后数据形状: {df_imputed.shape}.")
            remaining_nan_after_imputation = df_imputed[self.preprocessor.data_imputer.target_column].isna(
            ).sum()
            if remaining_nan_after_imputation > 0:
                logger.warning(f"Pipeline: 缺失值处理后，目标列 '{self.preprocessor.data_imputer.target_column}' 仍有 {
                               remaining_nan_after_imputation} 个缺失值.")

            # 保存插值数据
            df_imputed.to_csv(self.imputed_data_path, index=False)
            logger.info(f"Pipeline: 插值后的数据已保存至: {self.imputed_data_path}")

            # 绘制缺失值处理相关图
            try:
                logger.info("Pipeline: 绘制缺失值处理相关图...")
                # plot_per_battery_imputed 需要清洗前的数据来区分原始点和插值点
                # 这里传入清洗后的数据 df (作为原始点参考) 和插值后的数据 df_imputed
                self.plotter.plot_per_battery_imputed(df, df_imputed, self.plot_dir)
                self.plotter.plot_overall_sequence(
                    df_imputed,
                    '插值后总体序列',
                    self.overall_after_imputation_plot_path,
                    max_batteries=self.overall_plot_max_batteries)
            except Exception as e:
                logger.error(f"Pipeline: 绘制缺失值处理相关图时发生错误: {e}")

            return df_imputed
        except Exception as e:
            logger.error(f"Pipeline: 缺失值处理流程执行失败: {e}.")
            return None

    def _denoise_data(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """
        执行数据去噪步骤.

        Args:
            df (pd.DataFrame): 需要去噪的数据框 (通常是插值后的数据).

        Returns:
            pd.DataFrame | None: 去噪后的数据框，如果失败则返回 None.
        """
        logger.info("Pipeline: 执行数据去噪流程...")
        try:
            # 确保去噪的目标列存在于 df 中
            target_col_for_denoising = self.preprocessor.data_denoiser.target_column
            if target_col_for_denoising not in df.columns:
                logger.error(f"Pipeline: 数据去噪失败: 数据中缺少目标列 '{target_col_for_denoising}'.")
                return None

            df_denoised = self.preprocessor.denoise_data(df)
            logger.info(f"Pipeline: 数据去噪完成. 去噪后数据形状: {df_denoised.shape}.")
            denoised_col_name = self.preprocessor.data_denoiser.denoised_column_name
            if denoised_col_name not in df_denoised.columns:
                logger.warning(f"Pipeline: 数据去噪流程完成，但去噪列 '{denoised_col_name}' 未生成.")
            else:
                remaining_nan_after_denoising = df_denoised[denoised_col_name].isna().sum()
                if remaining_nan_after_denoising > 0:
                    logger.warning(f"Pipeline: 数据去噪后，去噪列 '{denoised_col_name}' 仍有 {
                                   remaining_nan_after_denoising} 个缺失值.")

            # 保存去噪数据
            df_denoised.to_csv(self.denoised_data_path, index=False)
            logger.info(f"Pipeline: 去噪后的数据已保存至: {self.denoised_data_path}")

            # 绘制去噪相关图
            try:
                logger.info("Pipeline: 绘制去噪相关图...")
                # 绘制原始插值数据与去噪后数据的对比
                self.plotter.plot_per_battery_denoised(
                    df_denoised,  # 传入去噪后的数据，其中包含原始列和去噪列
                    self.plot_dir,
                    original_column=target_col_for_denoising,  # 使用去噪的目标列作为原始列
                    denoised_column=denoised_col_name)
            except Exception as e:
                logger.error(f"Pipeline: 绘制去噪相关图时发生错误: {e}")

            return df_denoised
        except Exception as e:
            logger.error(f"Pipeline: 数据去噪流程执行失败: {e}.")
            return None

    def _decompose_data(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """
        执行趋势分解步骤.

        Args:
            df (pd.DataFrame): 需要进行趋势分解的数据框 (通常是去噪后的数据).

        Returns:
            pd.DataFrame | None: 趋势分解后的数据框，如果失败则返回 None.
        """
        logger.info("Pipeline: 开始趋势分解流程...")
        try:
            # 确保分解的目标列存在于 df 中
            target_col_for_decomposition = self.preprocessor.data_decomposer.target_column
            if target_col_for_decomposition not in df.columns:
                logger.error(f"Pipeline: 趋势分解失败: 数据中缺少目标列 '{
                             target_col_for_decomposition}'.")
                return None

            df_decomposed = self.preprocessor.decompose_data(df)
            logger.info(f"Pipeline: 趋势分解完成. 分解后数据形状: {df_decomposed.shape}.")

            trend_col_name = self.preprocessor.data_decomposer.trend_column_name
            residual_col_name = self.preprocessor.data_decomposer.residual_column_name
            mode_prefix = self.preprocessor.data_decomposer.mode_column_prefix

            if trend_col_name not in df_decomposed.columns:
                logger.warning(f"Pipeline: 趋势分解流程完成，但趋势列 '{trend_col_name}' 未生成.")
            else:
                remaining_nan_trend = df_decomposed[trend_col_name].isna().sum()
                if remaining_nan_trend > 0:
                    logger.warning(f"Pipeline: 趋势分解后，趋势列 '{trend_col_name}' 仍有 {
                                   remaining_nan_trend} 个缺失值.")

            if residual_col_name not in df_decomposed.columns:
                logger.warning(f"Pipeline: 趋势分解流程完成，但残差列 '{residual_col_name}' 未生成.")
            else:
                remaining_nan_residual = df_decomposed[residual_col_name].isna().sum()
                if remaining_nan_residual > 0:
                    logger.warning(f"Pipeline: 趋势分解后，残差列 '{residual_col_name}' 仍有 {
                                   remaining_nan_residual} 个缺失值.")

            mode_cols_check = [col for col in df_decomposed.columns if col.startswith(mode_prefix)]
            logger.info(f"Pipeline: 分解出的模态数量: {len(mode_cols_check)}")

            # 保存分解数据
            df_decomposed.to_csv(self.decomposed_data_path, index=False)
            logger.info(f"Pipeline: 趋势分解后的数据已保存至: {self.decomposed_data_path}")

            # 绘制趋势分解相关图
            try:
                logger.info("Pipeline: 绘制趋势分解相关图...")
                self.plotter.plot_per_battery_decomposed(
                    df_decomposed,
                    self.plot_dir,
                    target_column=target_col_for_decomposition,
                    show_modes=True  # 根据需要控制是否绘制模态
                )
            except Exception as e:
                logger.error(f"Pipeline: 绘制趋势分解相关图时发生错误: {e}")

            return df_decomposed
        except Exception as e:
            logger.error(f"Pipeline: 趋势分解流程执行失败: {e}.")
            return None
