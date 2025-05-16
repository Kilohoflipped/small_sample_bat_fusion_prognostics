import logging
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 假设这个模块存在
from src.modules.visualization.style_setter import StyleSetter

logger = logging.getLogger(__name__)


class PreprocessPlotter:
    """
    负责绘制电池数据图表的类
    使用 StyleSetter 获取风格并在绘制时使用上下文应用风格
    """

    def __init__(self, style_setter: StyleSetter):
        """
        初始化 PreprocessPlotter.

        Args:
            style_setter (StyleSetter): 绘图风格设置对象.
        """
        if not isinstance(style_setter, StyleSetter):
            raise TypeError("style_setter 必须是 StyleSetter 类的实例")

        self.style_setter = style_setter
        self.style: Dict[str, Any] = self.style_setter.get_style()

    def _apply_plot_style(self, ax: plt.Axes, title: str, xlabel: str, ylabel: str):
        """
        应用通用绘图风格到 Axes 对象 (主要设置文本).

        Args:
            ax (matplotlib.axes.Axes): 要应用风格的 Axes 对象.
            title (str): 图表标题.
            xlabel (str): X轴标签.
            ylabel (str): Y轴标签.
        """
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def _save_plot(self, fig: plt.Figure, base_output_path_without_extension: str):
        """
        保存图表到指定路径，根据 rcParams 自动添加文件扩展名.

        Args:
            fig (matplotlib.figure.Figure): 要保存的 Figure 对象.
            base_output_path_without_extension (str): 图片保存路径 (不包含文件扩展名).
        """
        try:
            save_format = plt.rcParams.get("savefig.format", "png")
            full_output_path = f"{base_output_path_without_extension}.{save_format}"
            fig.savefig(full_output_path, bbox_inches="tight")
            logger.info(f"已保存图表至: {full_output_path}")
        except Exception as e:
            logger.error(f"保存图表 '{full_output_path}' 时发生错误: {e}")
            raise
        finally:
            plt.close(fig)

    def plot_per_battery_raw(
        self, df: pd.DataFrame, output_dir: str, target_column: str = "target"
    ):
        """
        为每个 battery_id 绘制原始数据的折线图.

        Args:
            df (pd.DataFrame): 原始数据，包含 battery_id, cycle_idx 和 target 列.
            output_dir (str): 图片保存目录.
            target_column (str): 目标列名，默认为 'target'.
        """
        required_cols = ["battery_id", "cycle_idx", target_column]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"绘制原始数据图失败: 输入 DataFrame 缺少必要列 {missing}")
            return

        rc_params_dict = self.style_setter.get_rc_params_dict()
        df["battery_id"] = df["battery_id"].astype(str)
        valid_battery_ids = df[df["battery_id"] != "nan"]["battery_id"].unique()

        if not valid_battery_ids.size > 0:
            logger.warning("没有有效的 battery_id 可用于绘制原始数据单电池图.")
            return

        for bid in valid_battery_ids:
            df_bid = df[df["battery_id"] == bid].copy()

            with plt.style.context(rc_params_dict):
                fig, ax = plt.subplots()

                ax.plot(
                    df_bid["cycle_idx"],
                    df_bid[target_column],
                    alpha=0.7,
                    label="原始数据",
                )
                ax.legend()

                self._apply_plot_style(
                    ax, f"电池 {bid} - 原始数据", "循环索引 (cycles)", "容量 (Ah)"
                )

                plot_path = os.path.join(output_dir, f"battery_{bid}_raw_{target_column}")
                self._save_plot(fig, plot_path)

    def plot_per_battery_comparison(
        self,
        df_cleaned: pd.DataFrame,  # 清洗后的数据 (不含异常点)
        df_anomalies_only: pd.DataFrame,  # 仅包含异常点的数据
        output_dir: str,
        target_column: str = "target",
    ):
        """
        为每个 battery_id 绘制原始数据 (清洗后 + 异常点) 与异常点的对比图.

        Args:
            df_cleaned (pd.DataFrame): 清洗后的数据 (不含异常点).
            df_anomalies_only (pd.DataFrame): 仅包含异常点的数据.
            output_dir (str): 图片保存目录.
            target_column (str): 目标列名，默认为 'target'.
        """
        # 检查清洗后的 DataFrame 是否包含必要列
        required_cols_cleaned = ["battery_id", "cycle_idx", target_column]
        if not all(col in df_cleaned.columns for col in required_cols_cleaned):
            missing = [col for col in required_cols_cleaned if col not in df_cleaned.columns]
            logger.error(f"绘制数据对比图失败: 清洗后的 DataFrame 缺少必要列 {missing}")
            raise ValueError(f"清洗后的 DataFrame 缺少必要列 {missing}")

        # 检查仅包含异常点的 DataFrame 是否包含必要列
        # 仅需要 battery_id, cycle_idx, target_column 来定位和绘制异常点
        required_cols_anomalies = ["battery_id", "cycle_idx", target_column]
        if not all(col in df_anomalies_only.columns for col in required_cols_anomalies):
            missing = [
                col for col in required_cols_anomalies if col not in df_anomalies_only.columns
            ]
            logger.error(f"绘制数据对比图失败: 仅包含异常点的 DataFrame 缺少必要列 {missing}")
            raise ValueError(f"仅包含异常点的 DataFrame 缺少必要列 {missing}")

        # 获取绘图参数
        rc_params_dict = self.style_setter.get_rc_params_dict()

        # 确保 battery_id 是字符串类型，以便后续筛选和合并
        df_cleaned["battery_id"] = df_cleaned["battery_id"].astype(str)
        df_anomalies_only["battery_id"] = df_anomalies_only["battery_id"].astype(str)

        # 合并清洗后的数据和异常点数据，以重建完整的原始数据序列
        # 使用 concat 合并，然后按 battery_id 和 cycle_idx 排序
        df_original_combined = pd.concat([df_cleaned, df_anomalies_only], ignore_index=True)
        df_original_combined = df_original_combined.sort_values(
            by=["battery_id", "cycle_idx"]
        ).reset_index(drop=True)

        # 获取所有有效的 battery_id，从合并后的完整数据中获取
        valid_battery_ids = df_original_combined[df_original_combined["battery_id"] != "nan"][
            "battery_id"
        ].unique()

        if not valid_battery_ids.size > 0:
            # 使用 logger 记录警告信息
            logger.warning("没有有效的 battery_id 可用于绘制数据对比图.")
            return

        # 遍历每个 battery_id 进行绘图
        for bid in valid_battery_ids:
            # 筛选出当前 battery_id 的完整原始数据
            df_bid_original_combined = df_original_combined[
                df_original_combined["battery_id"] == bid
            ].copy()
            # 筛选出当前 battery_id 的异常点数据 (从仅包含异常点的原始输入 DataFrame 中筛选)
            df_bid_anomalies_only = df_anomalies_only[df_anomalies_only["battery_id"] == bid].copy()

            # 如果完整原始数据为空，则跳过该电池
            if df_bid_original_combined.empty:
                logger.warning(f"电池 {bid} 的完整原始数据为空，跳过绘图.")
                continue

            # 使用绘图样式上下文
            with plt.style.context(rc_params_dict):
                fig, ax = plt.subplots()

                # 绘制完整的原始数据曲线
                ax.plot(
                    df_bid_original_combined["cycle_idx"],
                    df_bid_original_combined[target_column],
                    alpha=0.7,
                    label="原始数据",  # 标签使用“原始数据”
                    zorder=1,
                )

                # 如果存在当前 battery_id 的异常点，则绘制散点
                if not df_bid_anomalies_only.empty:
                    ax.scatter(
                        df_bid_anomalies_only["cycle_idx"],
                        df_bid_anomalies_only[target_column],
                        color=self.style.get("anomaly_color", "#d64f38"),
                        marker="x",
                        s=self.style.get("anomaly_markersize", 100),
                        alpha=0.9,
                        label="异常点",
                        zorder=2,
                    )

                # 显示图例
                ax.legend()

                # 应用自定义绘图样式（标题、标签等）
                self._apply_plot_style(
                    ax,
                    f"电池 {bid} - 原始数据与异常点",  # 标题
                    "循环索引 (cycles)",
                    "容量 (Ah)",
                )

                # 构造图片保存路径
                plot_path = os.path.join(
                    output_dir, f"battery_{bid}_anomaly_comparison_{target_column}"  # 文件名
                )
                # 保存图片
                self._save_plot(fig, plot_path)

    def plot_per_battery_cleaned(
        self, df_cleaned: pd.DataFrame, output_dir: str, target_column: str = "target"
    ):
        """
        为每个 battery_id 绘制清洗后的数据图.

        Args:
            df_cleaned (pd.DataFrame): 清洗后的数据.
            output_dir (str): 图片保存目录.
            target_column (str): 目标列名，默认为 'target'.
        """
        required_cols = ["battery_id", "cycle_idx", target_column]
        if not all(col in df_cleaned.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_cleaned.columns]
            logger.error(f"绘制清洗后数据图失败: 输入 DataFrame 缺少必要列 {missing}")
            return

        rc_params_dict = self.style_setter.get_rc_params_dict()
        df_cleaned["battery_id"] = df_cleaned["battery_id"].astype(str)
        valid_battery_ids = df_cleaned[df_cleaned["battery_id"] != "nan"]["battery_id"].unique()

        if not valid_battery_ids.size > 0:
            logger.warning("没有有效的 battery_id 可用于绘制清洗后数据图.")
            return

        for bid in valid_battery_ids:
            df_bid = df_cleaned[df_cleaned["battery_id"] == bid].copy()

            with plt.style.context(rc_params_dict):
                fig, ax = plt.subplots()

                ax.plot(
                    df_bid["cycle_idx"],
                    df_bid[target_column],
                    alpha=0.7,
                    label="清洗后数据",
                )
                ax.legend()

                self._apply_plot_style(
                    ax, f"电池 {bid} - 清洗后数据", "循环索引 (cycles)", "容量 (Ah)"
                )

                plot_path = os.path.join(output_dir, f"battery_{bid}_cleaned_{target_column}")
                self._save_plot(fig, plot_path)

    def plot_overall_sequence(
        self,
        df: pd.DataFrame,
        title: str,
        output_path: str,
        target_column: str = "target",
        hue: str = "battery_id",
    ):
        """
        绘制总体序列图.

        Args:
            df (pd.DataFrame): 数据.
            title (str): 图表标题.
            output_path (str): 保存路径.
            target_column (str): 目标列名，用于 Y 轴，默认为 'target'.
            hue (str): 分组列，默认为 'battery_id'.
        """
        required_cols = ["cycle_idx", target_column, hue]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"绘制总体序列图失败: 输入 DataFrame 缺少必要列 {missing}")
            return

        df_to_plot = df.copy()
        if hue in df_to_plot.columns:
            df_to_plot[hue] = df_to_plot[hue].astype(str)
            df_to_plot = df_to_plot[df_to_plot[hue] != "nan"].copy()
        else:
            logger.warning(f"总体序列图的 hue 列 '{hue}' 不存在，将不进行分组绘制.")
            hue = None

        if df_to_plot.empty and hue is not None:
            logger.warning(f"过滤无效 '{hue}' 值后数据为空，无法绘制总体序列图.")
            return

        rc_params_dict = self.style_setter.get_rc_params_dict()

        with plt.style.context(rc_params_dict):
            fig, ax = plt.subplots()

            num_batteries_to_plot = df_to_plot[hue].nunique() if hue else 1
            palette = (
                sns.color_palette("tab20", num_batteries_to_plot)
                if num_batteries_to_plot <= 20
                else sns.color_palette("hsv", num_batteries_to_plot)
            )

            sns.lineplot(
                data=df_to_plot,
                x="cycle_idx",
                y=target_column,
                hue=hue,
                palette=palette if hue else None,
                marker="o",
                markersize=self.style.get("interpolated_markersize", 50) / 5,
                alpha=0.7,
                ax=ax,
                legend="full" if hue else False,
            )

            self._apply_plot_style(ax, title, "循环索引 (cycles)", "容量 (Ah)")

            self._save_plot(fig, output_path)

    def plot_per_battery_imputed(
        self,
        df_original: pd.DataFrame,
        df_imputed: pd.DataFrame,
        output_dir: str,
        original_column: str,
        imputed_column: str,
    ):
        """
        为每个 battery_id 绘制插值前与插值后的数据对比图.

        Args:
            df_original (pd.DataFrame): 插值前的数据 DataFrame.
            df_imputed (pd.DataFrame): 插值后的数据 DataFrame.
            output_dir (str): 图片保存目录.
            original_column (str): 插值前数据列名.
            imputed_column (str): 插值后数据列名.
        """
        required_cols_original = ["battery_id", "cycle_idx", original_column]
        required_cols_imputed = ["battery_id", "cycle_idx", imputed_column]

        if not all(col in df_original.columns for col in required_cols_original):
            missing = [col for col in required_cols_original if col not in df_original.columns]
            logger.error(f"绘制插值对比图失败: 原始 DataFrame 缺少必要列 {missing}")
            return
        if not all(col in df_imputed.columns for col in required_cols_imputed):
            missing = [col for col in required_cols_imputed if col not in df_imputed.columns]
            logger.error(f"绘制插值对比图失败: 插值后 DataFrame 缺少必要列 {missing}")
            return

        rc_params_dict = self.style_setter.get_rc_params_dict()

        df_original["battery_id"] = df_original["battery_id"].astype(str)
        df_imputed["battery_id"] = df_imputed["battery_id"].astype(str)

        valid_battery_ids_original = df_original[df_original["battery_id"] != "nan"][
            "battery_id"
        ].unique()
        valid_battery_ids_imputed = df_imputed[df_imputed["battery_id"] != "nan"][
            "battery_id"
        ].unique()

        valid_battery_ids = list(set(valid_battery_ids_original) & set(valid_battery_ids_imputed))

        if not valid_battery_ids:
            logger.warning("没有共同的有效 battery_id 可用于绘制插值对比图.")
            return

        for bid in valid_battery_ids:
            df_bid_original = df_original[df_original["battery_id"] == bid].copy()
            df_bid_imputed = df_imputed[df_imputed["battery_id"] == bid].copy()

            # 确定插值点：在 df_bid_imputed 中存在，但在 df_bid_original 中是 NaN
            # 需要根据 cycle_idx 对齐两个 DataFrame 进行比较
            merged_df = pd.merge(
                df_bid_original[["cycle_idx", original_column]],
                df_bid_imputed[["cycle_idx", imputed_column]],
                on="cycle_idx",
                how="inner",
            )

            df_interpolated_points = merged_df[
                merged_df[original_column].isna() & merged_df[imputed_column].notna()
            ].copy()

            with plt.style.context(rc_params_dict):
                fig, ax = plt.subplots()

                ax.plot(
                    df_bid_original["cycle_idx"],
                    df_bid_original[original_column],
                    alpha=0.7,
                    label=f"{original_column} (插值前)",
                    zorder=1,
                )

                ax.plot(
                    df_bid_imputed["cycle_idx"],
                    df_bid_imputed[imputed_column],
                    alpha=0.7,
                    label=f"{imputed_column} (插值后)",
                    zorder=2,
                )

                if not df_interpolated_points.empty:
                    ax.scatter(
                        df_interpolated_points["cycle_idx"],
                        df_interpolated_points[imputed_column],
                        color=self.style.get("interpolated_color", "#d64f38"),
                        marker=self.style.get("interpolated_mark_type", "o"),
                        s=self.style.get("interpolated_markersize", 50),
                        label="插值生成值",
                        alpha=0.8,
                        zorder=3,
                    )

                ax.legend()
                self._apply_plot_style(
                    ax,
                    f"电池 {bid} - 插值对比",
                    "循环索引 (cycles)",
                    "容量 (Ah)",
                )

                plot_path = os.path.join(
                    output_dir,
                    f"battery_{bid}_imputed_compare_{original_column}_{imputed_column}",
                )
                self._save_plot(fig, plot_path)

    def plot_per_battery_denoised(
        self,
        df_original: pd.DataFrame,
        df_denoised: pd.DataFrame,
        output_dir: str,
        original_column: str,
        denoised_column: str,
    ):
        """
        为每个 battery_id 绘制去噪前与去噪后的数据对比图.

        Args:
            df_original (pd.DataFrame): 去噪前的数据 DataFrame.
            df_denoised (pd.DataFrame): 去噪后的数据 DataFrame.
            output_dir (str): 图片保存目录.
            original_column (str): 去噪前数据列名.
            denoised_column (str): 去噪后数据列名.
        """
        required_cols_original = ["battery_id", "cycle_idx", original_column]
        required_cols_denoised = ["battery_id", "cycle_idx", denoised_column]

        if not all(col in df_original.columns for col in required_cols_original):
            missing = [col for col in required_cols_original if col not in df_original.columns]
            logger.error(f"绘制去噪对比图失败: 原始 DataFrame 缺少必要列 {missing}")
            return
        if not all(col in df_denoised.columns for col in required_cols_denoised):
            missing = [col for col in required_cols_denoised if col not in df_denoised.columns]
            logger.error(f"绘制去噪对比图失败: 去噪后 DataFrame 缺少必要列 {missing}")
            return

        rc_params_dict = self.style_setter.get_rc_params_dict()

        df_original["battery_id"] = df_original["battery_id"].astype(str)
        df_denoised["battery_id"] = df_denoised["battery_id"].astype(str)

        valid_battery_ids_original = df_original[df_original["battery_id"] != "nan"][
            "battery_id"
        ].unique()
        valid_battery_ids_denoised = df_denoised[df_denoised["battery_id"] != "nan"][
            "battery_id"
        ].unique()

        valid_battery_ids = list(set(valid_battery_ids_original) & set(valid_battery_ids_denoised))

        if not valid_battery_ids:
            logger.warning("没有共同的有效 battery_id 可用于绘制去噪对比图.")
            return

        for bid in valid_battery_ids:
            df_bid_original = df_original[df_original["battery_id"] == bid].copy()
            df_bid_denoised = df_denoised[df_denoised["battery_id"] == bid].copy()

            with plt.style.context(rc_params_dict):
                fig, ax = plt.subplots()

                ax.plot(
                    df_bid_original["cycle_idx"],
                    df_bid_original[original_column],
                    alpha=0.7,
                    label=f"去噪前",
                    zorder=1,
                )
                ax.plot(
                    df_bid_denoised["cycle_idx"],
                    df_bid_denoised[denoised_column],
                    alpha=0.7,
                    label=f"去噪后",
                    zorder=2,
                )

                ax.legend()
                self._apply_plot_style(
                    ax,
                    f"电池 {bid} - 去噪前后对比图",
                    "循环索引 (cycles)",
                    "容量 (Ah)",
                )

                plot_path = os.path.join(
                    output_dir,
                    f"battery_{bid}_denoised_compare_{original_column}_{denoised_column}",
                )
                self._save_plot(fig, plot_path)

    def plot_per_battery_standardized(
        self, df_standardized: pd.DataFrame, output_dir: str, standardized_column: str
    ):
        """
        为每个 battery_id 绘制标准化后的数据折线图.

        Args:
            df_standardized (pd.DataFrame): 包含标准化后数据的 DataFrame.
            output_dir (str): 图片保存目录.
            standardized_column (str): 标准化后的列名.
        """
        required_cols = ["battery_id", "cycle_idx", standardized_column]
        if not all(col in df_standardized.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_standardized.columns]
            logger.error(f"绘制标准化数据图失败: 输入 DataFrame 缺少必要列 {missing}")
            return

        rc_params_dict = self.style_setter.get_rc_params_dict()
        df_standardized["battery_id"] = df_standardized["battery_id"].astype(str)
        valid_battery_ids = df_standardized[df_standardized["battery_id"] != "nan"][
            "battery_id"
        ].unique()

        if not valid_battery_ids.size > 0:
            logger.warning("没有有效的 battery_id 可用于绘制标准化数据图.")
            return

        for bid in valid_battery_ids:
            df_bid = df_standardized[df_standardized["battery_id"] == bid].copy()

            with plt.style.context(rc_params_dict):
                fig, ax = plt.subplots()

                ax.plot(
                    df_bid["cycle_idx"],
                    df_bid[standardized_column] * 100,
                    alpha=0.7,
                    label="标准化后数据",
                )
                ax.legend()

                self._apply_plot_style(
                    ax,
                    f"电池 {bid} - 标准化后数据",
                    "循环索引 (cycles)",
                    f"SOH (%)",
                )

                plot_path = os.path.join(
                    output_dir, f"battery_{bid}_standardized_{standardized_column}"
                )
                self._save_plot(fig, plot_path)

    def plot_per_battery_decomposed(
        self,
        df: pd.DataFrame,
        output_dir: str,
        target_column_before_decompose: str,
        show_modes: bool = False,
    ):
        """
        为每个 battery_id 绘制分解结果图，包含原始数据、趋势、残差和可选的模态.

        Args:
            df (pd.DataFrame): 包含分解前、趋势、残差和模态数据的 DataFrame.
            output_dir (str): 图片保存目录.
            target_column_before_decompose (str): 进行分解的原始列名，用于查找相关列.
            show_modes (bool): 是否绘制所有模态（模态 2 及以后），默认为 False.
        """
        trend_col = f"{target_column_before_decompose}_trend"
        residual_col = f"{target_column_before_decompose}_residual"
        mode_prefix = f"{target_column_before_decompose}_mode_"
        mode1_col = f"{mode_prefix}1"

        required_cols = [
            "battery_id",
            "cycle_idx",
            target_column_before_decompose,
            trend_col,
        ]

        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"绘制趋势分解图失败: 输入 DataFrame 缺少必要列 {missing}")
            return

        has_residual = residual_col in df.columns
        has_mode1 = mode1_col in df.columns
        all_mode_cols = [col for col in df.columns if col.startswith(mode_prefix)]
        remaining_mode_cols = [col for col in all_mode_cols if col != mode1_col]
        has_remaining_modes = len(remaining_mode_cols) > 0

        rc_params_dict = self.style_setter.get_rc_params_dict()
        full_style = self.style_setter.get_style()

        df["battery_id"] = df["battery_id"].astype(str)
        valid_battery_ids = df[df["battery_id"] != "nan"]["battery_id"].unique()

        if not valid_battery_ids.size > 0:
            logger.warning("没有有效的 battery_id 可用于绘制趋势分解图.")
            return

        for bid in valid_battery_ids:
            df_bid = df[df["battery_id"] == bid].copy()

            with plt.style.context(rc_params_dict):
                fig, ax = plt.subplots()
                ax2 = ax.twinx()

                # --- 绘制到主 Axes (ax) ---
                ax.plot(
                    df_bid["cycle_idx"],
                    df_bid[target_column_before_decompose],
                    label=f"{target_column_before_decompose} (分解前)",
                    zorder=1,
                    alpha=0.7,
                )

                ax.plot(
                    df_bid["cycle_idx"],
                    df_bid[trend_col],
                    label="趋势",
                    zorder=full_style.get("trend_zorder", 2),
                    alpha=full_style.get("trend_alpha", 0.7),
                    color=full_style.get("trend_line_color", "orange"),
                    linestyle=full_style.get("trend_linestyle", "-"),
                    linewidth=full_style.get("trend_linewidth", 2.0),
                )

                if has_mode1 and mode1_col in df_bid.columns:
                    ax.plot(
                        df_bid["cycle_idx"],
                        df_bid[mode1_col],
                        label="模态 1",
                        linestyle=full_style.get("mode_linestyle", "--"),
                        linewidth=full_style.get("mode_linewidth", 1.0),
                        color=full_style.get("mode1_line_color", "purple"),
                        zorder=full_style.get("mode_zorder_ax1", 3),
                        alpha=full_style.get("mode_alpha", 0.7),
                    )

                # --- 绘制到第二个 Axes (ax2) ---
                if has_residual and residual_col in df_bid.columns:
                    ax2.plot(
                        df_bid["cycle_idx"],
                        df_bid[residual_col],
                        color=full_style.get("residual_line_color", "gray"),
                        linestyle=full_style.get("residual_linestyle", "--"),
                        linewidth=full_style.get("residual_linewidth", 1),
                        label="残差",
                        zorder=full_style.get("residual_zorder", 3),
                        alpha=full_style.get("residual_alpha", 0.65),
                    )

                if show_modes and has_remaining_modes:
                    remaining_modes_color = full_style.get("remaining_modes_line_color", "teal")
                    remaining_modes_linestyle = full_style.get("remaining_modes_linestyle", "--")
                    remaining_modes_linewidth = full_style.get("remaining_modes_linewidth", 1.0)
                    remaining_modes_zorder = full_style.get("mode_zorder_ax2", 2)
                    remaining_modes_alpha = full_style.get("mode_alpha", 0.7)

                    for i, col in enumerate(remaining_mode_cols):
                        label = "其余模态" if i == 0 else "_nolegend_"
                        ax2.plot(
                            df_bid["cycle_idx"],
                            df_bid[col],
                            linestyle=remaining_modes_linestyle,
                            linewidth=remaining_modes_linewidth,
                            color=remaining_modes_color,
                            label=label,
                            zorder=remaining_modes_zorder,
                            alpha=remaining_modes_alpha,
                        )
                elif show_modes and not has_remaining_modes and has_mode1:
                    logger.warning(f"电池 {bid} 只有模态 1，没有其他模态需要绘制为 '其余模态'")
                elif show_modes and not has_mode1 and has_remaining_modes:
                    all_found_mode_cols = [
                        col for col in df_bid.columns if col.startswith(mode_prefix)
                    ]
                    if all_found_mode_cols:
                        remaining_modes_color = full_style.get("remaining_modes_line_color", "teal")
                        remaining_modes_linestyle = full_style.get(
                            "remaining_modes_linestyle", "--"
                        )
                        remaining_modes_linewidth = full_style.get("remaining_modes_linewidth", 1.0)
                        remaining_modes_zorder = full_style.get("mode_zorder_ax2", 2)
                        remaining_modes_alpha = full_style.get("mode_alpha", 0.7)

                        for i, col in enumerate(all_found_mode_cols):
                            label = "其余模态" if i == 0 else "_nolegend_"
                            ax2.plot(
                                df_bid["cycle_idx"],
                                df_bid[col],
                                linestyle=remaining_modes_linestyle,
                                linewidth=remaining_modes_linewidth,
                                color=remaining_modes_color,
                                label=label,
                                zorder=remaining_modes_zorder,
                                alpha=remaining_modes_alpha,
                            )
                    else:
                        logger.warning(f"电池 {bid} 没有找到任何模态列需要绘制.")
                elif show_modes and not all_mode_cols:
                    logger.warning(f"电池 {bid} 没有找到任何模态列，跳过绘制模态")

                self._apply_plot_style(
                    ax,
                    f"电池 {bid} - 分解结果 ({target_column_before_decompose})",
                    "循环索引 (cycles)",
                    f"{target_column_before_decompose} / 趋势 / 模态 1",
                )

                secondary_ylabel_text = full_style.get("secondary_ylabel", "分解分量值")
                ax2.set_ylabel(
                    secondary_ylabel_text, fontsize=full_style.get("rcParams.axes.labelsize", 20)
                )
                ax2.tick_params(axis="y", labelsize=full_style.get("rcParams.ytick.labelsize", 18))

                ax2_line_color = (
                    full_style.get("residual_line_color", "gray")
                    if has_residual
                    else (
                        full_style.get("remaining_modes_line_color", "teal")
                        if (
                            show_modes
                            and (has_remaining_modes or (not has_mode1 and all_mode_cols))
                        )
                        else "#d64f38"
                    )
                )
                ax2.spines["right"].set_color(ax2_line_color)
                ax2.yaxis.label.set_color(ax2_line_color)
                ax2.tick_params(axis="y", colors=ax2_line_color)
                ax2.spines["right"].set_linewidth(full_style.get("rcParams.axes.linewidth", 1))

                handles1, labels1 = ax.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                combined_handles = handles1 + handles2
                combined_labels = labels1 + labels2

                ax.legend(combined_handles, combined_labels)

                plot_path = os.path.join(
                    output_dir, f"battery_{bid}_decomposed_{target_column_before_decompose}"
                )
                self._save_plot(fig, plot_path)
