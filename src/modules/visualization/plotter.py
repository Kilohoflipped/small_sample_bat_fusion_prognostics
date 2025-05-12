import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any

# 从 StyleSetter 导入 StyleSetter 类
from src.modules.visualization.style_setter import StyleSetter

logger = logging.getLogger(__name__)


class Plotter:
    """
    负责绘制电池数据图表的类.
    使用 StyleSetter 获取风格，并在绘制时临时应用风格。
    """

    def __init__(self, style_setter: StyleSetter, plot_size_cm: Tuple[float, float], dpi: int):
        """
        初始化 Plotter.

        Args:
            style_setter (StyleSetter): 绘图风格设置对象.
            plot_size_cm (tuple): 坐标轴区域大小 (width, height)，单位厘米.
            dpi (int): 图片分辨率.
        """
        if not isinstance(style_setter, StyleSetter):
            raise TypeError("style_setter 必须是 StyleSetter 类的实例")

        self.style_setter = style_setter
        self.style = self.style_setter.get_style()  # 获取风格字典
        self.plot_size_cm = plot_size_cm
        self.dpi = dpi
        self.plot_size_in = (
            self._cm_to_inches(
                plot_size_cm[0]), self._cm_to_inches(
                plot_size_cm[1]))

        # 在这里不设置全局 Matplotlib 参数

    def _cm_to_inches(self, cm: float) -> float:
        """将厘米转换为英寸."""
        return cm / 2.54

    def _apply_plot_style(self, ax: plt.Axes, title: str, xlabel: str,
                          ylabel: str, legend_loc: str = None):
        """
        应用通用绘图风格到 Axes 对象.

        Args:
            ax (matplotlib.axes.Axes): 要应用风格的 Axes 对象.
            title (str): 图表标题.
            xlabel (str): X轴标签.
            ylabel (str): Y轴标签.
            legend_loc (str, optional): 图例位置. 如果为 None, 使用 StyleSetter 中的默认值.
        """
        # 应用从 StyleSetter 获取的风格
        ax.set_title(title, fontsize=self.style.get('title_fontsize', 22), fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=self.style.get('label_fontsize', 20))
        ax.set_ylabel(ylabel, fontsize=self.style.get('label_fontsize', 20))
        ax.tick_params(axis='both', labelsize=self.style.get('tick_fontsize', 18))

        # 设置边框颜色和可见性
        spine_color = self.style.get('spine_color', 'black')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(spine_color)

        # 控制网格可见性
        ax.grid(
            self.style.get(
                'grid_visible', False), color=self.style.get(
                'grid_color', '#cccccc'), alpha=self.style.get(
                'grid_alpha', 0.5))

        # 设置图例风格
        current_legend_loc = legend_loc if legend_loc is not None else self.style.get(
            'legend_loc', 'best')
        if ax.get_legend():
            ax.legend(prop={'size': self.style.get('legend_fontsize', 19)},
                      frameon=self.style.get('legend_frameon', False),
                      loc=current_legend_loc)

        # 设置图形和坐标轴背景颜色
        ax.set_facecolor(self.style.get('axes_facecolor', 'white'))
        ax.figure.patch.set_facecolor(self.style.get('figure_facecolor', 'white'))

    def _save_plot(self, fig: plt.Figure, output_path: str):
        """
        保存图表到指定路径.

        Args:
            fig (matplotlib.figure.Figure): 要保存的 Figure 对象.
            output_path (str): 图片保存路径.
        """
        try:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"已保存图表至: {output_path}")
        except Exception as e:
            logger.error(f"保存图表 '{output_path}' 时发生错误: {e}")
        finally:
            plt.close(fig)  # 确保关闭图表，释放内存

    def plot_per_battery_comparison(self, df_original: pd.DataFrame,
                                    df_with_anomaly: pd.DataFrame, output_dir: str):
        """
        为每个 battery_id 绘制原始数据与异常点的对比图.

        Args:
            df_original (pd.DataFrame): 原始数据.
            df_with_anomaly (pd.DataFrame): 包含异常标记的数据.
            output_dir (str): 图片保存目录.
        """
        required_cols = ['battery_id', 'cycle_idx', 'target']
        if not all(col in df_original.columns for col in required_cols):
            logger.error(f"绘制原始数据与异常点对比图失败: 输入 DataFrame 缺少必要列 {required_cols}")
            return
        if 'anomaly' not in df_with_anomaly.columns:
            logger.error("绘制原始数据与异常点对比图失败: 包含异常标记的 DataFrame 缺少 'anomaly' 列")
            return

        for bid in df_original['battery_id'].unique():
            df_bid = df_original[df_original['battery_id'] == bid].copy()
            df_anomaly_bid = df_with_anomaly[
                (df_with_anomaly['battery_id'] == bid) & (df_with_anomaly['anomaly'] == -1)
            ].copy()

            # 使用 style context 临时应用风格
            with plt.style.context({'font.family': self.style.get('font_family', 'sans-serif'),
                                    'axes.unicode_minus': self.style.get('axes_unicode_minus', False)}):
                fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)

                ax.plot(df_bid['cycle_idx'], df_bid['target'], color=self.style.get('line_color_primary', '#3a80f8'),
                        linestyle='-', linewidth=self.style.get('line_linewidth', 1.5), label='原始数据', alpha=0.6)

                if not df_anomaly_bid.empty:
                    ax.scatter(df_anomaly_bid['cycle_idx'], df_anomaly_bid['target'],
                               color=self.style.get('anomaly_color', '#d64f38'), marker='x',
                               s=self.style.get('anomaly_markersize', 100), label='异常点')

                # 应用通用风格
                self._apply_plot_style(ax,
                                       f'电池 {bid} - 原始数据与异常点',
                                       '循环索引 (cycles)',
                                       '容量 (Ah)')

                plot_path = os.path.join(output_dir, f'battery_{bid}_step_0_0_comparison.png')
                self._save_plot(fig, plot_path)

    def plot_per_battery_cleaned(self, df_cleaned: pd.DataFrame, output_dir: str):
        """
        为每个 battery_id 绘制清洗后的数据图.

        Args:
            df_cleaned (pd.DataFrame): 清洗后的数据.
            output_dir (str): 图片保存目录.
        """
        required_cols = ['battery_id', 'cycle_idx', 'target']
        if not all(col in df_cleaned.columns for col in required_cols):
            logger.error(f"绘制清洗后数据图失败: 输入 DataFrame 缺少必要列 {required_cols}")
            return

        for bid in df_cleaned['battery_id'].unique():
            df_bid = df_cleaned[df_cleaned['battery_id'] == bid].copy()

            with plt.style.context({'font.family': self.style.get('font_family', 'sans-serif'),
                                    'axes.unicode_minus': self.style.get('axes_unicode_minus', False)}):
                fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)

                ax.plot(df_bid['cycle_idx'], df_bid['target'], color=self.style.get('line_color_primary', '#3a80f8'),
                        linestyle='-', linewidth=self.style.get('line_linewidth', 1.5), label='清洗后数据', alpha=0.8)

                # 应用通用风格
                self._apply_plot_style(ax,
                                       f'电池 {bid} - 清洗后数据',
                                       '循环索引 (cycles)',
                                       '容量 (Ah)')

                plot_path = os.path.join(output_dir, f'battery_{bid}_step_0_1_cleaned.png')
                self._save_plot(fig, plot_path)

    def plot_per_battery_raw(self, df: pd.DataFrame, output_dir: str):
        """
        为每个 battery_id 绘制原始数据的折线图.

        Args:
            df (pd.DataFrame): 原始数据，包含 battery_id, cycle_idx 和 target 列.
            output_dir (str): 图片保存目录.
        """
        required_cols = ['battery_id', 'cycle_idx', 'target']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"绘制原始数据图失败: 输入 DataFrame 缺少必要列 {required_cols}")
            return

        for bid in df['battery_id'].unique():
            df_bid = df[df['battery_id'] == bid].copy()

            with plt.style.context({'font.family': self.style.get('font_family', 'sans-serif'),
                                    'axes.unicode_minus': self.style.get('axes_unicode_minus', False)}):
                fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)

                ax.plot(df_bid['cycle_idx'], df_bid['target'], color=self.style.get('line_color_primary', '#3a80f8'),
                        linestyle='-', linewidth=self.style.get('line_linewidth', 1.5), label='原始数据', alpha=0.8)

                # 应用通用风格
                self._apply_plot_style(ax,
                                       f'电池 {bid} - 原始数据',
                                       '循环索引 (cycles)',
                                       '容量 (Ah)')

                plot_path = os.path.join(output_dir, f'battery_{bid}_step_0_0_a_raw.png')
                self._save_plot(fig, plot_path)

    def plot_overall_sequence(self, df: pd.DataFrame, title: str,
                              output_path: str, hue: str = 'battery_id', max_batteries: int = 20):
        """
        绘制总体序列图.

        Args:
            df (pd.DataFrame): 数据.
            title (str): 图表标题.
            output_path (str): 保存路径.
            hue (str): 分组列，默认为 'battery_id'.
            max_batteries (int): 绘制的最大电池数量，避免图表过于拥挤. 默认为 20.
        """
        required_cols = ['cycle_idx', 'target', hue]
        if not all(col in df.columns for col in required_cols):
            logger.error(f"绘制总体序列图失败: 输入 DataFrame 缺少必要列 {required_cols}")
            return

        unique_batteries = df[hue].unique()
        if len(unique_batteries) > max_batteries:
            logger.warning(f"电池数量 ({len(unique_batteries)}) 超过最大绘制数量 ({
                max_batteries}), 仅绘制前 {max_batteries} 个电池")
            batteries_to_plot = unique_batteries[:max_batteries]
            df_to_plot = df[df[hue].isin(batteries_to_plot)].copy()
        else:
            df_to_plot = df.copy()

        with plt.style.context({'font.family': self.style.get('font_family', 'sans-serif'),
                                'axes.unicode_minus': self.style.get('axes_unicode_minus', False)}):
            fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)

            # 使用 seaborn lineplot 绘制，hue 参数用于按电池ID分组
            # palette 可以根据电池数量调整
            num_batteries_to_plot = len(df_to_plot[hue].unique())
            palette = sns.color_palette(
                "tab20", num_batteries_to_plot) if num_batteries_to_plot <= 20 else sns.color_palette(
                "hsv", num_batteries_to_plot)

            sns.lineplot(data=df_to_plot, x='cycle_idx', y='target', hue=hue, palette=palette,
                         # 总体图标记小一些
                         linewidth=self.style.get('line_linewidth', 1.5), marker='o', markersize=self.style.get('scatter_markersize', 50) / 5, alpha=0.7, ax=ax)

            # 应用通用风格
            self._apply_plot_style(ax,
                                   title,
                                   '循环索引 (cycles)',
                                   '容量 (Ah)')

            self._save_plot(fig, output_path)

    def plot_per_battery_imputed(self, df_cleaned: pd.DataFrame,
                                 df_imputed: pd.DataFrame, output_dir: str):
        """
        为每个 battery_id 绘制清洗后与插值后的数据对比图，原始值和插值值用不同颜色区分.

        Args:
            df_cleaned (pd.DataFrame): 清洗后的数据（原始值）.
            df_imputed (pd.DataFrame): 插值后的数据（包含原始值和新生成的插值点）.
            output_dir (str): 图片保存目录.
        """
        required_cols_cleaned = ['battery_id', 'cycle_idx', 'target']
        required_cols_imputed = ['battery_id', 'cycle_idx', 'target']
        if not all(col in df_cleaned.columns for col in required_cols_cleaned):
            logger.error(f"绘制插值对比图失败: 清洗后 DataFrame 缺少必要列 {required_cols_cleaned}")
            return
        if not all(col in df_imputed.columns for col in required_cols_imputed):
            logger.error(f"绘制插值对比图失败: 插值后 DataFrame 缺少必要列 {required_cols_imputed}")
            return

        for bid in df_cleaned['battery_id'].unique():
            df_clean_bid = df_cleaned[df_cleaned['battery_id'] == bid].copy()
            df_impute_bid = df_imputed[df_imputed['battery_id'] == bid].copy()

            # 确保按 cycle_idx 排序
            df_clean_bid = df_clean_bid.sort_values('cycle_idx')
            df_impute_bid = df_impute_bid.sort_values('cycle_idx')

            # 确定原始值和插值值
            # 使用 merge 来找到插值后的数据中哪些 cycle_idx 不在清洗后的数据中
            merged = pd.merge(df_impute_bid[['battery_id', 'cycle_idx', 'target']],
                              df_clean_bid[['battery_id', 'cycle_idx']],
                              on=['battery_id', 'cycle_idx'],
                              how='left',
                              indicator=True)

            df_impute_bid['is_original'] = merged['_merge'] == 'both'

            # 分离原始值和插值值
            df_original_points = df_impute_bid[df_impute_bid['is_original']].copy()
            df_interpolated_points = df_impute_bid[~df_impute_bid['is_original']].copy()

            with plt.style.context({'font.family': self.style.get('font_family', 'sans-serif'),
                                    'axes.unicode_minus': self.style.get('axes_unicode_minus', False)}):
                fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)

                # 绘制所有点的折线图（确保连续性）作为背景
                ax.plot(df_impute_bid['cycle_idx'], df_impute_bid['target'], color=self.style.get('background_line_color', '#d3d3d3'),
                        linestyle='-', linewidth=1.0, alpha=0.5, label='_nolegend_')  # 背景线透明度高一些

                # 绘制原始值（散点）
                if not df_original_points.empty:
                    ax.scatter(df_original_points['cycle_idx'], df_original_points['target'],
                               color=self.style.get('line_color_primary', '#3a80f8'), marker='o',
                               s=self.style.get('scatter_markersize', 50), label='原始值', alpha=0.8)

                # 绘制插值值（散点）
                if not df_interpolated_points.empty:
                    ax.scatter(df_interpolated_points['cycle_idx'], df_interpolated_points['target'],
                               color=self.style.get('interpolated_color', '#d64f38'), marker='o',
                               s=self.style.get('scatter_markersize', 50), label='插值生成值', alpha=0.8)

                # 应用通用风格
                self._apply_plot_style(ax,
                                       f'电池 {bid} - 清洗后值与插值生成值',
                                       '循环索引 (cycles)',
                                       '容量 (Ah)',
                                       legend_loc='lower left')  # 插值图例放左下角可能更好

                # 保存图像
                plot_path = os.path.join(output_dir, f'battery_{bid}_step_1_imputed.png')
                self._save_plot(fig, plot_path)

    def plot_per_battery_denoised(self, df: pd.DataFrame, output_dir: str,
                                  original_column: str = 'target', denoised_column: str = 'target_denoised'):
        """
        为每个 battery_id 绘制原始数据与去噪后数据的对比图.

        Args:
            df (pd.DataFrame): 包含原始和去噪数据的 DataFrame.
            output_dir (str): 图片保存目录.
            original_column (str): 原始数据列名.
            denoised_column (str): 去噪后数据列名.
        """
        required_cols = ['battery_id', 'cycle_idx', original_column, denoised_column]
        if not all(col in df.columns for col in required_cols):
            logger.error(f"绘制去噪对比图失败: 输入 DataFrame 缺少必要列 {required_cols}")
            return

        for bid in df['battery_id'].unique():
            df_bid = df[df['battery_id'] == bid].copy()

            with plt.style.context({'font.family': self.style.get('font_family', 'sans-serif'),
                                    'axes.unicode_minus': self.style.get('axes_unicode_minus', False)}):
                fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)

                ax.plot(df_bid['cycle_idx'], df_bid[original_column], color=self.style.get('line_color_primary', '#3a80f8'),
                        linestyle='-', linewidth=self.style.get('line_linewidth', 1.5), label='原始数据', alpha=0.6)
                ax.plot(df_bid['cycle_idx'], df_bid[denoised_column], color=self.style.get('line_color_secondary', '#d64f38'),
                        linestyle='-', linewidth=self.style.get('line_linewidth', 1.5), label='去噪后数据', alpha=0.8)

                # 应用通用风格
                self._apply_plot_style(ax,
                                       f'电池 {bid} - 原始数据与去噪后数据',
                                       '循环索引 (cycles)',
                                       '容量 (Ah)')

                plot_path = os.path.join(output_dir, f'battery_{bid}_step_2_denoised.png')
                self._save_plot(fig, plot_path)

    def plot_per_battery_decomposed(self, df: pd.DataFrame, output_dir: str,
                                    target_column: str = 'target_denoised', show_modes: bool = False):
        """
        为每个 battery_id 绘制去噪数据与趋势分解结果的对比图，可选显示所有模态.

        Args:
            df (pd.DataFrame): 包含去噪、趋势、残差和所有模态数据的 DataFrame.
            output_dir (str): 图片保存目录.
            target_column (str): 进行分解的原始列名，用于查找趋势、残差和模态列.
            show_modes (bool): 是否绘制所有模态，默认为 False.
        """
        required_cols = ['battery_id', 'cycle_idx', target_column, f'{target_column}_trend']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"绘制趋势分解图失败: 输入 DataFrame 缺少必要列 {required_cols}")
            return

        for bid in df['battery_id'].unique():
            df_bid = df[df['battery_id'] == bid].copy()

            with plt.style.context({'font.family': self.style.get('font_family', 'sans-serif'),
                                    'axes.unicode_minus': self.style.get('axes_unicode_minus', False)}):
                fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)

                # 绘制去噪后数据
                ax.plot(df_bid['cycle_idx'], df_bid[target_column], color=self.style.get('line_color_primary', '#3a80f8'),
                        linestyle='-', linewidth=self.style.get('line_linewidth', 1.5), label='去噪后数据', alpha=0.6)

                # 绘制趋势
                ax.plot(df_bid['cycle_idx'], df_bid[f'{target_column}_trend'], color=self.style.get('line_color_secondary', '#d64f38'),
                        linestyle='-', linewidth=self.style.get('line_linewidth', 1.5), label='趋势', alpha=0.8)

                # 绘制残差 (可选，如果需要可以添加)
                # if f'{target_column}_residual' in df_bid.columns:
                #      ax.plot(df_bid['cycle_idx'], df_bid[f'{target_column}_residual'], color='gray', linestyle='--',
                #              linewidth=1.0, label='残差', alpha=0.5)

                if show_modes:
                    # 绘制所有模态，从 DataFrame 中读取已计算好的模态列
                    mode_cols = [
                        col for col in df_bid.columns if col.startswith(
                            f'{target_column}_mode_')]
                    if mode_cols:
                        # 使用不同的颜色和样式绘制模态
                        colors = sns.color_palette("tab10", len(mode_cols))  # 使用 seaborn 调色板
                        for i, col in enumerate(mode_cols):
                            ax.plot(df_bid['cycle_idx'], df_bid[col], linestyle='--', linewidth=1,
                                    color=colors[i % len(colors)], label=f'模态 {i + 1}', alpha=0.7)
                    else:
                        logger.warning(f"电池 {bid} 没有找到模态列，跳过绘制模态")

                # 应用通用风格
                self._apply_plot_style(ax,
                                       f'电池 {bid} - {target_column} 与趋势分解',  # 修改标题以反映分解内容
                                       '循环索引 (cycles)',
                                       '容量 (Ah)')

                plot_path = os.path.join(output_dir, f'battery_{bid}_step_3_decomposed.png')
                self._save_plot(fig, plot_path)
