import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np # 导入 numpy

class Plotter:
    """
    负责绘制电池数据图表的类。
    修改后使用 StyleSetter 获取风格，并抽象通用绘图设置。
    """

    def __init__(self, style_setter, plot_size_cm, dpi):
        """
        初始化 Plotter。

        Args:
            style_setter (StyleSetter): 绘图风格设置对象。
            plot_size_cm (tuple): 坐标轴区域大小 (width, height)，单位厘米。
            dpi (int): 图片分辨率。
        """
        # 直接从 StyleSetter 获取风格字典
        self.style = style_setter.get_style()
        self.plot_size_cm = plot_size_cm
        self.dpi = dpi
        self.plot_size_in = (self._cm_to_inches(plot_size_cm[0]), self._cm_to_inches(plot_size_cm[1]))

        # 设置 Matplotlib 字体和 unicode 支持，仅在 Plotter 实例化时设置一次
        # 注意：这仍然会影响全局，但比 StyleSetter 里每次实例化都设置要好。
        # 更好的方法是使用 plt.style.context() 在每个图表绘制时临时应用风格。
        # 为了简化，这里暂时保留一次性设置，但理想情况下应使用 context。
        plt.rcParams['font.family'] = self.style.get('font_family', 'sans-serif')
        plt.rcParams['axes.unicode_minus'] = self.style.get('axes_unicode_minus', True)


    def _cm_to_inches(self, cm):
        """将厘米转换为英寸。"""
        return cm / 2.54

    def _apply_plot_style(self, ax, title, xlabel, ylabel, legend_loc='best'):
        """
        应用通用绘图风格到 Axes 对象。

        Args:
            ax (matplotlib.axes.Axes): 要应用风格的 Axes 对象。
            title (str): 图表标题。
            xlabel (str): X轴标签。
            ylabel (str): Y轴标签。
            legend_loc (str): 图例位置，默认为 'best'。
        """
        ax.set_title(title, fontsize=self.style['title_fontsize'], fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=self.style['label_fontsize'])
        ax.set_ylabel(ylabel, fontsize=self.style['label_fontsize'])
        ax.tick_params(axis='both', labelsize=self.style['tick_fontsize'])

        # 设置边框颜色和可见性
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(self.style.get('spine_color', 'black'))

        # 控制网格可见性，这里选择不显示网格
        ax.grid(False)

        # 设置图例风格
        if ax.get_legend(): # 只有当存在图例时才设置
             ax.legend(prop={'size': self.style['legend_fontsize']}, frameon=False, loc=legend_loc)

        # 确保布局紧凑
        plt.tight_layout()


    def plot_per_battery_comparison(self, df, df_with_anomaly, output_dir):
        """
        为每个 battery_id 绘制原始数据与异常点的对比图。
        不再负责创建目录。

        Args:
            df (pd.DataFrame): 原始数据。
            df_with_anomaly (pd.DataFrame): 包含异常标记的数据。
            output_dir (str): 图片保存目录。
        """
        # 目录创建由调用者负责
        for bid in df['battery_id'].unique():
            df_bid = df[df['battery_id'] == bid]
            df_anomaly_bid = df_with_anomaly[
                (df_with_anomaly['battery_id'] == bid) & (df_with_anomaly['anomaly'] == -1)
                ]
            fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)

            ax.plot(df_bid['cycle_idx'], df_bid['target'], color=self.style['line_color_primary'], linestyle='-',
                    linewidth=1.5, label='原始数据', alpha=0.6)

            if not df_anomaly_bid.empty:
                ax.scatter(df_anomaly_bid['cycle_idx'], df_anomaly_bid['target'],
                           color=self.style['anomaly_color'], marker='x', s=100, label='异常点')

            # 应用通用风格
            self._apply_plot_style(ax,
                                   f'电池 {bid} - 原始数据与异常点',
                                   '循环索引 (cycles)',
                                   '容量 (Ah)')

            plot_path = os.path.join(output_dir, f'battery_{bid}_step_0_0_comparison.png')
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"已保存电池 {bid} 对比图至: {plot_path}")

    def plot_per_battery_cleaned(self, df_cleaned, output_dir):
        """
        为每个 battery_id 绘制清洗后的数据图。
        不再负责创建目录。

        Args:
            df_cleaned (pd.DataFrame): 清洗后的数据。
            output_dir (str): 图片保存目录。
        """
        # 目录创建由调用者负责
        for bid in df_cleaned['battery_id'].unique():
            df_bid = df_cleaned[df_cleaned['battery_id'] == bid]
            fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)

            ax.plot(df_bid['cycle_idx'], df_bid['target'], color=self.style['line_color_primary'], linestyle='-',
                    linewidth=1.5, label='清洗后数据', alpha=0.8)

            # 应用通用风格
            self._apply_plot_style(ax,
                                   f'电池 {bid} - 清洗后数据',
                                   '循环索引 (cycles)',
                                   '容量 (Ah)')

            plot_path = os.path.join(output_dir, f'battery_{bid}_step_0_1_cleaned.png')
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"已保存电池 {bid} 清洗后序列图至: {plot_path}")

    def plot_per_battery_raw(self, df, output_dir):
        """
        为每个 battery_id 绘制原始数据的折线图。
        不再负责创建目录。

        Args:
            df (pd.DataFrame): 原始数据，包含 battery_id, cycle_idx 和 target 列。
            output_dir (str): 图片保存目录。
        """
        # 目录创建由调用者负责
        for bid in df['battery_id'].unique():
            df_bid = df[df['battery_id'] == bid]
            fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)

            ax.plot(df_bid['cycle_idx'], df_bid['target'], color=self.style['line_color_primary'], linestyle='-',
                    linewidth=1.5, label='原始数据', alpha=0.8)

            # 应用通用风格
            self._apply_plot_style(ax,
                                   f'电池 {bid} - 原始数据',
                                   '循环索引 (cycles)',
                                   '容量 (Ah)')

            plot_path = os.path.join(output_dir, f'battery_{bid}_step_0_0_a_raw.png')
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"已保存电池 {bid} 原始数据图至: {plot_path}")

    def plot_overall_sequence(self, df, title, output_path, hue='battery_id'):
        """
        绘制总体序列图。
        不再负责创建目录。

        Args:
            df (pd.DataFrame): 数据。
            title (str): 图表标题。
            output_path (str): 保存路径。
            hue (str): 分组列，默认为 'battery_id'。
        """
        fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)

        # 注意：当电池数量很多时，此图可能变得非常拥挤和难以辨认。
        # 考虑限制绘制的电池数量或使用其他可视化方法。
        sns.lineplot(data=df, x='cycle_idx', y='target', hue=hue, palette='tab20',
                     linewidth=1.5, marker='o', alpha=0.7, ax=ax)

        # 应用通用风格
        self._apply_plot_style(ax,
                               title,
                               '循环索引 (cycles)',
                               '容量 (Ah)')

        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"已保存序列图至: {output_path}")

    def plot_per_battery_imputed(self, df_cleaned, df_imputed, output_dir):
        """
        为每个 battery_id 绘制清洗后与插值后的数据对比图，原始值和插值值用不同颜色区分。
        不再负责创建目录。

        Args:
            df_cleaned (pd.DataFrame): 清洗后的数据（原始值）。
            df_imputed (pd.DataFrame): 插值后的数据（包含原始值和新生成的插值点）。
            output_dir (str): 图片保存目录。
        """
        # 目录创建由调用者负责
        for bid in df_cleaned['battery_id'].unique():
            # 获取当前电池的清洗后和插值后数据
            df_clean_bid = df_cleaned[df_cleaned['battery_id'] == bid].copy()
            df_impute_bid = df_imputed[df_imputed['battery_id'] == bid].copy()

            # 确保按 cycle_idx 排序
            df_clean_bid = df_clean_bid.sort_values('cycle_idx')
            df_impute_bid = df_impute_bid.sort_values('cycle_idx')

            # 确定原始值和插值值 - 更加清晰的逻辑
            # 使用 merge 来找到插值后的数据中哪些 cycle_idx 不在清洗后的数据中
            merged = pd.merge(df_impute_bid[['battery_id', 'cycle_idx', 'target']],
                              df_clean_bid[['battery_id', 'cycle_idx']],
                              on=['battery_id', 'cycle_idx'],
                              how='left',
                              indicator=True)

            df_impute_bid['is_original'] = merged['_merge'] == 'both'

            # 分离原始值和插值值
            df_original = df_impute_bid[df_impute_bid['is_original']].copy()
            df_interpolated = df_impute_bid[~df_impute_bid['is_original']].copy()

            # 创建画布
            fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)

            # 绘制所有点的折线图（确保连续性）作为背景
            ax.plot(df_impute_bid['cycle_idx'], df_impute_bid['target'], color=self.style['background_line_color'],
                    linestyle='-', linewidth=1.0, alpha=0.3, label='_nolegend_')  # 背景线

            # 绘制原始值（散点+线）
            if not df_original.empty:
                ax.plot(df_original['cycle_idx'], df_original['target'],
                        color=self.style['line_color_primary'], linestyle='-', label='原始值', alpha=0.8)
                # 同时绘制散点以更清晰地标记原始点
                ax.scatter(df_original['cycle_idx'], df_original['target'],
                           color=self.style['line_color_primary'], marker='o', s=30, alpha=0.8, label='_nolegend_')


            # 绘制插值值（散点）
            if not df_interpolated.empty:
                ax.scatter(df_interpolated['cycle_idx'], df_interpolated['target'],
                           color=self.style['interpolated_color'], marker='o', s=50, label='插值生成值', alpha=0.8)

            # 应用通用风格
            self._apply_plot_style(ax,
                                   f'电池 {bid} - 清洗后值与插值生成值',
                                   '循环索引 (cycles)',
                                   '容量 (Ah)',
                                   legend_loc='lower left') # 插值图例放左下角可能更好

            # 保存图像
            plot_path = os.path.join(output_dir, f'battery_{bid}_step_1_imputed.png')
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"已保存电池 {bid} 插值对比图至: {plot_path}")


    def plot_per_battery_denoised(self, df, output_dir):
        """
        为每个 battery_id 绘制插值后数据与去噪后数据的对比图。
        不再负责创建目录。

        Args:
            df (pd.DataFrame): 包含插值和去噪数据的 DataFrame。
            output_dir (str): 图片保存目录。
        """
        # 目录创建由调用者负责
        for bid in df['battery_id'].unique():
            df_bid = df[df['battery_id'] == bid]
            fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)

            ax.plot(df_bid['cycle_idx'], df_bid['target'], color=self.style['line_color_primary'], linestyle='-',
                    linewidth=1.5, label='插值后数据', alpha=0.6)
            ax.plot(df_bid['cycle_idx'], df_bid['target_denoised'], color=self.style['line_color_secondary'], linestyle='-',
                    linewidth=1.5, label='去噪后数据', alpha=0.8)

            # 应用通用风格
            self._apply_plot_style(ax,
                                   f'电池 {bid} - 插值后数据与去噪后数据',
                                   '循环索引 (cycles)',
                                   '容量 (Ah)')

            plot_path = os.path.join(output_dir, f'battery_{bid}_step_2_denoised.png')
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"已保存电池 {bid} 去噪对比图至: {plot_path}")

    def plot_per_battery_decomposed(self, df, output_dir, show_modes=False, target_column='target_denoised'):
        """
        为每个 battery_id 绘制去噪数据与趋势分解结果的对比图，可选显示所有模态。
        不再在绘图方法内重新计算 VMD。

        Args:
            df (pd.DataFrame): 包含去噪、趋势、残差和所有模态数据的 DataFrame。
            output_dir (str): 图片保存目录。
            show_modes (bool): 是否绘制所有模态，默认为 False。
            target_column (str): 进行分解的原始列名，用于查找模态列。
        """
        # 目录创建由调用者负责
        for bid in df['battery_id'].unique():
            df_bid = df[df['battery_id'] == bid]
            fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)

            # 绘制去噪后数据
            ax.plot(df_bid['cycle_idx'], df_bid[target_column], color=self.style['line_color_primary'], linestyle='-',
                    linewidth=1.5, label='去噪后数据', alpha=0.6)

            # 绘制趋势
            ax.plot(df_bid['cycle_idx'], df_bid[f'{target_column}_trend'], color=self.style['line_color_secondary'], linestyle='-',
                    linewidth=1.5, label='趋势', alpha=0.8)

            # 绘制残差 (可选，如果需要可以添加)
            # ax.plot(df_bid['cycle_idx'], df_bid[f'{target_column}_residual'], color='gray', linestyle='--',
            #         linewidth=1.0, label='残差', alpha=0.5)


            if show_modes:
                # 绘制所有模态，从 DataFrame 中读取已计算好的模态列
                mode_cols = [col for col in df_bid.columns if col.startswith(f'{target_column}_mode_')]
                # 使用不同的颜色和样式绘制模态
                colors = sns.color_palette("tab10", len(mode_cols)) # 使用 seaborn 调色板
                for i, col in enumerate(mode_cols):
                    ax.plot(df_bid['cycle_idx'], df_bid[col], linestyle='--', linewidth=1,
                            color=colors[i], label=f'模态 {i + 1}', alpha=0.7)

            # 应用通用风格
            self._apply_plot_style(ax,
                                   f'电池 {bid} - 去噪后数据与趋势分解', # 修改标题以反映分解内容
                                   '循环索引 (cycles)',
                                   '容量 (Ah)')

            plot_path = os.path.join(output_dir, f'battery_{bid}_step_3_decomposed.png')
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"已保存电池 {bid} 趋势分解图至: {plot_path}")
