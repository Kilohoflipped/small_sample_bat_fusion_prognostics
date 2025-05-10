import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Plotter:
    """负责绘制电池数据图表的类。"""

    def __init__(self, style_setter, plot_size_cm, dpi):
        """
        初始化 Plotter。

        Args:
            style_setter (StyleSetter): 绘图风格设置对象。
            plot_size_cm (tuple): 坐标轴区域大小 (width, height)，单位厘米。
            dpi (int): 图片分辨率。
        """
        self.style = style_setter.get_style()
        self.plot_size_cm = plot_size_cm
        self.dpi = dpi
        self.plot_size_in = (self._cm_to_inches(plot_size_cm[0]), self._cm_to_inches(plot_size_cm[1]))

    def _cm_to_inches(self, cm):
        """将厘米转换为英寸。"""
        return cm / 2.54

    def plot_per_battery_comparison(self, df, df_with_anomaly, output_dir):
        """
        为每个 battery_id 绘制原始数据与异常点的对比图。

        Args:
            df (pd.DataFrame): 原始数据。
            df_with_anomaly (pd.DataFrame): 包含异常标记的数据。
            output_dir (str): 图片保存目录。
        """
        os.makedirs(output_dir, exist_ok=True)
        for bid in df['battery_id'].unique():
            df_bid = df[df['battery_id'] == bid]
            df_anomaly_bid = df_with_anomaly[
                (df_with_anomaly['battery_id'] == bid) & (df_with_anomaly['anomaly'] == -1)
                ]
            fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)
            ax.plot(df_bid['cycle_idx'], df_bid['target'], color='#3a80f8', linestyle='-',
                    linewidth=1.5, label='原始数据', alpha=0.6)
            if not df_anomaly_bid.empty:
                ax.scatter(df_anomaly_bid['cycle_idx'], df_anomaly_bid['target'],
                           color='#d64f38', marker='x', s=100, label='异常点')
            ax.set_title(f'电池 {bid} - 原始数据与异常点', fontsize=self.style['title_fontsize'],
                         fontweight='bold')
            ax.set_xlabel('循环索引 (cycles)', fontsize=self.style['label_fontsize'])
            ax.set_ylabel('容量 (Ah)', fontsize=self.style['label_fontsize'])
            ax.tick_params(axis='both', labelsize=self.style['tick_fontsize'])
            ax.legend(prop={'size': self.style['legend_fontsize']}, frameon=False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
            ax.grid(False)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'battery_{bid}_step_0_0_comparison.png')
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"已保存电池 {bid} 对比图至: {plot_path}")

    def plot_per_battery_cleaned(self, df_cleaned, output_dir):
        """
        为每个 battery_id 绘制清洗后的数据图。

        Args:
            df_cleaned (pd.DataFrame): 清洗后的数据。
            output_dir (str): 图片保存目录。
        """
        os.makedirs(output_dir, exist_ok=True)
        for bid in df_cleaned['battery_id'].unique():
            df_bid = df_cleaned[df_cleaned['battery_id'] == bid]
            fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)
            ax.plot(df_bid['cycle_idx'], df_bid['target'], color='#3a80f8', linestyle='-',
                    linewidth=1.5, label='清洗后数据', alpha=0.8)
            ax.set_title(f'电池 {bid} - 清洗后数据', fontsize=self.style['title_fontsize'],
                         fontweight='bold')
            ax.set_xlabel('循环索引 (cycles)', fontsize=self.style['label_fontsize'])
            ax.set_ylabel('容量 (Ah)', fontsize=self.style['label_fontsize'])
            ax.tick_params(axis='both', labelsize=self.style['tick_fontsize'])
            ax.legend(prop={'size': self.style['legend_fontsize']}, frameon=False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
            ax.grid(False)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'battery_{bid}_step_0_1_cleaned.png')
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"已保存电池 {bid} 清洗后序列图至: {plot_path}")

    def plot_per_battery_raw(self, df, output_dir):
        """
        为每个 battery_id 绘制原始数据的折线图。

        Args:
            df (pd.DataFrame): 原始数据，包含 battery_id, cycle_idx 和 target 列。
            output_dir (str): 图片保存目录。
        """
        os.makedirs(output_dir, exist_ok=True)
        for bid in df['battery_id'].unique():
            df_bid = df[df['battery_id'] == bid]
            fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)
            ax.plot(df_bid['cycle_idx'], df_bid['target'], color='#3a80f8', linestyle='-',
                    linewidth=1.5, label='原始数据', alpha=0.8)
            ax.set_title(f'电池 {bid} - 原始数据', fontsize=self.style['title_fontsize'],
                         fontweight='bold')
            ax.set_xlabel('循环索引 (cycles)', fontsize=self.style['label_fontsize'])
            ax.set_ylabel('容量 (Ah)', fontsize=self.style['label_fontsize'])
            ax.tick_params(axis='both', labelsize=self.style['tick_fontsize'])
            ax.legend(prop={'size': self.style['legend_fontsize']}, frameon=False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
            ax.grid(False)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'battery_{bid}_step_0_0_a_raw.png')
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"已保存电池 {bid} 原始数据图至: {plot_path}")
    def plot_overall_sequence(self, df, title, output_path, hue='battery_id'):
        """
        绘制总体序列图。

        Args:
            df (pd.DataFrame): 数据。
            title (str): 图表标题。
            output_path (str): 保存路径。
            hue (str): 分组列，默认为 'battery_id'。
        """
        fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)
        sns.lineplot(data=df, x='cycle_idx', y='target', hue=hue, palette='tab20',
                     linewidth=1.5, marker='o', alpha=0.7, ax=ax)
        ax.set_title(title, fontsize=self.style['title_fontsize'], fontweight='bold')
        ax.set_xlabel('循环索引 (cycles)', fontsize=self.style['label_fontsize'])
        ax.set_ylabel('容量 (Ah)', fontsize=self.style['label_fontsize'])
        ax.tick_params(axis='both', labelsize=self.style['tick_fontsize'])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"已保存序列图至: {output_path}")

    def plot_per_battery_imputed(self, df_cleaned, df_imputed, output_dir):
        """
        为每个 battery_id 绘制清洗后与插值后的数据对比图，原始值和插值值用不同颜色区分。

        Args:
            df_cleaned (pd.DataFrame): 清洗后的数据（原始值）。
            df_imputed (pd.DataFrame): 插值后的数据（包含原始值和新生成的插值点）。
            output_dir (str): 图片保存目录。
        """
        os.makedirs(output_dir, exist_ok=True)
        for bid in df_cleaned['battery_id'].unique():
            # 获取当前电池的清洗后和插值后数据
            df_clean_bid = df_cleaned[df_cleaned['battery_id'] == bid].copy()
            df_impute_bid = df_imputed[df_imputed['battery_id'] == bid].copy()

            # 确保按 cycle_idx 排序
            df_clean_bid = df_clean_bid.sort_values('cycle_idx')
            df_impute_bid = df_impute_bid.sort_values('cycle_idx')

            # 确定原始值和插值值
            # 原始值：df_clean_bid 中的 cycle_idx 对应的 target
            # 插值值：df_impute_bid 中不在 df_clean_bid 的 cycle_idx
            original_idxs = set(df_clean_bid['cycle_idx'])
            df_impute_bid['is_original'] = df_impute_bid['cycle_idx'].apply(
                lambda x: x in original_idxs
            )

            # 分离原始值和插值值
            df_original = df_impute_bid[df_impute_bid['is_original']].copy()
            df_interpolated = df_impute_bid[~df_impute_bid['is_original']].copy()

            # 创建画布
            fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)

            # 绘制所有点的折线图（确保连续性）
            ax.plot(df_impute_bid['cycle_idx'], df_impute_bid['target'], color='#d3d3d3',
                    linestyle='-', linewidth=1.0, alpha=0.3, label='_nolegend_')  # 背景线

            # 绘制原始值（散点+线）
            if not df_original.empty:
                ax.plot(df_original['cycle_idx'], df_original['target'],
                           color='#3a80f8', linestyle='-', label='原始值', alpha=0.8)

            # 绘制插值值（散点+线）
            if not df_interpolated.empty:
                ax.scatter(df_interpolated['cycle_idx'], df_interpolated['target'],
                           color='#d64f38', marker='o', s=50, label='插值生成值', alpha=0.8)

            # 设置图表属性
            ax.set_title(f'电池 {bid} - 清洗后值与插值生成值', fontsize=self.style['title_fontsize'],
                         fontweight='bold')
            ax.set_xlabel('循环索引 (cycles)', fontsize=self.style['label_fontsize'])
            ax.set_ylabel('容量 (Ah)', fontsize=self.style['label_fontsize'])
            ax.tick_params(axis='both', labelsize=self.style['tick_fontsize'])
            ax.legend(prop={'size': self.style['legend_fontsize']}, frameon=False, loc='lower left')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
            ax.grid(False)
            plt.tight_layout()

            # 保存图像
            plot_path = os.path.join(output_dir, f'battery_{bid}_step_1_imputed.png')
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"已保存电池 {bid} 插值对比图至: {plot_path}")

    def plot_per_battery_denoised(self, df, output_dir):
        """
        为每个 battery_id 绘制原始数据与去噪后数据的对比图。

        Args:
            df (pd.DataFrame): 包含原始和去噪数据的 DataFrame。
            output_dir (str): 图片保存目录。
        """
        os.makedirs(output_dir, exist_ok=True)
        for bid in df['battery_id'].unique():
            df_bid = df[df['battery_id'] == bid]
            fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)
            ax.plot(df_bid['cycle_idx'], df_bid['target'], color='#3a80f8', linestyle='-',
                    linewidth=1.5, label='插值后数据', alpha=0.6)
            ax.plot(df_bid['cycle_idx'], df_bid['target_denoised'], color='#d64f38', linestyle='-',
                    linewidth=1.5, label='去噪后数据', alpha=0.8)
            ax.set_title(f'电池 {bid} - 插值后数据与去噪后数据', fontsize=self.style['title_fontsize'],
                         fontweight='bold')
            ax.set_xlabel('循环索引 (cycles)', fontsize=self.style['label_fontsize'])
            ax.set_ylabel('容量 (Ah)', fontsize=self.style['label_fontsize'])
            ax.tick_params(axis='both', labelsize=self.style['tick_fontsize'])
            ax.legend(prop={'size': self.style['legend_fontsize']}, frameon=False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
            ax.grid(False)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'battery_{bid}_step_2_denoised.png')
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"已保存电池 {bid} 去噪对比图至: {plot_path}")

    def plot_per_battery_decomposed(self, df, output_dir, show_modes=False):
        """
        为每个 battery_id 绘制去噪数据与趋势分解结果的对比图，可选显示所有模态。

        Args:
            df (pd.DataFrame): 包含去噪和趋势分解数据的 DataFrame。
            output_dir (str): 图片保存目录。
            show_modes (bool): 是否绘制所有模态，默认为 False。
        """
        os.makedirs(output_dir, exist_ok=True)
        for bid in df['battery_id'].unique():
            df_bid = df[df['battery_id'] == bid]
            fig, ax = plt.subplots(figsize=self.plot_size_in, dpi=self.dpi)
            ax.plot(df_bid['cycle_idx'], df_bid['target_denoised'], color='#3a80f8', linestyle='-',
                    linewidth=1.5, label='去噪后数据', alpha=0.6)
            ax.plot(df_bid['cycle_idx'], df_bid['target_trend'], color='#d64f38', linestyle='-',
                    linewidth=1.5, label='趋势', alpha=0.8)
            if show_modes:
                # 重新分解以获取所有模态（仅用于可视化）
                from sktime.transformations.series.vmd import VmdTransformer
                signal = df_bid['target_denoised'].values
                transformer = VmdTransformer(K=5, alpha=1000, returned_decomp="u")
                modes = transformer.fit_transform(pd.Series(signal))
                for i in range(modes.shape[1]):
                    ax.plot(df_bid['cycle_idx'], modes.iloc[:, i], linestyle='--', linewidth=1,
                            label=f'模态 {i + 1}', alpha=0.5)
            ax.set_title(f'电池 {bid} - 去噪后数据与趋势', fontsize=self.style['title_fontsize'],
                         fontweight='bold')
            ax.set_xlabel('循环索引 (cycles)', fontsize=self.style['label_fontsize'])
            ax.set_ylabel('容量 (Ah)', fontsize=self.style['label_fontsize'])
            ax.tick_params(axis='both', labelsize=self.style['tick_fontsize'])
            ax.legend(prop={'size': self.style['legend_fontsize']}, frameon=False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
            ax.grid(False)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'battery_{bid}_step_3_decomposed.png')
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"已保存电池 {bid} 趋势分解图至: {plot_path}")