import logging
import os
from typing import Any, Dict, Tuple

import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.modules.visualization.style_setter import StyleSetter

logger = logging.getLogger(__name__)


class PreprocessPaperPlotter:
    """
    负责绘制电池数据图表，特别适用于论文展示，包含内嵌图展示噪声。
    使用 StyleSetter 获取风格，并在绘制时使用上下文应用风格。
    """

    def __init__(self, style_setter: StyleSetter, csv_path: str, output_dir: str):
        """
        初始化 PreprocessPaperPlotter.

        Args:
            style_setter (StyleSetter): 绘图风格设置对象.
                                       图表尺寸和分辨率将从 StyleSetter 管理的 rcParams 中获取。
            csv_path (str): 输入的 CSV 数据文件路径。需要包含 'cycle_idx', 'target', 'battery_id' 列。
            output_dir (str): 图片保存目录。
        """
        if not isinstance(style_setter, StyleSetter):
            raise TypeError("style_setter 必须是 StyleSetter 类的实例")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"输入 CSV 文件未找到: {csv_path}")
        if not os.path.isdir(output_dir):
            logger.warning(f"输出目录不存在，尝试创建: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        self.style_setter = style_setter
        self.style: Dict[str, Any] = self.style_setter.get_style()
        self.output_dir = output_dir
        self._df = self._load_data(csv_path)

    def _load_data(self, csv_path: str) -> pd.DataFrame:
        """
        从 CSV 文件加载数据并验证必要列。

        Args:
            csv_path (str): CSV 文件路径。

        Returns:
            pd.DataFrame: 加载并验证后的 DataFrame.

        Raises:
            ValueError: 如果 CSV 文件缺少必要列。
            Exception: 读取文件时发生的其他错误。
        """
        required_cols = ["cycle_idx", "target", "battery_id"]
        try:
            df = pd.read_csv(csv_path)
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"输入 CSV 文件 '{csv_path}' 缺少必要列: {missing}")
            logger.info(f"成功从 '{csv_path}' 加载数据.")
            return df[
                required_cols + [col for col in df.columns if col not in required_cols]
            ]  # 确保必要列在前
        except FileNotFoundError:
            # 这个异常已经在 __init__ 中处理了，但作为内部方法，防御性地保留
            logger.error(f"数据文件未找到: {csv_path}")
            raise
        except ValueError as e:
            logger.error(f"数据验证失败: {e}")
            raise
        except Exception as e:
            logger.error(f"读取 CSV 文件 '{csv_path}' 时发生错误: {e}")
            raise

    def _apply_plot_style(self, ax: plt.Axes, title: str, xlabel: str, ylabel: str):
        """
        应用通用绘图风格到 Axes 对象 (主要设置文本).
        大部分 Axes 和 Figure 风格通过 style context (rcParams) 控制

        Args:
            ax (matplotlib.axes.Axes): 要应用风格的 Axes 对象.
            title (str): 图表标题.
            xlabel (str): X轴标签.
            ylabel (str): Y轴标签.
        """
        # 设置标题和标签文本
        # 字体大小等属性由 rcParams context 控制
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # 注意：如果需要设置内嵌图的标题或标签，需要在这里添加逻辑或在绘图方法中单独设置

    def _save_plot(self, fig: plt.Figure, base_output_path_without_extension: str):
        """
        保存图表到指定路径，根据 rcParams 自动添加文件扩展名。

        Args:
            fig (matplotlib.figure.Figure): 要保存的 Figure 对象.
            base_output_path_without_extension (str): 图片保存路径 (不包含文件扩展名).
                                                     这个路径将与 rcParams['savefig.format'] 结合使用。
        """
        try:
            # 获取当前 rcParams 中的 savefig.format 设置
            # 这个设置应该是由 plot_style context 设置的
            save_format = plt.rcParams.get(
                "savefig.format", "svg"
            )  # 如果rcParams中没有设置，默认为svg

            # 构造完整的带有扩展名的文件路径
            full_output_path = f"{base_output_path_without_extension}.{save_format}"

            # savefig 将根据 full_output_path 的扩展名来确定保存格式
            # 通过构造带有正确扩展名的路径，我们确保了格式的正确推断和文件命名的规范性。
            # bbox_inches="tight" 和 transparent 由 rcParams 控制更灵活，但这里显式写出确保效果
            # 如果需要在rcParams中设置 bbox_inches 和 transparent，则这里可以移除
            fig.savefig(
                full_output_path,
                bbox_inches=plt.rcParams.get("savefig.bbox", "tight"),  # 从 rcParams 获取
                transparent=plt.rcParams.get("savefig.transparent", False),  # 从 rcParams 获取
                dpi=plt.rcParams.get("savefig.dpi", 300),  # 从 rcParams 获取 dpi
            )

            logger.info(f"已保存图表至: {full_output_path}")
        except Exception as e:
            logger.error(f"保存图表 '{full_output_path}' 时发生错误: {e}")
            # 视情况选择是否重新抛出异常
            # raise
        finally:
            plt.close(fig)  # 确保关闭图表，释放内存

    def plot_per_battery_paper_style(self, noise_range: Tuple[float, float], inset_location: str):
        """
        为每个 battery_id 绘制 cycle_idx 对 target 的曲线图，并包含指定范围的内嵌图。

        Args:
            noise_range (Tuple[float, float]): 一个元组，表示内嵌图展示的 cycle_idx 范围 (min, max)。
        """
        if self._df is None or self._df.empty:
            logger.warning("没有数据可用于绘制，跳过论文风格图表绘制。")
            return

        rc_params_dict = self.style_setter.get_rc_params_dict()

        # 可以从 style 中获取论文风格标题模板
        main_title_template = self.style.get(
            "paper_style_main_title", "电池 {bid} 原始数据与局部噪声细节"
        )
        xlabel_text = self.style.get("paper_style_xlabel", "循环索引 (cycles)")
        ylabel_text = self.style.get("paper_style_ylabel", "容量 (Ah)")
        # 从 style 中获取内嵌图尺寸和位置配置
        inset_width = self.style.get("paper_style_inset_width", "30%")  # 内嵌图宽度占父 Axes 的比例
        inset_height = self.style.get(
            "paper_style_inset_height", "20%"
        )  # 内嵌图高度占父 Axes 的比例
        # 内嵌图位置，例如 'upper right', 'upper left', 'lower left', 'lower right' 等
        # inset_location = self.style.get("paper_style_inset_location", "center")
        # 内嵌图单位，如果 width/height 是绝对值，需要指定单位 "inch" 或 "figure fraction"
        # 如果 width/height 是比例字符串（如 "40%"），通常不需要 unit
        # inset_unit = self.style.get("paper_style_inset_unit", None)

        battery_ids = self._df["battery_id"].unique()
        logger.info(f"开始绘制 {len(battery_ids)} 个电池的论文风格图表...")

        for bid in battery_ids:
            logger.info(f"绘制电池 ID: {bid}")
            df_bid = self._df[self._df["battery_id"] == bid].copy()

            if df_bid.empty:
                logger.warning(f"电池 ID {bid} 没有数据，跳过。")
                continue

            # 过滤内嵌图所需的数据范围
            df_inset_data = df_bid[
                (df_bid["cycle_idx"] >= noise_range[0]) & (df_bid["cycle_idx"] <= noise_range[1])
            ].copy()

            if df_inset_data.empty:
                logger.warning(f"电池 ID {bid} 在范围 {noise_range} 内没有数据，将不绘制内嵌图。")
                plot_inset = False
            else:
                plot_inset = True
                # 确保内嵌图数据按 cycle_idx 排序，绘图时连线正确
                df_inset_data = df_inset_data.sort_values("cycle_idx")
                # 为了设置内嵌图的Y轴范围，获取该范围内数据的实际Y值范围
                y_min_inset_data = df_inset_data["target"].min()
                y_max_inset_data = df_inset_data["target"].max()
                # 为内嵌图Y轴留一点边距
                y_inset_margin = (
                    (y_max_inset_data - y_min_inset_data) * 0.2
                    if (y_max_inset_data - y_min_inset_data) > 0
                    else 0.1
                )
                y_start_inset = y_min_inset_data - y_inset_margin
                y_end_inset = y_max_inset_data + y_inset_margin

            with plt.style.context(rc_params_dict):
                # 使用 StyleSetter 中定义的 figure size 和 dpi
                fig, ax = plt.subplots()  # ax 是主 Axes

                # 绘制主曲线
                ax.plot(
                    df_bid["cycle_idx"],
                    df_bid["target"],
                    # 曲线风格由 rcParams 控制，如 lines.color, lines.linewidth 等
                    label="原始数据",  # 主图的标签可以用来区分电池，或者在单电池图时省略
                    alpha=0.7,
                    zorder=1,
                )

                # 应用主图通用风格 (标题和标签文本，由 rcParams 控制字体大小等)
                # 在应用通用风格前绘制，确保图例等不受影响（虽然这里单电池图没图例）
                self._apply_plot_style(
                    ax,
                    main_title_template.format(bid=bid),  # 使用模板和电池ID填充标题
                    xlabel_text,
                    ylabel_text,
                )
                # 添加主图图例 (如果需要)
                ax.legend()

                if plot_inset:
                    # --- 修改这里：使用 inset_locator.inset_axes 进行自动定位 ---
                    # 创建内嵌 Axes
                    # 第一个参数是父 Axes 对象 (ax)
                    # width 和 height 指定内嵌图的尺寸，可以是比例字符串或绝对数值
                    # loc 指定内嵌图的自动位置
                    inset_ax = inset_axes(
                        ax,  # 父 Axes 对象
                        width=inset_width,  # 内嵌图宽度 (来自 style config)
                        height=inset_height,  # 内嵌图高度 (来自 style config)
                        loc=inset_location,  # 内嵌图位置 (来自 style config)
                        # unit=inset_unit # 如果 width/height 是绝对值，可能需要指定单位
                    )

                    # 绘制内嵌图曲线
                    # 注意：这里绘制的是整个电池的数据 df_bid，然后通过设置 inset_ax 的 xlim/ylim 来“放大”
                    # 或者只绘制过滤后的 df_inset_data 也可以，但通常绘制 df_bid 更通用，放大靠 set_xlim/ylim
                    inset_ax.plot(
                        df_bid["cycle_idx"],  # 绘制整个电池的数据
                        df_bid["target"],
                        marker="o",  # 在内嵌图上通常会显示点来突出噪声
                        linestyle="-",
                        markersize=self.style.get("inset_markersize", 3),  # 从 style 获取内嵌点大小
                        alpha=self.style.get("inset_alpha", 0.7),  # 从 style 获取透明度
                        # 内嵌图曲线颜色等其他风格继续受 rcParams 控制
                        zorder=1,  # 与主图 Zorder 可能需要区分
                    )

                    # 设置内嵌图的 x 轴范围 (使用传入的 noise_range)
                    inset_ax.set_xlim(noise_range)
                    # 设置内嵌图的 y 轴范围 (使用计算好的范围和边距)
                    inset_ax.set_ylim(y_start_inset, y_end_inset)

                    # --- 应用内嵌图的风格 (边框、刻度等) ---
                    # 从 StyleSetter 的 get_style() 中获取自定义参数
                    inset_border_color = self.style.get("inset_border_color", "#333333")
                    inset_linewidth = self.style.get("inset_linewidth", 1.5)
                    inset_zoom_alpha = self.style.get(
                        "inset_zoom_alpha", 0.9
                    )  # indicate_inset_zoom 可能有 alpha 参数

                    # 设置内嵌图框的边缘风格
                    for spine in inset_ax.spines.values():
                        spine.set_edgecolor(inset_border_color)
                        spine.set_linewidth(inset_linewidth)
                        spine.set_alpha(inset_zoom_alpha)

                    # 设置内嵌图的刻度参数
                    inset_ax.tick_params(
                        axis="both",
                        which="major",
                        labelsize=self.style.get("inset_tick_labelsize", 8),
                    )
                    # 可以选择隐藏内嵌图的 x, y 轴标签
                    inset_ax.set_xlabel("")
                    inset_ax.set_ylabel("")
                    # 可以选择隐藏内嵌图的刻度标签，只保留框和内容
                    inset_ax.set_xticklabels([])
                    inset_ax.set_yticklabels([])
                    # 可以选择隐藏内嵌图的刻度线
                    inset_ax.tick_params(axis="both", which="both", length=0)

                    # --- 连接主图和内嵌图区域 ---
                    # 使用 indicate_inset_zoom 连接，或者 mark_inset
                    # indicate_inset_zoom 是 ax 对象的方法
                    # mark_inset 是 inset_locator 模块的函数
                    # mark_inset 功能更强大，可以控制连接线的起点终点和样式
                    # indicator_color = self.style.get("inset_indicator_color", "red")
                    # indicator_linestyle = self.style.get("inset_indicator_linestyle", "--")

                    # 使用 mark_inset 来绘制连接框和线，功能更灵活
                    # loc1, loc2 控制连接线在两个 Axes 上的连接点位置
                    from mpl_toolkits.axes_grid1.inset_locator import (
                        mark_inset,
                    )  # 确保导入 mark_inset

                    mark_inset(
                        ax,
                        inset_ax,
                        loc1=self.style.get("inset_connector_loc1", 1),
                        loc2=self.style.get("inset_connector_loc2", 2),
                        fc="none",  # 框内部不填充颜色
                        ec=inset_border_color,  # 框和连接线颜色与内嵌图框边缘颜色一致
                        lw=inset_linewidth,  # 线宽一致
                        linestyle=self.style.get("inset_connector_linestyle", "--"),  # 连接线样式
                        alpha=self.style.get(
                            "inset_connector_alpha", inset_zoom_alpha
                        ),  # 连接线透明度
                    )

                    # 内嵌图通常不需要自己的标题和完整的轴标签，刻度足够了
                    # inset_ax.set_title("Noise Detail", fontsize=10)
                    # inset_ax.set_xlabel("Cycle Index")
                    # inset_ax.set_ylabel("Capacity (Ah)")

            # 调整布局以适应内嵌图
            fig.tight_layout()

            # 保存图表
            # 构建基于 battery_id 的输出文件名，并获取 StyleSetter 中的保存格式
            save_format = self.style_setter.get_style().get("rcParams.savefig.format", "png")
            # plot_filename = f"battery_{bid}_capacity_paper_style.{save_format}" # 直接带上格式
            # plot_path = os.path.join(self.output_dir, plot_filename)

            # 或者如果您 _save_plot 支持只传基础文件名并自动加扩展名
            plot_filename_base = f"battery_{bid}_capacity_paper_style"
            plot_path = os.path.join(
                self.output_dir, plot_filename_base
            )  # 传递不带扩展名的基础路径

            # 使用 _save_plot 保存，它会处理 dpi 和 bbox_inches
            # 请确保 _save_plot 内部会根据 rcParams['savefig.format'] 自动添加扩展名
            self._save_plot(fig, plot_path)

        logger.info("所有论文风格图表绘制完成")
