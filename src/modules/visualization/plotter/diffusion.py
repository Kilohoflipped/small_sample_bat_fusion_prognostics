import logging
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.modules.visualization.style_setter import StyleSetter

logger = logging.getLogger(__name__)


class DiffusionPlotter:
    """
    负责绘制扩散模型生成结果图表的类.
    使用 StyleSetter 获取风格并在绘制时使用上下文应用风格.
    """

    def __init__(self, style_setter: StyleSetter):
        """
        初始化 DiffusionPlotter.

        Args:
            style_setter (StyleSetter): 绘图风格设置对象.
        """
        if not isinstance(style_setter, StyleSetter):
            raise TypeError("style_setter 必须是 StyleSetter 类的实例")

        self.style_setter = style_setter
        # 获取完整的风格字典，可能包含 rcParams 之外的自定义风格
        self.style: Dict[str, Any] = self.style_setter.get_style()
        # 获取 rcParams 字典，用于 plt.style.context
        self.rc_params_dict: Dict[str, Any] = self.style_setter.get_rc_params_dict()

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
            save_format = self.rc_params_dict.get("savefig.format", "png")
            bbox_inches_param = self.rc_params_dict.get("savefig.bbox", "tight")
            full_output_path = f"{base_output_path_without_extension}.{save_format}"
            fig.savefig(
                full_output_path,
                bbox_inches=bbox_inches_param,
            )
            logger.info(f"已保存图表至: {full_output_path}")
        except Exception as e:
            logger.error(f"保存图表 '{full_output_path}' 时发生错误: {e}")
            # 重新抛出异常以便调用者知晓保存失败
            raise
        finally:
            # 总是关闭图表以释放内存
            plt.close(fig)

    def plot_generated_vs_real(
        self,
        real_sequence: np.ndarray,
        generated_sequence: np.ndarray,
        battery_id: Any,
        conditions: torch.Tensor,
        output_dir: str,
        sample_index: int,
    ):
        """
        绘制扩散模型生成的序列与真实序列的对比图.

        Args:
            real_sequence (np.ndarray): 真实的序列数据 (例如，电池容量曲线).
            generated_sequence (np.ndarray): 扩散模型生成的序列数据.
            battery_id (Any): 对应样本的电池 ID.
            conditions (torch.Tensor): 用于生成该样本的条件张量.
            output_dir (str): 图片保存目录.
            sample_index (int): 当前处理的样本在可视化列表中的索引 (用于文件名).
        """
        with plt.style.context(self.rc_params_dict):
            fig, ax = plt.subplots()

            # 绘制真实数据
            ax.plot(
                real_sequence,
                label="实测曲线",
                zorder=2,
                alpha=0.7,
            )

            # 绘制生成数据
            ax.plot(
                generated_sequence,
                label="生成曲线",
                zorder=1,
                alpha=0.7,
                linestyle="--",
            )

            title = f"电池 {battery_id} - DDPM生成曲线对比"
            xlabel = "循环索引 (cycles)"
            ylabel = "SOH (%)"

            self._apply_plot_style(ax, title, xlabel, ylabel)

            ax.legend()
            ax.grid(self.rc_params_dict.get("axes.grid", False))  # 根据 rcParams 设置网格

            # 构造图片保存路径
            # 文件名包含电池ID和样本索引
            plot_filename = f"battery_{battery_id}_gen_vs_real_{sample_index+1}"
            plot_filepath = os.path.join(output_dir, plot_filename)

            # 保存图表
            self._save_plot(fig, plot_filepath)
