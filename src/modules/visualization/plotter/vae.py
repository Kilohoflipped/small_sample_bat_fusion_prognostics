import logging
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.modules.visualization.style_setter import StyleSetter

logger = logging.getLogger(__name__)


class VAEPlotter:
    """
    负责绘制 VAE 生成结果图表的类.
    使用 StyleSetter 获取风格并在绘制时使用上下文应用风格.
    """

    def __init__(self, style_setter: StyleSetter):
        """
        初始化 VAEPlotter.

        Args:
            style_setter (StyleSetter): 绘图风格设置对象.
        """
        if not isinstance(style_setter, StyleSetter):
            raise TypeError("style_setter 必须是 StyleSetter 类的实例")

        self.style_setter = style_setter
        self.style: Dict[str, Any] = self.style_setter.get_style()
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
            # 使用 StyleSetter 获取保存格式，如果没有设置则默认为 png
            save_format = self.rc_params_dict.get("savefig.format", "png")
            full_output_path = f"{base_output_path_without_extension}.{save_format}"
            fig.savefig(full_output_path, bbox_inches="tight")
            logger.info(f"已保存图表至: {full_output_path}")
        except Exception as e:
            logger.error(f"保存图表 '{full_output_path}' 时发生错误: {e}")
            raise
        finally:
            plt.close(fig)

    def plot_generated_vs_original(
        self,
        original_sequences: List[np.ndarray],
        generated_sequences: List[np.ndarray],
        battery_ids: List[str],
        lengths: List[int],
        output_dir: str,
    ) -> Dict[str, float] | None:
        """
        绘制原始序列与 VAE 生成序列的对比图，并计算 MSE.

        Args:
            original_sequences (List[np.ndarray]): 原始序列列表.
            generated_sequences (List[np.ndarray]): VAE 生成序列列表.
            battery_ids (List[str]): 对应的电池 ID 列表.
            lengths (List[int]): 对应的序列实际长度列表.
            output_dir (str): 图片保存目录.

        Returns:
            Dict[str, float] | None: 包含每个电池 MSE 的字典，如果发生错误则返回 None.
        """
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                logger.info(f"已创建绘图输出目录: {output_dir}")
            except OSError as e:
                logger.error(f"创建绘图输出目录失败: {output_dir} - {e}", exc_info=True)
                return None

        if not original_sequences or not generated_sequences or not battery_ids or not lengths:
            logger.warning("绘图数据为空或不完整, 跳过绘图.")
            return None

        if not (
            len(original_sequences) == len(generated_sequences) == len(battery_ids) == len(lengths)
        ):
            logger.error("绘图数据长度不一致, 无法进行对比绘图.")
            logger.error(
                f"原始序列数量: {len(original_sequences)}, 生成序列数量: {len(generated_sequences)}, 电池ID数量: {len(battery_ids)}, 长度数量: {len(lengths)}"
            )
            return None

        mse_dict: Dict[str, float] = {}

        # 遍历对齐后的序列进行绘图
        for i in range(len(battery_ids)):
            battery_id = battery_ids[i]
            orig_seq = original_sequences[i]
            gen_seq = generated_sequences[i]
            seq_len = lengths[i]

            # 确保序列长度一致，只取有效部分
            # 这里的逻辑应该与 inference_vae.py 中的对齐逻辑一致
            # 如果长度不一致，日志警告并在绘图和 MSE 计算中使用最小长度
            if len(orig_seq) < seq_len or len(gen_seq) < seq_len:
                logger.warning(
                    f"电池 {battery_id} 数据长度与记录的长度不符: 原始={len(orig_seq)}, 生成={len(gen_seq)}, 记录={seq_len}. 将使用最小长度."
                )
                current_seq_len = min(len(orig_seq), len(gen_seq), seq_len)
                orig_seq_plot = orig_seq[:current_seq_len]
                gen_seq_plot = gen_seq[:current_seq_len]
            else:
                current_seq_len = seq_len
                orig_seq_plot = orig_seq[:current_seq_len]
                gen_seq_plot = gen_seq[:current_seq_len]

            # 计算 MSE (只计算有效长度部分)
            try:
                mse = np.mean((orig_seq_plot - gen_seq_plot) ** 2).item()  # 确保转换为 Python float
                mse_dict[str(battery_id)] = mse  # 确保 battery_id 是字符串作为字典键
            except Exception as e:
                logger.error(f"计算电池 {battery_id} 的 MSE 时发生错误: {e}", exc_info=True)
                mse_dict[str(battery_id)] = float("nan")  # 记录为 NaN

            # 绘制图表
            with plt.style.context(self.rc_params_dict):
                fig, ax = plt.subplots()

                ax.plot(orig_seq_plot, label="原始序列", alpha=0.7)
                ax.plot(gen_seq_plot, label="生成序列", linestyle="--", alpha=0.7)

                self._apply_plot_style(
                    ax,
                    f"电池 {battery_id} - VAE训练结果",
                    "循环索引 (cycles)",
                    "容量 (Ah)",
                )

                ax.legend()

                plot_path_base = os.path.join(
                    output_dir, f"battery_{battery_id}_generated_vs_original"
                )
                try:
                    self._save_plot(fig, plot_path_base)
                except Exception as e:
                    logger.error(f"保存电池 {battery_id} 的生成对比图失败: {e}", exc_info=True)

        # 保存 MSE 摘要
        mse_summary_path = os.path.join(output_dir, "mse_summary_generated.txt")
        try:
            with open(mse_summary_path, "w") as f:
                for bid, mse_value in mse_dict.items():
                    # 使用 {:.4f} 格式化浮点数，即使是 NaN 或 Inf 也能处理
                    f.write(f"电池 {bid}: MSE = {mse_value:.4f}\n")

                # 计算并写入平均 MSE (只计算有效的 MSE 值)
                valid_mses = [v for v in mse_dict.values() if not np.isnan(v) and not np.isinf(v)]
                if valid_mses:
                    average_mse = float(np.mean(valid_mses))
                    f.write(f"平均 MSE: {average_mse:.4f}\n")
                else:
                    f.write("无有效 MSE 数据可计算平均值.\n")
            logger.info(f"已保存 MSE 摘要至: {mse_summary_path}")

        except Exception as e:
            logger.error(f"保存 MSE 摘要文件失败: {e}", exc_info=True)

        return mse_dict
