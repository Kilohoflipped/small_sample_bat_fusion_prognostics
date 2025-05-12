import matplotlib.pyplot as plt
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class StyleSetter:
    """
    设置绘图风格的类。负责加载和管理绘图风格参数。
    不再修改全局 Matplotlib 参数，而是提供一个方法来获取风格字典，
    由 PreprocessPlotter 在绘制每个图表时临时应用。
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化 StyleSetter.

        Args:
            config (Dict[str, Any], optional): 包含绘图风格参数的配置字典.
                                               如果为 None, 则使用默认风格.
        """
        # 定义默认风格
        self._default_style = {
            "title_fontsize": 22,
            "label_fontsize": 20,
            "tick_fontsize": 18,
            "legend_fontsize": 19,
            "line_color_primary": "#3a80f8",  # 主线条颜色 (蓝色)
            "line_color_secondary": "#d64f38",  # 次要线条颜色 (红色)
            "anomaly_color": "#d64f38",  # 异常点颜色 (红色)
            "interpolated_color": "#d64f38",  # 插值点颜色 (红色)
            "background_line_color": "#d3d3d3",  # 背景线颜色 (浅灰色)
            "grid_color": "#cccccc",  # 网格颜色
            "spine_color": "black",  # 边框颜色
            "font_family": "SimSun",  # 字体家族
            "axes_unicode_minus": False,  # 支持 Unicode 负号
            "legend_frameon": False,  # 图例无边框
            "legend_loc": "best",  # 默认图例位置
            "figure_facecolor": "white",  # 图形背景颜色
            "axes_facecolor": "white",  # 坐标轴背景颜色
            "grid_alpha": 0.5,  # 网格透明度
            "line_linewidth": 1.5,  # 默认线条宽度
            "scatter_markersize": 50,  # 默认散点大小
            "anomaly_markersize": 100,  # 异常点散点大小
        }

        # 如果提供了配置, 则更新默认风格
        self._style = self._default_style.copy()
        if config:
            self.load_style_from_config(config)
            logger.info("StyleSetter 已从配置加载风格")
        else:
            logger.info("StyleSetter 使用默认风格")

    def load_style_from_config(self, config: Dict[str, Any]):
        """
        从配置字典加载绘图风格参数.

        Args:
            config (Dict[str, Any]): 包含绘图风格参数的配置字典.
        """
        # 仅更新配置中存在的风格参数
        for key, value in config.items():
            if key in self._style:
                self._style[key] = value
            else:
                logger.warning(f"配置中存在未知风格参数: '{key}'")

    def get_style(self) -> Dict[str, Any]:
        """
        获取当前绘图风格参数字典.

        Returns:
            Dict[str, Any]: 包含所有风格参数的字典.
        """
        return self._style

    def apply_global_style(self):
        """
        临时应用全局 Matplotlib 风格设置.
        注意: 更好的做法是在 PreprocessPlotter 中使用 context manager.
        这个方法仅用于兼容 PreprocessPlotter 当前的设计.
        """
        # 仅设置那些通常需要全局设置的参数
        plt.rcParams["font.family"] = self._style.get("font_family", "sans-serif")
        plt.rcParams["axes.unicode_minus"] = self._style.get(
            "axes_unicode_minus", False
        )
        # 其他风格参数将在 PreprocessPlotter 的 _apply_plot_style 中应用到具体的 Axes 对象
        logger.debug("已应用全局 Matplotlib 风格设置")

    # 不再有 _set_global_style 方法
