"""
该模块提供 StyleSetter 类, 用于管理和应用 Matplotlib 绘图风格参数
支持从配置字典加载自定义风格, 并提供获取和应用风格的方法
"""

import logging
from typing import Any, Dict

from matplotlib.rcsetup import cycler

logger = logging.getLogger(__name__)


# 辅助函数：展平嵌套字典
def _flatten_dict(d, parent_key="", sep="."):
    """
    将嵌套字典展平为单层字典，使用分隔符连接键。
    例如：{'a': {'b': 1}} 展平为 {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        # 如果值是字典，则递归展平
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        # 否则，添加键值对
        else:
            items.append((new_key, v))
    # 返回展平后的字典
    return dict(items)


class StyleSetter:
    """
    设置绘图风格的类。负责加载和管理绘图风格参数。
    定义包含 rcParams 参数和自定义参数的风格字典
    由 Plotter 在绘制每个图表时通过上下文管理应用
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化 StyleSetter

        Args:
            config (Dict[str, Any], optional): 包含绘图风格参数的配置字典.
                                               这个字典应该对应 YAML 中 'plot_style' 的内容。
                                               如果为 None, 则使用默认风格.
        """
        # 定义默认风格
        self._default_style = {
            "rcParams.font.family": ["Times New Roman", "SimSun"],  # 字体家族
            "rcParams.axes.unicode_minus": False,  # 支持 Unicode 负号
            "rcParams.axes.titlesize": 22,  # 标题字体大小
            "rcParams.axes.labelsize": 20,  # 轴标签字体大小
            "rcParams.axes.facecolor": "white",  # 坐标轴背景颜色
            "rcParams.axes.edgecolor": "black",  # 边框颜色
            "rcParams.axes.linewidth": 1,  # 轴线宽度
            "rcParams.axes.spines.right": True,  # 四周边框
            "rcParams.axes.spines.left": True,
            "rcParams.axes.spines.bottom": True,
            "rcParams.axes.spines.top": True,
            "rcParams.axes.prop_cycle": cycler(color=["#2a9d90", "#f3a260", "#72bdd7"]),
            "rcParams.axes.grid": False,  # 网格可见性
            "rcParams.grid.color": "#e0e0e0",  # 网格颜色
            "rcParams.grid.alpha": 0.5,  # 网格透明度
            "rcParams.grid.linewidth": 0.7,
            "rcParams.grid.linestyle": "-",
            "rcParams.xtick.labelsize": 18,  # X轴刻度标签字体大小
            "rcParams.xtick.direction": "in",  # 刻度向内
            "rcParams.xtick.top": False,  # 不在顶部显示刻度
            "rcParams.xtick.bottom": True,  # 显示底部主刻度
            "rcParams.xtick.labelbottom": True,  # 显示底部刻度标签
            "rcParams.ytick.labelsize": 18,  # Y轴刻度标签字体大小
            "rcParams.ytick.direction": "in",  # 刻度向内
            "rcParams.ytick.left": True,  # 显示左侧主刻度
            "rcParams.ytick.right": False,  # 不在右侧显示刻度
            "rcParams.ytick.labelleft": True,  # 显示左侧刻度标签
            "rcParams.legend.loc": "best",  # 默认图例位置
            "rcParams.legend.fontsize": 19,  # 图例字体大小
            "rcParams.legend.frameon": False,  # 图例无边框
            "rcParams.legend.fancybox": False,  # 图例边框是否圆角
            "rcParams.legend.shadow": False,  # 图例是否有阴影
            "rcParams.lines.linewidth": 1.5,  # 默认线条宽度
            "rcParams.lines.linestyle": "-",  # 默认实线
            "rcParams.lines.color": "#2a9d90",  # 默认线条颜色
            "rcParams.lines.marker": None,  # 默认不带标记
            "rcParams.lines.markersize": 6.0,  # 默认标记大小
            "rcParams.lines.markeredgewidth": 1.0,  # 标记边缘线宽
            "rcParams.figure.facecolor": "white",  # 图形背景颜色
            "rcParams.figure.figsize": [6.2992, 3.5433],  # 16cm, 9cm / 2.54 变成英寸
            "rcParams.figure.dpi": 300,  # 创建图窗时的默认 DPI
            "rcParams.savefig.dpi": 300,  # 保存图窗时的默认 DPI
            "rcParams.savefig.format": "svg",  # 默认以svg存储
            "rcParams.savefig.bbox": "tight",  # 控制保存图片时是否包含图窗周围的空白区域
            "rcParams.savefig.transparent": False,  # 默认不透明 (使用布尔值 False)
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
        从**可能嵌套**的配置字典加载绘图风格参数.
        会将嵌套字典展平为点号分隔的键，并更新内部风格字典。

        Args:
            config (Dict[str, Any]): 包含绘图风格参数的配置字典 (可以是嵌套结构).
                                     例如，可以直接传入 YAML 中 'plot_style' 下的内容。
        """
        # --- 调用展平函数处理输入的配置 ---
        flat_config = _flatten_dict(config)

        # --- 然后使用展平后的字典来更新风格 ---
        for key, value in flat_config.items():
            if key in self._style:
                self._style[key] = value
            else:
                self._style[key] = value
                logger.warning(f"配置中存在 StyleSetter 不识别的风格参数: '{key}' (展平后键)")

    def get_style(self) -> Dict[str, Any]:
        """
        获取当前绘图风格参数字典.

        Returns:
            Dict[str, Any]: 包含所有风格参数的字典.
        """
        return self._style

    def get_rc_params_dict(self) -> Dict[str, Any]:
        """
        从当前风格字典中提取适用于 plt.style.context 的 rcParams 参数.
        返回一个移除了 'rcParams.' 前缀的字典。
        这个字典可以直接用于 plt.rcParams.update() 或 plt.style.context().
        """
        # 这个列表应该包含所有希望作为 rcParams 应用的展平后的键
        # 与 _default_style 中 rcParams 部分的键一致
        # 列表本身可以保持带有前缀的键，用于从 self._style 中筛选
        rc_params_keys_with_prefix = [
            "rcParams.font.family",
            "rcParams.axes.unicode_minus",
            "rcParams.axes.titlesize",
            "rcParams.axes.labelsize",
            "rcParams.axes.facecolor",
            "rcParams.axes.edgecolor",
            "rcParams.axes.linewidth",
            "rcParams.axes.spines.right",
            "rcParams.axes.spines.left",
            "rcParams.axes.spines.bottom",
            "rcParams.axes.spines.top",
            "rcParams.axes.grid",
            "rcParams.grid.color",
            "rcParams.grid.alpha",
            "rcParams.grid.linewidth",
            "rcParams.grid.linestyle",
            "rcParams.xtick.labelsize",
            "rcParams.xtick.direction",
            "rcParams.xtick.top",
            "rcParams.xtick.bottom",
            "rcParams.xtick.labelbottom",
            "rcParams.ytick.labelsize",
            "rcParams.ytick.direction",
            "rcParams.ytick.left",
            "rcParams.ytick.right",
            "rcParams.ytick.labelleft",
            "rcParams.legend.loc",
            "rcParams.legend.fontsize",
            "rcParams.legend.frameon",
            "rcParams.legend.fancybox",
            "rcParams.legend.shadow",
            "rcParams.lines.linewidth",
            "rcParams.lines.linestyle",
            "rcParams.lines.color",
            "rcParams.lines.marker",
            "rcParams.lines.markersize",
            "rcParams.lines.markeredgewidth",
            "rcParams.figure.facecolor",
            "rcParams.figure.figsize",
            "rcParams.figure.dpi",
            "rcParams.savefig.dpi",
            "rcParams.savefig.format",
            "rcParams.savefig.bbox",
            "rcParams.savefig.transparent",
            "rcParams.axes.prop_cycle",
        ]

        rc_params_dict_for_context = {}
        for key_with_prefix in rc_params_keys_with_prefix:
            # 检查带有前缀的键是否存在于内部风格字典中
            if key_with_prefix in self._style:
                value = self._style[key_with_prefix]
                # 移除 'rcParams.' 前缀，得到标准的 rcParams 键
                # 确保键确实以 'rcParams.' 开头
                if key_with_prefix.startswith("rcParams."):
                    standard_rc_key = key_with_prefix[len("rcParams.") :]
                    rc_params_dict_for_context[standard_rc_key] = value
                else:
                    # 理论上不会发生，但作为备用处理
                    logger.warning(
                        f"内部风格字典中发现非预期的rcParam键格式（无rcParams.前缀）: {key_with_prefix}"
                    )
                    rc_params_dict_for_context[key_with_prefix] = value  # 直接使用原键

        return rc_params_dict_for_context
