import matplotlib.pyplot as plt

class StyleSetter:
    """
    设置绘图风格的类。不再修改全局 Matplotlib 参数，
    而是提供一个方法来获取风格字典，由 Plotter 应用到具体的 Axes 对象上。
    """

    def __init__(self, title_fontsize=22):
        """
        初始化 StyleSetter。

        Args:
            title_fontsize (int): 标题字体大小，默认为 22。
        """
        self.title_fontsize = title_fontsize
        self.label_fontsize = title_fontsize - 2
        self.tick_fontsize = title_fontsize - 2
        self.legend_fontsize = title_fontsize - 3

    def get_style(self):
        """
        获取绘图风格参数字典。

        Returns:
            dict: 包含字体大小和其他风格参数的字典。
        """
        # 返回一个包含所有风格参数的字典
        return {
            'title_fontsize': self.title_fontsize,
            'label_fontsize': self.label_fontsize,
            'tick_fontsize': self.tick_fontsize,
            'legend_fontsize': self.legend_fontsize,
            # 添加其他可能的风格参数，例如颜色、线条样式等
            'line_color_primary': '#3a80f8',
            'line_color_secondary': '#d64f38',
            'anomaly_color': '#d64f38',
            'interpolated_color': '#d64f38',
            'background_line_color': '#d3d3d3',
            'grid_color': '#cccccc',
            'spine_color': 'black',
            'font_family': 'SimSun',
            'axes_unicode_minus': False,
        }

    # 不再有 _set_global_style 方法，避免污染全局状态
