import matplotlib.pyplot as plt


class StyleSetter:
    """设置全局绘图风格的类。"""

    def __init__(self, title_fontsize=22):
        """
        初始化 StyleSetter。

        Args:
            title_fontsize (int): 标题字体大小，默认为 18。
        """
        self.title_fontsize = title_fontsize
        self.label_fontsize = title_fontsize - 2
        self.tick_fontsize = title_fontsize - 2
        self.legend_fontsize = title_fontsize - 3
        self._set_global_style()

    def _set_global_style(self):
        """设置全局绘图参数。"""
        plt.rcParams['font.family'] = 'SimSun'
        plt.rcParams['axes.unicode_minus'] = False

    def get_style(self):
        """
        获取绘图风格参数。

        Returns:
            dict: 包含字体大小的字典。
        """
        return {
            'title_fontsize': self.title_fontsize,
            'label_fontsize': self.label_fontsize,
            'tick_fontsize': self.tick_fontsize,
            'legend_fontsize': self.legend_fontsize,
        }
