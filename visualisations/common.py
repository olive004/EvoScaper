

import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import font_manager


def set_theme():
    
    font_path = os.path.join('..', 'notebooks', 'Harding_Regular.ttf')
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['font.sans-serif'] = [font_name]

    sns.set_theme(font=font_name, style='white')


def get_colors(pal, n):
    cmap = cm.get_cmap(pal, n)
    return [cmap(i) for i in range(n)]


def make_pastel(color: tuple) -> tuple:
    r, g, b = color[:3]
    _alpha = 0.4
    pastel_rgb = (1 - _alpha + _alpha * r,
                    1 - _alpha + _alpha * g,
                    1 - _alpha + _alpha * b)
    color = (*pastel_rgb, 1.0)
    return color