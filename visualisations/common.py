

import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager


def set_theme():
    
    font_path = os.path.join('..', 'notebooks', 'Harding_Regular.ttf')
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['font.sans-serif'] = [font_name]

    sns.set_theme(font=font_name, style='white')
