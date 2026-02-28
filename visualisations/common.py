

import os
import numpy as np
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


def add_sample_names(df, names_species_input, n_species_input, n_species_total):
    df['sample_name'] = (np.arange(n_species_total) * np.ones((int(len(df)/n_species_total), n_species_total))).flatten()
    df = df[df['sample_name'].isin(list(np.arange(n_species_total)[-n_species_input:]))].reset_index(drop=True)
    df['sample_name'] = df['sample_name'].map(lambda x: names_species_input[int(x - (n_species_total - n_species_input))])
    return df


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


