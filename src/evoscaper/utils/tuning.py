from evoscaper.utils.normalise import NormalizationSettings, FilterSettings


def make_settings(hpos, keys_dataset):
    x_norm_settings = NormalizationSettings(**{s.replace('prep_x_', ''): hpos[s] for s in keys_dataset if 'prep_x' in s})
    y_norm_settings = NormalizationSettings(**{s.replace('prep_y_', ''): hpos[s] for s in keys_dataset if 'prep_y' in s})
    filter_settings = FilterSettings(**{s: hpos[s] for s in keys_dataset if 'filt' in s})
    return x_norm_settings, y_norm_settings, filter_settings