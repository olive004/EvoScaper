

import jax
from evoscaper.utils.dataclasses import FilterSettings, NormalizationSettings
from evoscaper.utils.dataset import make_xy, prep_data

import pandas as pd
import numpy as np


def make_fake_data():
    data = {
        "id": [1, 2, 3, 4, 5] * 2,
        "x": [1, 2, 3, 4, 5] * 2,
        "y": [1, 2, 3, 4, 5] * 2,
        "z": [1, 2, 3, 4, 5] * 2,
        "label": [1, 0, 1, 0, 1] * 2
    }
    data["a"] = [-1, 2, 3, 4, 5] * 2
    data["b"] = [1, 0, np.nan, 4, 5] * 2
    data["c"] = [1, 2, 3, 0, np.nan] * 2
    data["cat1"] = ["A", "B", "C", "A", "B"] * 2
    data["cat2"] = ["X", "Y", "X", "Y", "X"] * 2
    data["cat3"] = ["foo", "bar", "baz", "foo", "bar"] * 2
    data["sample_name"] = ["foo", "bar", "baz", "foo", "bar"] * 2
    data['sensitivity_wrt_species-6'] = [0, 2, 3, 4, 5] * 2
    data['precision_wrt_species-6'] = [0, 2, 3, 4, 5] * 2
    return pd.DataFrame(data)


def test_prep_data():

    data = make_fake_data()
    
    df = prep_data(data=data, output_species=['foo'], col_y='Log sensitivity', cols_x=['x', 'y', 'z'], filter_settings=FilterSettings())
    assert len(df) == 2, 'Filtering did not work as expected'
    
    df = prep_data(data=data, output_species=['foo'], col_y='adaptation', cols_x=['x', 'y', 'z'], filter_settings=FilterSettings())
    assert len(df) == 1, 'Filtering adaptation did not work as expected'
    
    
def test_make_xy():
    
    data = make_fake_data()
    x_cols = ['x', 'y', 'z']
    y_col = 'Log sensitivity'
    df = prep_data(data=data, output_species=['foo'], col_y=y_col, cols_x=x_cols, filter_settings=FilterSettings())
    rng = jax.random.PRNGKey(0)
    TOTAL_DS = 2
    x_norm_settings = NormalizationSettings()
    y_norm_settings = NormalizationSettings()
    
    x, cond, x_datanormaliser, x_methods_preprocessing, y_datanormaliser, y_methods_preprocessing = make_xy(df, TOTAL_DS, x_cols, y_col,
                                                                                                            x_norm_settings, y_norm_settings)
    
    assert np.allclose(x_datanormaliser.create_chain_preprocessor_inverse(x_methods_preprocessing)(x), df[x_cols].values), 'Inverse transformation of x did not work as expected'
    assert np.all(np.abs(y_datanormaliser.create_chain_preprocessor_inverse(y_methods_preprocessing)(cond).flatten() - df[y_col].values) < 1e-3), 'Inverse transformation of x did not work as expected'

    
    
    
test_prep_data()
test_make_xy()
