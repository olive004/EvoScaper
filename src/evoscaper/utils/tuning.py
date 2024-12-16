

import pandas as pd

from evoscaper.utils.dataclasses import (
    NormalizationSettings, FilterSettings, DatasetConfig, ModelConfig, TrainingConfig, OptimizationConfig)
    

def make_config_model(x, hpos):
    return ModelConfig(
        seed_arch=hpos['seed_arch'],
        decoder_head=x.shape[-1],
        activation=hpos['activation'],
        model=hpos['model'],
        enc_layers=hpos['enc_layers'],
        dec_layers=hpos['dec_layers'],
        hidden_size=hpos['hidden_size'],
        use_sigmoid_decoder=hpos['use_sigmoid_decoder'],
        enc_init=hpos['enc_init'],
        dec_init=hpos['dec_init'],
        call_kwargs={},
        init_model_with_random=hpos['init_model_with_random'],
    )
    

def make_configs_initial(hpos: pd.Series):
    config_norm_x = NormalizationSettings(
        **{s.replace('prep_x_', ''): hpos[s] for s in hpos.index if 'prep_x' in s})
    config_norm_y = NormalizationSettings(
        **{s.replace('prep_y_', ''): hpos[s] for s in hpos.index if 'prep_y' in s})
    config_filter = FilterSettings(
        **{s: hpos[s] for s in hpos.index if 'filt' in s})
    config_training = TrainingConfig(
        batch_size=hpos['batch_size'],
        epochs=hpos['epochs'],
        patience=hpos['patience'],
        learning_rate=hpos['learning_rate'],
        loss_func=hpos['loss_func'],
        use_dropout=hpos['use_dropout'],
        dropout_rate=hpos['dropout_rate'],
        use_l2_reg=hpos['use_l2_reg'],
        l2_reg_alpha=hpos['l2_reg_alpha'],
        use_kl_div=hpos['use_kl_div'],
        kl_weight=hpos['kl_weight'],
        print_every=hpos['print_every']
    )
    config_optimisation = OptimizationConfig(**{s: hpos[s] for s in OptimizationConfig.__annotations__.keys()})
    config_dataset = DatasetConfig(**{s: hpos[s] for s in DatasetConfig.__annotations__.keys()})
    
    return config_norm_x, config_norm_y, config_filter, config_optimisation, config_dataset, config_training
