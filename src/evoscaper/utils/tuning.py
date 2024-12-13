

from evoscaper.utils.dataclasses import (
    NormalizationSettings, FilterSettings, DatasetConfig, ModelConfig, TrainingConfig, OptimizationConfig)
from evoscaper.model.shared import get_activation_fn


def make_config_dataset(hpos: dict):
    return DatasetConfig(
        seed=hpos['seed_dataset'],
        include_diffs=hpos['include_diffs'],
        objective_col=hpos['objective_col'],
        output_species=hpos['output_species'],
        total_ds_max=hpos['total_ds_max'],
        train_split=hpos['train_split'],
        x_type=hpos['x_type'],
        filenames_train_table=hpos['filenames_train_table'],
        filenames_verify_table=hpos['filenames_verify_table'],
    )


def make_configs_initial(hpos: dict):
    x_norm_settings = NormalizationSettings(
        **{s.replace('prep_x_', ''): hpos[s] for s in hpos.columns if 'prep_x' in s})
    y_norm_settings = NormalizationSettings(
        **{s.replace('prep_y_', ''): hpos[s] for s in hpos.columns if 'prep_y' in s})
    filter_settings = FilterSettings(
        **{s: hpos[s] for s in hpos.columns if 'filt' in s})
    config_training = TrainingConfig(
        batch_size=hpos['batch_size'],
        epochs=hpos['epochs'],
        patience=hpos['patience'],
        learning_rate=hpos['learning_rate'],
        learning_rate_sched=hpos['learning_rate_sched'],
        loss_func=hpos['loss_func'],
        use_dropout=hpos['use_dropout'],
        dropout_rate=hpos['dropout_rate'],
        use_l2_reg=hpos['use_l2_reg'],
        l2_reg_alpha=hpos['l2_reg_alpha'],
        use_kl_div=hpos['use_kl_div'],
        kl_weight=hpos['kl_weight'],
        use_warmup=hpos['use_warmup'],
        warmup_epochs=hpos['warmup_epochs'],
        print_every=hpos['print_every']
    )
    OptimizationConfig(**{s: hpos[s] for s in OptimizationConfig.__dict__.keys()})
    
    return x_norm_settings, y_norm_settings, filter_settings, make_config_dataset(hpos), config_training


def make_configs(x, hpos: dict):

    model_settings = ModelConfig(
        seed=hpos['seed_arch'],
        decoder_head=x.shape[-1],
        activation=get_activation_fn(hpos['activation']),
        use_sigmoid_decoder=hpos['use_sigmoid_decoder'],
        model=hpos['model'],
        enc_layers=hpos['enc_layers'],
        dec_layers=hpos['dec_layers'],
        decoder_head=hpos['decoder_head'],
        hidden_size=hpos['hidden_size'],
        use_sigmoid_decoder=hpos['use_sigmoid_decoder'],
        enc_init=hpos['enc_init'],
        dec_init=hpos['dec_init'],
        activation=hpos['activation'],
        call_kwargs={},
        init_model_with_random=hpos['init_model_with_random']
    )
