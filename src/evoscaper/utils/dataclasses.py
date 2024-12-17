

from dataclasses import dataclass
from typing import List


@dataclass
class DatasetConfig:
    seed_dataset: int
    include_diffs: bool
    objective_col: str
    output_species: List[str]
    signal_species: List[str]
    total_ds_max: int
    train_split: float
    x_type: str
    filenames_train_table: List[str]
    filenames_verify_table: List[str]
    filenames_train_config: List[str]
    filenames_verify_config: List[str]
    use_test_data: bool


@dataclass
class FilterSettings:
    """ Filter data before creating the dataset. """
    filt_x_nans: bool = True
    filt_y_nans: bool = True
    filt_sensitivity_nans: bool = True
    filt_precision_nans: bool = True
    filt_n_same_x_max: int = 1
    filt_n_same_x_max_bins: int = 15


@dataclass
class NormalizationSettings:
    """
    Configuration settings for data normalization.
    """
    negative: bool = False
    logscale: bool = False
    standardise: bool = True
    min_max: bool = False
    robust_scaling: bool = False
    categorical: bool = False
    categorical_onehot: bool = False
    categorical_n_bins: int = 10
    categorical_method: str = 'equal_width'


@dataclass
class ModelConfig:
    seed_arch: int
    model: str
    enc_layers: List[int]
    dec_layers: List[int]
    decoder_head: int
    hidden_size: int
    use_sigmoid_decoder: str
    enc_init: str
    dec_init: str
    activation: str
    call_kwargs: dict
    init_model_with_random: bool


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    loss_func: str
    use_dropout: bool
    dropout_rate: float
    use_l2_reg: bool
    l2_reg_alpha: float
    use_kl_div: bool
    kl_weight: float
    print_every: int


@dataclass
class OptimizationConfig:
    seed_opt: int
    opt_method: str
    opt_min_lr: float
    opt_min_delta: float
    learning_rate_sched: str
    use_warmup: bool
    warmup_epochs: int
