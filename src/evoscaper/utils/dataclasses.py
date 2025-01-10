

from dataclasses import dataclass, field
from typing import List


@dataclass
class DatasetConfig:
    seed_dataset: int
    include_diffs: bool
    objective_col: List[str]
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
    
    def __post_init__(self):
        self.objective_col = [self.objective_col] if isinstance(self.objective_col, str) else list(self.objective_col)


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
    standardise: bool = False
    min_max: bool = True
    robust_scaling: bool = True
    categorical: bool = False
    categorical_onehot: bool = False
    categorical_n_bins: int = 10
    categorical_method: str = 'equal_width'


@dataclass
class ModelConfig:
    seed_arch: int
    decoder_head: int
    hidden_size: int
    call_kwargs: dict
    enc_ls: int
    dec_ls: int
    num_enc_layers: int
    num_dec_layers: int
    model: str = 'CVAE'
    use_sigmoid_decoder: bool = False
    enc_init: str = 'HeNormal'
    dec_init: str = 'HeNormal'
    activation: str = 'leaky_relu'
    init_model_with_random: bool = False
    factor_expanding_ls: int = 1
    factor_contracting_ls: int = 1
    enc_layers: List[int] = field(init=False)
    dec_layers: List[int] = field(init=False)

    def __post_init__(self):
        self.enc_layers = [self.enc_ls] * self.num_enc_layers
        self.dec_layers = [self.dec_ls] * self.num_dec_layers
        if self.num_enc_layers > 1:
            self.enc_layers[0] = self.enc_layers[0] * self.factor_expanding_ls
            self.enc_layers[-1] = self.enc_layers[-1] * self.factor_contracting_ls
        if self.num_dec_layers > 1:
            self.dec_layers[0] = self.dec_layers[0] * self.factor_contracting_ls
            self.dec_layers[-1] = self.dec_layers[-1] * self.factor_expanding_ls


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
