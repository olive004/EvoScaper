# %%
# # Simple representation space tests with an FCN

# %% [markdown]
# ## Imports
#

# %%
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.results.analytics.naming import get_true_interaction_cols
from synbio_morpher.utils.data.data_format_tools.common import write_json
from typing import List
from functools import partial

import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
import equinox as eqx
import optax  # https://github.com/deepmind/optax
from jaxtyping import Array, Float, Int  # https://github.com/google/jaxtyping
import argparse

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import r2_score

import nni
import wandb

from datetime import datetime
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

jax.config.update('jax_platform_name', 'cpu')

logger = logging.getLogger('MLP')

jax.devices()

# %%
# Make sure GPU is actually working
jnp.arange(9)

# %% [markdown]
# ## Load data
#

# %%


def load_data():
    fn = '../../data/processed/ensemble_mutation_effect_analysis/2023_07_17_105328/tabulated_mutation_info.csv'
    fn_test_data = '../../data/raw/ensemble_mutation_effect_analysis/2023_10_03_204819/tabulated_mutation_info.csv'
    data = pd.read_csv(fn)
    try:
        data.drop(columns=['Unnamed: 0'], inplace=True)
    except:
        pass
    return data
# %%
# https://coderzcolumn.com/tutorials/artificial-intelligence/haiku-cnn


class MLP(hk.Module):

    def __init__(self, layer_sizes: List[int], n_head: int, use_categorical: bool):
        super().__init__(name="FCN")
        self.layers = self.create_layers(layer_sizes, n_head, use_categorical)

    def create_layers(self, layer_sizes: List[int], n_head: int, use_categorical: bool):
        sizes = layer_sizes + [n_head]
        l = []
        for i, s in enumerate(sizes):
            if l:
                l.append(jax.nn.relu)
                if np.mod(i, 2) == 0:
                    l.append(jax.nn.sigmoid)
            # if sj == n_head:
            #     l.append(eqx.nn.Dropout(p=0.4))

            # He initialisation
            l.append(
                hk.Linear(s, w_init=hk.initializers.VarianceScaling(scale=2.0))
            )

        if use_categorical:
            l.append(jax.nn.log_softmax)
        return l

    def __call__(self, x: Float[Array, " num_interactions"], inference: bool = False, seed: int = 0) -> Float[Array, " n_head"]:
        for i, layer in enumerate(self.layers):
            kwargs = {} if not type(layer) == eqx.nn.Dropout else {
                'inference': inference, 'key': jax.random.PRNGKey(seed)}

            x = layer(x, **kwargs)

            if inference:
                df = pd.DataFrame(data=np.array(x), columns=['0'])
                wandb.log({f'emb_{i}_{type(layer)}': df})
        return x


def MLP_fn(x, init_kwargs: dict = {}, call_kwargs: dict = {}):
    model = MLP(**init_kwargs)
    return model(x, **call_kwargs)

# %% [markdown]
# ## Losses
#

# %%


def loss_fn(
    params, rng,
    model: MLP, x: Float[Array, " batch n_interactions"], y: Int[Array, " batch"],
    l2_reg_alpha: Float,
    loss_type: str = 'categorical'
) -> Float[Array, ""]:

    pred_y = model.apply(params, rng, x)
    if loss_type == 'categorical':
        loss = cross_entropy(y, pred_y, num_classes=pred_y.shape[-1]) / len(x)
    else:
        loss = mse_loss(y, pred_y.flatten())

    # Add L2 loss
    # loss += sum(
    #     l2_loss(w, alpha=l2_reg_alpha)
    #     for w in jax.tree_util.tree_leaves(params)
    # )
    return loss


def l2_loss(weights, alpha):
    return alpha * (weights ** 2).mean()


def cross_entropy(y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"], num_classes: int):
    one_hot_actual = jax.nn.one_hot(y, num_classes=num_classes)
    return optax.softmax_cross_entropy(pred_y, one_hot_actual).sum()


def mse_loss(y, pred_y):
    return jnp.mean(jnp.square(pred_y - y))


def update_params(optimiser, params, grads):
    updates, optimizer_state = optimiser.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params


@eqx.filter_jit
def compute_accuracy_categorical(
    params, rng, model: MLP, x: Float[Array, "batch num_interactions"], y: Int[Array, " batch n_head"]
) -> Float[Array, ""]:
    pred_y = model.apply(params, rng, x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)


@eqx.filter_jit
def compute_accuracy_regression(
    params, rng, model: MLP, x: Float[Array, "batch num_interactions"], y: Int[Array, " batch n_head"],
    threshold=0.1
) -> Float[Array, ""]:
    pred_y = model.apply(params, rng, x)
    return jnp.mean(jnp.abs(y - pred_y) <= threshold)

# %% [markdown]
# ## Hyperparameters
#

# %%


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='MLP')
    parser.add_argument("--n_batches", type=int, default=None)
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='Total number of steps or epochs to run training for')
    parser.add_argument('--linear_layer_size', type=int, default=None,
                        help='Architecural parameter. Size of output of second linear layer')
    parser.add_argument("--num_linear_layers", type=int, default=None)
    parser.add_argument("--make_encoder_layers", type=bool, default=False)

    args, _ = parser.parse_known_args()
    return args

# %% [markdown]
# ## Initialise
#

# %% [markdown]
# ### Input
#
# To make sure that there is little repetition in the dataset, the uniqueness of each sample will be judged. For genetic circuits,there is a lot of sparsity, as most biological sequences do not interact. Therefore, there may be an overrepresentation of some circuit topologies.
#

# %%


def custom_round(x, base=5):
    return base * round(x/base)


def convert_to_scientific_exponent(x, numerical_resolution: dict):
    exp_not = f'{x:.0e}'.split('e')
    resolution = numerical_resolution[int(exp_not[1])]
    base = int(10 / resolution)
    pre = custom_round(int(exp_not[0]), base=base)
    return int(exp_not[1]) + pre / 10


def drop_duplicates_keep_first_n(df, column, n):
    """ GCG """
    indices = df[df.duplicated(subset=column, keep=False)].groupby(
        column).head(n).index
    all_duplicates_indices = df[df.duplicated(subset=column, keep=False)].index
    to_drop = list(set(all_duplicates_indices) - set(indices))
    df2 = df.drop(to_drop)
    return df2

# %% [markdown]
# ## Train
#

# %%


def train_step(params, rng, model, x, y, optimiser, optimiser_state, l2_reg_alpha):

    loss, grads = jax.value_and_grad(loss_fn)(
        params, rng, model, x, y, l2_reg_alpha)

    updates, optimiser_state = optimiser.update(grads, optimiser_state)
    params = optax.apply_updates(params, updates)

    return params, loss, grads


def eval_step(params, rng, model: MLP, x, y, l2_reg_alpha):
    """ Return the average of loss and accuracy on validation data """
    # pred_y = model.apply(params, rng, x)
    # return accuracy_score(y, jnp.argmax(pred_y, axis=1))
    loss = loss_fn(params, rng, model, x, y, l2_reg_alpha)
    acc = compute_accuracy(params, rng, model, x, y)
    return acc, loss


def train(params, rng, model, x_train, y_train, x_val, y_val,
          optimiser, optimiser_state,
          l2_reg_alpha, epochs, batch_size: int,
          save_every: int = 50):
    saves = {}
    n_batches = (x_train.shape[0]//batch_size)+1
    for e in range(epochs):

        for batch in range(n_batches):
            start = int(batch*batch_size)
            end = int((batch+1)*batch_size) if batch != n_batches - 1 else None

            # Single batch of data
            x_batch, y_batch = x_train[start:end], y_train[start:end]

            if len(x_batch) and len(y_batch):
                params, train_loss, grads = train_step(
                    params, rng, model, x_batch, y_batch, optimiser, optimiser_state, l2_reg_alpha)

        val_acc, val_loss = eval_step(
            params, rng, model, x_val, y_val, l2_reg_alpha)

        if np.mod(e, save_every) == 0:
            saves[e] = {
                'params': params,
                'grads': grads,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }
            print(
                f'Epoch {e} / {epochs} -\t\t Train loss: {train_loss}\tVal loss: {val_loss}\tVal accuracy: {val_acc}')

            nni.report_intermediate_result({
                'train_loss': np.float32(train_loss),
                'val_acc': np.float32(val_acc),
                'default': np.float32(val_loss)
            })
    nni.report_final_result({
        'train_loss': np.float32(train_loss),
        'val_acc': np.float32(val_acc),
        'default': np.float32(val_loss)
    })
    return params, saves


USE_CATEGORICAL = False
compute_accuracy = compute_accuracy_categorical if USE_CATEGORICAL else compute_accuracy_regression
loss_fn = partial(
    loss_fn, loss_type='categorical' if USE_CATEGORICAL else 'mse')

# %%


def main(args):

    data = load_data()

    BATCH_SIZE = 128
    N_BATCHES = args['n_batches']
    TOTAL_DS = BATCH_SIZE * N_BATCHES
    MAX_TOTAL_DS = TOTAL_DS
    train_split_perc = 0.8
    TRAIN_SPLIT = int(train_split_perc * TOTAL_DS)
    TEST_SPLIT = TOTAL_DS - TRAIN_SPLIT
    LEARNING_RATE = args['lr']
    LEARNING_RATE_SCHED = 'cosine_decay'
    # LEARNING_RATE_SCHED = 'constant'
    WARMUP_EPOCHS = 20
    L2_REG_ALPHA = 0.01
    EPOCHS = args['epochs']
    PRINT_EVERY = EPOCHS // 30
    SEED = 1
    INPUT_SPECIES = 'RNA_1'
    target_circ_func = 'sensitivity'

    # MLP Architecture
    layer_size = int(args['linear_layer_size'])
    LAYER_SIZES = [layer_size] * int(args['num_linear_layers'])
    if args['make_encoder_layers']:
        LAYER_SIZES[1:-1] = (np.array(LAYER_SIZES[1:-1]) / 2).astype(int)

    USE_DROPOUT = False
    USE_L2_REG = False
    USE_WARMUP = False

    save_path = 'saves_' + str(datetime.now()).split(' ')[0].replace(
        '-', '_') + '__' + str(datetime.now()).split(' ')[-1].split('.')[0].replace(':', '_')

    rng = jax.random.PRNGKey(SEED)

    # %%
    vectorized_convert_to_scientific_exponent = np.vectorize(
        convert_to_scientific_exponent)
    filt = data['sample_name'] == INPUT_SPECIES
    numerical_resolution = 2

    # Balance the dataset
    df = drop_duplicates_keep_first_n(data[filt], get_true_interaction_cols(
        data, 'energies', remove_symmetrical=True), n=200)
    df['sensitivity'] = df['sensitivity'].round(
        np.abs(int(f'{df["sensitivity"].min():.0e}'.split('e')[1]))-1)
    df = drop_duplicates_keep_first_n(
        df, column='sensitivity', n=200)

    TOTAL_DS = np.min([TOTAL_DS, MAX_TOTAL_DS, len(df)])

    # %%
    x = df[get_true_interaction_cols(
        data, 'energies', remove_symmetrical=True)].iloc[:TOTAL_DS].values
    y = df['sensitivity'].iloc[:TOTAL_DS].to_numpy()

    if USE_CATEGORICAL:
        y_map = {k: numerical_resolution for k in np.arange(int(f'{y[y != 0].min():.0e}'.split(
            'e')[1])-1, np.max([int(f'{y.max():.0e}'.split('e')[1])+1, 0 + 1]))}
        y_map[-6] = 1
        y_map[-5] = 1
        y_map[-4] = 4
        y_map[-3] = 2
        y_map[-1] = 3
        y = jax.tree_util.tree_map(partial(
            vectorized_convert_to_scientific_exponent, numerical_resolution=y_map), y)
        y = np.interp(y, sorted(np.unique(y)), np.arange(
            len(sorted(np.unique(y))))).astype(int)
    else:
        zero_log_replacement = -10.0
        y = np.where(y != 0, np.log10(y), zero_log_replacement)

    x, y = shuffle(x, y, random_state=SEED)

    N_HEAD = len(np.unique(y)) if USE_CATEGORICAL else 1

    if x.shape[0] < TOTAL_DS:
        print(
            f'WARNING: The filtered data is not as large as the requested total dataset size: {x.shape[0]} vs. requested {TOTAL_DS}')

    # %% [markdown]
    # #### Scale input
    #

    # %%
    xscaler, yscaler = MinMaxScaler(), MinMaxScaler()
    x = xscaler.fit_transform(x)
    # %%
    x_train, y_train = x[:TRAIN_SPLIT], y[:TRAIN_SPLIT]
    x_val, y_val = x[-TEST_SPLIT:], y[-TEST_SPLIT:]

    # %% [markdown]
    # ### Initialise model
    #

    # %%
    model = hk.transform(partial(MLP_fn, init_kwargs={
        'layer_sizes': LAYER_SIZES, 'n_head': N_HEAD, 'use_categorical': USE_CATEGORICAL}))

    params = model.init(rng, x[:2])

    # %% [markdown]
    # ### Optimiser
    #

    # %%
    if LEARNING_RATE_SCHED == 'cosine_decay':
        learning_rate_scheduler = optax.cosine_decay_schedule(
            LEARNING_RATE, decay_steps=EPOCHS, alpha=L2_REG_ALPHA)
    else:
        learning_rate_scheduler = LEARNING_RATE
    optimiser = optax.sgd(learning_rate=learning_rate_scheduler)

    if USE_WARMUP:
        warmup_fn = optax.linear_schedule(
            init_value=0., end_value=LEARNING_RATE,
            transition_steps=WARMUP_EPOCHS * N_BATCHES)
        cosine_epochs = max(EPOCHS - WARMUP_EPOCHS, 1)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=LEARNING_RATE,
            decay_steps=cosine_epochs * N_BATCHES)
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[WARMUP_EPOCHS * N_BATCHES])
        optimiser = optax.sgd(learning_rate=schedule_fn)

    optimiser_state = optimiser.init(x)

    # %%
    params, saves = train(params, rng, model, x_train, y_train, x_val, y_val, optimiser, optimiser_state,
                          l2_reg_alpha=L2_REG_ALPHA, epochs=EPOCHS, batch_size=BATCH_SIZE,
                          save_every=PRINT_EVERY)  # int(STEPS // 15))

    # %% [markdown]
    # ## Visualise
    #

    # %%
    plt.figure(figsize=(7*3, 6))
    ax = plt.subplot(1, 3, 1)
    plt.plot(list(saves.keys()), [v['train_loss'] for v in saves.values()])
    plt.ylabel('train_loss')
    plt.xlabel('step')
    ax = plt.subplot(1, 3, 2)
    plt.plot(list(saves.keys()), [v['val_loss'] for v in saves.values()])
    plt.ylabel('val_loss')
    plt.xlabel('step')
    ax = plt.subplot(1, 3, 3)
    plt.plot(list(saves.keys()), [v['val_accuracy'] for v in saves.values()])
    plt.ylabel('val_accuracy')
    plt.xlabel('step')

    plt.savefig('training_summary.png')

    # %%
    # params = arrayise(params)
    predicted = model.apply(params, rng, x_val)
    if USE_CATEGORICAL:
        predicted = jnp.argmax(predicted, axis=1)

    sns.scatterplot(x=y_val, y=predicted.flatten(), alpha=0.1)
    plt.title('Predicted vs. actual labels')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')

    print('The R2 score is ', r2_score(y_val, predicted))
    print('The R2 score with weighted variance is ', r2_score(
        y_val, predicted, multioutput='variance_weighted'))

    plt.savefig('predicted_actual.png')

    # %%
    # write_json(saves, out_path=save_path)


if __name__ == '__main__':
    # Go to http://127.0.0.1:8080
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(nni.utils.merge_parameter(get_params(), tuner_params))
        # params = {"lr": 0.0001, "epochs": 2, "n_batches": 500,
        #           "linear_layer_size": 32, "num_linear_layers": 4, "make_encoder_layers": False}

        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
