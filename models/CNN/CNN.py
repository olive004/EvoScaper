"""
A simple CNN for predicting dynamics from synthetic genetic circuits.
Script mirroring the `01_simple_fcn.ipynb`

"""

import os
import argparse
import logging
import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nni.utils import merge_parameter
from torchvision import datasets, transforms

logger = logging.getLogger('mnist_AutoML')

from typing import Optional, List, Callable, Dict, Any, Tuple
from dataclasses import dataclass
import os

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
import tensorflow as tf
from tensorboard.plugins import projector
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
from sklearn.manifold import TSNE

import equinox as eqx
import wandb

import seaborn as sns
import matplotlib.pyplot as plt

jax.config.update('jax_platform_name', 'cpu')

from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.misc.numerical import make_symmetrical_matrix_from_sequence
from synbio_morpher.utils.misc.string_handling import convert_liststr_to_list
from synbio_morpher.utils.misc.type_handling import flatten_listlike
from synbio_morpher.utils.results.analytics.naming import get_true_names_analytics, get_true_interaction_cols
from synbio_morpher.utils.parameter_inference.interpolation_grid import create_parameter_range

from src.utils.math import calculate_conv_output, convert_to_scientific_exponent

jax.devices()


class CNN(eqx.Module):
    layers: list

    def __init__(self, key, n_channels: int, out_channels: int, n_head: int,
                 kernel_size: int = 3, in_dim1: int = 3,
                 max_pool_kernel_size: int = 2):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.

        # [(W−K+2P)/S]+1
        # 1D
        # conv_out = (out_channels - 1 * (kernel_size - 1) - 1) // 1 + 1
        # 2D
        # conv_out = (in_dim1 - kernel_size + 2 * 0) // 1 + 1
        out1 = calculate_conv_output(in_dim1, kernel_size, padding=0, stride=1)
        out2 = calculate_conv_output(out1, max_pool_kernel_size, padding=0, stride=1)

        self.layers = [
            eqx.nn.Conv2d(in_channels=n_channels, out_channels=out_channels,
                          kernel_size=kernel_size, key=key1),
            eqx.nn.MaxPool2d(kernel_size=max_pool_kernel_size, stride=1),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(np.power(out2, 2) * out_channels, 
                          np.power(out2, 2) * out_channels * 4, key=key2),
            jax.nn.sigmoid,
            eqx.nn.Linear(np.power(out2, 2) * out_channels * 4,
                          np.power(out2, 2) * out_channels, key=key3),
            jax.nn.relu,
            # eqx.nn.Dropout(p=0.4),
            eqx.nn.Linear(np.power(out2, 2) * out_channels, n_head, key=key4),
            jax.nn.log_softmax
        ]

    def __call__(self, x: Float[Array, "1 28 28"], inference: bool = False) -> Float[Array, "10"]:
        for i, layer in enumerate(self.layers):
            kwargs = {} if not type(layer) == eqx.nn.Dropout else {
                'inference': inference, 'key': jax.random.PRNGKey(0)}

            x = layer(x, **kwargs)
            
            # wandb.log({f'emb_{i}_{type(layer)}': x})
        return x
    
    
def loss(
    model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    # Our input has the shape (BATCH_SIZE, 1, 28, 28), but our model operations on
    # a single input input image of shape (1, 28, 28).
    #
    # Therefore, we have to use jax.vmap, which in this case maps our model over the
    # leading (batch) axis.
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)


def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    # y are the true targets, and should be integers 0-9.
    # pred_y are the log-softmax'd predictions.
    pred_y = jnp.take_along_axis(pred_y, y, axis=1)
    # pred_y = jnp.take_along_axis(pred_y, y, axis=1)
    return -jnp.mean(pred_y)


@eqx.filter_jit
def compute_accuracy(
    model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    pred_y = jax.vmap(model)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)

def evaluate(model: CNN, testloader: torch.utils.data.DataLoader):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    avg_loss = 0
    avg_acc = 0
    for x, y in zip(*testloader):
        # x = x.numpy()
        # y = y.numpy()
        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        avg_loss += loss(model, x, y)
        avg_acc += compute_accuracy(model, x, y)
    return avg_loss / len(testloader), avg_acc / len(testloader)


def train(
    model: CNN,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
) -> CNN:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: CNN,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, " batch"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    saves = {}
    for step, (x, y) in zip(range(steps), infinite_trainloader()):
    # for step, (x, y) in zip(range(steps), trainloader):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        # x = x.numpy()
        # y = y.numpy()
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model, testloader)
            # print(
            #     f"{step=}, train_loss={train_loss.item()}, "
            #     f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            # )
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                step, step * len(x), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
            saves[step] = {
                'opt_state': opt_state,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }
    return model, saves


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy


def load_data(filepath_data):
    """ The file path could be something like 
    'data/processed/ensemble_mutation_effect_analysis/2023_07_17_105328/tabulated_mutation_info.csv'
    """
    data = pd.read_csv(filepath_data)


def main(args):
    
    BATCH_SIZE = 128
    N_BATCHES = 10000
    TRAIN_SPLIT = int(0.8 * N_BATCHES)
    TEST_SPLIT = N_BATCHES - TRAIN_SPLIT
    LEARNING_RATE = 3e-4
    STEPS = 10000
    PRINT_EVERY = 100
    SEED = 5678
    TOTAL_DS = BATCH_SIZE * N_BATCHES

    # Architecture
    N_CHANNELS = 1
    OUT_CHANNELS = 3
    KERNEL_SIZE = 1
    MAX_POOL_KERNEL_SIZE = 1
    
    
    data = load_data(args['filepath_data'])
    n_samples = len(data['sample_name'].unique())

    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key, 2)

    x = data[get_true_interaction_cols(data, 'binding_rates_dissociation', remove_symmetrical=True)].iloc[:TOTAL_DS].values
    x = np.expand_dims(np.array([make_symmetrical_matrix_from_sequence(xx, n_samples) for xx in x]), axis=1)

    y = data['sensitivity_wrt_species-6'].iloc[:TOTAL_DS].to_numpy()
    y = np.array([convert_to_scientific_exponent(yy) for yy in y])[None, :] * -1

    N_HEAD = len(np.unique(y))
    
    ######### Make sure things work
    
    model = CNN(subkey, n_channels=N_CHANNELS, out_channels=OUT_CHANNELS, n_head=N_HEAD, kernel_size=KERNEL_SIZE, max_pool_kernel_size=MAX_POOL_KERNEL_SIZE)
    
    # Example loss
    loss_value = loss(model, x[:10], y[:10])
    print(loss_value.shape)  # scalar loss
    # Example inference
    output = jax.vmap(model)(x[:10])
    print(output.shape)  # batch of predictions
    
    params, static = eqx.partition(model, eqx.is_array)

    def loss2(params, static, x, y):
        model = eqx.combine(params, static)
        return loss(model, x, y)

    loss_value, grads = jax.value_and_grad(loss2)(params, static, x[:5], y[:5])
    print(loss_value)
    
    ##########

    loss = eqx.filter_jit(loss)  # JIT our loss function from earlier!

    dataloader = (x.reshape(*[N_BATCHES, BATCH_SIZE] + list(x.shape[1:])), y.reshape(N_BATCHES, BATCH_SIZE, 1))
    evaluate(model, dataloader)
    
    optim = optax.adamw(LEARNING_RATE)
    
    #########
    
    trainloader = zip(dataloader[0][:TRAIN_SPLIT], dataloader[1][:TRAIN_SPLIT])
    testloader = (dataloader[0][:TEST_SPLIT], dataloader[1][:TEST_SPLIT])
    
    model, saves = train(model, trainloader, testloader, optim, STEPS, PRINT_EVERY)
    
    ##############################################3
    
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data_dir = args['data_dir']

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True, **kwargs)

    hidden_size = args['hidden_size']

    model = Net(hidden_size=hidden_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'],
                          momentum=args['momentum'])

    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader)

        # report intermediate result
        nni.report_intermediate_result(test_acc)
        logger.debug('test accuracy %g', test_acc)
        logger.debug('Pipe send intermediate result done.')

    # report final result
    nni.report_final_result(test_acc)
    logger.debug('Final result is %g', test_acc)
    logger.debug('Send final result done.')


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--data_dir", type=str,
                        default='./data', help="data directory")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--batch_num", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=512, metavar='N',
                        help='hidden layer size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')


    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
