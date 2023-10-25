from .configuration_utils import load_item_from_dict

from typing import Union, Dict, Tuple

import numpy as np
import torch

NxM_JSON_CONFIG_KEY = "nxm_config"
N_JSON_KEY = "N"
M_JSON_KEY = "M"
DEFAULT_N = 4
DEFAULT_M = 2

TensorType = Union[torch.Tensor, np.ndarray]


def load_nxm_config(
    config: Dict,
    logger
) -> Tuple[int, int]:

    n_config = load_item_from_dict(
        config,
        N_JSON_KEY,
        DEFAULT_N,
        logger
    )
    m_config = load_item_from_dict(
        config,
        M_JSON_KEY,
        DEFAULT_M,
        logger
    )

    if m_config > n_config:
        logger.error("M should not be larger than N")
        raise ValueError

    return n_config, m_config


def add_nxm_config(
    arguments: Dict,
    n: int = DEFAULT_N,
    m: int = DEFAULT_M
) -> Dict:
    """
    Add the appropriate N:M arguments to the provided dictionary.

    Parameters:
        arguments (Dict): Dictionary of any other options that may need to be added
        n (int): Chunk size for N:M compression
        m (int): Number of values to retain
    """
    arguments[N_JSON_KEY] = n
    arguments[M_JSON_KEY] = m
    return arguments


def maskNxM(
    parameter: TensorType,
    n: int,
    m: int
) -> TensorType:
    """
    Accepts either a torch.Tensor or numpy.ndarray and generates a floating point mask of 1's and 0's
    corresponding to the locations that should be retained for NxM pruning. The appropriate ranking mechanism
    should already be built into the parameter when this method is called.
    """

    if type(parameter) is torch.Tensor:
        out_neurons, in_neurons = parameter.size()

        with torch.no_grad():
            groups = parameter.reshape(out_neurons, -1, n)
            zeros = torch.zeros(1, 1, 1, device=parameter.device)
            ones = torch.ones(1, 1, 1, device=parameter.device)

            percentile = m / n
            quantiles = torch.quantile(groups, percentile, -1, keepdim=True)
            mask = torch.where(groups > quantiles, ones, zeros).reshape(out_neurons, in_neurons)
    else:
        out_neurons, in_neurons = parameter.shape
        percentile = (100 * m) / n

        groups = parameter.reshape(out_neurons, -1, n)
        group_thresholds = np.percentile(groups, percentile, axis=-1, keepdims=True)
        mask = (groups > group_thresholds).astype(np.float32).reshape(out_neurons, in_neurons)

    return mask
