from typing import Tuple, Dict

import numpy as np
import torch

from .configuration_utils import load_item_from_dict

TensorType = torch.Tensor

QUANTIZATION_CONFIG_KEY = "quantization_config"

BITS_KEY = "precision"
DEFAULT_BITS = 8
# Should probably be an enum (but not sure that's iterable which is necessary for now)
VALID_QUANTIZATIONS = [1, 2, 4, 8]

NORM_KEY = "norm"
DEFAULT_NORM = 2
VALID_NORMS = [1, 2]

CHUNK_SIZE_KEY = "chunk_size"
DEFAULT_CHUNK_SIZE = -1

GROUP_KEY = "groups"
DEFAULT_GROUPS = 1


def load_quantization_config(
    config: Dict,
    logger
) -> Tuple[int, int]:

    num_bits = load_item_from_dict(
        config,
        BITS_KEY,
        DEFAULT_BITS,
        logger,
        validation_values=VALID_QUANTIZATIONS
    )
    norm = load_item_from_dict(
        config,
        NORM_KEY,
        DEFAULT_NORM,
        logger,
        validation_values=VALID_NORMS
    )
    chunk_size = load_item_from_dict(
        config,
        CHUNK_SIZE_KEY,
        DEFAULT_CHUNK_SIZE,
        logger
    )
    num_groups = load_item_from_dict(
        config,
        GROUP_KEY,
        DEFAULT_GROUPS,
        logger
    )

    return num_bits, norm, chunk_size, num_groups


def add_quantization_config(
    arguments: Dict,
    num_bits: int = DEFAULT_BITS,
    norm: int = DEFAULT_NORM,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    num_groups: int = DEFAULT_GROUPS
) -> Dict:
    """
    Add the appropriate quantization arguments to the provided dictionary.

    Parameters:
        arguments (Dict): Dictionary of any other options that may need to be added
        num_bits (int): Integer compression size
        norm (int): Norm for measuring quantization quality
        chunk_size (int): How many neurons in each quantization group
        groups (int): Number of quantization groups in each neuron
    """
    arguments[BITS_KEY] = num_bits
    arguments[NORM_KEY] = norm
    arguments[CHUNK_SIZE_KEY] = chunk_size
    arguments[GROUP_KEY] = num_groups
    return arguments


def quantizeTensor(
    parameter: TensorType,
    scale: TensorType,
    offset: int,
    num_bits: int
) -> TensorType:
    """
    Quantization method that quantizes using round to nearest integer
    rounding. This method will work effectively through 16-bit quantization
    but can fail at higher integer quantizations since there is an implicit
    assumption that the valid range.

    This method supports either PyTorch or Numpy as the backend for the computation.
    PyTorch will use the device in which the parameter tensor is passed, while numpy is
    always single-threaded CPU. While the numpy version should be more debuggable,
    it suffers from serious performance degradation compared to the PyTorch versions.
    """

    min_num_bit_representation = -2 ** (num_bits - 1)
    max_num_bit_representation = (2 ** (num_bits - 1)) - 1

    if (offset < min_num_bit_representation or
       offset > max_num_bit_representation):
        raise ValueError("Offset of {} not within valid range for {}"
                         .format(offset, num_bits))

    min_integer_representation = min_num_bit_representation + offset
    max_integer_representation = max_num_bit_representation + offset

    with torch.no_grad():
        unbounded_integers = torch.round(parameter / scale)
        bounded_integers = torch.clamp(unbounded_integers,
                                       min=min_integer_representation,
                                       max=max_integer_representation)
        result = bounded_integers * scale

    return result


def evaluateQuantization(
    quantizedParameter: TensorType,
    rawParameter: TensorType,
    norm=2
) -> float:
    """
    Wrapper method for calculating norms for PyTorch.
    """

    with torch.no_grad():
        score = torch.norm(quantizedParameter - rawParameter, p=norm, dim=-1).float()

    return score


def fixedOffsetQuantizationSearch(
    parameter: TensorType,
    num_bits: int,
    offset: int,
    chunk_size: int,
    evaluation_norm: int
) -> Tuple[float, TensorType]:

    float_info = np.finfo(np.float32)
    step_size = 256 * float_info.eps
    out_neurons, in_neurons = parameter.size()
    current_device = parameter.device

    if chunk_size == -1:
        chunk_size = out_neurons

    if out_neurons % chunk_size != 0:
        raise ValueError("Dimensions of out_neurons ({}) and chunk_size ({}) are incompatible".format(
            out_neurons, chunk_size
        ))

    # Same as [out_neurons // chunk_size, -1]
    reshaped_parameter = parameter.reshape([-1, chunk_size * in_neurons])

    with torch.no_grad():
        max_float = parameter.abs().max()
        max_val = parameter.max()

    if max_float == max_val:
        current_scale = max_float / positiveScaleFactor(num_bits)
    else:
        current_scale = max_float / negativeScaleFactor(num_bits)

    # Tensor shapes are of [num_quantization_chunks, 1] in order to ensure that
    # dimension broadcasting occurs in the appropriate manner.
    best_scores = torch.full((out_neurons // chunk_size, 1), float_info.max, device=current_device)
    optimal_scales = torch.zeros((out_neurons // chunk_size, 1), device=current_device)

    while current_scale > 0:
        quantizing_scale = torch.full((out_neurons // chunk_size, 1), current_scale, device=current_device)
        quantized_parameter = quantizeTensor(reshaped_parameter, quantizing_scale, offset, num_bits)

        # Dimension of scores as returned is [num_quantization_chunks], which does not broadcast
        # correctly, so we need to reshape
        scores = evaluateQuantization(quantized_parameter, reshaped_parameter, norm=evaluation_norm)
        scores = scores.reshape((out_neurons // chunk_size, 1))

        replacement_mask = scores < best_scores

        best_scores_temp = torch.where(replacement_mask, scores, best_scores)
        optimal_scales = torch.where(replacement_mask, quantizing_scale, optimal_scales)

        best_scores = best_scores_temp

        current_scale -= step_size

        if replacement_mask.sum() == 0:
            break

    optimal_quantization = quantizeTensor(reshaped_parameter, optimal_scales, offset, num_bits)
    # Return tensor to original [out_neurons, in_neurons] shape
    optimal_quantization = optimal_quantization.reshape((out_neurons, in_neurons))
    return optimal_scales, optimal_quantization


def positiveScaleFactor(num_bits: int):
    return 2 ** (num_bits - 1) - 1


def negativeScaleFactor(num_bits: int):
    return 2 ** (num_bits - 1)


def quantizeSymmetricSimple(
    raw_tensor: TensorType,
    num_bits: int,
    chunk_size: int
) -> TensorType:

    original_shape = raw_tensor.size()

    outer_dimension = 1
    for dim_index in range(raw_tensor.dim() - 1):
        outer_dimension *= raw_tensor.size(dim_index)
    inner_dimension = raw_tensor.size(-1)

    if chunk_size == -1:
        chunk_size = outer_dimension

    if outer_dimension % chunk_size != 0:
        raise ValueError("Dimensions of out_neurons ({}) and chunk_size ({}) are incompatible".format(
            outer_dimension, chunk_size
        ))

    with torch.no_grad():
        # Could use -1 in either side, but this makes it more apparent what the size should be
        # and should be less error prone.
        reshaped_tensor = raw_tensor.view((outer_dimension // chunk_size, chunk_size * inner_dimension))
        zeros = torch.zeros(outer_dimension // chunk_size, device=reshaped_tensor.device)

        # Since integer range is not symmetrical, we need to handle the positive
        # and negative cases slightly differently. We bias ourselves to negative values
        # in the case of a tie since that results in a smaller scale (could be the
        # wrong choice sincce it causes us to underrepresent a large positive value
        # versus over-representing a large negative value).
        max_absolute_values, _ = reshaped_tensor.abs().max(-1)
        max_values, _ = reshaped_tensor.max(-1)
        scale_factors = torch.where(
            max_absolute_values != max_values,
            negativeScaleFactor(num_bits),
            positiveScaleFactor(num_bits)
        )

        scales = max_absolute_values / scale_factors

    quantized_weights = torch.fake_quantize_per_channel_affine(
        reshaped_tensor,
        scales,
        zeros,
        0,
        -negativeScaleFactor(num_bits),
        positiveScaleFactor(num_bits)
    )

    return quantized_weights.view(original_shape)
