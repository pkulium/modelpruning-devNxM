import torch

from admm_ds import nxm_utils, quantization_utils


def sparsify_then_quantize(
    parameter: torch.Tensor,
    num_bits: int,
    offset: int,
    n: int,
    m: int,
    norm=2
):
    with torch.no_grad():
        scores = parameter.abs()

        mask = nxm_utils.maskNxM(scores, n, m)

        sparse_view = parameter * mask

        optimal_scale, quantized_tensor = quantization_utils.fixedOffsetQuantizationSearch(
            sparse_view,
            num_bits,
            offset,
            norm
        )

        pruning_diff = quantization_utils.evaluateQuantization(sparse_view, parameter, norm=norm)
        quantized_diff = quantization_utils.evaluateQuantization(quantized_tensor, sparse_view, norm=norm)
        total_diff = quantization_utils.evaluateQuantization(quantized_tensor, parameter, norm=norm)

        print("{},{},{},{}".format(total_diff, quantized_diff, pruning_diff, optimal_scale))


def quantize_then_sparsify(
    parameter: torch.Tensor,
    num_bits: int,
    offset: int,
    n: int,
    m: int,
    norm=2
):
    with torch.no_grad():

        optimal_scale, quantized_tensor = quantization_utils.fixedOffsetQuantizationSearch(
            parameter,
            num_bits,
            offset,
            norm
        )

        scores = quantized_tensor.abs()
        mask = nxm_utils.maskNxM(scores, n, m)
        pruned_tensor = quantized_tensor * mask

        quantized_diff = quantization_utils.evaluateQuantization(quantized_tensor, parameter, norm=norm)
        pruning_diff = quantization_utils.evaluateQuantization(pruned_tensor, quantized_tensor, norm=norm)
        total_diff = quantization_utils.evaluateQuantization(pruned_tensor, parameter, norm=norm)

        print("{},{},{},{}".format(total_diff, quantized_diff, pruning_diff, optimal_scale))
