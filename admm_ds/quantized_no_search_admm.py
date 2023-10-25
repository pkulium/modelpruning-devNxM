import logging
from typing import Dict

import torch

from .admm_projection import ADMMQuantizedFCLProjection, ADMMMaskedQuantizedFCLProjection
from .layers import FakeQuantizedFullyConnectedLayer, MaskedFakeQuantizedFullyConnectedLayer

from .quantization_utils import (
    add_quantization_config,
    quantizeSymmetricSimple,
    load_quantization_config
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SymmetricModifiedQuantizedProjection(ADMMQuantizedFCLProjection):
    """
    This modified quantized projection technique differes from the original
    ``SymmetricQuantizedProjection`` by modifying the quantization constraint such
    that we are predicated on our scale being solely determined by the largest absolute
    magnitude of a parameter within a chunk. Basically, rather than formulating this as
    the optimization of the parameters into evenly spaced groups, its optimization of the
    parameters such that the error is minimized for the largest magnitude parameter.
    """

    ProjectionName = "MOD_QUANT"

    _module: FakeQuantizedFullyConnectedLayer
    _num_bits: int
    _norm: int
    _chunk_size: int
    _num_groups: int

    def __init__(self, model: torch.nn.Module, targeted_module: str, compression_args: Dict) -> None:
        super(SymmetricModifiedQuantizedProjection, self).__init__(model, targeted_module, compression_args)

        self._num_bits, self._norm, self._chunk_size, self._num_groups = load_quantization_config(compression_args, logger)
        self._module.weights_bits = self._num_bits

    def project(self):
        with torch.no_grad():
            self._z = quantizeSymmetricSimple(
                self._module.weight + self._u,
                self._num_bits,
                self._chunk_size
            )

    @classmethod
    def createConfig(cls, num_bits: int, chunk_size: int) -> Dict:
        from admm_ds.compression_configurations import TransformerADMMConfig

        return TransformerADMMConfig.buildEntry(
            cls,
            add_quantization_config({}, num_bits=num_bits, chunk_size=chunk_size)
        )

    @staticmethod
    def compression_ratio(compression_args: Dict) -> float:
        num_bits, _, _, _ = load_quantization_config(compression_args, logger)
        return float(num_bits) / float(32)


class SymmetricMaskedModifiedQuantizedProjection(ADMMMaskedQuantizedFCLProjection):
    """
    This modified quantized projection technique differes from the above
    ``SymmetricModifiedQuantizedProjection`` by including a masking stage on
    the forward pass to maintain NxM semi-structured sparsity. At this point, the
    system assumes N=4 and M=2 and will parse quantization configurations only.
    """

    ProjectionName = "MASKED_MOD_QUANT"

    _module: MaskedFakeQuantizedFullyConnectedLayer
    _num_bits: int
    _norm: int
    _chunk_size: int
    _num_groups: int

    def __init__(self, model: torch.nn.Module, targeted_module: str, compression_args: Dict) -> None:
        super(SymmetricMaskedModifiedQuantizedProjection, self).__init__(model, targeted_module, compression_args)

        self._num_bits, self._norm, self._chunk_size, self._num_groups = load_quantization_config(compression_args, logger)
        self._module.weights_bits = self._num_bits

    def project(self):
        with torch.no_grad():
            self._z = quantizeSymmetricSimple(
                self._module.weight + self._u,
                self._num_bits,
                self._chunk_size
            )

    @classmethod
    def createConfig(cls, num_bits: int, chunk_size: int) -> Dict:
        from admm_ds.compression_configurations import TransformerADMMConfig

        return TransformerADMMConfig.buildEntry(
            cls,
            add_quantization_config({}, num_bits=num_bits, chunk_size=chunk_size)
        )

    @staticmethod
    def compression_ratio(compression_args: Dict) -> float:
        num_bits, _, _, _ = load_quantization_config(compression_args, logger)
        return float(num_bits + 2) / (2 * float(32))
